import os
import glob
import torch
import random
import numpy as np

from PIL import Image
from argparse import ArgumentParser

import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

from ood_metrics import fpr_at_95_tpr

from models.eomt import EoMT
from models.vit import ViT

# =========================================================
# SEED
# =========================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =========================================================
# CONFIG
# =========================================================

NUM_CLASSES = 20

input_transform = Compose([
    Resize((640, 640), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

# =========================================================
# LOAD CHECKPOINT
# =========================================================

def extract_state_dict(checkpoint):

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()

    for name, param in state_dict.items():

        if name.startswith("network."):
            name = name.replace("network.", "")

        if name not in own_state:

            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)

            else:
                print(name, "not loaded")
                continue

        else:

            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)

    return model


# =========================================================
# LOAD MODEL
# =========================================================

def load_eomt(checkpoint_path, device):

    encoder = ViT(
        img_size=(640, 640),
        patch_size=14,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100,
        num_blocks=3,
        masked_attn_enabled=False,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    checkpoint = extract_state_dict(checkpoint)

    model = load_my_state_dict(model, checkpoint)

    model.eval()

    return model


# =========================================================
# MAIN
# =========================================================

def main():

    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Input image glob"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path"
    )

    args = parser.parse_args()

    anomaly_score_list = []

    ood_gts_list = []

    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()

    file = open("results.txt", "a")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = load_eomt(args.checkpoint, device)

    # =====================================================
    # LOOP
    # =====================================================

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):

        print(path)

        images = input_transform(
            Image.open(path).convert("RGB")
        ).unsqueeze(0).float().to(device)

        # =================================================
        # FORWARD
        # =================================================

        with torch.no_grad():

            result = model(images)

        mask_logits = result[0][-1]

        class_logits = result[1][-1]

        # =================================================
        # UPSAMPLE
        # =================================================

        H = 512
        W = 1024

        mask_logits = F.interpolate(
            mask_logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        # =================================================
        # PIXEL LOGITS
        # =================================================

        mask_probs = torch.sigmoid(mask_logits)

        class_probs = torch.softmax(
            class_logits,
            dim=-1
        )

        Mat_Class = class_probs.transpose(1, 2)

        Mat_Mask = torch.flatten(
            mask_probs,
            start_dim=2
        )

        pixel_logits = torch.matmul(
            Mat_Class,
            Mat_Mask
        )

        pixel_logits = pixel_logits.unflatten(
            2,
            (H, W)
        )

        pixel_logits = pixel_logits.squeeze(0)

        # =================================================
        # RBA SCORE
        # =================================================

        anomaly_result = -torch.sum(
            torch.tanh(pixel_logits.cpu()),
            dim=0
        ).numpy()

        # =================================================
        # GT PATH
        # =================================================

        pathGT = path.replace(
            "images",
            "labels_masks"
        )

        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")

        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        # =================================================
        # GT
        # =================================================

        mask = Image.open(pathGT)

        mask = target_transform(mask)

        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)

        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where(
                (ood_gts > 1) & (ood_gts < 201),
                1,
                ood_gts
            )

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # =================================================
        # STORE
        # =================================================

        if 1 not in np.unique(ood_gts):

            continue

        else:

            ood_gts_list.append(ood_gts)

            anomaly_score_list.append(anomaly_result)

        del result
        del anomaly_result
        del ood_gts
        del mask

        torch.cuda.empty_cache()

    # =====================================================
    # METRICS
    # =====================================================

    file.write("\n")

    ood_gts = np.array(ood_gts_list)

    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)

    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]

    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))

    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))

    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(
        val_label,
        val_out
    )

    fpr = fpr_at_95_tpr(
        val_out,
        val_label
    )

    # =====================================================
    # RESULTS
    # =====================================================

    print(f'AUPRC score: {prc_auc * 100.0}')

    print(f'FPR@TPR95: {fpr * 100.0}')

    file.write(
        '    AUPRC score:' + str(prc_auc * 100.0) +
        '   FPR@TPR95:' + str(fpr * 100.0)
    )

    file.close()


if __name__ == '__main__':

    main()