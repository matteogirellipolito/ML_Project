import os
import glob
import random
import torch
import numpy as np

from PIL import Image
from argparse import ArgumentParser

import torch.nn.functional as F

from torchvision.transforms import Compose, Resize, ToTensor

from sklearn.metrics import (
    average_precision_score,
    roc_curve,
)

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
    Resize((256, 512), Image.NEAREST),
])

# =========================================================
# METRICS
# =========================================================

def compute_fpr95(labels, scores):

    fpr, tpr, thresholds = roc_curve(labels, scores)

    idx = np.argmin(np.abs(tpr - 0.95))

    return fpr[idx]


# =========================================================
# CHECKPOINT
# =========================================================

def extract_state_dict(checkpoint):

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()

    loaded = 0

    for name, param in state_dict.items():

        if name.startswith("network."):
            name = name.replace("network.", "")

        if name in own_state:

            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1

    print(f"Loaded {loaded} parameters")

    return model


# =========================================================
# MODEL
# =========================================================

def load_eomt(checkpoint_path, device):

    print("Loading EoMT...")

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
        help="Glob path"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_eomt(args.checkpoint, device)

    image_paths = sorted(glob.glob(args.input))

    print(f"Found {len(image_paths)} images")

    all_labels = []

    all_rba_scores = []

    for path in image_paths:

        print(path)

        # =====================================================
        # IMAGE
        # =====================================================

        image = Image.open(path).convert("RGB")

        image_tensor = input_transform(image).unsqueeze(0).to(device)

        # =====================================================
        # FORWARD
        # =====================================================

        with torch.no_grad():

            result = model(image_tensor)

        mask_logits = result[0][-1]

        class_logits = result[1][-1]

        # =====================================================
        # UPSAMPLE
        # =====================================================

        H = 256
        W = 512

        mask_logits = F.interpolate(
            mask_logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        # =====================================================
        # PIXEL LOGITS
        # =====================================================

        mask_probs = torch.sigmoid(mask_logits)

        class_probs = torch.softmax(class_logits, dim=-1)

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

        # =====================================================
        # RBA SCORE
        # =====================================================

        anomaly_score = -torch.sum(
            torch.tanh(pixel_logits.cpu()),
            dim=0
        )

        anomaly_score = anomaly_score.numpy()

        print(
            "Anomaly score stats:",
            anomaly_score.min(),
            anomaly_score.max()
        )

        # =====================================================
        # GT
        # =====================================================

        gt_path = path.replace(
            "images",
            "labels_masks"
        )

        gt_path = gt_path.replace(".jpg", ".png")
        gt_path = gt_path.replace(".jpeg", ".png")
        gt_path = gt_path.replace(".webp", ".png")

        gt = Image.open(gt_path)

        gt = target_transform(gt)

        gt = np.array(gt)

        # =====================================================
        # DATASET-SPECIFIC LABEL FIX
        # =====================================================

        if "RoadAnomaly" in gt_path:
            gt = np.where(gt == 2, 1, gt)

        if "LostAndFound" in gt_path:
            gt = np.where(gt == 0, 255, gt)
            gt = np.where(gt == 1, 0, gt)
            gt = np.where((gt > 1) & (gt < 201), 1, gt)

        if "Streethazard" in gt_path:
            gt = np.where(gt == 14, 255, gt)
            gt = np.where(gt < 20, 0, gt)
            gt = np.where(gt == 255, 1, gt)

        # =====================================================
        # VALID PIXELS
        # =====================================================

        valid_mask = gt != 255

        labels = gt[valid_mask]

        scores = anomaly_score[valid_mask]

        all_labels.append(labels.flatten())

        all_rba_scores.append(scores.flatten())

        del result
        torch.cuda.empty_cache()

    # =========================================================
    # CONCAT
    # =========================================================

    all_labels = np.concatenate(all_labels)

    all_rba_scores = np.concatenate(all_rba_scores)

    # =========================================================
    # METRICS
    # =========================================================

    auprc = average_precision_score(
        all_labels,
        all_rba_scores
    )

    fpr95 = compute_fpr95(
        all_labels,
        all_rba_scores
    )

    # =========================================================
    # RESULTS
    # =========================================================

    print("\n==============================")
    print("RBA RESULTS")
    print("==============================")

    print(f"AuPRC   : {auprc * 100:.2f}")
    print(f"FPR95   : {fpr95 * 100:.2f}")

    print("==============================\n")


if __name__ == "__main__":
    main()