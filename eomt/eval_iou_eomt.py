import random
import numpy as np
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from datasets.cityscapes_semantic import CityscapesSemantic
from datasets.transforms import Transforms

from models.eomt import EoMT
from models.vit import ViT

from iouEval import iouEval, getColorEntry

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

    encoder = ViT(
        img_size=(640, 640),
        patch_size=14,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )

    model = EoMT(
        encoder=encoder,
        num_classes=20,
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
# TARGET CONVERSION
# =========================================================

def target_to_semantic(target):

    masks = target["masks"]
    labels = target["labels"]

    H, W = masks.shape[-2:]

    semantic = torch.ones((H, W), dtype=torch.long) * 19

    for mask, cls in zip(masks, labels):

        semantic[mask.bool()] = cls

    return semantic


# =========================================================
# MAIN
# =========================================================

def main():

    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading EoMT...")

    model = load_eomt(args.checkpoint, device)

    # =====================================================
    # DATASET
    # =====================================================

    transforms = Transforms(
        img_size=(640, 640),
        color_jitter_enabled=False,
        scale_range=(1.0, 1.0),
    )

    dataset = CityscapesSemantic(
        path=args.data_path,
        split="val",
        transforms=transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    print(f"Found {len(dataset)} validation images")

    # =====================================================
    # IOU
    # =====================================================

    iouEvalVal = iouEval(NUM_CLASSES)

    # =====================================================
    # LOOP
    # =====================================================

    for step, (image, target) in enumerate(loader):

        print(step)

        image = image.to(device)

        # =================================================
        # TARGET
        # =================================================

        semantic_gt = target_to_semantic(target[0])

        semantic_gt = semantic_gt.unsqueeze(0).unsqueeze(0)

        # =================================================
        # FORWARD
        # =================================================

        with torch.no_grad():

            result = model(image)

        mask_logits = result[0][-1]

        class_logits = result[1][-1]

        # =================================================
        # PIXEL LOGITS
        # =================================================

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

        H = mask_logits.shape[-2]
        W = mask_logits.shape[-1]

        pixel_logits = pixel_logits.unflatten(
            2,
            (H, W)
        )

        pixel_logits = pixel_logits.squeeze(0)

        # remove void class
        pixel_logits = pixel_logits[:-1]

        prediction = torch.argmax(
            pixel_logits,
            dim=0
        )

        prediction = prediction.unsqueeze(0).unsqueeze(0).cpu()

        # =================================================
        # IOU
        # =================================================

        iouEvalVal.addBatch(
            prediction,
            semantic_gt
        )

    # =====================================================
    # RESULTS
    # =====================================================

    iouVal, iou_classes = iouEvalVal.getIoU()

    print("---------------------------------------")
    print("Per-Class IoU:")

    for i in range(iou_classes.size(0)):

        iouStr = getColorEntry(
            iou_classes[i]
        ) + '{:0.2f}'.format(
            iou_classes[i] * 100
        ) + '\033[0m'

        print(f"Class {i}: {iouStr}")

    print("=======================================")

    iouStr = getColorEntry(
        iouVal.item()
    ) + '{:0.2f}'.format(
        iouVal.item() * 100
    ) + '\033[0m'

    print("MEAN IoU: ", iouStr, "%")


if __name__ == "__main__":
    main()