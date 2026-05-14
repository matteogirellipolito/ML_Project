import random
import numpy as np
import torch
import torch.nn.functional as F

from argparse import ArgumentParser

from datasets.cityscapes_semantic import CityscapesSemantic

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

# 19 semantic classes for Cityscapes
# ignore label = 19
NUM_CLASSES = 19

# =========================================================
# CHECKPOINT
# =========================================================

def extract_state_dict(checkpoint):

    print(checkpoint.keys())

    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]

    elif "model" in checkpoint:
        sd = checkpoint["model"]

    else:
        sd = checkpoint

    print("NUM KEYS:", len(sd))

    for k in list(sd.keys())[:50]:
        print(k)

    return sd


def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()

    loaded = 0

    missing_keys = []

    shape_mismatch = []

    for name, param in state_dict.items():

        original_name = name

        if name.startswith("network."):
            name = name.replace("network.", "")

        if name not in own_state:

            missing_keys.append(original_name)

            continue

        if own_state[name].shape != param.shape:

            shape_mismatch.append(
                (
                    original_name,
                    param.shape,
                    own_state[name].shape
                )
            )

            continue

        own_state[name].copy_(param)

        loaded += 1

    print(f"\nLoaded parameters: {loaded}")

    print(f"\nMissing keys: {len(missing_keys)}")

    for k in missing_keys:
        print("MISSING:", k)

    print(f"\nShape mismatches: {len(shape_mismatch)}")

    for k, ckpt_shape, model_shape in shape_mismatch:

        print(
            f"SHAPE MISMATCH: {k} | "
            f"ckpt={ckpt_shape} model={model_shape}"
        )

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
# TARGET CONVERSION
# =========================================================

def target_to_semantic(target):

    masks = target["masks"]

    labels = target["labels"]

    H, W = masks.shape[-2:]

    # ignore label = 19
    semantic = torch.ones(
        (H, W),
        dtype=torch.long
    ) * 19

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

    # =====================================================
    # MODEL
    # =====================================================

    print("Loading EoMT...")

    model = load_eomt(
        args.checkpoint,
        device
    )

    # =====================================================
    # DATASET
    # =====================================================

    datamodule = CityscapesSemantic(
        path=args.data_path,
        batch_size=1,
        num_workers=2,
        img_size=(640, 640),
    )

    datamodule.setup()

    loader = datamodule.val_dataloader()

    print(
        f"Found {len(datamodule.cityscapes_val_dataset)} validation images"
    )

    # =====================================================
    # IOU
    # =====================================================

    iouEvalVal = iouEval(
        NUM_CLASSES,
        ignoreIndex=19
    )

    # =====================================================
    # LOOP
    # =====================================================

    for step, batch in enumerate(loader):

        print(step)

        images, targets = batch

        image = images[0].unsqueeze(0)

        image = F.interpolate(
            image,
            size=(640, 640),
            mode="bilinear",
            align_corners=False
        )

        image = image.to(device)

        target = targets[0]

        # =================================================
        # TARGET
        # =================================================

        semantic_gt = target_to_semantic(target)

        semantic_gt = (
            semantic_gt
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )

        semantic_gt = F.interpolate(
            semantic_gt,
            size=(640, 640),
            mode="nearest"
        )

        semantic_gt = semantic_gt.long().cpu()

        # =================================================
        # FORWARD
        # =================================================

        with torch.no_grad():

            result = model(image)

        mask_logits = result[0][-1]

        class_logits = result[1][-1]

        # =================================================
        # UPSAMPLE MASKS
        # =================================================

        mask_logits = F.interpolate(
            mask_logits,
            size=semantic_gt.shape[-2:],
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

        H = semantic_gt.shape[-2]

        W = semantic_gt.shape[-1]

        pixel_logits = pixel_logits.unflatten(
            2,
            (H, W)
        )

        pixel_logits = pixel_logits.squeeze(0)

        # =================================================
        # PREDICTION
        # =================================================

        prediction = torch.argmax(
            pixel_logits,
            dim=0
        )

        prediction = (
            prediction
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
        )

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

    class_names = [
        "Road",
        "Sidewalk",
        "Building",
        "Wall",
        "Fence",
        "Pole",
        "Traffic Light",
        "Traffic Sign",
        "Vegetation",
        "Terrain",
        "Sky",
        "Person",
        "Rider",
        "Car",
        "Truck",
        "Bus",
        "Train",
        "Motorcycle",
        "Bicycle"
    ]

    for i in range(iou_classes.size(0)):

        iouStr = (
            getColorEntry(iou_classes[i].item())
            + '{:0.2f}'.format(iou_classes[i] * 100)
            + '\033[0m'
        )

        print(f"{class_names[i]}: {iouStr}")

    print("=======================================")

    iouStr = (
        getColorEntry(iouVal.item())
        + '{:0.2f}'.format(iouVal.item() * 100)
        + '\033[0m'
    )

    print("MEAN IoU:", iouStr, "%")


if __name__ == "__main__":
    main()
