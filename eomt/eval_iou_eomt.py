import os
import yaml
import torch
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

from models.eomt import EoMT
from models.vit import ViT

from datasets.cityscapes_semantic import CityscapesSemantic
from iouEval import iouEval


# ============================================================
# SEED
# ============================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# ============================================================
# CONSTANTS
# ============================================================

NUM_CLASSES = 19
IGNORE_INDEX = 255

IMG_SIZE = 1024
BATCH_SIZE = 1


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def extract_state_dict(checkpoint):

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()

    loaded = []
    missing = []
    mismatched = []
    unused = []

    print("\n================ CHECKPOINT DEBUG ================\n")

    for name, param in state_dict.items():

        original_name = name

        if name.startswith("network."):
            name = name.replace("network.", "")

        if name not in own_state:

            unused.append(original_name)
            continue

        if own_state[name].shape != param.shape:

            mismatched.append({
                "key": name,
                "checkpoint": tuple(param.shape),
                "model": tuple(own_state[name].shape)
            })

            print(f"[SHAPE MISMATCH] {name}")
            print(f"checkpoint: {tuple(param.shape)}")
            print(f"model:      {tuple(own_state[name].shape)}")
            print()

            continue

        own_state[name].copy_(param)

        loaded.append(name)

    for name in own_state.keys():

        checkpoint_name = f"network.{name}"

        if checkpoint_name not in state_dict and name not in state_dict:

            missing.append(name)

    print("\n================ SUMMARY ================\n")

    print(f"Loaded params: {len(loaded)}")
    print(f"Unused checkpoint keys: {len(unused)}")
    print(f"Missing model keys: {len(missing)}")
    print(f"Shape mismatches: {len(mismatched)}")

    print("\n=========================================\n")

    if len(unused) > 0:

        print("\n========== UNUSED KEYS ==========\n")

        for k in unused:
            print(k)

    if len(missing) > 0:

        print("\n========== MISSING MODEL KEYS ==========\n")

        for k in missing:
            print(k)

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_eomt(args, device):

    print("Creating ViT backbone...")

    encoder = ViT(
        img_size=(IMG_SIZE, IMG_SIZE),
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )

    print("Creating EoMT...")

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100,
        num_blocks=3,
        masked_attn_enabled=True,
    ).to(device)

    print("\nLoading checkpoint...")
    print(args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )

    checkpoint = extract_state_dict(checkpoint)

    model = load_my_state_dict(model, checkpoint)

    model.eval()

    return model


# ============================================================
# MAIN
# ============================================================

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDEVICE: {device}")

    # ========================================================
    # MODEL
    # ========================================================

    model = load_eomt(args, device)

    # ========================================================
    # DATA
    # ========================================================

    print("\nCreating datamodule...\n")

    datamodule = CityscapesSemantic(
        path=args.data_dir,
        batch_size=BATCH_SIZE,
        num_workers=2,
        img_size=IMG_SIZE,
    )

    datamodule.setup()

    val_loader = datamodule.val_dataloader()

    print(f"Found {len(val_loader.dataset)} validation images")

    # ========================================================
    # IOU
    # ========================================================

    iouEvalVal = iouEval(NUM_CLASSES, IGNORE_INDEX)

    # ========================================================
    # LOOP
    # ========================================================

    with torch.no_grad():

        for step, batch in enumerate(tqdm(val_loader)):

            images_tuple = batch[0]
            targets_tuple = batch[1]

            # ====================================================
            # STACK IMAGES
            # ====================================================

            images = torch.stack([
                img.float()
                for img in images_tuple
            ], dim=0)

            # ====================================================
            # BUILD SEMANTIC GT
            # ====================================================

            semantic_gt_list = []

            for target in targets_tuple:

                masks = target["masks"]
                labels = target["labels"]

                H, W = masks.shape[-2:]

                semantic_mask = torch.zeros(
                    (H, W),
                    dtype=torch.long
                )

                for instance_idx in range(len(labels)):

                    class_id = labels[instance_idx].item()

                    instance_mask = masks[instance_idx]

                    semantic_mask[instance_mask] = class_id

                semantic_gt_list.append(semantic_mask)

            semantic_gt = torch.stack(
                semantic_gt_list,
                dim=0
            )

            semantic_gt = semantic_gt.unsqueeze(1)

            # ====================================================
            # NORMALIZE
            # ====================================================

            images = images.float() / 255.0

            # ====================================================
            # RESIZE
            # ====================================================

            images = resize(
                images,
                size=[1024, 1024],
                interpolation=InterpolationMode.BILINEAR,
            )

            semantic_gt = resize(
                semantic_gt.float(),
                size=[1024, 1024],
                interpolation=InterpolationMode.NEAREST,
            ).long()

            # ====================================================
            # MOVE TO DEVICE
            # ====================================================

            images = images.to(device)

            semantic_gt = semantic_gt.to(device)

            # ====================================================
            # FORWARD
            # ====================================================

            result = model(images)

            mask_logits = result[0][-1]
            class_logits = result[1][-1]

            # ====================================================
            # UPSAMPLE MASKS
            # ====================================================

            mask_logits = F.interpolate(
                mask_logits,
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False
            )

            # ====================================================
            # QUERY -> PIXEL CONVERSION
            # ====================================================

            # remove no-object class
            class_logits = class_logits[..., :-1]

            pixel_logits = torch.einsum(
                "bqc,bqhw->bchw",
                class_logits,
                mask_logits
            )

            # ====================================================
            # PREDICTION
            # ====================================================

            prediction = torch.argmax(
                pixel_logits,
                dim=1,
                keepdim=True
            )

            # ====================================================
            # DEBUG
            # ====================================================

            if step % 20 == 0:

                print(f"\n================ STEP {step} ================\n")

                print("images shape:")
                print(images.shape)

                print("\nsemantic_gt unique:")
                print(torch.unique(semantic_gt))

                print("\nmask_logits shape:")
                print(mask_logits.shape)

                print("\nclass_logits shape:")
                print(class_logits.shape)

                print("\npixel_logits shape:")
                print(pixel_logits.shape)

                print("\npixel logits min/max:")
                print(pixel_logits.min().item())
                print(pixel_logits.max().item())

                print("\nPrediction unique:")
                print(torch.unique(prediction))

                print("\nPrediction distribution:")

                uniq, counts = torch.unique(
                    prediction,
                    return_counts=True
                )

                for u, c in zip(uniq, counts):

                    print(f"class {u.item()} -> {c.item()}")

                print("\nCUDA MEMORY:")
                print(
                    torch.cuda.memory_allocated() / 1024**3,
                    "GB"
                )

            # ====================================================
            # IOU UPDATE
            # ====================================================

            iouEvalVal.addBatch(
                prediction.long(),
                semantic_gt.long()
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    mean_iou, per_class_iou = iouEvalVal.getIoU()

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

    print("\n=======================================")
    print("Per-Class IoU:\n")

    for i, name in enumerate(class_names):

        print(f"{name}: {per_class_iou[i] * 100:.2f}")

    print("\n=======================================")

    print(f"MEAN IoU: {mean_iou * 100:.2f}")

    print("=======================================\n")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )

    args = parser.parse_args()

    main(args)