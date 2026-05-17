import os
import yaml
import torch
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from torchvision.transforms import Compose, Resize, ToTensor

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
BATCH_SIZE = 16


# ============================================================
# TRANSFORMS
# ============================================================

input_transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE), Image.NEAREST),
])


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

            print(f"[UNUSED] {original_name}")

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

    if len(mismatched) > 0:

        print("\n========== SHAPE MISMATCHES ==========\n")

        for item in mismatched:

            print(item["key"])
            print(f'checkpoint: {item["checkpoint"]}')
            print(f'model:      {item["model"]}')
            print()

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
        num_workers=4,
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

            print(f"\n================ STEP {step} ================\n")

            # ====================================================
            # BATCH DEBUG
            # ====================================================

            print("\nBATCH TYPE:")
            print(type(batch))

            print("\nBATCH LENGTH:")
            print(len(batch))

            for i, item in enumerate(batch):

                print(f"\nLEVEL 1 ITEM {i}")
                print(type(item))

                if isinstance(item, (list, tuple)):

                    print(f"LENGTH: {len(item)}")

                    for j, subitem in enumerate(item):

                        print(f"\n  LEVEL 2 ITEM {j}")
                        print(type(subitem))

                        if torch.is_tensor(subitem):

                            print(f"  SHAPE: {subitem.shape}")
                            print(f"  DTYPE: {subitem.dtype}")

                        elif isinstance(subitem, (list, tuple)):

                            print(f"  NESTED LENGTH: {len(subitem)}")

                            for k, deepitem in enumerate(subitem):

                                print(f"\n    LEVEL 3 ITEM {k}")
                                print(type(deepitem))

                                if torch.is_tensor(deepitem):

                                    print(f"    SHAPE: {deepitem.shape}")
                                    print(f"    DTYPE: {deepitem.dtype}")

            # ====================================================
            # SMART UNPACK
            # ====================================================

            def find_first_tensor(obj):

                if torch.is_tensor(obj):
                    return obj

                if isinstance(obj, (list, tuple)):

                    for item in obj:

                        result = find_first_tensor(item)

                        if result is not None:
                            return result

                return None


            images = find_first_tensor(batch[0])
            semantic_gt = find_first_tensor(batch[1])

            print("\nFOUND IMAGE TENSOR:")
            print(images.shape)

            print("\nFOUND GT TENSOR:")
            print(semantic_gt.shape)

            images = images.to(device)
            semantic_gt = semantic_gt.to(device)

            # ====================================================
            # FORWARD
            # ====================================================

            result = model(images)

            mask_logits = result[0][-1]
            class_logits = result[1][-1]

            print("\nmask_logits shape:")
            print(mask_logits.shape)

            print("class_logits shape:")
            print(class_logits.shape)

            print("\nmask logits min/max:")
            print(mask_logits.min().item())
            print(mask_logits.max().item())

            print("\nclass logits min/max:")
            print(class_logits.min().item())
            print(class_logits.max().item())

            print("\nNaN checks:")
            print(torch.isnan(mask_logits).any())
            print(torch.isnan(class_logits).any())

            # ====================================================
            # UPSAMPLE MASKS
            # ====================================================

            mask_logits = F.interpolate(
                mask_logits,
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False
            )

            print("\nUpsampled mask logits:")
            print(mask_logits.shape)

            # ====================================================
            # QUERY -> PIXEL CONVERSION
            # ====================================================

            mask_probs = torch.sigmoid(mask_logits)

            class_probs = torch.softmax(
                class_logits,
                dim=-1
            )

            print("\nclass_probs shape:")
            print(class_probs.shape)

            # REMOVE VOID / NO-OBJECT CLASS
            class_probs = class_probs[..., :-1]

            print("\nclass_probs without void:")
            print(class_probs.shape)

            Mat_Class = class_probs.transpose(1, 2)

            Mat_Mask = mask_probs.flatten(2)

            print("\nMat_Class shape:")
            print(Mat_Class.shape)

            print("Mat_Mask shape:")
            print(Mat_Mask.shape)

            pixel_logits = torch.matmul(
                Mat_Class,
                Mat_Mask
            )

            pixel_logits = pixel_logits.unflatten(
                2,
                (IMG_SIZE, IMG_SIZE)
            )

            print("\npixel_logits shape:")
            print(pixel_logits.shape)

            print("\npixel logits min/max:")
            print(pixel_logits.min().item())
            print(pixel_logits.max().item())

            print("\nPixel logits NaN:")
            print(torch.isnan(pixel_logits).any())

            # ====================================================
            # FINAL PREDICTION
            # ====================================================

            prediction = torch.argmax(
                pixel_logits,
                dim=1,
                keepdim=True
            )

            print("\nPrediction unique:")
            print(torch.unique(prediction))

            print("\nPrediction distribution:")

            uniq, counts = torch.unique(
                prediction,
                return_counts=True
            )

            for u, c in zip(uniq, counts):

                print(f"class {u.item()} -> {c.item()}")

            # ====================================================
            # IOU DEBUG
            # ====================================================

            print("\nIOU INPUT DEBUG")

            print("\nprediction shape:")
            print(prediction.shape)

            print("semantic_gt shape:")
            print(semantic_gt.shape)

            print("\nprediction dtype:")
            print(prediction.dtype)

            print("semantic_gt dtype:")
            print(semantic_gt.dtype)

            print("\nprediction min/max:")
            print(prediction.min())
            print(prediction.max())

            print("\nsemantic_gt min/max:")
            print(semantic_gt.min())
            print(semantic_gt.max())

            # ====================================================
            # IOU UPDATE
            # ====================================================

            iouEvalVal.addBatch(
                prediction.long(),
                semantic_gt.long()
            )

            print(f"\nStep {step} completed")

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