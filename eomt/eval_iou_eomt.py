import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor, Resize
from torchvision.transforms import InterpolationMode
import os
import time
import random

from argparse import ArgumentParser

from datasets.cityscapes_semantic import CityscapesSemantic

from models.eomt import EoMT
from models.vit import ViT

from iouEval import iouEval, getColorEntry

# =========================================================
# CONFIG
# =========================================================

NUM_CLASSES = 20

# =========================================================
# SEED
# =========================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =========================================================
# LOAD STATE DICT
# =========================================================

def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()

    loaded = 0
    missing = []
    mismatch = []

    for name, param in state_dict.items():

        original_name = name

        if name.startswith("network."):
            name = name.replace("network.", "")

        if name not in own_state:
            missing.append(original_name)
            continue

        if own_state[name].shape != param.shape:
            mismatch.append(
                (
                    original_name,
                    param.shape,
                    own_state[name].shape
                )
            )
            continue

        own_state[name].copy_(param)

        loaded += 1

    print("\n================ MODEL LOAD ================")
    print(f"Loaded params: {loaded}")
    print(f"Missing keys: {len(missing)}")
    print(f"Shape mismatches: {len(mismatch)}")

    if len(mismatch) > 0:

        print("\n--- SHAPE MISMATCHES ---")

        for name, ckpt_shape, model_shape in mismatch:

            print(
                f"{name}\n"
                f"  checkpoint: {ckpt_shape}\n"
                f"  model:      {model_shape}"
            )

    print("============================================\n")

    return model

# =========================================================
# MODEL
# =========================================================

def load_eomt(weightspath, device):

    print("Creating ViT backbone...")

    encoder = ViT(
        img_size=(1024, 1024),
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )

    print("Creating EoMT...")

    model = EoMT(
        encoder=encoder,
        num_classes=19,
        num_q=100,
        num_blocks=3,
        masked_attn_enabled=False,
    )

    print("\nLoading checkpoint...")
    print(weightspath)

    checkpoint = torch.load(
        weightspath,
        map_location="cpu"
    )

    print("\nCheckpoint type:")
    print(type(checkpoint))

    if isinstance(checkpoint, dict):

        print("\nCheckpoint keys:")
        print(checkpoint.keys())

    if "state_dict" in checkpoint:

        print("\nUsing checkpoint['state_dict']")
        checkpoint = checkpoint["state_dict"]

    elif "model" in checkpoint:

        print("\nUsing checkpoint['model']")
        checkpoint = checkpoint["model"]

    print("\nFirst 20 parameter keys:\n")

    for i, k in enumerate(checkpoint.keys()):

        print(k)

        if i >= 20:
            break

    model = load_my_state_dict(
        model,
        checkpoint
    )

    model = model.to(device)

    model.eval()

    return model

# =========================================================
# TARGET -> SEMANTIC
# =========================================================

def target_to_semantic(target):

    masks = target["masks"]
    labels = target["labels"]

    H, W = masks.shape[-2:]

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

def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print("\nDEVICE:", device)

    # =====================================================
    # MODEL
    # =====================================================

    print("\nLoading EoMT...")

    model = load_eomt(
        args.loadWeights,
        device
    )

    # =====================================================
    # DATASET
    # =====================================================

    print("\nCreating datamodule...")

    datamodule = CityscapesSemantic(
        path=args.datadir,
        batch_size=1,
        num_workers=args.num_workers,
        img_size=(1024, 1024),
    )

    datamodule.setup()

    loader = datamodule.val_dataloader()

    # =====================================================
    # RESIZE LIKE ERFNET
    # =====================================================

    image_resize = Resize(
        (1024, 1024),
        interpolation=InterpolationMode.BILINEAR
    )

    target_resize = Resize(
        (1024, 1024),
        interpolation=InterpolationMode.NEAREST
    )

    print(
        f"\nFound {len(datamodule.cityscapes_val_dataset)} validation images"
    )

    # =====================================================
    # IOU
    # =====================================================

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    # =====================================================
    # LOOP
    # =====================================================

    for step, batch in enumerate(loader):

        images, targets = batch

        image = images[0]

        # =====================================================
        # RESIZE IMAGE
        # =====================================================

        image = image_resize(image)

        if not args.cpu:
            image = image.cuda()

        target = targets[0]

        semantic_gt = target_to_semantic(target)

        semantic_gt = semantic_gt.unsqueeze(0)

        # =====================================================
        # RESIZE GT
        # =====================================================

        semantic_gt = target_resize(semantic_gt)

        semantic_gt = semantic_gt.long()

        semantic_gt = semantic_gt.unsqueeze(0).cpu()

        # =============================================
        # DEBUG GT
        # =============================================

        if step == 0:

            print("\n================ GT DEBUG ================")

            print("GT shape:")
            print(semantic_gt.shape)

            print("GT unique classes:")
            print(torch.unique(semantic_gt))

            print("==========================================")

        # =============================================
        # FORWARD
        # =============================================

        with torch.no_grad():

            outputs = model(image.unsqueeze(0))

        mask_logits = outputs[0][-1]
        class_logits = outputs[1][-1]

        if step == 0:

            print("\n================ OUTPUT DEBUG ================")

            print("mask_logits shape:")
            print(mask_logits.shape)

            print("class_logits shape:")
            print(class_logits.shape)

            print("\nmask logits stats:")
            print(mask_logits.min().item())
            print(mask_logits.max().item())

            print("\nclass logits stats:")
            print(class_logits.min().item())
            print(class_logits.max().item())

            print("==============================================")

        # =============================================
        # QUERY -> PIXEL
        # =============================================

        mask_probs = torch.sigmoid(mask_logits)

        class_probs = torch.softmax(
            class_logits,
            dim=-1
        )

        class_probs = class_probs.transpose(1, 2)

        mask_probs = torch.flatten(
            mask_probs,
            start_dim=2
        )

        pixel_logits = torch.matmul(
            class_probs,
            mask_probs
        )

        H = mask_logits.shape[-2]
        W = mask_logits.shape[-1]

        pixel_logits = pixel_logits.unflatten(
            2,
            (H, W)
        )

        # remove void
        pixel_logits = pixel_logits[:, :-1]

        prediction = pixel_logits.max(1)[1]

        prediction = prediction.unsqueeze(1).cpu()

        if step == 0:
            
            print("\n================ PRED DEBUG ================")

            print("Prediction shape:")
            print(prediction.shape)

            print("Prediction unique classes:")
            print(torch.unique(prediction))

            print("============================================")

        iouEvalVal.addBatch(
            prediction,
            semantic_gt
        )

        print(step)

    # =====================================================
    # RESULTS
    # =====================================================

    iouVal, iou_classes = iouEvalVal.getIoU()

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")

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

    for i in range(19):

        value = iou_classes[i].item()

        iouStr = (
            getColorEntry(value)
            + '{:0.2f}'.format(value * 100)
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

# =========================================================
# ENTRY
# =========================================================

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--loadWeights',
        required=True
    )

    parser.add_argument(
        '--datadir',
        required=True
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4
    )

    parser.add_argument(
        '--cpu',
        action='store_true'
    )

    main(parser.parse_args())