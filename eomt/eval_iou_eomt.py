import numpy as np
import torch
import torch.nn.functional as F
import random
import time

from argparse import ArgumentParser
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from datasets.cityscapes_semantic import CityscapesSemantic

from models.eomt import EoMT
from models.vit import ViT

from iouEval import iouEval, getColorEntry

# =========================================================
# CONFIG
# =========================================================

NUM_CLASSES = 20
IMG_SIZE = (1024, 1024)

# =========================================================
# SEED
# =========================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =========================================================
# LOAD CHECKPOINT
# =========================================================

def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()

    loaded = 0

    print("\n================ LOAD DEBUG ================\n")

    for name, param in state_dict.items():

        original_name = name

        if name.startswith("network."):
            name = name.replace("network.", "")

        if "criterion.empty_weight" in name:
            continue

        if name not in own_state:
            print(f"[NOT FOUND] {original_name}")
            continue

        if own_state[name].shape != param.shape:

            print(f"\n[SHAPE MISMATCH]")
            print(name)
            print(f"ckpt : {param.shape}")
            print(f"model: {own_state[name].shape}")

            continue

        own_state[name].copy_(param)
        loaded += 1

    print(f"\nLoaded params: {loaded}")
    print("\n============================================\n")

    return model

# =========================================================
# MODEL
# =========================================================

def load_eomt(weightspath, device):

    print("Creating ViT backbone...")

    encoder = ViT(
        img_size=IMG_SIZE,
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

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

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
        batch_size=16,
        num_workers=args.num_workers,
        img_size=IMG_SIZE,
    )

    datamodule.setup()

    loader = datamodule.val_dataloader()

    print(
        f"\nFound {len(datamodule.cityscapes_val_dataset)} validation images"
    )

    # =====================================================
    # RESIZE
    # =====================================================

    image_resize = Resize(
        IMG_SIZE,
        interpolation=InterpolationMode.BILINEAR
    )

    target_resize = Resize(
        IMG_SIZE,
        interpolation=InterpolationMode.NEAREST
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

        print(f"\n================ STEP {step} ================\n")

        image = images[0]

        image = image_resize(image)

        if not args.cpu:
            image = image.cuda()

        target = targets[0]

        semantic_gt = target_to_semantic(target)

        semantic_gt = semantic_gt.unsqueeze(0).float()

        semantic_gt = target_resize(semantic_gt)

        semantic_gt = semantic_gt.long()

        semantic_gt = semantic_gt.unsqueeze(0).cpu()

        print("GT unique:")
        print(torch.unique(semantic_gt))

        # =====================================================
        # FORWARD
        # =====================================================

        with torch.no_grad():

            outputs = model(image.unsqueeze(0))

        mask_logits = outputs[0][-1]
        class_logits = outputs[1][-1]

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

        # =====================================================
        # NAN CHECK
        # =====================================================

        print("\nNaN checks:")

        print(torch.isnan(mask_logits).any())
        print(torch.isnan(class_logits).any())

        # =====================================================
        # QUERY -> PIXEL
        # =====================================================

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

        pixel_logits = pixel_logits[:, :-1]

        prediction = pixel_logits.max(1)[1]

        prediction = prediction.unsqueeze(1).float()

        prediction = resize(
            prediction,
            size=semantic_gt.shape[-2:],
            interpolation=InterpolationMode.NEAREST
        )

        prediction = prediction.long().cpu()

        print("\nPrediction unique:")
        print(torch.unique(prediction))

        # =====================================================
        # PIXEL STATS
        # =====================================================

        unique, counts = torch.unique(
            prediction,
            return_counts=True
        )

        print("\nPrediction distribution:")

        for u, c in zip(unique, counts):

            print(
                f"class {u.item()} -> {c.item()}"
            )

        # =====================================================
        # IOU
        # =====================================================

        try:

            iouEvalVal.addBatch(
                prediction.squeeze(1),
                semantic_gt.squeeze(1)
            )

        except Exception as e:

            print("\nIOU ERROR:")
            print(e)

            print("\nPrediction shape:")
            print(prediction.shape)

            print("\nGT shape:")
            print(semantic_gt.shape)

            raise e

        print(f"\nStep {step} completed")

    # =====================================================
    # RESULTS
    # =====================================================

    iouVal, iou_classes = iouEvalVal.getIoU()

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")

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

    print("\nPer-Class IoU:\n")

    for i in range(19):

        value = iou_classes[i].item()

        iouStr = (
            getColorEntry(value)
            + '{:0.2f}'.format(value * 100)
            + '\033[0m'
        )

        print(f"{class_names[i]}: {iouStr}")

    print("\n=======================================")

    iouStr = (
        getColorEntry(iouVal.item())
        + '{:0.2f}'.format(iouVal.item() * 100)
        + '\033[0m'
    )

    print("MEAN IoU:", iouStr)

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
        default=2
    )

    parser.add_argument(
        '--cpu',
        action='store_true'
    )

    main(parser.parse_args())