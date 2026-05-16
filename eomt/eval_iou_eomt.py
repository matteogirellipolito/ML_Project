import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode
from torchvision.transforms.functional import resize
import time
import random

from argparse import ArgumentParser

from datasets.cityscapes_semantic import CityscapesSemantic

from models.eomt import EoMT
from models.vit import ViT

from iouEval import iouEval, getColorEntry

# =========================================================
# SHARED LABEL SPACE
# =========================================================

IGNORE_INDEX = 255

SHARED_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic light",
]

CITYSCAPES_LABEL_TO_ID = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
}

COCO_LABEL_TO_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic light": 9,
}

SHARED_NAME_TO_ID = {
    name: idx for idx, name in enumerate(SHARED_CLASSES)
}

CITYSCAPES_TO_SHARED = {
    CITYSCAPES_LABEL_TO_ID["person"]: SHARED_NAME_TO_ID["person"],
    CITYSCAPES_LABEL_TO_ID["car"]: SHARED_NAME_TO_ID["car"],
    CITYSCAPES_LABEL_TO_ID["truck"]: SHARED_NAME_TO_ID["truck"],
    CITYSCAPES_LABEL_TO_ID["bus"]: SHARED_NAME_TO_ID["bus"],
    CITYSCAPES_LABEL_TO_ID["motorcycle"]: SHARED_NAME_TO_ID["motorcycle"],
    CITYSCAPES_LABEL_TO_ID["bicycle"]: SHARED_NAME_TO_ID["bicycle"],
    CITYSCAPES_LABEL_TO_ID["traffic light"]: SHARED_NAME_TO_ID["traffic light"],
}

COCO_TO_SHARED = {
    COCO_LABEL_TO_ID["person"]: SHARED_NAME_TO_ID["person"],
    COCO_LABEL_TO_ID["car"]: SHARED_NAME_TO_ID["car"],
    COCO_LABEL_TO_ID["truck"]: SHARED_NAME_TO_ID["truck"],
    COCO_LABEL_TO_ID["bus"]: SHARED_NAME_TO_ID["bus"],
    COCO_LABEL_TO_ID["motorcycle"]: SHARED_NAME_TO_ID["motorcycle"],
    COCO_LABEL_TO_ID["bicycle"]: SHARED_NAME_TO_ID["bicycle"],
    COCO_LABEL_TO_ID["traffic light"]: SHARED_NAME_TO_ID["traffic light"],
}

NUM_SHARED_CLASSES = len(SHARED_CLASSES)

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

    checkpoint_keys = set()

    for k in state_dict.keys():

        if k.startswith("network."):
            k = k.replace("network.", "")

        checkpoint_keys.add(k)

    model_keys = set(own_state.keys())

    real_missing_model_keys = sorted(
        list(model_keys - checkpoint_keys)
    )

    print("\n================ MODEL LOAD ================")

    print(f"Loaded params: {loaded}")
    print(f"Missing checkpoint keys: {len(missing)}")
    print(f"Missing model keys: {len(real_missing_model_keys)}")
    print(f"Shape mismatches: {len(mismatch)}")

    if len(missing) > 0:

        print("\n--- CHECKPOINT KEYS NOT USED ---")

        for k in missing:
            print(k)

    if len(real_missing_model_keys) > 0:

        print("\n--- MODEL KEYS MISSING FROM CHECKPOINT ---")

        for k in real_missing_model_keys:
            print(k)

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

def load_eomt(weightspath, device, num_blocks):

    print("Creating ViT backbone...")

    encoder = ViT(
        img_size=(640, 640),
        patch_size=14,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )

    print("Creating EoMT...")

    model = EoMT(
        encoder=encoder,
        num_classes=19,
        num_q=100,
        num_blocks=num_blocks,
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
# REMAP TARGET TO SHARED SPACE
# =========================================================

def remap_target_to_shared(target):

    shared = torch.ones_like(target) * IGNORE_INDEX

    for src_id, dst_id in CITYSCAPES_TO_SHARED.items():

        shared[target == src_id] = dst_id

    return shared

# =========================================================
# REMAP LOGITS TO SHARED SPACE
# =========================================================

def remap_logits_to_shared(logits):

    """
    logits shape:
    [19, H, W]
    """

    H, W = logits.shape[-2:]

    shared_logits = torch.zeros(
        (NUM_SHARED_CLASSES, H, W),
        device=logits.device
    )

    for src_id, dst_id in CITYSCAPES_TO_SHARED.items():

        shared_logits[dst_id] += logits[src_id]

    return shared_logits

# =========================================================
# MAIN
# =========================================================

def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print("\nDEVICE:", device)

    print("\nLoading EoMT...")

    model = load_eomt(
        args.loadWeights,
        device,
        args.num_blocks
    )

    print("\nCreating datamodule...")

    datamodule = CityscapesSemantic(
        path=args.datadir,
        batch_size=1,
        num_workers=args.num_workers,
        img_size=(640, 640),
    )

    datamodule.setup()

    loader = datamodule.val_dataloader()

    image_resize = Resize(
        (640, 640),
        interpolation=InterpolationMode.BILINEAR
    )

    target_resize = Resize(
        (640, 640),
        interpolation=InterpolationMode.NEAREST
    )

    print(
        f"\nFound {len(datamodule.cityscapes_val_dataset)} validation images"
    )

    iouEvalVal = iouEval(NUM_SHARED_CLASSES)

    start = time.time()

    # =====================================================
    # LOOP
    # =====================================================

    for step, batch in enumerate(loader):

        images, targets = batch

        image = images[0]

        image = image_resize(image)

        if not args.cpu:
            image = image.cuda()

        target = targets[0]

        semantic_gt = target_to_semantic(target)

        semantic_gt = semantic_gt.unsqueeze(0)

        semantic_gt = target_resize(semantic_gt)

        semantic_gt = semantic_gt.long()

        # =====================================================
        # REMAP GT TO SHARED LABEL SPACE
        # =====================================================

        semantic_gt = remap_target_to_shared(
            semantic_gt
        )

        semantic_gt = semantic_gt.unsqueeze(0).cpu()

        if step == 0:

            print("\n================ GT DEBUG ================")

            print("GT shape:")
            print(semantic_gt.shape)

            print("GT unique classes:")
            print(torch.unique(semantic_gt))

            print("==========================================")

        # =====================================================
        # FORWARD
        # =====================================================

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

        # remove void
        pixel_logits = pixel_logits[:, :-1]

        # =====================================================
        # RESIZE PIXEL LOGITS TO GT SIZE
        # =====================================================

        pixel_logits = F.interpolate(
            pixel_logits,
            size=semantic_gt.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # =====================================================
        # REMAP TO SHARED SPACE
        # =====================================================

        shared_logits = remap_logits_to_shared(
            pixel_logits[0]
        )

        prediction = shared_logits.argmax(0)

        prediction = prediction.unsqueeze(0).unsqueeze(0)

        prediction = prediction.long().cpu()

        if step == 0:

            print("\n================ FINAL DEBUG ================")

            print("Prediction shape:")
            print(prediction.shape)

            print("GT shape:")
            print(semantic_gt.shape)

            print("\nPrediction unique:")
            print(torch.unique(prediction))

            print("\nGT unique:")
            print(torch.unique(semantic_gt))

            print("=============================================")

        # =====================================================
        # IGNORE PIXELS
        # =====================================================

        valid_mask = semantic_gt != IGNORE_INDEX

        pred_valid = prediction[valid_mask]
        gt_valid = semantic_gt[valid_mask]

        iouEvalVal.addBatch(
            pred_valid.unsqueeze(0),
            gt_valid.unsqueeze(0)
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

    for i in range(NUM_SHARED_CLASSES):

        value = iou_classes[i].item()

        iouStr = (
            getColorEntry(value)
            + '{:0.2f}'.format(value * 100)
            + '\033[0m'
        )

        print(f"{SHARED_CLASSES[i]}: {iouStr}")

    print("=======================================")

    iouStr = (
        getColorEntry(iouVal.item())
        + '{:0.2f}'.format(iouVal.item() * 100)
        + '\033[0m'
    )

    print("SHARED MEAN IoU:", iouStr, "%")

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

    parser.add_argument(
        '--num-blocks',
        type=int,
        default=3
    )

    main(parser.parse_args())