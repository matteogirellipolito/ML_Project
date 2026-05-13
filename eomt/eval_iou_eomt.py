import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from datasets.cityscapes_semantic import CityscapesSemantic
from datasets.transforms import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

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

input_transform_cityscapes = Compose([
    Resize((640, 640), Image.BILINEAR),
    ToTensor(),
])

target_transform_cityscapes = Compose([
    Resize((512, 1024), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),  # ignore label
])

# =========================================================
# CHECKPOINT UTILS
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

def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("Loading EoMT...")

    model = load_eomt(args.checkpoint, device)

    # =====================================================
    # DATASET
    # =====================================================

    loader = DataLoader(
        CityscapesSemantic(
            args.datadir,
            input_transform_cityscapes,
            target_transform_cityscapes,
            subset=args.subset
        ),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    # =====================================================
    # IOU
    # =====================================================

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    # =====================================================
    # LOOP
    # =====================================================

    for step, (images, labels, filename, filenameGt) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            result = model(images)

        mask_logits = result[0][-1]
        class_logits = result[1][-1]

        # =================================================
        # UPSAMPLE MASKS
        # =================================================

        H = labels.shape[2]
        W = labels.shape[3]

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

        # remove batch dimension
        pixel_logits = pixel_logits.squeeze(0)

        # =================================================
        # REMOVE VOID CLASS
        # =================================================

        pixel_logits = pixel_logits[:-1]

        # =================================================
        # PREDICTION
        # =================================================

        prediction = pixel_logits.argmax(0).unsqueeze(0)

        iouEvalVal.addBatch(
            prediction.data,
            labels
        )

        filenameSave = filename[0].split("leftImg8bit/")[1]

        print(step, filenameSave)

    # =====================================================
    # RESULTS
    # =====================================================

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []

    for i in range(iou_classes.size(0)):

        iouStr = (
            getColorEntry(iou_classes[i])
            + '{:0.2f}'.format(iou_classes[i] * 100)
            + '\033[0m'
        )

        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time() - start, "seconds")
    print("=======================================")

    print("Per-Class IoU:")

    class_names = [
        "Road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    for i in range(19):
        print(iou_classes_str[i], class_names[i])

    print("=======================================")

    iouStr = (
        getColorEntry(iouVal)
        + '{:0.2f}'.format(iouVal * 100)
        + '\033[0m'
    )

    print("MEAN IoU: ", iouStr, "%")


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--checkpoint',
        required=True
    )

    parser.add_argument(
        '--subset',
        default="val"
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
        '--batch-size',
        type=int,
        default=1
    )

    parser.add_argument(
        '--cpu',
        action='store_true'
    )

    main(parser.parse_args())