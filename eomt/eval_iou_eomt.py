import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import time

from PIL import Image
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# Aggiungi la cartella eomt al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eomt'))

from models.eomt import EoMT
from models.vit import ViT
from dataset import cityscapes
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20  # 19 classi Cityscapes + 1 ignore

# EoMT vuole input quadrato 640x640
input_transform_cityscapes = Compose([
    Resize((640, 640), Image.BILINEAR),
    ToTensor(),
])

# Le label le teniamo a risoluzione 512 (altezza), ratio preservato → 512x1024
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   # pixel ignore → classe 19
])


# =========================================================
# CARICAMENTO CHECKPOINT
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
    print(f"Loaded {loaded}/{len(own_state)} parameters")
    return model


# =========================================================
# MAIN
# =========================================================

def main(args):

    device = torch.device("cuda" if not args.cpu else "cpu")

    print("Loading EoMT model...")

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

    checkpoint = torch.load(args.loadWeights, map_location=device, weights_only=True)
    checkpoint = extract_state_dict(checkpoint)
    model = load_my_state_dict(model, checkpoint)
    model.eval()

    print("Model and weights LOADED successfully")

    if not os.path.exists(args.datadir):
        print("Error: datadir could not be loaded")
        return

    loader = DataLoader(
        cityscapes(
            args.datadir,
            input_transform_cityscapes,
            target_transform_cityscapes,
            subset=args.subset
        ),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            result = model(images)

        # Prendi l'output dell'ultimo layer
        mask_logits  = result[0][-1]   # [B, num_q, H_small, W_small]
        class_logits = result[1][-1]   # [B, num_q, num_classes+1]

        # Upscala le maschere alla risoluzione delle label (512x1024)
        H, W = labels.shape[-2], labels.shape[-1]
        mask_logits = F.interpolate(
            mask_logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        # Converti in mappa semantica pixel-wise
        mask_probs  = torch.sigmoid(mask_logits)          # [B, num_q, H, W]
        class_probs = torch.softmax(class_logits, dim=-1) # [B, num_q, num_classes+1]

        Mat_Class = class_probs.transpose(1, 2)            # [B, num_classes+1, num_q]
        Mat_Mask  = torch.flatten(mask_probs, start_dim=2) # [B, num_q, H*W]

        pixel_logits = torch.matmul(Mat_Class, Mat_Mask)   # [B, num_classes+1, H*W]
        pixel_logits = pixel_logits.unflatten(2, (H, W))   # [B, num_classes+1, H, W]

        # Argmax solo sulle prime NUM_CLASSES classi (escludi void all'indice -1)
        pred = pixel_logits[:, :NUM_CLASSES, :, :].argmax(1).unsqueeze(1)

        iouEvalVal.addBatch(pred.data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1]
        print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i]) + '{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    print("Per-Class IoU:")
    print(iou_classes_str[0],  "Road")
    print(iou_classes_str[1],  "Sidewalk")
    print(iou_classes_str[2],  "Building")
    print(iou_classes_str[3],  "Wall")
    print(iou_classes_str[4],  "Fence")
    print(iou_classes_str[5],  "Pole")
    print(iou_classes_str[6],  "Traffic light")
    print(iou_classes_str[7],  "Traffic sign")
    print(iou_classes_str[8],  "Vegetation")
    print(iou_classes_str[9],  "Terrain")
    print(iou_classes_str[10], "Sky")
    print(iou_classes_str[11], "Person")
    print(iou_classes_str[12], "Rider")
    print(iou_classes_str[13], "Car")
    print(iou_classes_str[14], "Truck")
    print(iou_classes_str[15], "Bus")
    print(iou_classes_str[16], "Train")
    print(iou_classes_str[17], "Motorcycle")
    print(iou_classes_str[18], "Bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal*100) + '\033[0m'
    print("MEAN IoU: ", iouStr, "%")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadWeights', required=True,
                        help="Path to EoMT checkpoint (.pth o .bin)")
    parser.add_argument('--subset',      default="val")
    parser.add_argument('--datadir',     required=True,
                        help="Root Cityscapes (contiene leftImg8bit/ e gtFine/)")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size',  type=int, default=1)
    parser.add_argument('--cpu',         action='store_true')
    main(parser.parse_args())