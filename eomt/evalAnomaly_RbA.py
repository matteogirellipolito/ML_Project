import os
import glob
import yaml
import torch
import random
import importlib
import warnings
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from argparse import ArgumentParser
from lightning import seed_everything
from torchvision.transforms import Compose, Resize, ToTensor
from scipy.special import softmax
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from sklearn.metrics import average_precision_score

from ood_metrics import fpr_at_95_tpr

seed_everything(42, verbose=False)

IGNORE_INDEX = 255

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


def compute_metrics(anomaly_scores, ood_gts):

    anomaly_scores = np.array(anomaly_scores)
    ood_gts = np.array(ood_gts)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out)).flatten()
    val_label = np.concatenate((ind_label, ood_label)).flatten()

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"AUPRC score: {prc_auc*100:.4f}")
    print(f"FPR@TPR95: {fpr*100:.4f}")


def main():

    parser = ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    device = "cuda"

    output_dir = "outputs_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)

    encoder = encoder_cls(
        img_size=(512, 1024),
        **encoder_cfg.get("init_args", {})
    )

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)

    network_kwargs = {
        k: v for k, v in network_cfg["init_args"].items()
        if k != "encoder"
    }

    network = network_cls(
        masked_attn_enabled=False,
        num_classes=19,
        encoder=encoder,
        **network_kwargs,
    )

    # Load lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)

    model_kwargs = {
        k: v for k, v in config["model"]["init_args"].items()
        if k != "network"
    }

    model = lit_cls(
        img_size=(512, 1024),
        num_classes=19,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    print("EoMT model loaded")

    anomaly_scores = []
    ood_gts_list = []

    for path in glob.glob(args.input):

        print("Processing:", path)

        image = Image.open(path).convert("RGB")
        image_np = np.array(image)

        tensor_img = input_transform(image).unsqueeze(0).to(device)

        with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):

            mask_logits_per_layer, class_logits_per_layer = model(tensor_img)

            mask_logits = mask_logits_per_layer[-1]

            # RbA score
            probs = softmax(mask_logits.squeeze(0).cpu().numpy(), axis=0)

            top2 = np.partition(probs, -2, axis=0)[-2:]
            rba_score = top2[-1] - top2[-2]

            anomaly_map = -rba_score

        anomaly_scores.append(anomaly_map)

        # Ground truth
        pathGT = path.replace("images", "labels_masks")
        pathGT = pathGT.replace("jpg", "png").replace("webp", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)

        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)

        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # filtro originale
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)

        # Heatmap
        norm_map = normalize(anomaly_map)

        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.imshow(image_np)
        plt.axis("off")
        plt.title("Original")

        plt.subplot(1,2,2)
        plt.imshow(norm_map, cmap="jet")
        plt.axis("off")
        plt.title("RbA")

        save_path = os.path.join(
            output_dir,
            os.path.basename(path).split(".")[0] + "_rba.png"
        )

        plt.savefig(save_path)
        plt.close()

    compute_metrics(anomaly_scores, np.array(ood_gts_list))

    print(f"Heatmaps saved in {output_dir}")


if __name__ == "__main__":
    main()