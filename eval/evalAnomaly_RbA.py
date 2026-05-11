import os
import glob
import torch
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
from scipy.special import softmax

from ood_metrics import fpr_at_95_tpr

# EoMT
from eomt.models.eomt import EoMT

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


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

    print("\n===== RbA =====")
    print(f"AUPRC score: {prc_auc*100:.4f}")
    print(f"FPR@TPR95: {fpr*100:.4f}")


def compute_rba(class_logits):

    probs = softmax(class_logits, axis=-1)

    max_prob = np.max(probs[..., :-1], axis=-1)

    rba = 1.0 - max_prob

    return rba


def build_model():

    """
    QUI devi mettere la costruzione reale del modello.
    """

    encoder = ...   # TODO
    num_q = ...     # TODO

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=num_q
    )

    return model


def main():

    parser = ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()

    output_dir = "outputs_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    anomaly_scores = []
    ood_gts_list = []

    model = build_model()

    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint)

    model = model.cuda()
    model.eval()

    print("EoMT Model LOADED")

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):

        print("Processing:", path)

        image = Image.open(path).convert("RGB")
        image_np = np.array(image)

        tensor_img = input_transform(image).unsqueeze(0).float().cuda()

        with torch.no_grad():
            mask_logits_layers, class_logits_layers = model(tensor_img)

        class_logits = class_logits_layers[-1].squeeze(0).cpu().numpy()

        rba_map = compute_rba(class_logits)

        anomaly_scores.append(rba_map)

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

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)

        heatmap = normalize(rba_map)

        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.imshow(image_np)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("RbA")
        plt.axis("off")

        filename = os.path.basename(path).split('.')[0]
        plt.savefig(f"{output_dir}/{filename}_rba.png")
        plt.close()

    compute_metrics(anomaly_scores, ood_gts_list)

    print(f"\nHeatmaps saved in {output_dir}")


if __name__ == "__main__":
    main()