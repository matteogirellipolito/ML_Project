import os
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr

from erfnet import ERFNet


def compute_metrics(anomaly_scores, ood_gts, name):

    anomaly_scores = np.array(anomaly_scores)
    ood_gts = np.array(ood_gts)

    min_len = min(len(anomaly_scores), len(ood_gts))
    anomaly_scores = anomaly_scores[:min_len]
    ood_gts = ood_gts[:min_len]

    val_out = anomaly_scores.flatten()
    val_label = ood_gts.flatten()

    valid_mask = val_label != 255
    val_out = val_out[valid_mask]
    val_label = val_label[valid_mask]

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"\n{name}")
    print(f"AUPRC: {prc_auc * 100:.4f}")
    print(f"FPR@TPR95: {fpr * 100:.4f}")


def save_heatmap(image, logit_map, entropy_map, filename):

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(logit_map, cmap="jet")
    axs[1].set_title("Heatmap 1")
    axs[1].axis("off")

    axs[2].imshow(entropy_map, cmap="jet")
    axs[2].set_title("Heatmap 2")
    axs[2].axis("off")

    os.makedirs("outputs_heatmaps", exist_ok=True)
    plt.savefig(f"outputs_heatmaps/{filename}")
    plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ERFNet(20)
    model = torch.nn.DataParallel(model)

    model.load_state_dict(
        torch.load("../trained_models/erfnet_pretrained.pth")
    )

    model.to(device)
    model.eval()

    print("Model LOADED")

    input_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_paths = glob.glob(args.input)

    anomaly_logit_list = []
    anomaly_entropy_list = []
    ood_gts_array = []

    for img_path in image_paths:

        print("Processing:", img_path)

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        image_tensor = input_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)

        logits = output.squeeze(0).cpu().numpy()

        probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

        max_logit = -np.max(logits, axis=0)

        entropy = -np.sum(
            probs * np.log(probs + 1e-12),
            axis=0
        )

        anomaly_logit_list.append(max_logit)
        anomaly_entropy_list.append(entropy)

        label_path = img_path.replace("images", "labels_masks")

        if label_path.endswith(".jpg"):
            label_path = label_path.replace(".jpg", ".png")

        label = np.array(Image.open(label_path))

        ood_gts = (label == 2).astype(np.uint8)

        ood_gts_array.append(ood_gts)

        filename = os.path.basename(img_path)
        save_heatmap(image_np, max_logit, entropy, filename)

    compute_metrics(anomaly_logit_list, ood_gts_array, "Metric 1")
    compute_metrics(anomaly_entropy_list, ood_gts_array, "Metric 2")


if __name__ == "__main__":
    main()