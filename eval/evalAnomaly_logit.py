import os
import glob
import torch
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from erfnet import ERFNet
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
from scipy.special import softmax

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS =3
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

def compute_metrics(anomaly_scores, ood_gts, method_name):
    anomaly_scores = np.array(anomaly_scores)

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

    print(f"\n===== {method_name} =====")
    print(f"AUPRC score: {prc_auc * 100:.4f}")
    print(f"FPR@TPR95: {fpr * 100:.4f}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    output_dir = "outputs_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    anomaly_logit_list = []
    anomaly_entropy_list = []
    ood_gts_list = []

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
            else:
                own_state[name].copy_(param)
        return model

    weightspath = args.loadDir + args.loadWeights
    model = load_my_state_dict(
        model,
        torch.load(weightspath, map_location=lambda storage, loc: storage)
    )

    print("Model LOADED")
    model.eval()

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print("Processing:", path)

        image = Image.open(path).convert('RGB')
        image_np = np.array(image)

        tensor_img = input_transform(image).unsqueeze(0).float().cuda()

        with torch.no_grad():
            result = model(tensor_img)

        logits = result.squeeze(0).data.cpu().numpy()

        # MAX LOGIT
        maxlogit = -np.max(logits, axis=0)

        # MAX ENTROPY
        probs = softmax(logits, axis=0)
        entropy = np.sum(-probs * np.log(probs + 1e-9), axis=0)
        
        anomaly_logit_list.append(maxlogit)
        anomaly_entropy_list.append(entropy)

        # Ground Truth
        pathGT = path.replace("images", "labels_masks")
        pathGT = pathGT.replace("jpg", "png").replace("webp", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        ##riaggiunte come da originale
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)
        ##

        ood_gts_list.append(ood_gts)


        # Heatmaps
        maxlogit_norm = normalize(maxlogit)
        entropy_norm = normalize(entropy)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image_np)
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(maxlogit_norm, cmap='jet')
        axs[1].set_title("MaxLogit")
        axs[1].axis('off')

        axs[2].imshow(entropy_norm, cmap='jet')
        axs[2].set_title("MaxEntropy")
        axs[2].axis('off')

        filename = os.path.basename(path).split('.')[0]
        save_path = os.path.join(output_dir, f"{filename}_heatmap.png")

        plt.savefig(save_path)
        plt.close()

    ood_gts_array = np.array(ood_gts_list)

    compute_metrics(anomaly_logit_list, ood_gts_array, "MaxLogit")
    compute_metrics(anomaly_entropy_list, ood_gts_array, "MaxEntropy")

    print(f"\nHeatmaps saved in: {output_dir}")

if __name__ == '__main__':
    main()