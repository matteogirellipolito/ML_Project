import os
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

# Configurazione della riproducibilità e parametri globali
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

NUM_CLASSES = 20

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

def normalize(x):
    """Normalizza i valori in un intervallo [0, 1] per la visualizzazione."""
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

def compute_rba_map(mask_logits, mask_pred, temp=1.0):
    """
    Calcola la mappa di anomalia tramite l'algoritmo Rejected by All (RbA).
    Determina l'incertezza pesando le maschere spaziali per la confidenza 
    delle query e identificando le aree non coperte da predizioni sicure.
    """
    probs = torch.softmax(mask_logits / temp, dim=-1)
    conf, _ = torch.max(probs, dim=-1) 
    masks = mask_pred.sigmoid() 
    weighted_masks = masks * conf.view(-1, 1, 1)
    confidence_map = torch.sum(weighted_masks, dim=0)
    return 1.0 - torch.clamp(confidence_map, 0, 1)

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="model_weights.pth")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    output_dir = "outputs_rba"
    os.makedirs(output_dir, exist_ok=True)

    anomaly_score_list = []
    ood_gts_list = []

    # Inizializzazione del modello 
    from model_lib import MyModel 
    model = MyModel(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        """Mappa e carica i pesi nel modello, gestendo le discrepanze nei nomi dei parametri."""
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
            else:
                own_state[name].copy_(param)
        return model

    weightspath = os.path.join(args.loadDir, args.loadWeights)
    model = load_my_state_dict(model, torch.load(weightspath, map_location='cpu'))
    model.eval()

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print("Processing:", path)
        image = Image.open(path).convert('RGB')
        tensor_img = input_transform(image).unsqueeze(0).float().cuda()

        with torch.no_grad():
            output = model(tensor_img)
            # Estrazione dei componenti RbA dall'output del modello
            anomaly_map = compute_rba_map(output['pred_logits'][0], output['pred_masks'][0])
            anomaly_result = anomaly_map.cpu().numpy()

        # Caricamento e processamento della Ground Truth
        pathGT = path.replace("images", "labels_masks").replace("jpg", "png").replace("webp", "png")
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # Normalizzazione delle etichette per i diversi dataset di anomalia
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

        # Accumulo dei punteggi per i soli pixel validi (escludendo l'etichetta 255)
        valid_mask = ood_gts != 255
        if 1 in np.unique(ood_gts):
            anomaly_score_list.append(anomaly_result[valid_mask])
            ood_gts_list.append(ood_gts[valid_mask])

        # Generazione e salvataggio della mappa di calore
        plt.imshow(normalize(anomaly_result), cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(path)}_rba.png"), bbox_inches='tight')
        plt.close()

    # Calcolo statistico delle metriche di performance globali
    y_scores = np.concatenate(anomaly_score_list)
    y_true = np.concatenate(ood_gts_list)
    
    auprc = average_precision_score(y_true, y_scores)
    fpr95 = fpr_at_95_tpr(y_scores, y_true)

    print(f"AUPRC score: {auprc * 100:.2f}")
    print(f"FPR@TPR95: {fpr95 * 100:.2f}")

if __name__ == '__main__':
    main()
