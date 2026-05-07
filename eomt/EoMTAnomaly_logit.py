import os
import sys
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Fix per i moduli della repo EoMT
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from training.lightning_module import LightningModule
    from ood_metrics import fpr_at_95_tpr
    from sklearn.metrics import average_precision_score
    from torchvision import transforms
except ImportError:
    print("Assicurati di avere ood_metrics.py e scikit-learn installati.")

# --- LOGICA DI CALCOLO ANOMALIA (Punto 5) ---

def compute_anomaly_score(logits, masks, method='RbA', temp=1.0):
    """
    Calcola lo score di anomalia adattato alle Mask Architectures.
    """
    # Applicazione Temperature Scaling sui logit delle query
    logits = logits / temp
    
    if method == 'MSP':
        probs = F.softmax(logits, dim=-1)
        # 1 - Max Softmax Probability
        query_scores = 1 - torch.max(probs, dim=-1)[0]
        return torch.einsum('q,qhw->hw', query_scores, masks)

    elif method == 'MaxLogit':
        query_scores = -torch.max(logits, dim=-1)[0]
        return torch.einsum('q,qhw->hw', query_scores, masks)

    elif method == 'MaxEntropy':
        probs = F.softmax(logits, dim=-1)
        query_scores = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return torch.einsum('q,qhw->hw', query_scores, masks)

    elif method == 'RbA':
        # Rejected by All: somma pesata delle confidenze sulle maschere
        probs = F.softmax(logits, dim=-1)
        mask_weights, _ = torch.max(probs, dim=-1)
        weighted_masks = masks * mask_weights.view(-1, 1, 1)
        rba_map = torch.sum(weighted_masks, dim=0)
        return 1 - rba_map

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

# --- FUNZIONE METRICHE (Dal tuo Punto 4) ---

def compute_final_metrics(anomaly_scores, ood_gts, method_name):
    scores = np.array(anomaly_scores)
    gts = np.array(ood_gts)

    # Filtraggio dei pixel validi 
    mask = gts != 255
    val_out = scores[mask].flatten()
    val_label = gts[mask].flatten()

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"\n===== Risultati {method_name} =====")
    print(f"AUPRC: {prc_auc * 100:.4f}")
    print(f"FPR@TPR95: {fpr * 100:.4f}")
    return prc_auc, fpr

# --- MAIN SCRIPT ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path root dataset (es. SMIYC)")
    parser.add_argument("--ckpt", required=True, help="Path a eomt_cityscapes.bin")
    parser.add_argument("--dataset", default="SMIYC_RA-21", help="Nome del dataset")
    parser.add_argument("--method", choices=['MSP', 'MaxLogit', 'MaxEntropy', 'RbA'], default='RbA')
    parser.add_argument("--temp", type=float, default=1.0, help="Scaling T (0.5, 0.75, 1.1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("heatmaps_eomt", exist_ok=True)

    # 1. Caricamento Modello
    print("Loading EoMT Model...")
    model = LightningModule.load_from_checkpoint(args.ckpt).to(device)
    model.eval()

    # 2. Trasformazioni (EoMT usa 640x640 per DINOv2)
    input_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Resize((640, 640), Image.NEAREST)

    # 3. Loop di Valutazione
    img_dir = os.path.join(args.data_dir, "images")
    gt_dir = os.path.join(args.data_dir, "labels_masks")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    
    all_scores = []
    all_gts = []

    for filename in tqdm(img_files):
        # Carica Immagine
        img_path = os.path.join(img_dir, filename)
        raw_img = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(raw_img).unsqueeze(0).to(device)

        # Inferenza
        with torch.no_grad():
            output = model(img_tensor)
            logits = output['pred_logits']
            masks = output['pred_masks'].sigmoid()

        # Calcolo Anomalia
        anomaly_map = compute_anomaly_score(logits[0], masks[0], args.method, args.temp)
        anomaly_np = anomaly_map.cpu().numpy()

        # Carica Ground Truth 
        gt_path = os.path.join(gt_dir, filename.replace(".jpg", ".png"))
        gt_img = Image.open(gt_path)
        gt_res = np.array(target_transform(gt_img))
        
        # Mapping label (SMIYC esempio)
        ood_gt = np.where(gt_res == 2, 1, gt_res) # 1=Anomaly, 0=Ind, 255=Ignore
        
        all_scores.append(anomaly_np)
        all_gts.append(ood_gt)

        # Salvataggio Heatmap 
        if len(all_scores) % 20 == 0: # Ne salva una ogni 20 per non riempire il disco
            plt.imshow(normalize(anomaly_np), cmap='jet')
            plt.title(f"{args.method} T={args.temp}")
            plt.axis('off')
            plt.savefig(f"heatmaps_eomt/{filename}_heat.png")
            plt.close()

    # 4. Calcolo Finale
    compute_final_metrics(all_scores, all_gts, f"{args.method}_T{args.temp}")

if __name__ == "__main__":
    main()
