# =============================================================================
# VERSIONE SPERIMENTALE (Opzione B') - NON e' l'originale.
# Idea: nel ramo OE carichiamo la TERNA GREZZA (immagine contaminata + anomaly
# mask + gt label map) e applichiamo UN SOLO transform a tutti e tre insieme.
# Cosi' immagine, anomalia e gt ricevono la STESSA augmentation (stesso "lancio
# di dadi") e l'allineamento e' garantito.
#
# Trucco per far passare l'anomaly nello stesso transform senza modificare
# transforms.py: la agganciamo come riga extra dentro target["masks"] e
# teniamo un flag parallelo "is_anomaly" (gestito automaticamente da _filter,
# esattamente come "labels"/"is_crowd"). Dopo il transform la separiamo.
#
# Matching per FILENAME (non per indice): robusto a differenze di ordinamento.
# =============================================================================

from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from datasets.cityscapes_semantic import CityscapesSemantic


class CityscapesCOCO(Dataset):

    def __init__(
        self,
        image_dir,
        anomaly_dir,
        gt_dir,
        semantic_dataset,
        prob_oe=0.1,
    ):

        # mappe filename -> path (matching robusto per nome, non per indice)
        self.contam_by_name = {p.name: p for p in Path(image_dir).rglob("*.png")}
        self.anomaly_by_name = {p.name: p for p in Path(anomaly_dir).rglob("*.png")}
        self.gt_by_name = {p.name: p for p in Path(gt_dir).rglob("*.png")}

        self.semantic_dataset = semantic_dataset
        self.prob_oe = prob_oe

        # riuso LO STESSO oggetto transforms del dataset semantico:
        # il ramo "pulito" lo applica internamente, il ramo OE lo applichiamo noi
        # -> stesso identico comportamento di augmentation.
        self.transforms = semantic_dataset.transforms
        self.target_parser = CityscapesSemantic.target_parser

        print("FOUND CONTAM:", len(self.contam_by_name))
        print("FOUND ANOMALY:", len(self.anomaly_by_name))
        print("FOUND GT:", len(self.gt_by_name))

    def __len__(self):
        return len(self.semantic_dataset)

    def __getitem__(self, idx):

        # nome del file Cityscapes sottostante a questo indice
        name = Path(self.semantic_dataset.imgs[idx]).name

        # ---------------- RAMO PULITO (1 - prob_oe) ----------------
        if random.random() > self.prob_oe or name not in self.contam_by_name:
            sem_img, target = self.semantic_dataset[idx]
            H, W = sem_img.shape[-2:]
            target["anomaly_mask"] = tv_tensors.Mask(
                torch.zeros((H, W), dtype=torch.bool)
            )
            return sem_img, target

        # ---------------- RAMO OE: terna grezza + UN solo transform ----------------
        # 1) immagine contaminata grezza (uint8, come il dataset pulito)
        contam = np.array(Image.open(self.contam_by_name[name]).convert("RGB"))
        contam = tv_tensors.Image(
            torch.from_numpy(contam).permute(2, 0, 1).contiguous()
        )

        # 2) anomaly mask grezza [H,W] bool
        anomaly = np.array(Image.open(self.anomaly_by_name[name])) > 0
        anomaly = torch.from_numpy(anomaly).bool()

        # 3) gt label map grezza [1,H,W] -> maschere/labels via lo stesso parser
        gt = np.array(Image.open(self.gt_by_name[name]))
        gt = tv_tensors.Mask(torch.from_numpy(gt.astype("int64"))[None])
        masks, labels, is_crowd = self.target_parser(target=gt)

        # aggancio l'anomaly come riga extra: viaggia nello STESSO transform
        masks = tv_tensors.Mask(torch.stack(masks + [anomaly]))
        labels = torch.tensor(labels + [0], dtype=torch.long)        # label dummy per la riga anomaly
        is_crowd = torch.tensor(is_crowd + [False], dtype=torch.bool)
        is_anomaly = torch.zeros(masks.shape[0], dtype=torch.bool)
        is_anomaly[-1] = True

        target = {
            "masks": masks,
            "labels": labels,
            "is_crowd": is_crowd,
            "is_anomaly": is_anomaly,
        }

        # *** UNICO transform condiviso: immagine + maschere + anomaly insieme ***
        contam, target = self.transforms(contam, target)

        # separo l'anomaly dalle maschere semantiche
        is_anom = target["is_anomaly"].bool()
        if is_anom.any():
            anomaly = target["masks"][is_anom].any(dim=0).bool()
        else:
            # anomalia ritagliata fuori dal crop -> nessuna anomalia in questo campione
            anomaly = torch.zeros(contam.shape[-2:], dtype=torch.bool)

        keep = ~is_anom
        masks = target["masks"][keep].bool()
        labels = target["labels"][keep]
        is_crowd = target["is_crowd"][keep]

        # tolgo la regione anomala dalla supervisione semantica (come nell'originale)
        if masks.numel() > 0:
            visible = masks & (~anomaly)
            valid = visible.flatten(1).any(dim=1)
            masks = visible[valid]
            labels = labels[valid]
            is_crowd = is_crowd[valid]

        target = {
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "is_crowd": is_crowd,
            "anomaly_mask": tv_tensors.Mask(anomaly),
        }

        return contam, target
