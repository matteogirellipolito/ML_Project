#Training dataset wrapper that adds Outlier Exposure (OE) contamination on Cityscapes samples



##geometric alignment --> for contaminated image the triple (image, semantic mask, anomaly maks) go through the same SHARED augmentation for concistency between pixels

#Contaminated files are matched to the underlying Cityscapes sample by FILE NAME
#(no index for robustness to order changes in the setes)

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
        image_dir,          # contaminated images
        anomaly_dir,        # anomaly masks
        gt_dir,             # semantic label maps
        semantic_dataset,   # clean Cityscapes dataset (with transforms)
        prob_oe=0.1,        
    ):

        # Index triplet by file name
        self.contam_by_name = {p.name: p for p in Path(image_dir).rglob("*.png")}
        self.anomaly_by_name = {p.name: p for p in Path(anomaly_dir).rglob("*.png")}
        self.gt_by_name = {p.name: p for p in Path(gt_dir).rglob("*.png")}

        self.semantic_dataset = semantic_dataset
        self.prob_oe = prob_oe

        # Reuse SAME augmentation object as the clean dataset
        self.transforms = semantic_dataset.transforms
        self.target_parser = CityscapesSemantic.target_parser

        print("FOUND CONTAM:", len(self.contam_by_name))
        print("FOUND ANOMALY:", len(self.anomaly_by_name))
        print("FOUND GT:", len(self.gt_by_name))

    def __len__(self):
        return len(self.semantic_dataset)

    def __getitem__(self, idx):

        # File name of the underlying Cityscapes image at this index
        name = Path(self.semantic_dataset.imgs[idx]).name

        #  CLEAN branch (probability 1 - prob_oe) 
        if random.random() > self.prob_oe or name not in self.contam_by_name:
            sem_img, target = self.semantic_dataset[idx]
            H, W = sem_img.shape[-2:]
            target["anomaly_mask"] = tv_tensors.Mask(
                torch.zeros((H, W), dtype=torch.bool)
            )
            return sem_img, target

        #  OE--> raw triplet + one shared transform
        
        # 1) contaminated image
        contam = np.array(Image.open(self.contam_by_name[name]).convert("RGB"))
        contam = tv_tensors.Image(
            torch.from_numpy(contam).permute(2, 0, 1).contiguous()
        )

        # 2) anomaly mask --> boolean [H, W] tensor
        anomaly = np.array(Image.open(self.anomaly_by_name[name])) > 0
        anomaly = torch.from_numpy(anomaly).bool()

        # 3) label map [1, H, W] -> per-class masks/labels
        gt = np.array(Image.open(self.gt_by_name[name]))
        gt = tv_tensors.Mask(torch.from_numpy(gt.astype("int64"))[None])
        masks, labels, is_crowd = self.target_parser(target=gt)

        # Append the anomaly as an extra mask row for SAME transform as semantic masks
        masks = tv_tensors.Mask(torch.stack(masks + [anomaly]))
        labels = torch.tensor(labels + [0], dtype=torch.long)
        is_crowd = torch.tensor(is_crowd + [False], dtype=torch.bool)
        is_anomaly = torch.zeros(masks.shape[0], dtype=torch.bool)
        is_anomaly[-1] = True

        target = {
            "masks": masks,
            "labels": labels,
            "is_crowd": is_crowd,
            "is_anomaly": is_anomaly,
        }

        # Single shared augmentation: image + semantic masks + anomaly
        contam, target = self.transforms(contam, target)

        # Pull the anomaly row back out of the (now augmented) masks
        is_anom = target["is_anomaly"].bool()
        if is_anom.any():
            anomaly = target["masks"][is_anom].any(dim=0).bool()
        else:
            # The anomaly was cropped out entirely -> no anomaly in this sample
            anomaly = torch.zeros(contam.shape[-2:], dtype=torch.bool)

        keep = ~is_anom
        masks = target["masks"][keep].bool()
        labels = target["labels"][keep]
        is_crowd = target["is_crowd"][keep]

        # Remove the anomalous region from the semantic supervisio 
        # drop any mask that became empty as a result
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
