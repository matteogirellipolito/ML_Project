from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

class CityscapesCOCO(Dataset):

    def __init__(
        self,
        image_dir,
        anomaly_dir,
        semantic_dataset,
        prob_oe=0.1,
    ):

        self.images = sorted(Path(image_dir).rglob("*.png"))
        self.masks = sorted(Path(anomaly_dir).rglob("*.png"))

        self.semantic_dataset = semantic_dataset
        self.prob_oe = prob_oe

        assert len(self.images) == len(self.semantic_dataset)
        assert len(self.masks) == len(self.semantic_dataset)

        print("FOUND IMAGES:", len(self.images))
        print("FOUND MASKS:", len(self.masks))

    def __len__(self):
        return len(self.semantic_dataset)

    def __getitem__(self, idx):

        sem_img, target = self.semantic_dataset[idx]

        H, W = sem_img.shape[-2:]

        if random.random() > self.prob_oe:

            target["anomaly_mask"] = tv_tensors.Mask(
                torch.zeros((H, W), dtype=torch.bool)
            )

            return sem_img, target

        contam = np.array(Image.open(self.images[idx]).convert("RGB"))
        contam = torch.from_numpy(contam).permute(2,0,1).float()
        if contam.shape[-2:] != (H, W):
            contam = torch.nn.functional.interpolate(
                contam.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )[0]
 
        anomaly = np.array(Image.open(self.masks[idx])) > 0
        anomaly = torch.from_numpy(anomaly)
        if anomaly.shape != torch.Size([H, W]):
            anomaly = torch.nn.functional.interpolate(
                anomaly.float().unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="nearest",
            )[0, 0].bool()

        target["anomaly_mask"] = tv_tensors.Mask(anomaly)
        
        if target["masks"].numel() > 0:

            visible_masks = target["masks"].bool() & (~anomaly)

            valid = visible_masks.flatten(1).any(dim=1)

            target["visible_masks"] = visible_masks

            target["masks"] = tv_tensors.Mask(
                visible_masks[valid]
            )

            target["labels"] = target["labels"][valid]

            target["is_crowd"] = target["is_crowd"][valid]

        return contam, target