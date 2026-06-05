from pathlib import Path
from PIL import Image

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import tv_tensors


class CityscapesCOCO(Dataset):

    def __init__(
        self,
        image_dir,
        anomaly_dir,
        semantic_dataset=None,
    ):

        self.images = sorted(Path(image_dir).rglob("*.png"))
        self.masks = sorted(Path(anomaly_dir).rglob("*.png"))
        self.semantic_dataset = semantic_dataset

        print("FOUND IMAGES:",len(self.images))
        print("FOUND MASKS:",len(self.masks))

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        sem_img, target = self.semantic_dataset[idx]

        contam = np.array(Image.open(self.images[idx]).convert("RGB"))
        contam = torch.from_numpy(contam).permute(2,0,1).float()
        contam = torch.nn.functional.interpolate(
            contam.unsqueeze(0),
            size=sem_img.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        anomaly = np.array(Image.open(self.masks[idx])) > 0
        anomaly = torch.tensor(anomaly,dtype=torch.float32)[None,None]
        anomaly = torch.nn.functional.interpolate(
            anomaly,
            size=sem_img.shape[-2:],
            mode="nearest"
        )[0,0].bool()

        target["anomaly_mask"] = tv_tensors.Mask(anomaly)

        return contam, target
    
"""
File di prima:


FILE USATO:
/content/ML_Project/eomt/datasets/loader_cityscapesCOCO.py

INIZIO __getitem__:

    def __getitem__(self, idx):

        img = Image.open(self.images[idx]).convert("RGB")
        anomaly = np.array(Image.open(self.masks[idx])) > 0
        anomaly = tv_tensors.Mask(torch.tensor(anomaly,dtype=torch.bool))

        if self.semantic_dataset is None:
            img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0

            return img, anomaly

        _, target = self.semantic_dataset[idx]
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        target["anomaly_mask"] = anomaly

        return img, target


"""