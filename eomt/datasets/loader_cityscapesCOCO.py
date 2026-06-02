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

        image_dir = Path(image_dir)
        anomaly_dir = Path(anomaly_dir)

        self.images = sorted([p for p in image_dir.rglob("*.png") if p.is_file()])
        self.masks = sorted([p for p in anomaly_dir.rglob("*.png") if p.is_file()])

        print("FOUND IMAGES:", len(self.images))
        print("FOUND MASKS:", len(self.masks))

        assert len(self.images) > 0
        assert len(self.images) == len(self.masks)

        self.semantic_dataset = semantic_dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        anomaly = np.array(Image.open(self.masks[idx])) > 0
        anomaly = tv_tensors.Mask(torch.tensor(anomaly,dtype=torch.bool))

        if self.semantic_dataset is None:
            img = np.array(Image.open(self.images[idx]).convert("RGB"))

            return img, anomaly

        _, target = self.semantic_dataset[idx]

        contaminated_img = Image.open(self.images[idx]).convert("RGB")
        contaminated_img = np.array(contaminated_img)

        target["anomaly_mask"] = anomaly

        return contaminated_img, target