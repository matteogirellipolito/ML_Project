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

        self.images = sorted(
            list(Path(image_dir).glob("*.png"))
        )

        self.masks = sorted(
            list(Path(anomaly_dir).glob("*.png"))
        )

        self.semantic_dataset = semantic_dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = np.array(
            Image.open(
                self.images[idx]
            ).convert("RGB")
        )

        anomaly = np.array(
            Image.open(
                self.masks[idx]
            )
        ) > 0

        anomaly = tv_tensors.Mask(
            torch.tensor(
                anomaly,
                dtype=torch.bool
            )
        )

        if self.semantic_dataset is None:

            return img, anomaly

        _, target = self.semantic_dataset[idx]

        target["anomaly_mask"] = anomaly

        return img, target