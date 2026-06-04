from pathlib import Path
from PIL import Image

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import tv_tensors
import torchvision.transforms.functional as TF

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

        img = Image.open(
            self.images[idx]
        ).convert("RGB")

        anomaly = Image.open(
            self.masks[idx]
        )

        if self.semantic_dataset is None:

            img = torch.from_numpy(
                np.array(img)
            ).permute(2,0,1).float()/255.

            anomaly = torch.tensor(
                np.array(anomaly) > 0,
                dtype=torch.bool
            )

            anomaly = tv_tensors.Mask(anomaly)

            return img, anomaly

        _, target = self.semantic_dataset[idx]

        img = torch.from_numpy(
            np.array(img)
        ).permute(2,0,1).float()/255.

        target_h, target_w = target["masks"].shape[-2:]

        anomaly = TF.resize(
            anomaly,
            [target_h, target_w],
            interpolation=TF.InterpolationMode.NEAREST
        )

        anomaly = np.array(anomaly) > 0

        anomaly = tv_tensors.Mask(
            torch.tensor(
                anomaly,
                dtype=torch.bool
            )
        )

        target["anomaly_mask"] = anomaly

        return img, target