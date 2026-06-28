from pathlib import Path
from PIL import Image
import random
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
        prob_oe=0.1
    ):

        self.images = sorted(Path(image_dir).rglob("*.png"))
        self.masks = sorted(Path(anomaly_dir).rglob("*.png"))
        self.semantic_dataset = semantic_dataset
        self.prob_oe = prob_oe 

        print("FOUND IMAGES:",len(self.images))
        print("FOUND MASKS:",len(self.masks))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        sem_img, target = self.semantic_dataset[idx]

        if random.random() > self.prob_oe:
            target["anomaly_mask"] = tv_tensors.Mask(torch.zeros(sem_img.shape[-2:], dtype=torch.bool))
            return sem_img, target

        contam = np.array(Image.open(self.images[idx]).convert("RGB"))
        contam = torch.from_numpy(contam).permute(2,0,1).float()
        if contam.shape[-2:] != sem_img.shape[-2:]:
            contam = torch.nn.functional.interpolate(
                contam.unsqueeze(0),
                size=sem_img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]

        anomaly = np.array(Image.open(self.masks[idx])) > 0
        anomaly = torch.tensor(anomaly, dtype=torch.float32)[None,None]
        if anomaly.shape[-2:] != sem_img.shape[-2:]:
            anomaly = torch.nn.functional.interpolate(
                anomaly,
                size=sem_img.shape[-2:],
                mode="nearest"
            )
        anomaly = anomaly[0,0].bool()
        
        target["anomaly_mask"] = tv_tensors.Mask(anomaly)

        if "masks" in target and target["masks"].numel() > 0:
            visible_masks = (target["masks"].bool() & ~anomaly)
            target["visible_masks"] = visible_masks
            valid = visible_masks.flatten(1).any(dim=1)

            if valid.any():
                target["masks"] = tv_tensors.Mask(visible_masks[valid])

                if "labels" in target:
                    target["labels"] = target["labels"][valid]

                if "is_crowd" in target:
                    target["is_crowd"] = target["is_crowd"][valid]

        return contam, target 