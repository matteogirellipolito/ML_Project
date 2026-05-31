from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

import numpy as np
import glob
import os

class CityscapesCOCO(Dataset):

    def __init__(
        self,
        image_dir,
        anomaly_dir,
        semantic_dir,
    ):

        self.images = sorted(list(Path(image_dir).glob("*.png")))
        self.anomaly_masks = sorted(list(Path(anomaly_dir).glob("*.png")))
        self.semantic_dir = semantic_dir

        assert len(self.images) == len(self.anomaly_masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        filename = os.path.basename(img_path)
        city = filename.split("_")[0]
        semantic_path = glob.glob(
            self.semantic_dir
            + f"/{city}/"
            + filename.replace(
                city + "_",
                ""
            ).replace(
                "_leftImg8bit",
                "_gtFine_labelIds"
            )
        )[0]

        img = np.array(Image.open(img_path).convert("RGB"))
        anomaly = np.array(Image.open(self.anomaly_masks[idx])) > 0
        semantic = np.array(Image.open(semantic_path))

        return (
            img,
            semantic,
            anomaly
        )