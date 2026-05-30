from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

import numpy as np

class CityscapesOE(Dataset):
    def __init__(
        self,
        image_dir,
        anomaly_dir,
    ):
        self.images = sorted(list(Path(image_dir).glob("*.png")))
        self.masks = sorted(list(Path(anomaly_dir).glob("*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img = np.array(Image.open(self.images[idx]).convert("RGB"))
        anomaly = np.array(Image.open(self.masks[idx]))
        anomaly = anomaly > 0

        return img, anomaly