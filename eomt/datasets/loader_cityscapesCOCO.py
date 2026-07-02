"""Training dataset wrapper that adds Outlier Exposure (OE) contamination on top
of clean Cityscapes samples.

Each item comes from one of two branches:
  * clean  (probability 1 - prob_oe): a normal Cityscapes image with an empty
    anomaly mask, produced by the underlying semantic dataset;
  * OE     (probability prob_oe): a pre-built contaminated triplet -- the
    contaminated image, its anomaly mask and the semantic label map -- loaded
    RAW and augmented here.

Key idea (geometric alignment): in the OE branch the image, the semantic masks
and the anomaly mask go through ONE shared augmentation, so that every pixel
stays consistent across the three. To route the anomaly mask through the same
transform without modifying the transform pipeline, it is appended as an extra
row of ``target["masks"]`` and tracked with a parallel boolean flag
``is_anomaly`` (filtered automatically by the transform, exactly like
``labels`` / ``is_crowd``); it is separated again once the transform is done.

Contaminated files are matched to the underlying Cityscapes sample by FILE NAME
rather than by index, which is robust to any difference in ordering between the
two sets.
"""

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
        image_dir,          # folder with the pre-built contaminated images
        anomaly_dir,        # folder with the matching anomaly masks
        gt_dir,             # folder with the matching semantic label maps
        semantic_dataset,   # underlying clean Cityscapes dataset (with transforms)
        prob_oe=0.1,        # probability of returning a contaminated sample
    ):

        # Index every triplet element by file name, so a Cityscapes sample can
        # be matched to its contaminated counterpart regardless of ordering.
        self.contam_by_name = {p.name: p for p in Path(image_dir).rglob("*.png")}
        self.anomaly_by_name = {p.name: p for p in Path(anomaly_dir).rglob("*.png")}
        self.gt_by_name = {p.name: p for p in Path(gt_dir).rglob("*.png")}

        self.semantic_dataset = semantic_dataset
        self.prob_oe = prob_oe

        # Reuse the SAME augmentation object as the clean dataset: the clean
        # branch applies it internally, the OE branch applies it explicitly, so
        # both branches share identical augmentation behaviour.
        self.transforms = semantic_dataset.transforms
        self.target_parser = CityscapesSemantic.target_parser

        print("FOUND CONTAM:", len(self.contam_by_name))
        print("FOUND ANOMALY:", len(self.anomaly_by_name))
        print("FOUND GT:", len(self.gt_by_name))

    def __len__(self):
        return len(self.semantic_dataset)

    def __getitem__(self, idx):

        # File name of the underlying Cityscapes image at this index.
        name = Path(self.semantic_dataset.imgs[idx]).name

        # ---------------- CLEAN branch (probability 1 - prob_oe) ----------------
        # Also taken if no contaminated image exists for this file name.
        if random.random() > self.prob_oe or name not in self.contam_by_name:
            sem_img, target = self.semantic_dataset[idx]
            H, W = sem_img.shape[-2:]
            target["anomaly_mask"] = tv_tensors.Mask(
                torch.zeros((H, W), dtype=torch.bool)
            )
            return sem_img, target

        # ---------------- OE branch: raw triplet + one shared transform ---------
        # 1) contaminated image, kept as uint8 like the clean dataset.
        contam = np.array(Image.open(self.contam_by_name[name]).convert("RGB"))
        contam = tv_tensors.Image(
            torch.from_numpy(contam).permute(2, 0, 1).contiguous()
        )

        # 2) anomaly mask as a boolean [H, W] tensor.
        anomaly = np.array(Image.open(self.anomaly_by_name[name])) > 0
        anomaly = torch.from_numpy(anomaly).bool()

        # 3) label map [1, H, W] -> per-class masks/labels via the shared parser.
        gt = np.array(Image.open(self.gt_by_name[name]))
        gt = tv_tensors.Mask(torch.from_numpy(gt.astype("int64"))[None])
        masks, labels, is_crowd = self.target_parser(target=gt)

        # Append the anomaly as an extra mask row so it rides through the SAME
        # transform as the semantic masks. `is_anomaly` marks which row it is;
        # `labels`/`is_crowd` get dummy entries to keep all arrays the same length.
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

        # *** Single shared augmentation: image + semantic masks + anomaly ***
        contam, target = self.transforms(contam, target)

        # Pull the anomaly row back out of the (now augmented) masks.
        is_anom = target["is_anomaly"].bool()
        if is_anom.any():
            anomaly = target["masks"][is_anom].any(dim=0).bool()
        else:
            # The anomaly was cropped out entirely -> no anomaly in this sample.
            anomaly = torch.zeros(contam.shape[-2:], dtype=torch.bool)

        keep = ~is_anom
        masks = target["masks"][keep].bool()
        labels = target["labels"][keep]
        is_crowd = target["is_crowd"][keep]

        # Remove the anomalous region from the semantic supervision, then drop
        # any mask that became empty as a result.
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
