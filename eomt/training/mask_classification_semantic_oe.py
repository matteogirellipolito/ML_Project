from training.mask_classification_semantic import MaskClassificationSemantic

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskClassificationSemanticOE(
    MaskClassificationSemantic
):

    def __init__(
        self,
        anomaly_weight=0.5,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.anomaly_weight = anomaly_weight
        self.anomaly_bce = nn.BCEWithLogitsLoss()

    def anomaly_logits_from_masks(self,mask_logits):
        return mask_logits.max(dim=1)[0]

    def training_step(self,batch,batch_idx):
        imgs, targets = batch
        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = self(crops)

        semantic_targets = self.to_per_pixel_targets_semantic(targets,self.ignore_idx)

        loss = self.criterion(mask_logits_per_layer,class_logits_per_layer,semantic_targets,)

        anomaly_masks = torch.stack([t["anomaly_mask"].float() for t in targets]).to(loss.device)

        anomaly_logits = self.anomaly_logits_from_masks(mask_logits_per_layer[-1])
        anomaly_logits = F.interpolate(anomaly_logits.unsqueeze(1),anomaly_masks.shape[-2:],mode="bilinear",).squeeze(1)

        anomaly_loss = self.anomaly_bce(anomaly_logits,anomaly_masks)
        total_loss = loss + self.anomaly_weight * anomaly_loss

        self.log("train_anomaly_loss",anomaly_loss,prog_bar=True)
        self.log("train_total_loss",total_loss,prog_bar=True)

        return total_loss