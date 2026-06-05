import torch
import torch.nn.functional as F

from training.mask_classification_semantic import MaskClassificationSemantic

class MaskClassificationSemanticOE(
    MaskClassificationSemantic
):

    def __init__(
        self,
        lambda_oe=0.1,
        tuning_mode="head",
        **kwargs
    ):

        super().__init__(**kwargs)

        self.lambda_oe = lambda_oe
        self.tuning_mode = tuning_mode
        self.apply_freezing()

    def apply_freezing(self):
        mode = self.tuning_mode.lower()

        if mode == "joint":
            for p in self.network.parameters():
                p.requires_grad = True
            return

        for name,param in self.network.named_parameters():
            name = name.lower()
            trainable = False

            if mode == "head":
                if "class" in name or "mask" in name:
                    trainable = True
            elif mode == "query":
                if "query" in name:
                    trainable = True

            param.requires_grad = trainable

    def anomaly_loss(
        self,
        class_logits,
        anomaly_masks
    ):
        probs = class_logits.softmax(-1)
        confidence,_ = probs.max(dim=-1)

        anomaly_score = confidence.mean()
        anomaly_presence = anomaly_masks.float().mean()

        return anomaly_score * anomaly_presence

    def training_step(
        self,
        batch,
        batch_idx
    ):

        imgs, targets = batch

        mask_logits_layers, class_logits_layers = self(imgs)
        mask_logits = mask_logits_layers[-1]
        class_logits = class_logits_layers[-1]

        semantic_losses = self.criterion(
            mask_logits,
            targets,
            class_logits
        )
        
        semantic_loss = sum(semantic_losses.values())
        
        anomaly_masks = torch.stack([
            t["anomaly_mask"]
            for t in targets
        ]).to(self.device)

        oe_loss = self.anomaly_loss(
            class_logits,
            anomaly_masks
        )

        loss = semantic_loss + self.lambda_oe * oe_loss

        for name,val in semantic_losses.items():

            self.log(
                f"train/{name}",
                val,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
            )

            self.log(
                "train/semantic_loss",
                semantic_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

            self.log(
                "train/oe_loss",
                oe_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        return loss