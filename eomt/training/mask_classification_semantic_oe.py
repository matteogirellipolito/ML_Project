import torch
import torch.nn.functional as F

from training.mask_classification_semantic import MaskClassificationSemantic

class MaskClassificationSemanticOE(MaskClassificationSemantic):

    def __init__(
        self,
        lambda_rba=0.1,
        rba_alpha=5.0,
        tuning_mode="head",
        **kwargs
    ):

        super().__init__(**kwargs)

        self.save_hyperparameters(
            ignore=["network"]
        )

        self.lambda_rba = lambda_rba
        self.rba_alpha = rba_alpha
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

    @staticmethod
    def oe_loss(
        logits,
        anomaly_mask,
        alpha=5.0,
    ):
        anomaly_mask = anomaly_mask.bool()

        if not anomaly_mask.any():
            return logits.new_zeros(())

        score = torch.tanh(logits)
        rba = -score.sum(dim=1)
        loss_map = F.relu(alpha-rba).pow(2)

        return loss_map[anomaly_mask].mean()

    def training_step(
        self,
        batch,
        batch_idx
    ):

        imgs, targets = batch

        batch_size = imgs.shape[0]
        
        mask_logits_layers, class_logits_layers = self(imgs)
 
        losses_all_blocks = {}

        for block_idx in range(len(mask_logits_layers)):
            losses = self.criterion(
            masks_queries_logits=mask_logits_layers[block_idx],
            class_queries_logits=class_logits_layers[block_idx],
            targets=targets,
            )

            suffix = self.block_postfix(block_idx)

            for loss_name, loss_value in losses.items():
                losses_all_blocks[f"{loss_name}{suffix}"] = loss_value

        semantic_loss = self.criterion.loss_total(losses_all_blocks, lambda *args, **kwargs: None) 
        
        anomaly_masks = torch.stack([
            t["anomaly_mask"]
            for t in targets
        ]).to(self.device)

        final_mask_logits = mask_logits_layers[-1]
        if final_mask_logits.shape[-2:] != imgs.shape[-2:]:
            final_mask_logits = F.interpolate(
            final_mask_logits,
            size=imgs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        per_pixel_logits = self.to_per_pixel_logits_semantic(final_mask_logits,class_logits_layers[-1])
        ood_loss = self.oe_loss(per_pixel_logits, anomaly_masks, alpha=self.rba_alpha)

        loss = semantic_loss + self.lambda_rba * ood_loss

        self.log(
            "train/total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )

        self.log(
            "train/semantic_loss",
            semantic_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size
        )

        self.log(
            "train/ood_loss",
            ood_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size
        )

        self.log(
            "train/anomaly_pixels",
            anomaly_masks.float().mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size
        )

        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False
        )

        self.log(
            "train/weighted_ood",
            self.lambda_rba * ood_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size
        )

        return loss