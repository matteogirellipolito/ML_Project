import torch
import torch.nn.functional as F

from training.mask_classification_semantic import (
    MaskClassificationSemantic
)


class MaskClassificationSemanticOE(
    MaskClassificationSemantic
):

    def __init__(
        self,
        anomaly_weight=0.1,
        anomaly_margin=5.0,
        anomaly_reduction="mean",

        freeze_only_head=False,
        freeze_only_query=False,

        **kwargs
    ):

        super().__init__(**kwargs)

        self.anomaly_weight = anomaly_weight
        self.anomaly_margin = anomaly_margin
        self.anomaly_reduction = anomaly_reduction

        self.freeze_only_head = freeze_only_head
        self.freeze_only_query = freeze_only_query

        if freeze_only_head:
            self.freeze_only_heads()

        if freeze_only_query:
            self.freeze_only_queries()    

    def freeze_only_heads(self):

        for p in self.network.parameters():
            p.requires_grad = False

        for module_name in ["class_head", "mask_head"]:

            module = getattr(
                self.network,
                module_name,
                None
            )

            if module is None:
                continue

            for p in module.parameters():
                p.requires_grad = True

    def freeze_only_queries(self):

        for p in self.network.parameters():
            p.requires_grad = False

        found = False

        for name, p in self.network.named_parameters():

            lname = name.lower()

            if (
                "query" in lname
                or "queries" in lname
                or "query_embed" in lname
            ):

                p.requires_grad = True
                found = True

        if not found:
            raise RuntimeError(
                "No query parameters found."
            )

    def anomaly_hinge_loss(
        self,
        logits,
        anomaly_mask,
    ):

        anomaly_mask = anomaly_mask.bool()

        if not anomaly_mask.any():

            return logits.new_zeros(())

        score = torch.tanh(logits)

        anomaly_score = -score.sum(dim=1)

        loss_map = F.relu(
            self.anomaly_margin - anomaly_score
        ).pow(2)

        selected = loss_map[anomaly_mask]

        if self.anomaly_reduction == "sum":
            return selected.sum()

        return selected.mean()
    
    def clean_targets(
        self,
        targets
    ):

        cleaned = []

        for t in targets:

            cleaned.append(
                {
                    "masks":
                        t["masks"].to(
                            self.device
                        ).bool(),

                    "labels":
                        t["labels"].to(
                            self.device
                        ).long(),
                }
            )

        return cleaned

    def batch_anomaly_masks(
        self,
        targets,
        size,
    ):

        masks = []

        for t in targets:

            m = t.get("ood_mask")

            if m is None:

                m = torch.zeros(
                    size,
                    dtype=torch.bool,
                    device=self.device
                )

            else:

                m = m.to(self.device)

                if tuple(m.shape[-2:]) != tuple(size):

                    m = F.interpolate(
                        m[None,None].float(),
                        size=size,
                        mode="nearest"
                    )[0,0]

                m = m.bool()

            masks.append(m)

        return torch.stack(masks)  

    def training_step(
        self,
        batch,
        batch_idx
    ):

        imgs, targets = batch

        clean_targets = self.clean_targets(
            targets
        )

        anomaly_masks = self.batch_anomaly_masks(
            targets,
            imgs.shape[-2:]
        )

        mask_logits_blocks, class_logits_blocks = self(
            imgs
        )

        losses = {}

        for i, (m, c) in enumerate(

            zip(
                mask_logits_blocks,
                class_logits_blocks
            )
        ):

            block_losses = self.criterion(

                masks_queries_logits=m,

                class_queries_logits=c,

                targets=clean_targets
            )

            postfix = self.block_postfix(i)

            losses |= {

                f"{k}{postfix}": v

                for k,v in block_losses.items()
            }

        segmentation_loss = self.criterion.loss_total(
            losses,
            self.log
        )

        final_mask_logits = mask_logits_blocks[-1]

        if final_mask_logits.shape[-2:] != imgs.shape[-2:]:

            final_mask_logits = F.interpolate(

                final_mask_logits,

                size=imgs.shape[-2:],

                mode="bilinear",

                align_corners=False
            )

        pixel_logits = self.to_per_pixel_logits_semantic(

            final_mask_logits,

            class_logits_blocks[-1]
        )

        anomaly_loss = self.anomaly_hinge_loss(

            pixel_logits,

            anomaly_masks
        )

        total_loss = (

            segmentation_loss

            + self.anomaly_weight * anomaly_loss
        )

        self.log(
            "train/segmentation_loss",
            segmentation_loss,
            prog_bar=True
        )

        self.log(
            "train/anomaly_loss",
            anomaly_loss,
            prog_bar=True
        )

        self.log(
            "train/total_loss",
            total_loss,
            prog_bar=True
        )

        return total_loss