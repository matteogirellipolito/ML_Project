"""LightningDataModule for Cityscapes semantic segmentation with Outlier
Exposure (OE).

It extends the plain Cityscapes semantic data module: after the usual setup, the
training dataset is wrapped in ``CityscapesCOCO``, which mixes clean images with
contaminated ones (a COCO object pasted onto the road as a synthetic anomaly).
The three extra paths point to the pre-built OE dataset, and ``prob_oe`` controls
the fraction of training samples that are contaminated. Validation is left
unchanged (clean Cityscapes), so standard mIoU evaluation still applies.
"""

from datasets.cityscapes_semantic import CityscapesSemantic
from datasets.loader_cityscapesCOCO import CityscapesCOCO


class CityscapesSemanticOE(CityscapesSemantic):
    """Cityscapes semantic data module augmented with Outlier Exposure."""

    def __init__(
        self,
        contaminated_images,   # folder with the pre-built contaminated images
        anomaly_masks,         # folder with the matching anomaly masks
        gt_dir,                # folder with the matching semantic label maps
        prob_oe=0.1,           # fraction of training samples that are contaminated
        **kwargs               # remaining arguments forwarded to the base module
    ):
        super().__init__(**kwargs)

        # Store the OE-specific paths and probability; they are only consumed
        # later, in setup(), when the training dataset is built.
        self.contaminated_images = contaminated_images
        self.anomaly_masks = anomaly_masks
        self.gt_dir = gt_dir
        self.prob_oe = prob_oe

    def setup(self, stage=None):
        # First build the standard Cityscapes train/val datasets.
        super().setup(stage)

        # Then replace the training set with the OE wrapper: it keeps the clean
        # samples and, with probability `prob_oe`, serves a contaminated triplet
        # (image + anomaly mask + label map) instead. The clean semantic dataset
        # is passed in so the wrapper can reuse its images and augmentation.
        self.cityscapes_train_dataset = CityscapesCOCO(
            image_dir=self.contaminated_images,
            anomaly_dir=self.anomaly_masks,
            gt_dir=self.gt_dir,
            semantic_dataset=self.cityscapes_train_dataset,
            prob_oe=self.prob_oe,
        )

        return self
