#LightningDataModule for Cityscapes semantic segmentation with Outlier Exposure

#training dataset is wrapped in CityscapesCOCO which mixes clean images with contaminated ones

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

        # Store the OE-specific paths and probability, they are only consumed later, in setup(), when the training dataset is built.
        self.contaminated_images = contaminated_images
        self.anomaly_masks = anomaly_masks
        self.gt_dir = gt_dir
        self.prob_oe = prob_oe

    def setup(self, stage=None):
        # 1) build the standard Cityscapes train/val datasets
        super().setup(stage)

        #  2)replace the training set with the OE wrapper
        # keeps normal samples and with prob_oe gives a contaminated triplet (image + anomaly mask + label map)
        self.cityscapes_train_dataset = CityscapesCOCO(
            image_dir=self.contaminated_images,
            anomaly_dir=self.anomaly_masks,
            gt_dir=self.gt_dir,
            semantic_dataset=self.cityscapes_train_dataset,
            prob_oe=self.prob_oe,
        )

        return self
