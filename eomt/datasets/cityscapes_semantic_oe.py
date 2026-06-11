from torch.utils.data import DataLoader

from datasets.cityscapes_semantic import CityscapesSemantic
from datasets.loader_cityscapesCOCO import CityscapesCOCO

class CityscapesSemanticOE(CityscapesSemantic):

    def __init__(
        self,
        contaminated_images,
        anomaly_masks,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.contaminated_images = contaminated_images
        self.anomaly_masks = anomaly_masks

    def setup(
        self,
        stage=None
    ):

        super().setup(stage)

        self.cityscapes_train_dataset = CityscapesCOCO(
            image_dir=self.contaminated_images,
            anomaly_dir=self.anomaly_masks,
            semantic_dataset=self.cityscapes_train_dataset,
        )

        return self