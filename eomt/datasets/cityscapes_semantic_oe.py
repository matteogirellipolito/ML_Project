# =============================================================================
# VERSIONE SPERIMENTALE (Opzione B') - NON e' l'originale.
# Differenze rispetto a cityscapes_semantic_oe.py:
#   - usa PROVA_loader_prova (loader con transform unico sulla terna)
#   - passa al loader anche gt_dir
#   - FORWARD CORRETTO di prob_oe al loader (nell'originale era ignorato)
# =============================================================================

from datasets.cityscapes_semantic import CityscapesSemantic
from datasets.PROVA_loader_prova import CityscapesCOCO


class CityscapesSemanticOE(CityscapesSemantic):
    def __init__(
        self,
        contaminated_images,
        anomaly_masks,
        gt_dir,
        prob_oe=0.1,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.contaminated_images = contaminated_images
        self.anomaly_masks = anomaly_masks
        self.gt_dir = gt_dir
        self.prob_oe = prob_oe

    def setup(
        self,
        stage=None
    ):

        super().setup(stage)

        self.cityscapes_train_dataset = CityscapesCOCO(
            image_dir=self.contaminated_images,
            anomaly_dir=self.anomaly_masks,
            gt_dir=self.gt_dir,
            semantic_dataset=self.cityscapes_train_dataset,
            prob_oe=self.prob_oe,
        )

        return self
