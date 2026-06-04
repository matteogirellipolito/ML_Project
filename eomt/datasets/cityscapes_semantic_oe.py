import random
import torch

from datasets.cityscapes_semantic import CityscapesSemantic
from training.coco_paste import COCOPaster

class WrapperOE:

    def __init__(
        self,
        dataset,
        paster,
        p_ood,
    ):
        self.dataset = dataset
        self.paster = paster
        self.p_ood = p_ood

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        if random.random() < self.p_ood:

            img_hwc = img.permute(1,2,0).cpu().numpy()
            img_hwc, anomaly = self.paster.paste(img_hwc)
            img = torch.from_numpy(img_hwc).permute(2,0,1)

            target["ood_mask"] = torch.tensor(anomaly,dtype=torch.bool)

        else:
            h, w = img.shape[-2:]

            target["ood_mask"] = torch.zeros((h,w),dtype=torch.bool)

        return img, target

class CityscapesSemanticOE(
    CityscapesSemantic
):

    def __init__(
        self,
        coco_root,
        p_ood=0.1,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.p_ood = p_ood
        self.coco_root = coco_root

    def setup(self, stage=None):

        super().setup(stage)

        self.paster = COCOPaster(coco_root=self.coco_root)

        self.cityscapes_train_dataset = WrapperOE(
            dataset=self.cityscapes_train_dataset,
            paster=self.paster,
            p_ood=self.p_ood,
        )

        return self