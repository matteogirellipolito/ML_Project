import os
import json
import random
from tqdm import tqdm
from PIL import Image

import numpy as np

from training.coco_paste import contaminate_cityscapes_image
from datasets.cityscapes_semantic import CityscapesSemantic

random.seed(42)
np.random.seed(42)

def build_dataset(
    cityscapes_zip,
    pano_json,
    coco_dir,
    pano_mask_dir,
    out_dir,
):

    img_out = os.path.join(out_dir,"images")
    mask_out = os.path.join(out_dir,"anomaly_masks")
    
    os.makedirs(img_out,exist_ok=True)
    os.makedirs(mask_out,exist_ok=True)

    with open(pano_json) as f:
        pano = json.load(f)

    semantic_dm = CityscapesSemantic(
        path=cityscapes_zip,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
    )
    semantic_dm.setup()

    dataset = semantic_dm.cityscapes_train_dataset
    
    for idx in tqdm(range(len(dataset))):
            
        img_tensor, target = dataset[idx]          
        img = img_tensor.cpu()
        if img.max() <= 1.0:
            img = (img * 255).byte()
        else:
            img = img.clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        img, mask = contaminate_cityscapes_image(
            img,
            pano,
            coco_dir,
            pano_mask_dir,
        )

        path = dataset.imgs[idx]
        filename = os.path.basename(path)
        city = os.path.basename(os.path.dirname(path))

        os.makedirs(os.path.join(img_out,city),exist_ok=True)
        os.makedirs(os.path.join(mask_out,city),exist_ok=True)

        filename = os.path.basename(path)

        Image.fromarray(img).save(os.path.join(img_out,city,filename))
        Image.fromarray(mask.astype("uint8") * 255).save(os.path.join(mask_out,city,filename))