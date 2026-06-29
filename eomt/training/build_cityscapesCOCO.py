import os
import json
import random
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np

from training.coco_paste import contaminate_cityscapes_image

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

    city_imgs = sorted(
        glob.glob(
            os.path.join(
                cityscapes_zip,
                "leftImg8bit",
                "train",
                "*",
                "*_leftImg8bit.png"
            )
        )
    )

    for path in tqdm(city_imgs):

        img = np.array(Image.open(path).convert("RGB"))
        contam,mask = contaminate_cityscapes_image(
            img,
            pano,
            coco_dir,
            pano_mask_dir
        )

        city = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)

        os.makedirs(os.path.join(img_out,city), exist_ok=True)
        os.makedirs(os.path.join(mask_out,city), exist_ok=True)

        Image.fromarray(contam).save(os.path.join(img_out,city,filename))
        Image.fromarray(mask.astype(np.uint8)*255).save(os.path.join(mask_out,city,filename))