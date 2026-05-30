import os
import glob
import json

from tqdm import tqdm
from PIL import Image

from training.coco_paste import contaminate_cityscapes_image

def build_dataset(
    city_dir,
    pano_json,
    coco_dir,
    pano_mask_dir,
    out_dir,
):

    os.makedirs(out_dir + "/images",exist_ok=True)
    os.makedirs(out_dir + "/anomaly_masks",exist_ok=True)
    city_imgs = glob.glob(city_dir + "/*/*.png")

    with open(pano_json) as f:
        pano = json.load(f)

    for path in tqdm(city_imgs):
        img,mask = contaminate_cityscapes_image(
            path,
            pano,
            coco_dir,
            pano_mask_dir,
        )
        name = os.path.basename(path)
        Image.fromarray(img).save(out_dir + "/images/" + name)
        Image.fromarray(mask*255).save(out_dir + "/anomaly_masks/" + name)