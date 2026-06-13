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

    img_out = os.path.join(out_dir,"images")
    mask_out = os.path.join(out_dir,"anomaly_masks")
    
    os.makedirs(img_out,exist_ok=True)
    os.makedirs(mask_out,exist_ok=True)

    city_imgs = sorted(glob.glob(city_dir + "/*/*.png"))

    with open(pano_json) as f:
        pano = json.load(f)

    for path in tqdm(city_imgs):

        img, mask = contaminate_cityscapes_image(
            path,
            pano,
            coco_dir,
            pano_mask_dir,
        )

        city = path.split("/")[-2]

        os.makedirs(os.path.join(img_out,city),exist_ok=True)
        os.makedirs(os.path.join(mask_out,city),exist_ok=True)

        filename = os.path.basename(path)

        Image.fromarray(img).save(os.path.join(img_out,city,filename))
        Image.fromarray(mask.astype("uint8") * 255).save(os.path.join(mask_out,city,filename))