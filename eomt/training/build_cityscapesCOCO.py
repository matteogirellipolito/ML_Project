#Offline builder for the contaminated Cityscapes + COCO dataset used for Outlier Exposure training

import os
import json
import random
from tqdm import tqdm
from PIL import Image

import numpy as np

from training.coco_paste import contaminate_cityscapes_image
from datasets.cityscapes_semantic import CityscapesSemantic

# Fix the RNG so the chosen objects and paste positions are reproducible
random.seed(42)
np.random.seed(42)


def build_dataset(
    cityscapes_zip,   # folder containing the Cityscapes .zip files
    pano_json,        # COCO panoptic annotations file (.json)
    coco_dir,         # folder with the COCO source images (.jpg)
    pano_mask_dir,    # folder with the COCO panoptic masks (.png)
    out_dir,          # destination folder for the generated dataset
):
    # One output sub-folder per element of the triplet images, anomaly_mask, gt
    img_out = os.path.join(out_dir, "images")
    mask_out = os.path.join(out_dir, "anomaly_masks")
    gt_out = os.path.join(out_dir, "gt")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)

    # Load the COCO panoptic annotations used to select which object to paste on each image
    with open(pano_json) as f:
        pano = json.load(f)

    # Cityscapes data module is reused only to read images and label maps from the zip archives
    semantic_dm = CityscapesSemantic(
        path=cityscapes_zip,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
    )
    semantic_dm.setup()

    dataset = semantic_dm.cityscapes_train_dataset

    # Turn off augmentation
    # Augmentation is deferred to load time
    dataset.transforms = None

    # Open the target archive once to read the raw label maps 
    _, target_zip, _ = dataset._load_zips()

    for idx in tqdm(range(len(dataset))):

        # Read the raw image and convert it to an HxWx3 array (format expected by the paste function)
        img_tensor, _ = dataset[idx]
        img = img_tensor.cpu()
        if img.max() <= 1.0:                 # values in [0, 1] -> rescale to [0, 255]
            img = (img * 255).byte()
        else:                                # already in [0, 255]
            img = img.clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()

        # Paste a COCO object onto the road. 
        # Returns the contaminated image and the binary mask marking the pasted anomalous pixels
        img, mask = contaminate_cityscapes_image(
            img,
            pano,
            coco_dir,
            pano_mask_dir,
        )

        # Preserve the original layout: contaminated image, anomaly mask and label map 
        # are all keyed by the same file name
        path = dataset.imgs[idx]
        filename = os.path.basename(path)
        city = os.path.basename(os.path.dirname(path))

        os.makedirs(os.path.join(img_out, city), exist_ok=True)
        os.makedirs(os.path.join(mask_out, city), exist_ok=True)
        os.makedirs(os.path.join(gt_out, city), exist_ok=True)

        # Contaminated RGB image
        Image.fromarray(img).save(os.path.join(img_out, city, filename))

        # Anomaly mask
        Image.fromarray(mask.astype("uint8") * 255).save(os.path.join(mask_out, city, filename))

        # Original Cityscapes label map, saved under the SAME file name (image, mask and label aligned by name)
        with target_zip.open(dataset.targets[idx]) as gt_file:
            gt_img = Image.open(gt_file)
            gt_img.load()
        gt_img.save(os.path.join(gt_out, city, filename))
