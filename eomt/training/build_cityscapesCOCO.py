"""Offline builder for the contaminated Cityscapes + COCO dataset used for
Outlier Exposure (OE) training.

For every Cityscapes training image, a COCO object belonging to a category that
does not exist in Cityscapes is pasted onto the road to create a *synthetic
anomaly*. Three aligned outputs are written under ``out_dir``, all sharing the
same file name so they can be matched later:

    images/         the contaminated RGB image
    anomaly_masks/  the binary mask of the pasted (anomalous) region
    gt/             the original Cityscapes semantic label map

Design note: the images are generated WITHOUT data augmentation (raw, full
resolution). Augmentation (flip / scale jitter / crop) is instead applied at
load time, jointly to the image, the semantic ground truth and the anomaly
mask, so that the three remain geometrically aligned. Saving the label map here
makes the OE dataset self-contained and keeps image, mask and label indexed by
the same file name.
"""

import os
import json
import random
from tqdm import tqdm
from PIL import Image

import numpy as np

from training.coco_paste import contaminate_cityscapes_image
from datasets.cityscapes_semantic import CityscapesSemantic

# Fix the RNG so the chosen objects and paste positions are reproducible.
random.seed(42)
np.random.seed(42)


def build_dataset(
    cityscapes_zip,   # folder containing the Cityscapes .zip files
    pano_json,        # COCO panoptic annotations file (.json)
    coco_dir,         # folder with the COCO source images (.jpg)
    pano_mask_dir,    # folder with the COCO panoptic masks (.png)
    out_dir,          # destination folder for the generated dataset
):
    # One output sub-folder per element of the triplet.
    img_out = os.path.join(out_dir, "images")
    mask_out = os.path.join(out_dir, "anomaly_masks")
    gt_out = os.path.join(out_dir, "gt")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)

    # Load the COCO panoptic annotations once; they are used to select which
    # object to paste on each image.
    with open(pano_json) as f:
        pano = json.load(f)

    # The Cityscapes data module is reused only to read the images and label
    # maps from the zip archives; batch size and workers are irrelevant here.
    semantic_dm = CityscapesSemantic(
        path=cityscapes_zip,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
    )
    semantic_dm.setup()

    dataset = semantic_dm.cityscapes_train_dataset

    # Turn off augmentation: we deliberately want the raw, full-resolution
    # images. Augmentation is deferred to load time (see module docstring).
    dataset.transforms = None

    # Open the target archive once to read the raw label maps (single process).
    _, target_zip, _ = dataset._load_zips()

    for idx in tqdm(range(len(dataset))):

        # Read the raw image and convert it to an HxWx3 uint8 NumPy array,
        # which is the format expected by the paste function.
        img_tensor, _ = dataset[idx]
        img = img_tensor.cpu()
        if img.max() <= 1.0:                 # values in [0, 1] -> rescale to [0, 255]
            img = (img * 255).byte()
        else:                                # already in [0, 255]
            img = img.clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()

        # Paste a COCO object onto the road. Returns the contaminated image and
        # the binary mask marking the pasted (anomalous) pixels.
        img, mask = contaminate_cityscapes_image(
            img,
            pano,
            coco_dir,
            pano_mask_dir,
        )

        # Preserve the original city/file layout so that the contaminated image,
        # the anomaly mask and the label map are all keyed by the same file name.
        path = dataset.imgs[idx]
        filename = os.path.basename(path)
        city = os.path.basename(os.path.dirname(path))

        os.makedirs(os.path.join(img_out, city), exist_ok=True)
        os.makedirs(os.path.join(mask_out, city), exist_ok=True)
        os.makedirs(os.path.join(gt_out, city), exist_ok=True)

        # (1) contaminated RGB image
        Image.fromarray(img).save(os.path.join(img_out, city, filename))

        # (2) anomaly mask, stored as a 0/255 PNG
        Image.fromarray(mask.astype("uint8") * 255).save(os.path.join(mask_out, city, filename))

        # (3) original Cityscapes label map, saved under the SAME file name so
        # image, mask and label remain aligned by name.
        with target_zip.open(dataset.targets[idx]) as gt_file:
            gt_img = Image.open(gt_file)
            gt_img.load()
        gt_img.save(os.path.join(gt_out, city, filename))
