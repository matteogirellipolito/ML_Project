# =============================================================================
# VERSIONE SPERIMENTALE (Opzione B') - NON e' l'originale.
# Differenza chiave rispetto a build_cityscapesCOCO.py:
#   - Le immagini contaminate vengono costruite da immagini Cityscapes GREZZE
#     (transforms=None): NESSUNA augmentation al build.
#   - Viene salvata anche la GT (label map originale) allineata alla terna,
#     cosi' nel loader si applica UN SOLO transform a (immagine+anomaly+gt).
# Output: out_dir/images, out_dir/anomaly_masks, out_dir/gt  (stessi filename)
# =============================================================================

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

    img_out = os.path.join(out_dir, "images")
    mask_out = os.path.join(out_dir, "anomaly_masks")
    gt_out = os.path.join(out_dir, "gt")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)

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

    # <<< DIFFERENZA CHIAVE: disattiva l'augmentation al build, immagini GREZZE >>>
    dataset.transforms = None

    # accesso allo zip per estrarre la label map grezza (build single-process)
    _, target_zip, _ = dataset._load_zips()

    for idx in tqdm(range(len(dataset))):

        img_tensor, _ = dataset[idx]            # raw, uint8 [3,H,W] (no augmentation)
        img = img_tensor.cpu()
        if img.max() <= 1.0:
            img = (img * 255).byte()
        else:
            img = img.clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()      # HWC uint8

        img, mask = contaminate_cityscapes_image(
            img,
            pano,
            coco_dir,
            pano_mask_dir,
        )

        path = dataset.imgs[idx]
        filename = os.path.basename(path)       # ..._leftImg8bit.png  (chiave comune della terna)
        city = os.path.basename(os.path.dirname(path))

        os.makedirs(os.path.join(img_out, city), exist_ok=True)
        os.makedirs(os.path.join(mask_out, city), exist_ok=True)
        os.makedirs(os.path.join(gt_out, city), exist_ok=True)

        # 1) immagine contaminata grezza
        Image.fromarray(img).save(os.path.join(img_out, city, filename))
        # 2) anomaly mask grezza
        Image.fromarray(mask.astype("uint8") * 255).save(os.path.join(mask_out, city, filename))
        # 3) GT (label map originale Cityscapes) - salvata con lo STESSO filename
        with target_zip.open(dataset.targets[idx]) as gt_file:
            gt_img = Image.open(gt_file)
            gt_img.load()
        gt_img.save(os.path.join(gt_out, city, filename))
