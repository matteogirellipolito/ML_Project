import random
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

VALID_CATEGORIES = {

    # animals    
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",

    # accessories / sports 
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball",
    38: "kite", 39: "baseball bat", 40: "baseball glove",

    # household objects
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",

    #food
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    
    # furniture
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 70: "toilet", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 
    82: "refrigerator",
    
    # electronics
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone"
}

def rgb2id(color):

    return (
        color[:,:,0].astype(np.int64)
        + 256 * color[:,:,1].astype(np.int64)
        + 256*256 * color[:,:,2].astype(np.int64)
    )

class COCOPaster:

    def __init__(
        self,
        coco_root,
        target_height_range=(80,250),
    ):

        coco_root = Path(coco_root)

        self.coco_img_dir = (
            coco_root /
            "val2017"
        )

        self.panoptic_dir = (
            coco_root /
            "annotations" /
            "panoptic_val2017"
        )

        json_path = (
            coco_root /
            "annotations" /
            "panoptic_val2017.json"
        )

        with open(json_path) as f:
            self.panoptic = json.load(f)

        self.target_height_range = target_height_range

    def sample_object(self):

        while True:
            ann = random.choice(self.panoptic["annotations"])

            candidates = [
                s for s in ann["segments_info"]
                if s["category_id"] in VALID_CATEGORIES
                and s["area"] > 1500
                and s["iscrowd"] == 0
            ]

            if len(candidates):
                seg = random.choice(candidates)

                return ann, seg

    def paste(self, city_img):
        city_img = np.array(city_img)

        ann, seg = self.sample_object()

        mask_rgb = np.array(Image.open(self.panoptic_dir / ann["file_name"]))

        ids = rgb2id(mask_rgb)
        mask = ids == seg["id"]

        ys, xs = np.where(mask)

        coco_name = ann["file_name"].replace(".png",".jpg")
        coco_img = np.array(Image.open(self.coco_img_dir / coco_name).convert("RGB"))

        crop = coco_img[ ys.min():ys.max() , xs.min():xs.max() ]

        crop_mask = mask[ ys.min():ys.max() , xs.min():xs.max() ]

        target_h = random.randint(*self.target_height_range)
        scale = target_h / crop.shape[0]
        new_w = int(crop.shape[1] * scale)

        crop = np.array(Image.fromarray(crop).resize((new_w,target_h)))
        crop_mask = np.array(Image.fromarray(crop_mask.astype(np.uint8)*255)
                             .resize(
                                 (new_w,target_h),
                                    Image.NEAREST) 
                                    ) > 0

        H,W = city_img.shape[:2]

        if new_w >= W or target_h >= H:
            return self.paste(city_img)

        x = random.randint(W//3,W-new_w-1)
        y = random.randint(H//2,H-target_h-1)

        alpha = gaussian_filter(crop_mask.astype(float),sigma=1.0)
        alpha = alpha[...,None]

        region = city_img[y:y+target_h,x:x+new_w]
        blend = (region*(1-alpha) + crop*alpha).astype(np.uint8)
        region[crop_mask] = blend[crop_mask]
        city_img[y:y+target_h,x:x+new_w] = region

        anomaly_mask = np.zeros((H,W),dtype=np.uint8)
        anomaly_mask[y:y+target_h,x:x+new_w][crop_mask] = 1

        return city_img, anomaly_mask.astype(np.uint8)