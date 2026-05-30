import random
import numpy as np

from PIL import Image
from scipy.ndimage import gaussian_filter

VALID_CATEGORIES = {

    # animals    
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",

    # accessories / sports 
    27: "backpack",
    28: "umbrella",
    31: "handbag", 
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    
    # household objects
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    
    # furniture
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    70: "toilet",
    
    # electronics
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",

    # miscellaneous objects    
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

def rgb2id(color):
    if len(color.shape) == 3:
        return (
            color[:,:,0].astype(np.int64)
            + 256 * color[:,:,1].astype(np.int64)
            + 256*256*color[:,:,2].astype(np.int64)
        )
    return color

def adaptive_resize(
    obj_img,
    obj_mask,
    city_img,
    min_frac=0.18,
    max_frac=0.35,
):
    city_h = city_img.shape[0]
    target_size = np.random.uniform(min_frac,max_frac) * city_h
    h,w = obj_img.shape[:2]
    scale = target_size / max(h,w)

    new_h = max(60, int(h*scale))
    new_w = max(60, int(w*scale))
    max_pixels = int(city_h * 0.45)
    if max(new_h, new_w) > max_pixels:
        scale_down = max_pixels / max(new_h, new_w)
        new_h = int(new_h * scale_down)
        new_w = int(new_w * scale_down)

    obj_img = np.array(Image.fromarray(obj_img).resize((new_w,new_h),Image.BILINEAR))
    obj_mask = np.array(Image.fromarray(obj_mask.astype(np.uint8)*255).resize((new_w,new_h),Image.NEAREST)) > 0

    return obj_img,obj_mask

def contaminate_cityscapes_image(
    city_path,
    pano,
    coco_img_dir,
    panoptic_mask_dir,
):
    city_img = np.array(Image.open(city_path).convert("RGB"))
    valid = False

    while not valid:
        ann = random.choice(pano["annotations"])
        segments = [
            s for s in ann["segments_info"]
            if s["category_id"] in VALID_CATEGORIES.keys()
            and s["area"] > 1500
            and s["iscrowd"] == 0
        ]

        if len(segments) == 0:
            continue

        seg = random.choice(segments)
        bbox = seg["bbox"]
        
        if seg["area"] > 8000 and bbox[2]*bbox[3] > 10000:
            valid = True

    mask_rgb = np.array(Image.open(panoptic_mask_dir + "/" + ann["file_name"]))
    ids = rgb2id(mask_rgb)
    inst_mask = ids == seg["id"]
    ys,xs = np.where(inst_mask)

    if len(xs) == 0:
        return contaminate_cityscapes_image(
            city_path,
            pano,
            coco_img_dir,
            panoptic_mask_dir
        )

    coco_name = ann["file_name"].replace(".png",".jpg")
    coco_img = np.array(Image.open(coco_img_dir + "/" + coco_name).convert("RGB"))

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    obj_h = y2 - y1
    obj_w = x2 - x1

    pad_y = int(obj_h * 0.20)
    pad_x = int(obj_w * 0.20)

    y1 = max(0, y1 - pad_y)
    y2 = min(coco_img.shape[0], y2 + pad_y)

    x1 = max(0, x1 - pad_x)
    x2 = min(coco_img.shape[1], x2 + pad_x)

    crop_img = coco_img[
        y1:y2,
        x1:x2
    ]

    crop_mask = inst_mask[
        y1:y2,
        x1:x2
    ]

    coverage = crop_mask.sum() / crop_mask.size

    if coverage < 0.12:
        return contaminate_cityscapes_image(
            city_path,
            pano,
            coco_img_dir,
            panoptic_mask_dir
        )

    obj_img,obj_mask = adaptive_resize(
        crop_img,
        crop_mask,
        city_img
    )
    result = city_img.copy()
    H,W = result.shape[:2]
    oh,ow = obj_img.shape[:2]
    max_x = max(W-ow-20, W//3 + 1)
    max_y = max(H-oh-20, H//2 + 1)
    
    x = np.random.randint(W//3,max_x)
    y = np.random.randint(H//2,max_y)

    anomaly_mask = np.zeros((H,W),dtype=np.uint8)
    region = result[y:y+oh,x:x+ow]
    soft_mask = gaussian_filter(obj_mask.astype(float),sigma=2)
    soft_mask = np.expand_dims(soft_mask,axis=-1)

    blended = (region*(1-soft_mask) + obj_img*soft_mask).astype(np.uint8)

    result[y:y+oh,x:x+ow] = blended

    anomaly_mask[y:y+oh,x:x+ow][obj_mask] = 1

    return result, anomaly_mask