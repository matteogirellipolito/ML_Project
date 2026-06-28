import random
import numpy as np

from PIL import Image
from scipy.ndimage import gaussian_filter

random.seed(42)
np.random.seed(42)

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
        +256*color[:,:,1].astype(np.int64)
        +256*256*color[:,:,2].astype(np.int64)
    )

def contaminate_cityscapes_image(
    city_image,
    pano,
    coco_dir,
    pano_mask_dir
):

    city = city_image.copy()

    ann=random.choice(pano["annotations"])

    valid=[]

    for s in ann["segments_info"]:

        if (
            s["category_id"] in VALID_CATEGORIES
            and s["area"]>3000
            and s["iscrowd"]==0
        ):
            valid.append(s)

    if len(valid)==0:
        return contaminate_cityscapes_image(
            city,
            pano,
            coco_dir,   
            pano_mask_dir,
        )

    seg=random.choice(valid)

    mask_rgb=np.array(
        Image.open(
            pano_mask_dir+"/"+ann["file_name"]
        )
    )

    ids=rgb2id(mask_rgb)

    inst_mask=ids==seg["id"]

    ys,xs=np.where(inst_mask)

    coco_img=np.array(
        Image.open(
            coco_dir+"/"+
            ann["file_name"].replace(".png",".jpg")
        ).convert("RGB")
    )

    y1,y2=ys.min(),ys.max()
    x1,x2=xs.min(),xs.max()

    obj=coco_img[y1:y2,x1:x2]
    mask=inst_mask[y1:y2,x1:x2]

    H,W=city.shape[:2]

    scale=np.random.uniform(
        0.15,
        0.30
    )*H/max(obj.shape[:2])

    nh=int(obj.shape[0]*scale)
    nw=int(obj.shape[1]*scale)

    obj=np.array(
        Image.fromarray(obj).resize(
            (nw,nh)
        )
    )

    mask=np.array(
        Image.fromarray(
            mask.astype(np.uint8)*255
        ).resize(
            (nw,nh)
        )
    )>0

    px=random.randint(
        W//3,
        W-nw-1
    )

    py=random.randint(
        H//2,
        H-nh-1
    )

    result=city.copy()

    soft=gaussian_filter(
        mask.astype(float),
        sigma=1
    )

    alpha=soft[...,None]

    reg=result[
        py:py+nh,
        px:px+nw
    ]

    blend=(
        reg*(1-alpha)
        +obj*alpha
    ).astype(np.uint8)

    reg[mask]=blend[mask]

    result[
        py:py+nh,
        px:px+nw
    ]=reg

    anomaly=np.zeros(
        (H,W),
        dtype=np.uint8
    )

    anomaly[
        py:py+nh,
        px:px+nw
    ][mask]=1

    return result,anomaly