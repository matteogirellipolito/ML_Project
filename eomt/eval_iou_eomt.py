import os
import time
import torch
from argparse import ArgumentParser
import random
import numpy as np
import sys
import yaml
import importlib

from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from eval.dataset import cityscapes
from eval.transform import Relabel, ToLabel
from eval.iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20 
IMG_SIZE = 1024

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

image_transform = ToPILImage()
input_transform = ToTensor()

target_transform = Compose([
    ToLabel(),
    Relabel(255, 19),   
])

def main(args):

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    config_path = 'configs/dinov2/cityscapes/semantic/eomt_base_640.yaml'
    config = yaml.safe_load(open(config_path))

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=(IMG_SIZE, IMG_SIZE), **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(masked_attn_enabled=False,num_classes=19,encoder=encoder,**network_kwargs)

    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = lit_cls(img_size=(IMG_SIZE, IMG_SIZE),num_classes=19,network=network,**model_kwargs).eval().to(device)

    if device.type == 'cpu':
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    else:
        state_dict = torch.load(args.checkpoint, map_location=f"cuda:{0}", weights_only=True)

    model.load_state_dict(state_dict, strict=False)

    print('Model weights loaded succesfully')
    
    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    model.eval()

    dataset = cityscapes(args.datadir,input_transform,target_transform,subset=args.subset)

    loader = DataLoader(dataset,num_workers=args.num_workers,batch_size=args.batch_size,shuffle=False)

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (image, labels, filename,filenameGt) in enumerate(loader):
        image = image.to(device)
        labels = labels.long().to(device)

        with torch.no_grad():
            image = image.squeeze(0)
            image = [(image * 255).to(torch.uint8)]
            img_sizes = [img.shape[-2:] for img in image]
        
            crops, origins = model.window_imgs_semantic(image)

            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = F.interpolate(mask_logits_per_layer[-1], (IMG_SIZE, IMG_SIZE), mode="bilinear")
        
            crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_per_layer[-1])
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        prediction = logits[0].max(0)[1].unsqueeze(0).unsqueeze(0)

        iouEvalVal.addBatch(prediction, labels)

        filenameSave = filename[0].split("leftImg8bit/")[-1]

        print (step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)
    
    
    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
        
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--datadir",default="/content/drive/MyDrive/Anomaly_Validation_Datasets/Cityscapes_val")

    parser.add_argument("--subset",default="val",choices=["train", "val", "test"])
    
    parser.add_argument("--cpu", action="store_true")
    
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--batch-size', type=int, default=1)
    
    parser.add_argument("--checkpoint",type=str,default="/content/drive/MyDrive/ML_Project/eomt_cityscapes.bin")

    args = parser.parse_args()
    main(args)