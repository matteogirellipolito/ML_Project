import os
import glob
import yaml
import torch
import random
import importlib
import numpy as np
import torch.nn.functional as F

from PIL import Image
from argparse import ArgumentParser
from torch.amp import autocast

from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr

from torchvision.transforms import ToTensor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMG_SIZE = 1024

input_transform = ToTensor()

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

    anomaly_score_list = []
    ood_gts_list = []

    image_paths = glob.glob(os.path.expanduser(str(args.input[0])))

    for path in image_paths:

        print(path)
        image = Image.open(path).convert("RGB")
        image = input_transform(image)
        image = image.unsqueeze(0).float().to(device)

        with torch.no_grad():
            image = image.squeeze(0)
            image = (image * 255).to(torch.uint8)
            image = [image.to(device)]
            img_sizes = [img.shape[-2:] for img in image]

            with autocast(dtype=torch.float16,device_type="cuda"):

                crops, origins = model.window_imgs_semantic(image)

                mask_logits_per_layer, class_logits_per_layer = model(crops)
                mask_logits = F.interpolate(mask_logits_per_layer[-1],(IMG_SIZE, IMG_SIZE),mode="bilinear")

                crop_logits = model.to_per_pixel_logits_semantic(mask_logits,class_logits_per_layer[-1])

                logits = model.revert_window_logits_semantic(crop_logits,origins,img_sizes)

        temperature_results = {}

        for temp in args.temperatures:
            scaled_logits = logits[0] / temp
            softmax_probs = torch.softmax(scaled_logits,dim=0)
            msp_map = 1.0 - torch.max(softmax_probs,dim=0)[0]
            anomaly_result = msp_map.cpu().numpy()

            if temp not in temperature_results:
                temperature_results[temp] = []
            temperature_results[temp].append(anomaly_result)

        pathGT = path.replace("images","labels_masks")

        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp","png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg","png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg","png")

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2),1,ood_gts)

        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0),255,ood_gts)
            ood_gts = np.where((ood_gts == 1),0,ood_gts)
            ood_gts = np.where((ood_gts > 1)&(ood_gts < 201),1,ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14),255,ood_gts)
            ood_gts = np.where((ood_gts < 20),0,ood_gts)
            ood_gts = np.where((ood_gts == 255),1,ood_gts)

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)

    ood_gts = np.array(ood_gts_list)

    for temp in args.temperatures:

        anomaly_scores = np.array(temperature_results[temp])

        ood_mask = ood_gts == 1
        ind_mask = ood_gts == 0

        ood_out = anomaly_scores[ood_mask[temp]]
        ind_out = anomaly_scores[ind_mask[temp]]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label,val_out)
        fpr = fpr_at_95_tpr(val_out,val_label)

        print(f"\nTemperature: {temp}")
        print(f"AUPRC score: {prc_auc * 100.0}")
        print(f"FPR@TPR95: {fpr * 100.0}\n")

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  

    parser.add_argument("--checkpoint",type=str,default="/content/drive/MyDrive/ML_Project/eomt_cityscapes.bin")

    parser.add_argument("--temperatures",type=float,nargs="+",default=[0.5, 0.75, 1.0, 1.1])

    parser.add_argument("--cpu",action="store_true")

    args = parser.parse_args()
    main(args)