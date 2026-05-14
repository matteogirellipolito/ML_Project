# Copyright (c) OpenMMLab. All rights reserved.

import os
import cv2
import yaml
import glob
import torch
import random
import warnings
import importlib
import numpy as np
import os.path as osp

from PIL import Image
from argparse import ArgumentParser

from models.eomt import EoMT
from models.vit import ViT

from ood_metrics import (
    fpr_at_95_tpr,
    calc_metrics,
    plot_roc,
    plot_pr,
    plot_barcode,
)

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
)

import torch.nn.functional as F


seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 19

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# image preprocessing
input_transform = Compose([
    Resize((1024, 1024), Image.BILINEAR),
    ToTensor(),
        # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Standard ImageNet/DINO
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])


## MODEL LOADING 

def load_my_state_dict(model, state_dict):

    model_state = model.state_dict()

    loaded_keys = []
    skipped_keys = []

    for name, param in state_dict.items():

        # remove DataParallel prefix
        if name.startswith("module."):
            name = name[len("module."):]

        if name not in model_state:
            skipped_keys.append(name)
            continue

        if model_state[name].shape != param.shape:
            print(
                f"Shape mismatch for {name}: "
                f"{param.shape} vs {model_state[name].shape}"
            )
            skipped_keys.append(name)
            continue

        model_state[name].copy_(param)
        loaded_keys.append(name)

    print(f"\nLoaded keys: {len(loaded_keys)}")
    print(f"Skipped keys: {len(skipped_keys)}")

    if len(skipped_keys) > 0:
        print("\nFirst skipped keys:")
        for k in skipped_keys[:20]:
            print(k)

    return model

# extract state_dict from checkpoint 
def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model" in checkpoint:
            return checkpoint["model"]

    return checkpoint

# funtion to load eomt model
def load_eomt(args, device):

    yaml_path = osp.join(args.loadConfigDir, args.loadConfig)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading config from: {yaml_path}")
    
    network_cfg = (
        config["model"]
        ["init_args"]
        ["network"]
        ["init_args"]
    )

    encoder_cfg = (
        network_cfg
        ["encoder"]
        ["init_args"]
    )

    backbone_name = encoder_cfg["backbone_name"]

    num_queries = network_cfg["num_q"]

    num_blocks = network_cfg["num_blocks"]

    img_size=(1024, 1024) #or (512, 1024) ????

    print(f"Backbone: {backbone_name}")
    print(f"Num queries: {num_queries}")
    print(f"Num blocks: {num_blocks}")

    encoder = ViT(
        img_size=img_size,
        patch_size=16,
        backbone_name=backbone_name,
    )

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=num_queries,
        num_blocks=num_blocks,
        masked_attn_enabled=True,
    ).to(device)

    state_dict_path = args.loadWeights

    print(f"Loading checkpoint from: {state_dict_path}")

    checkpoint = torch.load(
        state_dict_path,
        map_location=device
    )

    checkpoint = extract_state_dict(checkpoint)

    model = load_my_state_dict(model, checkpoint)

    model.eval()

    return model


#MAIN

def main():

    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help=(
            "A list of space separated input images; "
            "or a single glob pattern such as 'directory/*.jpg'"
        ),
    )

    parser.add_argument('--loadDir', default="../eomt")
    parser.add_argument('--loadModel', default="eomt.py")

    parser.add_argument('--loadConfigDir', default='../eomt/configs/dinov2/cityscapes/semantic/')
    parser.add_argument('--loadConfig', default='eomt_base_640.yaml')
    parser.add_argument('--loadWeights',default='/content/drive/MyDrive/eomt_cityscapes.bin')

    parser.add_argument('--subset', default="val")

    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)

    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    anomaly_score_MSP_list = []
    anomaly_score_MaxLogit_list = []
    anomaly_score_Entropy_list = []
    anomaly_score_Rba_list = []

    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()

    file = open('results.txt', 'a')

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Using device: {device}")

    
    model = load_eomt(args, device)

    
    #inference loop
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        #images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)

        mask_logits = result[0][-1] #last layer mask logits
        class_logits = result[1][-1] #last layer class logits 

        H=512
        W=1024
        target_size = (H, W) #TO DO: chek size with config model

        # masks upsampling
        mask_logits = F.interpolate(
            mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

         # mask probabilities
        mask_probs = torch.sigmoid(mask_logits)

        # class probabilities
        class_probs = torch.softmax(class_logits, dim=-1) #-1 is the class dimension class_logits.shape=[B,Q,C+1]


        #pixel-wise segmentation
        Mat_Class=class_probs.transpose(1,2)
        Mat_Mask=torch.flatten(input=mask_probs, start_dim=2)

        pixel_logits=torch.matmul(Mat_Class, Mat_Mask)
        pixel_logits = pixel_logits.unflatten(2, (H, W)) #return to H x W map from H*W vector
        pixel_logits = pixel_logits.squeeze(0) #loose Batch size dimension
        print("pixel_logits shape:", pixel_logits.shape)
        pixel_logits=pixel_logits[:-1, :, :]
        print("pixel_logits shape after pixel_logits=pixel_logits[:-1, :, :]:", pixel_logits.shape)

        
        # pixel probabilities
        pixel_probs = torch.softmax(pixel_logits, dim=0)

        pixel_probs_np = pixel_probs.cpu().numpy()

        pixel_logits_np = pixel_logits.cpu().numpy()

        #evaluation scores

        anomaly_result_MSP = (
            1.0 - np.max(pixel_probs_np, axis=0)
        )

        anomaly_result_MaxLogit = (
            -np.max(pixel_logits_np, axis=0)
        )

        anomaly_result_Entropy = (
            -np.sum(
                pixel_probs_np *
                np.log(pixel_probs_np + 1e-9),
                axis=0
            )
        )

        anomaly_result_Rba = (
            -torch.sum(
                torch.tanh(pixel_logits.cpu()),
                dim=0
            ).numpy()
        )


        #Ground Truth
        pathGT = path.replace("images", "labels_masks")

        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # dataset specific processing

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where(
                (ood_gts > 1) & (ood_gts < 201),
                1,
                ood_gts
            )
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):    # skip images without anomalies
            continue

        # store
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_MSP_list.append(anomaly_result_MSP)
             anomaly_score_MaxLogit_list.append(anomaly_result_MaxLogit)
             anomaly_score_Entropy_list.append(anomaly_result_Entropy)
             anomaly_score_Rba_list.append(anomaly_result_Rba)
        del result, anomaly_result_MSP, anomaly_result_MaxLogit, anomaly_result_Entropy ,ood_gts, mask #, anomaly_result_Rba
        torch.cuda.empty_cache()

    file.write("\n")

  # metrics evaluation

    def eval_metrics(ood_gts_list, anomaly_score_list):

        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        return [prc_auc, fpr]

    #evaluation metrics for all scores

    [prc_auc_MSP, fpr_MSP] = eval_metrics(ood_gts_list, anomaly_score_MSP_list)
    [prc_auc_MaxLogit, fpr_MaxLogit] = eval_metrics(ood_gts_list, anomaly_score_MaxLogit_list)
    [prc_auc_Entropy, fpr_Entropy] = eval_metrics(ood_gts_list, anomaly_score_Entropy_list)
    [prc_auc_Rba, fpr_Rba] = eval_metrics(ood_gts_list, anomaly_score_Rba_list)

    print(f'AUPRC MSP score: {prc_auc_MSP*100.0}')
    print(f'FPR@TPR95 MSP: {fpr_MSP*100.0}')

    print(f'AUPRC MaxLogit score: {prc_auc_MaxLogit*100.0}')
    print(f'FPR@TPR95 MaxLogit: {fpr_MaxLogit*100.0}')

    print(f'AUPRC Entropy score: {prc_auc_Entropy*100.0}')
    print(f'FPR@TPR95 Entropy: {fpr_Entropy*100.0}')

    print(f'AUPRC Rba score: {prc_auc_Rba*100.0}')
    print(f'FPR@TPR95 Rba: {fpr_Rba*100.0}')


  
    file.write(
        (
            'AUPRC softmax score: '
            + str(prc_auc_MSP * 100.0)
            + '   FPR@TPR95 softmax: '
            + str(fpr_MSP * 100.0)

            + '\nAUPRC logit score: '
            + str(prc_auc_MaxLogit * 100.0)
            + '   FPR@TPR95 logit: '
            + str(fpr_MaxLogit * 100.0)

            + '\nAUPRC entropy score: '
            + str(prc_auc_Entropy * 100.0)
            + '   FPR@TPR95 entropy: '
            + str(fpr_Entropy * 100.0)

            + '\nAUPRC rba score: '
            + str(prc_auc_Rba * 100.0)
            + '   FPR@TPR95 rba: '
            + str(fpr_Rba * 100.0)

            + '\n'
        )
    )

    file.close()


if __name__ == '__main__':
    main()
