# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import yaml
import glob
import torch
import random
from PIL import Image
import numpy as np
from models.eomt import EoMT
from models.vit import ViT
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import warnings
import importlib

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
    Resize((1024, 1024), Image.BILINEAR), # EoMT Giant usa solitamente 1280
    ToTensor(),
    # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Standard ImageNet/DINO
])

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


# load custom state dict
def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():

        ##DUBBI SE FARLO##
        if name.startswith("network."):
            name = name.replace("network.", "")
        ##

        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model



# extract state_dict from checkpoint 
def extract_state_dict(checkpoint):
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint

# funtion to load eomt model

def load_eomt(args, device):
    
    #yaml_path = "eomt_base_640_2x.yaml" 
    yaml_path= args.config_path
    with open(yaml_path, 'r') as f: #TO dO: impostare yaml_path
        config = yaml.safe_load(f)

    print(f"Caricamento configurazione da: {yaml_path}")

    
    
    encoder = ViT(
        img_size=(1024, 1024), #TO DO: check con H,W upscaling line 165
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2",  
    )

    
    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100,
        num_blocks=3,  #TO DO: check 3 o 4??
        masked_attn_enabled=True,
    ).to(device)


    
    state_dict_path = args.dict_path
    

    # 5. Carica i pesi
    checkpoint = torch.load(state_dict_path, map_location=device, weights_only=True)
    checkpoint = extract_state_dict(checkpoint)
    model = load_my_state_dict(model, checkpoint)

    model.eval()
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument("--config_path", default="/content/drive/MyDrive/PROGETTO_ML/ML_Project/eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")
    parser.add_argument("--dict_path", default="/content/drive/MyDrive/PROGETTO_ML/ML_Project/trained_models/eomt_cityscapes.bin")
    parser.add_argument('--loadDir',default="../eomt")
    parser.add_argument('--loadModel', default="eomt.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################################
    model = load_eomt(args, device) # caricamento pesi e modello??????????????
    ##############################################

    

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
         print(path)
         images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
         #images = images.permute(0,3,1,2)
         with torch.no_grad():
             result = model(images)

         mask_logits = result[0][-1] #last layer mask logits
         class_logits = result[1][-1] #last layer class logits 

         #H=256
         #W=512
         target_size = (512, 1024) #TO DO: chek size with config model

         # upsampling
         mask_logits_upsampled = F.interpolate(
             mask_logits,
             size=target_size,
             mode="bilinear",
             align_corners=False
         )

         # mask probabilities
         mask_probs = torch.sigmoid(mask_logits_upsampled)

         # class probabilities
         class_probs = torch.softmax(class_logits, dim=-1) #-1 is the class dimension class_logits.shape=[B,Q,C+1]

         #final pixel-wise segmentation
         Mat_Class=class_probs.transpose(1,2)
         Mat_Mask=torch.flatten(input=mask_probs, start_dim=2)

         pixel_logits=torch.matmul(Mat_Class, Mat_Mask)
         pixel_logits = pixel_logits.unflatten(2, (H, W)) #return to H x W map from H*W vector
         pixel_logits = pixel_logits.squeeze(0) #loose Batch size dimension
         pixel_logits = pixel_logits[:-1, :, :] #remove last class
        

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


         # GESTIONE GROUND TRUTH 
         pathGT = path.replace("images", "labels_masks")    
         #anomaly_result_rba = - torch.sum( torch.tanh(pixel_logits.data.cpu()), dim = 0) 
         if "RoadObsticle21" in pathGT:
             pathGT = pathGT.replace("webp", "png")
         if "fs_static" in pathGT:
             pathGT = pathGT.replace("jpg", "png")                
         if "RoadAnomaly" in pathGT:
             pathGT = pathGT.replace("jpg", "png")  
         
         mask = Image.open(pathGT)
         mask = target_transform(mask)
         ood_gts = np.array(mask)
         if "RoadAnomaly" in pathGT:
             ood_gts = np.where((ood_gts==2), 1, ood_gts)
         if "LostAndFound" in pathGT:
             ood_gts = np.where((ood_gts==0), 255, ood_gts)
             ood_gts = np.where((ood_gts==1), 0, ood_gts)
             ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

         if "Streethazard" in pathGT:
             ood_gts = np.where((ood_gts==14), 255, ood_gts)
             ood_gts = np.where((ood_gts<20), 0, ood_gts)
             ood_gts = np.where((ood_gts==255), 1, ood_gts)

         if 1 not in np.unique(ood_gts):
             continue           
         else:
              ood_gts_list.append(ood_gts)
              anomaly_score_MSP_list.append(anomaly_result_MSP)
              anomaly_score_MaxLogit_list.append(anomaly_result_MaxLogit)
              anomaly_score_Entropy_list.append(anomaly_result_Entropy)
              anomaly_score_Rba_list.append(anomaly_result_Rba)
         del result, anomaly_result_MSP, anomaly_result_MaxLogit, anomaly_result_Entropy, anomaly_result_Rba, ood_gts, mask #, anomaly_result_Rba
         torch.cuda.empty_cache()



    file.write( "\n")

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
    
    #####
    # TRASFORMAZIONE LISTE IN ARRAY
    ood_gts_list = np.array(ood_gts_list)
    anomaly_score_MSP_list = np.array(anomaly_score_MSP_list)
    anomaly_score_MaxLogit_list = np.array(anomaly_score_MaxLogit_list)
    anomaly_score_Entropy_list = np.array(anomaly_score_Entropy_list)

    # CALCOLO METRICHE
    [prc_auc_MSP, fpr_MSP] = eval_metrics(ood_gts_list, anomaly_score_MSP_list)
    [prc_auc_MaxLogit, fpr_MaxLogit] = eval_metrics(ood_gts_list, anomaly_score_MaxLogit_list)
    [prc_auc_Entropy, fpr_Entropy] = eval_metrics(ood_gts_list, anomaly_score_Entropy_list)

    # STAMPA RISULTATI
    print(f"MSP: AUPRC {prc_auc_MSP*100:.2f}, FPR95 {fpr_MSP*100:.2f}")
    print(f"MaxLogit: AUPRC {prc_auc_MaxLogit*100:.2f}, FPR95 {fpr_MaxLogit*100:.2f}")
    print(f"Entropy: AUPRC {prc_auc_Entropy*100:.2f}, FPR95 {fpr_Entropy*100:.2f}")
    #####



   
    [prc_auc_Rba, fpr_Rba] = eval_metrics(ood_gts_list, anomaly_score_Rba_list)
    print(f'AUPRC rba score: {prc_auc_Rba*100.0}', f'FPR@TPR95 rba: {fpr_Rba*100.0}')


    file.write(('      AUPRC softmax score:' + str(prc_auc_MSP*100.0) + '   FPR@TPR95 softmax:' + str(fpr_MSP*100.0) +
                '\n    AUPRC logit score:' + str(prc_auc_MaxLogit*100.0) + '   FPR@TPR95 logit:' + str(fpr_MaxLogit*100.0) +
                '\n    AUPRC entropy score:' + str(prc_auc_Entropy*100.0) + '   FPR@TPR95 entropy:' + str(fpr_Entropy*100.0) + '\n'
                '\n    AUPRC rba score:' + str(prc_auc_Rba*100.0) + '   FPR@TPR95 rba:' + str(fpr_Rba*100.0) + '\n'
                ))

    file.close()

if __name__ == '__main__':
    main()
