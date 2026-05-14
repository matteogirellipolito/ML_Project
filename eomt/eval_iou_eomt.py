import random
import numpy as np
import torch
import torch.nn.functional as F

from argparse import ArgumentParser

from datasets.cityscapes_semantic import CityscapesSemantic

from models.eomt import EoMT
from models.vit import ViT

from iouEval import iouEval, getColorEntry

# =========================================================
# SEED
# =========================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =========================================================
# CONFIG
# =========================================================

# 19 classi Cityscapes valide + 1 ignore (indice 19) = 20 per iouEval
NUM_CLASSES = 20

# =========================================================
# CHECKPOINT
# =========================================================

def extract_state_dict(checkpoint):

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


def load_my_state_dict(model, state_dict):

    own_state = model.state_dict()

    loaded = 0
    missing_keys = []
    shape_mismatch = []

    for name, param in state_dict.items():

        original_name = name

        if name.startswith("network."):
            name = name.replace("network.", "")

        if name not in own_state:
            missing_keys.append(original_name)
            continue

        if own_state[name].shape != param.shape:
            shape_mismatch.append((original_name, param.shape, own_state[name].shape))
            continue

        own_state[name].copy_(param)
        loaded += 1

    print(f"\nLoaded parameters: {loaded}")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Shape mismatches: {len(shape_mismatch)}")

    for k, ckpt_shape, model_shape in shape_mismatch:
        print(f"  SHAPE MISMATCH: {k} | ckpt={ckpt_shape} model={model_shape}")

    return model


# =========================================================
# MODEL
# =========================================================

def load_eomt(checkpoint_path, device):

    print("Loading EoMT...")

    encoder = ViT(
        img_size=(640, 640),
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )

    model = EoMT(
        encoder=encoder,
        num_classes=19,       # 19 classi Cityscapes → class_head avrà 20 output (19+1 void)
        num_q=100,
        num_blocks=3,
        masked_attn_enabled=False,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    checkpoint = extract_state_dict(checkpoint)

    model = load_my_state_dict(model, checkpoint)

    model.eval()

    return model


# =========================================================
# TARGET CONVERSION
# =========================================================

def target_to_semantic(target):
    """Converte le maschere binarie in mappa semantica 2D."""

    masks  = target["masks"]
    labels = target["labels"]

    H, W = masks.shape[-2:]

    # Inizializza tutto a 19 (ignore label)
    semantic = torch.ones((H, W), dtype=torch.long) * 19

    for mask, cls in zip(masks, labels):
        semantic[mask.bool()] = cls

    return semantic


# =========================================================
# MAIN
# =========================================================

def main():

    parser = ArgumentParser()

    parser.add_argument("--data_path",  required=True,
                        help="Cartella con i zip di Cityscapes")
    parser.add_argument("--checkpoint", required=True,
                        help="Path al checkpoint EoMT (.bin o .pth)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================
    # MODEL
    # =====================================================

    model = load_eomt(args.checkpoint, device)

    # =====================================================
    # DATASET
    # =====================================================

    datamodule = CityscapesSemantic(
        path=args.data_path,
        batch_size=1,
        num_workers=2,
        img_size=(640, 640),
    )

    datamodule.setup()

    loader = datamodule.val_dataloader()

    print(f"Found {len(datamodule.cityscapes_val_dataset)} validation images")

    # =====================================================
    # IOU EVAL
    # iouEval(20, ignoreIndex=19):
    #   - valuta classi 0-18 (le 19 classi Cityscapes)
    #   - ignora pixel dove GT=19 (ignore label)
    # =====================================================

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=19)

    # =====================================================
    # LOOP
    # =====================================================

    for step, batch in enumerate(loader):

        print(f"Step {step}")

        images, targets = batch

        image = images[0].unsqueeze(0)

        image = F.interpolate(
            image,
            size=(640, 640),
            mode="bilinear",
            align_corners=False
        )

        image = image.to(device)

        target = targets[0]

        # =================================================
        # GROUND TRUTH
        # =================================================

        semantic_gt = target_to_semantic(target)   # [H, W], valori 0-19

        semantic_gt = (
            semantic_gt
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )

        semantic_gt = F.interpolate(
            semantic_gt,
            size=(640, 640),
            mode="nearest"
        )

        semantic_gt = semantic_gt.long().cpu()     # [1, 1, 640, 640]

        # =================================================
        # FORWARD PASS
        # =================================================

        with torch.no_grad():
            result = model(image)

        mask_logits  = result[0][-1]   # [1, num_q, H_small, W_small]
        class_logits = result[1][-1]   # [1, num_q, 20]  (19 classi + void)

        # =================================================
        # UPSAMPLE MASCHERE
        # =================================================

        mask_logits = F.interpolate(
            mask_logits,
            size=semantic_gt.shape[-2:],
            mode="bilinear",
            align_corners=False
        )                                          # [1, num_q, 640, 640]

        # =================================================
        # PIXEL LOGITS
        # =================================================

        mask_probs  = torch.sigmoid(mask_logits)           # [1, num_q, 640, 640]
        class_probs = torch.softmax(class_logits, dim=-1)  # [1, num_q, 20]

        Mat_Class = class_probs.transpose(1, 2)            # [1, 20, num_q]
        Mat_Mask  = torch.flatten(mask_probs, start_dim=2) # [1, num_q, 640*640]

        pixel_logits = torch.matmul(Mat_Class, Mat_Mask)   # [1, 20, 640*640]

        H, W = semantic_gt.shape[-2], semantic_gt.shape[-1]
        pixel_logits = pixel_logits.unflatten(2, (H, W))   # [1, 20, 640, 640]
        pixel_logits = pixel_logits.squeeze(0)             # [20, 640, 640]

        # Rimuovi la classe void (ultimo canale) prima dell'argmax
        # → predizioni nell'intervallo 0-18 (le 19 classi valide)
        pixel_logits = pixel_logits[:-1]                   # [19, 640, 640]

        # =================================================
        # PREDIZIONE
        # =================================================

        prediction = torch.argmax(pixel_logits, dim=0)    # [640, 640], valori 0-18

        prediction = (
            prediction
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
        )                                                  # [1, 1, 640, 640]

        # =================================================
        # AGGIORNA IOU
        # =================================================

        iouEvalVal.addBatch(prediction, semantic_gt)

    # =====================================================
    # RISULTATI
    # =====================================================

    iouVal, iou_classes = iouEvalVal.getIoU()

    class_names = [
        "Road", "Sidewalk", "Building", "Wall", "Fence",
        "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
        "Sky", "Person", "Rider", "Car", "Truck",
        "Bus", "Train", "Motorcycle", "Bicycle"
    ]

    print("---------------------------------------")
    print("Per-Class IoU:")

    for i in range(iou_classes.size(0)):
        iouStr = (
            getColorEntry(iou_classes[i].item())
            + '{:0.2f}'.format(iou_classes[i].item() * 100)
            + '\033[0m'
        )
        print(f"  {class_names[i]}: {iouStr}")

    print("=======================================")

    iouStr = (
        getColorEntry(iouVal.item())
        + '{:0.2f}'.format(iouVal.item() * 100)
        + '\033[0m'
    )

    print("MEAN IoU:", iouStr, "%")


if __name__ == "__main__":
    main()