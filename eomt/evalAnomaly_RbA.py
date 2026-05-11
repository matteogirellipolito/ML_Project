import os
import yaml
import torch
import importlib
import warnings
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from lightning import seed_everything
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

seed_everything(0, verbose=False)

# ==========================
# CONFIG
# ==========================
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True
)

parser.add_argument(
    "--input",
    type=str,
    required=True
)

parser.add_argument(
    "--device",
    type=int,
    default=0
)

args = parser.parse_args()

device = args.device
img_idx = 0
config_path = args.config
data_path = args.input

# ==========================
# LOAD CONFIG
# ==========================
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# ==========================
# LOAD DATASET
# ==========================
data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
data_module = getattr(importlib.import_module(data_module_name), class_name)

data_module_kwargs = config["data"].get("init_args", {})

data = data_module(
    path=data_path,
    batch_size=1,
    num_workers=0,
    check_empty_targets=False,
    **data_module_kwargs
)

data.setup()

# ==========================
# LOAD ENCODER
# ==========================
encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)

encoder = encoder_cls(
    img_size=data.img_size,
    **encoder_cfg.get("init_args", {})
)

# ==========================
# LOAD NETWORK (EoMT)
# ==========================
network_cfg = config["model"]["init_args"]["network"]
network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
network_cls = getattr(importlib.import_module(network_module_name), network_class_name)

network_kwargs = {
    k: v for k, v in network_cfg["init_args"].items()
    if k != "encoder"
}

network = network_cls(
    masked_attn_enabled=False,
    num_classes=data.num_classes,
    encoder=encoder,
    **network_kwargs
)

# ==========================
# LOAD LIGHTNING MODULE
# ==========================
lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)

model_kwargs = {
    k: v for k, v in config["model"]["init_args"].items()
    if k != "network"
}

if "stuff_classes" in config["data"].get("init_args", {}):
    model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

model = lit_cls(
    img_size=data.img_size,
    num_classes=data.num_classes,
    network=network,
    **model_kwargs
).eval().to(device)

# ==========================
# LOAD PRETRAINED WEIGHTS
# ==========================
warnings.filterwarnings("ignore")

name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

try:
    state_dict_path = hf_hub_download(
        repo_id=f"tue-mps/{name}",
        filename="pytorch_model.bin",
    )

    state_dict = torch.load(
        state_dict_path,
        map_location=f"cuda:{device}",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=False)

except RepositoryNotFoundError:
    raise RuntimeError("Checkpoint non trovato.")

# ==========================
# RbA SCORE
# ==========================
def compute_rba(mask_logits, class_logits):
    class_probs = class_logits.softmax(dim=-1)[..., :-1]
    uncertainty = 1.0 - class_probs.max(dim=-1)[0]

    mask_probs = mask_logits.sigmoid()

    rba = torch.einsum(
        "bqhw,bq->bhw",
        mask_probs,
        uncertainty
    )

    return rba

# ==========================
# INFERENCE
# ==========================
def infer_rba(img):
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):

        imgs = [img.to(device)]
        transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(transformed)

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1],
            model.img_size,
            mode="bilinear"
        )

        rba = compute_rba(
            mask_logits,
            class_logits_per_layer[-1]
        )[0]

        rba = F.interpolate(
            rba[None, None],
            img.shape[-2:],
            mode="bilinear"
        )[0, 0]

    return rba.cpu().numpy()

# ==========================
# HEATMAP
# ==========================
def plot_rba(img, rba):
    img_np = img.permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(16,6))

    plt.subplot(1,3,1)
    plt.imshow(img_np)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(rba, cmap="hot")
    plt.title("RbA Heatmap")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(img_np)
    plt.imshow(rba, cmap="hot", alpha=0.55)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==========================
# RUN
# ==========================
img, target = data.val_dataloader().dataset[img_idx]

rba = infer_rba(img)

print("RbA stats")
print("Min:", np.min(rba))
print("Max:", np.max(rba))
print("Mean:", np.mean(rba))

plot_rba(img, rba)