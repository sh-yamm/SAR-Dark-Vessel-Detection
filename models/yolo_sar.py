"""
YOLOv8 modified for 2-channel SAR input (VV + VH).

Why modification is needed
--------------------------
Pretrained YOLOv8 expects 3-channel RGB images. SAR gives us 2 channels
(VV and VH polarisation). The mismatch is both:
  - Architectural: first Conv2d has weight shape [filters, 3, k, k]
  - Statistical:   ImageNet features learned colour/texture statistics
                   that do not transfer directly to SAR backscatter.

What we do
----------
1. Instantiate YOLOv8n with ch=2 via the ultralytics DetectionModel API
   (the architecture natively supports arbitrary input channels via `ch`).
2. Load pretrained RGB weights from yolov8n.pt for ALL layers EXCEPT the
   first conv, which has shape mismatch.
3. For the first conv, initialise the 2-channel weights by averaging the
   three pretrained RGB channels:
       VV channel ← average(R, G)   (luminance proxy)
       VH channel ← B channel       (blue ≈ least correlated with VV)
   This preserves the spatial filter structure while adapting to 2 channels.
4. All layers are unfrozen for fine-tuning with differential learning rates
   (backbone slower, head faster — see train.py).

Expected improvement over RGB pretrained baseline
--------------------------------------------------
The 2-channel model avoids injecting a meaningless "third channel" and
reduces the first-layer parameter count by 33 %, which helps when training
data is limited (xView3 has ~20 K labelled vessels).
"""

import torch
import torch.nn as nn
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel
import yaml


# --------------------------------------------------------------------------- #
#  Model construction                                                          #
# --------------------------------------------------------------------------- #

YOLO_CFG = "yolov8n.yaml"   # architecture definition (bundled with ultralytics)


def build_sar_yolo(pretrained_path: str = "yolov8n.pt",
                   num_classes: int = 1,
                   in_channels: int = 2) -> DetectionModel:
    """
    Build a YOLOv8n model with `in_channels` input channels.

    Parameters
    ----------
    pretrained_path : str
        Path to (or name of) pretrained YOLOv8 .pt file.
        Downloaded automatically by ultralytics if not present.
    num_classes : int
        Number of detection classes (1 for vessel / no-vessel).
    in_channels : int
        Number of SAR input channels (2 for VV + VH).

    Returns
    -------
    model : DetectionModel  (nn.Module subclass)
    """
    # ---- Step 1: build architecture with correct channel count ---- #
    model = DetectionModel(YOLO_CFG, ch=in_channels, nc=num_classes)

    # ---- Step 2: load pretrained weights ---- #
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    # ultralytics .pt files store the model under 'model' key
    pretrained_state = (ckpt["model"].float().state_dict()
                        if "model" in ckpt else ckpt)

    model_state = model.state_dict()

    # ---- Step 3: adapt first conv (3-ch → 2-ch) ---- #
    # Identify the first conv weight key (always model.0.conv.weight in YOLOv8)
    first_key = "model.0.conv.weight"
    if first_key in pretrained_state:
        w_rgb = pretrained_state[first_key]          # [C_out, 3, k, k]
        w_sar = torch.zeros(
            w_rgb.shape[0], in_channels,
            w_rgb.shape[2], w_rgb.shape[3]
        )
        # VV ← average of R and G (most luminance-related channels)
        w_sar[:, 0] = (w_rgb[:, 0] + w_rgb[:, 1]) / 2.0
        # VH ← B channel  (least correlated with luminance)
        w_sar[:, 1] = w_rgb[:, 2]

        pretrained_state[first_key] = w_sar
        print(f"  First conv adapted: [*, 3, *, *] -> [*, {in_channels}, *, *]")

    # ---- Step 4: load matching weights, skip shape mismatches ---- #
    loaded, skipped = [], []
    for k, v in pretrained_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    model.load_state_dict(model_state)
    print(f"  Loaded {len(loaded)} weight tensors, skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped keys (shape mismatch or new head): {skipped[:5]} ...")

    return model


# --------------------------------------------------------------------------- #
#  Optimiser with differential learning rates                                  #
# --------------------------------------------------------------------------- #

def build_optimiser(model: DetectionModel, cfg: dict) -> torch.optim.Optimizer:
    """
    Three-group AdamW:
      backbone  (model.0 – model.9)  → low lr  (preserve pretrained features)
      neck      (model.10 – model.21) → mid lr
      head      (model.22+)          → high lr (task-specific, train fast)

    The split indices are specific to YOLOv8n architecture.
    """
    lr_bb   = cfg["training"]["lr_backbone"]
    lr_neck = cfg["training"]["lr_neck"]
    lr_head = cfg["training"]["lr_head"]
    wd      = cfg["training"]["weight_decay"]

    backbone_params, neck_params, head_params = [], [], []

    for name, param in model.named_parameters():
        layer_idx = _layer_index(name)
        if layer_idx is None or layer_idx <= 9:
            backbone_params.append(param)
        elif layer_idx <= 21:
            neck_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr_bb,   "name": "backbone"},
        {"params": neck_params,     "lr": lr_neck,  "name": "neck"},
        {"params": head_params,     "lr": lr_head,  "name": "head"},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    print(f"  Optimiser groups: backbone={len(backbone_params)} params, "
          f"neck={len(neck_params)}, head={len(head_params)}")
    return optimizer


def _layer_index(param_name: str) -> int | None:
    """Extract numeric layer index from YOLOv8 parameter name."""
    parts = param_name.split(".")
    # YOLOv8 params look like  model.0.conv.weight, model.22.dfl.conv.weight
    if len(parts) >= 2 and parts[0] == "model":
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


# --------------------------------------------------------------------------- #
#  Save / load helpers                                                         #
# --------------------------------------------------------------------------- #

def save_checkpoint(model: DetectionModel,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    metrics: dict,
                    path: str) -> None:
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }, path)


def load_checkpoint(model: DetectionModel,
                    optimizer: torch.optim.Optimizer | None,
                    path: str,
                    device: str = "cpu") -> tuple[int, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# --------------------------------------------------------------------------- #
#  Quick smoke test                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Building 2-channel SAR YOLOv8 ...")
    model = build_sar_yolo(
        pretrained_path=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
    )
    model.eval()

    # Smoke test: forward pass with random 2-channel input
    dummy = torch.randn(2, 2, 512, 512)  # batch=2, channels=2, H=512, W=512
    with torch.no_grad():
        out = model(dummy)

    print(f"\nForward pass OK")
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            if hasattr(o, "shape"):
                print(f"  Output[{i}] shape: {o.shape}")
    else:
        print(f"  Output shape: {out.shape}")
