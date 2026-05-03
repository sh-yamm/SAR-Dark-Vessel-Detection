"""
Training loop for 2-channel SAR YOLOv8.

Uses the ultralytics DetectionModel forward pass and v8DetectionLoss.
Differential learning rates: backbone < neck < head (see models/yolo_sar.py).

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --resume checkpoints/last.pt
"""

import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset import build_dataloaders
from models.yolo_sar import build_sar_yolo, build_optimiser, save_checkpoint, load_checkpoint
from evaluate import evaluate_loader


# --------------------------------------------------------------------------- #
#  Loss wrapper                                                                #
# --------------------------------------------------------------------------- #

def get_loss_fn(model):
    """
    Wrap ultralytics v8DetectionLoss.

    v8DetectionLoss needs model.args (hyperparameters) which the ultralytics
    Trainer normally sets. We set it manually with default YOLOv8 values.
    """
    from ultralytics.utils.loss import v8DetectionLoss
    from types import SimpleNamespace

    if not hasattr(model, "args"):
        model.args = SimpleNamespace(
            box=10.0,   # increased from 7.5 — penalises localisation more (small objects)
            cls=0.3,    # decreased from 0.5 — single class needs less cls weight
            dfl=1.5,
        )
    return v8DetectionLoss(model)


def targets_to_ultralytics_batch(batch: dict,
                                  device: torch.device) -> dict:
    """
    Convert our collate_fn batch format to the dict ultralytics loss expects.

    Our targets tensor shape: (N, 6)  [batch_idx, cls, xc, yc, w, h]
    Ultralytics expects:
        batch['batch_idx'] : (N,)   long
        batch['cls']       : (N,)   long
        batch['bboxes']    : (N, 4) float  (xc, yc, w, h) normalised
        batch['img']       : (B, C, H, W)
    """
    imgs    = batch["imgs"].to(device)
    targets = batch["targets"].to(device)   # (N, 6)

    if targets.shape[0] > 0:
        batch_idx = targets[:, 0].long()
        cls       = targets[:, 1].long()
        bboxes    = targets[:, 2:]          # (N, 4)
    else:
        batch_idx = torch.zeros(0, dtype=torch.long, device=device)
        cls       = torch.zeros(0, dtype=torch.long, device=device)
        bboxes    = torch.zeros((0, 4),               device=device)

    return {
        "img":       imgs,
        "batch_idx": batch_idx,
        "cls":       cls,
        "bboxes":    bboxes,
    }


# --------------------------------------------------------------------------- #
#  Training loop                                                               #
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, loss_fn,
                    scaler, device, epoch) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        ul_batch = targets_to_ultralytics_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=scaler is not None):
            preds = model(ul_batch["img"])
            result = loss_fn(preds, ul_batch)

        # v8DetectionLoss returns (scalar_loss, loss_items) but the
        # return format changed across ultralytics versions — handle both.
        if isinstance(result, (tuple, list)):
            loss = result[0]
        else:
            loss = result
        # Ensure scalar for backward()
        if loss.dim() > 0:
            loss = loss.sum()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / n_batches


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- Setup ---- #
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir  = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs    = cfg["training"]["epochs"]
    patience  = cfg["training"]["patience"]
    val_every = cfg["training"]["val_interval"]

    print(f"Device: {device}")

    # ---- Data ---- #
    train_dl, val_dl = build_dataloaders(cfg)

    # ---- Model ---- #
    print("\nBuilding model ...")
    model = build_sar_yolo(
        pretrained_path=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
    ).to(device)

    optimizer = build_optimiser(model, cfg)
    loss_fn   = get_loss_fn(model)
    scaler    = GradScaler("cuda") if device.type == "cuda" else None

    # ---- Resume ---- #
    start_epoch = 0
    best_f1     = 0.0
    no_improve  = 0

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        start_epoch, prev_metrics = load_checkpoint(model, optimizer,
                                                     args.resume, str(device))
        best_f1 = prev_metrics.get("val_f1", 0.0)
        start_epoch += 1

    # ---- Training loop ---- #
    print(f"\nTraining for {epochs} epochs ...\n")
    history = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_dl, optimizer,
                                      loss_fn, scaler, device, epoch)

        metrics = {"epoch": epoch, "train_loss": train_loss}

        # Validation
        if (epoch + 1) % val_every == 0:
            val_metrics = evaluate_loader(model, val_dl, device, cfg)
            metrics.update(val_metrics)
            val_f1 = val_metrics.get("vessel_f1", 0.0)

            print(f"Epoch {epoch:3d} | "
                  f"loss={train_loss:.4f} | "
                  f"P={val_metrics.get('precision', 0):.3f} "
                  f"R={val_metrics.get('recall', 0):.3f} "
                  f"F1={val_f1:.3f} | "
                  f"{time.time()-t0:.1f}s")

            # Save best
            if val_f1 > best_f1:
                best_f1    = val_f1
                no_improve = 0
                save_checkpoint(model, optimizer, epoch, metrics,
                                 str(save_dir / "best.pt"))
                print(f"  [best] New best F1={best_f1:.4f} -> saved best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping after {patience} epochs without improvement.")
                    break
        else:
            print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                  f"{time.time()-t0:.1f}s")

        # Always save last checkpoint
        save_checkpoint(model, optimizer, epoch, metrics,
                         str(save_dir / "last.pt"))
        history.append(metrics)

    print(f"\nTraining complete. Best val F1 = {best_f1:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
