"""
Evaluation metrics for SAR ship detection.

Metrics reported
----------------
Vessel detection
    precision  : TP / (TP + FP)
    recall     : TP / (TP + FN)
    F1         : harmonic mean of P and R
    AP@0.5     : area under P-R curve at 500 m match threshold

Dark vessel classification
    dark_precision : of predicted darks, how many are truly dark
    dark_recall    : of true dark vessels, how many did we flag
    dark_f1        : primary metric for the interesting task

Matching rule (distance-based, not IoU)
    A detection is a TP if its centre falls within `match_radius_m` metres
    of a labelled vessel centre. IoU is inappropriate here because xView3
    labels are centre-points, not boxes.

Usage:
    python evaluate.py --config config/config.yaml --checkpoint checkpoints/best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import SARChipDataset, collate_fn
from models.yolo_sar import build_sar_yolo, load_checkpoint


# --------------------------------------------------------------------------- #
#  Distance-based matching                                                     #
# --------------------------------------------------------------------------- #

def match_detections_to_labels(pred_boxes: np.ndarray,
                                pred_scores: np.ndarray,
                                gt_boxes: np.ndarray,
                                chip_size: int = 512,
                                match_radius_px: float = 25.0
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predictions to ground-truth by distance.

    pred_boxes : (M, 4)  [xc, yc, w, h] normalised
    pred_scores: (M,)    confidence scores
    gt_boxes   : (N, 4)  [xc, yc, w, h] normalised (from labels)

    Returns
    -------
    tp : (M,) bool  — prediction is a true positive
    fp : (M,) bool  — prediction is a false positive
    fn_count : int  — number of unmatched ground-truth boxes
    """
    M = len(pred_boxes)
    N = len(gt_boxes)

    if M == 0:
        return np.array([], bool), np.array([], bool), N

    if N == 0:
        return np.zeros(M, bool), np.ones(M, bool), 0

    # Sort predictions by descending confidence
    order = np.argsort(-pred_scores)
    pred_boxes  = pred_boxes[order]
    pred_scores = pred_scores[order]

    # Convert normalised centres to pixel coords for distance computation
    pred_cx = pred_boxes[:, 0] * chip_size
    pred_cy = pred_boxes[:, 1] * chip_size
    gt_cx   = gt_boxes[:, 0] * chip_size
    gt_cy   = gt_boxes[:, 1] * chip_size

    matched_gt = set()
    tp = np.zeros(M, dtype=bool)
    fp = np.zeros(M, dtype=bool)

    for i in range(M):
        dists = np.hypot(pred_cx[i] - gt_cx, pred_cy[i] - gt_cy)
        best_j = int(dists.argmin())
        if dists[best_j] <= match_radius_px and best_j not in matched_gt:
            tp[i] = True
            matched_gt.add(best_j)
        else:
            fp[i] = True

    fn_count = N - len(matched_gt)
    return tp, fp, fn_count


# --------------------------------------------------------------------------- #
#  Precision-Recall curve and AP                                              #
# --------------------------------------------------------------------------- #

def compute_ap(tp_list: list[bool],
               fp_list: list[bool],
               n_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision from accumulated TP/FP lists.

    Returns (AP, precision_curve, recall_curve).
    """
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    tp_arr = np.array(tp_list, dtype=float)
    fp_arr = np.array(fp_list, dtype=float)

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)

    recall    = tp_cum / (n_gt + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    # Interpolated AP (11-point)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_t = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p_at_t / 11.0

    return ap, precision, recall


# --------------------------------------------------------------------------- #
#  Per-batch inference helper                                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict_batch(model, imgs: torch.Tensor,
                  conf_thresh: float = 0.30,
                  nms_iou: float = 0.50) -> list[dict]:
    """
    Run model on a batch of chips, apply NMS, return detections per image.

    Returns list of dicts (one per image):
        boxes  : (M, 4)  normalised [xc, yc, w, h]
        scores : (M,)
    """
    from ultralytics.utils.ops import non_max_suppression

    model.eval()
    raw = model(imgs)   # ultralytics returns list of head outputs in eval mode

    # Ultralytics postprocess expects (B, num_classes+4, num_anchors)
    # non_max_suppression returns list[Tensor] each (M, 6) [x1,y1,x2,y2,conf,cls]
    preds = non_max_suppression(raw, conf_thres=conf_thresh, iou_thres=nms_iou)

    results = []
    H = W = imgs.shape[-1]  # assume square chips

    for det in preds:
        if det is None or len(det) == 0:
            results.append({"boxes": np.zeros((0, 4)), "scores": np.zeros(0)})
            continue

        det = det.cpu().numpy()
        # xyxy → xywh normalised
        x1, y1, x2, y2 = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
        xc = ((x1 + x2) / 2) / W
        yc = ((y1 + y2) / 2) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        boxes  = np.stack([xc, yc, w, h], axis=1)
        scores = det[:, 4]
        results.append({"boxes": boxes, "scores": scores})

    return results


# --------------------------------------------------------------------------- #
#  Full evaluation over a DataLoader                                           #
# --------------------------------------------------------------------------- #

def evaluate_loader(model,
                    loader: DataLoader,
                    device: torch.device,
                    cfg: dict) -> dict:
    """
    Run evaluation over all chips in a DataLoader.

    Returns dict with precision, recall, vessel_f1, AP, dark_f1, etc.
    """
    conf_thresh    = cfg["inference"]["confidence_threshold"]
    nms_iou        = cfg["inference"]["nms_iou_threshold"]
    match_radius_m = cfg["evaluation"]["match_radius_m"]
    chip_size      = cfg["data"]["chip_size"]
    # Approximate: 10 m/px for Sentinel-1 IW
    match_radius_px = match_radius_m / 10.0

    all_tp, all_fp = [], []
    n_gt_total = 0

    model.eval()
    for batch in loader:
        imgs    = batch["imgs"].to(device)
        targets = batch["targets"]   # (N_total, 6)  cpu

        preds = predict_batch(model, imgs, conf_thresh, nms_iou)

        for b_idx in range(imgs.shape[0]):
            # Ground truth for this image
            gt_mask  = targets[:, 0] == b_idx
            gt_boxes = targets[gt_mask, 2:].numpy()   # (N, 4)
            n_gt_total += len(gt_boxes)

            # Predictions for this image
            p = preds[b_idx]
            pred_boxes  = p["boxes"]
            pred_scores = p["scores"]

            tp, fp, fn = match_detections_to_labels(
                pred_boxes, pred_scores, gt_boxes,
                chip_size, match_radius_px,
            )
            all_tp.extend(tp.tolist())
            all_fp.extend(fp.tolist())

    # ---- Compute metrics ---- #
    tp_arr = np.array(all_tp, dtype=float)
    fp_arr = np.array(all_fp, dtype=float)

    tp_sum = tp_arr.sum()
    fp_sum = fp_arr.sum()
    fn_sum = n_gt_total - tp_sum

    precision = float(tp_sum / (tp_sum + fp_sum + 1e-9))
    recall    = float(tp_sum / (tp_sum + fn_sum + 1e-9))
    vessel_f1 = float(2 * precision * recall / (precision + recall + 1e-9))

    ap, _, _ = compute_ap(all_tp, all_fp, n_gt_total)

    # Dark vessel F1 is computed at inference time via AIS join (not here).
    # Chip-level labels only carry the center-ship dark flag, not per-ship,
    # so we cannot compute it accurately during training evaluation.

    return {
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "vessel_f1":  round(vessel_f1, 4),
        "AP":         round(ap,        4),
        "dark_f1":    0.0,   # computed at inference time via AIS join
        "n_gt":       n_gt_total,
        "n_pred":     len(all_tp),
    }


# --------------------------------------------------------------------------- #
#  CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="Model checkpoint to evaluate")
    parser.add_argument("--split",      default="val",
                        choices=["train", "val"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model ...")
    model = build_sar_yolo(
        pretrained_path=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
    ).to(device)

    if Path(args.checkpoint).exists():
        load_checkpoint(model, None, args.checkpoint, str(device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[warn] Checkpoint not found: {args.checkpoint} — using random weights")

    ds = SARChipDataset(cfg["data"]["chips_dir"], split=args.split)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, num_workers=4, collate_fn=collate_fn)

    print(f"\nEvaluating on {args.split} split ({len(ds)} chips) ...")
    metrics = evaluate_loader(model, loader, device, cfg)

    print("\n=== Results ===")
    print(f"  Vessel detection")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1        : {metrics['vessel_f1']:.4f}")
    print(f"    AP        : {metrics['AP']:.4f}")
    print(f"  Dark vessel")
    print(f"    F1        : {metrics['dark_f1']:.4f}")
    print(f"  Counts")
    print(f"    GT vessels: {metrics['n_gt']}")
    print(f"    Predictions:{metrics['n_pred']}")


if __name__ == "__main__":
    main()
