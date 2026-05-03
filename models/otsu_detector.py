"""
Otsu global thresholding detector for SAR ship detection.

Otsu's method finds the single intensity threshold that maximises
between-class variance (foreground vs background). Unlike CFAR, it
uses one global threshold for the entire scene — no sliding window,
no parameters to tune.

Weakness vs CFAR: cannot adapt to spatially varying clutter (calm vs
rough sea). One bright storm cell can dominate the histogram and push
the threshold too high everywhere else.

Weakness vs YOLO: no semantic understanding — treats any bright pixel
as a ship candidate regardless of shape, texture, or context.
"""

import numpy as np
from scipy.ndimage import label as nd_label


def otsu_threshold(image: np.ndarray, n_bins: int = 512) -> float:
    """
    Compute Otsu's optimal global threshold via between-class variance.

    Works on any float image — no need to quantise to uint8 first.
    """
    flat = image[np.isfinite(image)].flatten()
    counts, edges = np.histogram(flat, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    total     = counts.sum()
    sum_total = (counts * centers).sum()

    sum_bg   = 0.0
    w_bg     = 0
    best_var = -1.0
    best_t   = centers[0]

    for i in range(len(counts)):
        w_bg   += counts[i]
        w_fg    = total - w_bg
        if w_bg == 0 or w_fg == 0:
            continue

        sum_bg   += counts[i] * centers[i]
        mean_bg   = sum_bg / w_bg
        mean_fg   = (sum_total - sum_bg) / w_fg

        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t   = centers[i]

    return float(best_t)


def detect(vv: np.ndarray,
           min_area: int = 4,
           downsample: int = 4,
           thresh_floor: float = 0.3) -> list[dict]:
    """
    Detect ships in a VV channel image using Otsu thresholding.

    SAR scenes are not bimodal — ocean dominates the histogram and pulls
    Otsu's threshold too low (flagging >50% of pixels as foreground).
    We apply a floor: the threshold is never lower than `thresh_floor`
    in normalised [-1, 1] space (≈ ships only, not ocean noise).

    We also downsample before labelling to keep memory and runtime bounded.
    """
    from scipy.ndimage import find_objects

    H, W = vv.shape
    vv_small = vv[::downsample, ::downsample]

    thresh = max(float(otsu_threshold(vv_small)), thresh_floor)
    # Ships occupy <0.2% of a typical SAR scene — clamp aggressively
    if (vv_small > thresh).mean() > 0.002:
        thresh = float(np.percentile(vv_small, 99.8))
    flagged_pct = float((vv_small > thresh).mean() * 100)
    print(f"  Otsu threshold={thresh:.4f}, flagged {flagged_pct:.1f}% of pixels")

    det_map = (vv_small > thresh).astype(np.uint8)
    struct  = np.ones((3, 3), dtype=int)
    labeled, n_components = nd_label(det_map, structure=struct)

    # Fast size filter via bincount (no per-component loop)
    sizes     = np.bincount(labeled.ravel())          # shape (n_components+1,)
    valid_ids = np.where(sizes >= min_area)[0]
    valid_ids = valid_ids[valid_ids > 0]              # exclude background label 0
    print(f"  {n_components} raw components, {len(valid_ids)} pass size filter")

    # Use find_objects bounding boxes for fast per-component peak lookup
    bboxes = find_objects(labeled)
    detections = []
    for comp_id in valid_ids:
        bb = bboxes[comp_id - 1]
        if bb is None:
            continue
        sub_lbl = labeled[bb]
        sub_vv  = vv_small[bb]
        mask    = sub_lbl == comp_id
        # Find peak within the bounding box
        vals         = np.where(mask, sub_vv, -np.inf)
        pr, pc       = np.unravel_index(vals.argmax(), vals.shape)
        row          = (bb[0].start + pr) * downsample
        col          = (bb[1].start + pc) * downsample
        score        = float(sub_vv[pr, pc] / (abs(thresh) + 1e-9))
        detections.append({"row": int(row), "col": int(col), "score": score})

    return detections
