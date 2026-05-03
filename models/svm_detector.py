"""
SVM ship detector using handcrafted SAR features.

Pipeline
--------
Training  (train_svm.py):
  1. Load preprocessed chips (pos = has ship, neg = background).
  2. Extract statistical + texture features from each chip.
  3. Fit a LinearSVC classifier. Save to checkpoints/svm_model.pkl.

Inference (inference.py):
  1. Tile the full scene into overlapping 512x512 chips.
  2. Extract features from each chip.
  3. SVM predicts positive (ship present) or negative.
  4. Within positive chips, localise ships by thresholding at
     mean + 2*std and running connected-component labelling.
  5. Convert chip-local pixel coordinates to scene coordinates.

Why SVM?
--------
SVM with handcrafted features represents the pre-deep-learning era of
SAR ship detection (approx. 2005-2016). Common features were polarimetric
ratios, GLCM texture, and statistical moments. The comparison with YOLO
shows how much learned feature extraction improves over manual engineering.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis as scipy_kurt
from scipy.ndimage import label as nd_label


# --------------------------------------------------------------------------- #
#  Feature extraction                                                          #
# --------------------------------------------------------------------------- #

def extract_features(chip: np.ndarray) -> np.ndarray:
    """
    Extract a 1D feature vector from a 2-channel SAR chip.

    Parameters
    ----------
    chip : np.ndarray  shape (2, H, W)  [VV, VH] normalised to [-1, 1]

    Returns
    -------
    features : np.ndarray  shape (F,)  float32
    """
    feats = []
    for ch in range(chip.shape[0]):   # VV then VH
        flat = chip[ch].flatten().astype(np.float64)
        mu   = flat.mean()
        sig  = flat.std()
        # Skip higher-order moments for near-constant chips (avoids catastrophic
        # cancellation in scipy skew/kurtosis on uniform background patches)
        if sig > 1e-6:
            sk = float(skew(flat))
            ku = float(scipy_kurt(flat))
        else:
            sk, ku = 0.0, 0.0
        sig += 1e-9
        feats += [
            float(mu),
            float(sig),
            sk,
            ku,
            float(np.percentile(flat, 95)),
            float(np.percentile(flat, 99)),
            float(flat.max()),
            float(flat.max() / (abs(mu) + 1e-9)),         # peak-to-mean
            float(((flat - mu) > 2 * sig).mean()),         # bright fraction
        ]

    # Cross-channel features (polarimetric)
    vv = chip[0].flatten().astype(np.float64)
    vh = chip[1].flatten().astype(np.float64)
    feats += [
        float((vv - vh).mean()),                           # polarimetric diff
        float(np.clip(vv / (vh + 1e-3), -10, 10).mean()), # ratio (clipped)
    ]

    return np.array(feats, dtype=np.float32)


def extract_features_batch(chips: np.ndarray) -> np.ndarray:
    """Extract features for a batch of chips. chips shape: (N, 2, H, W)."""
    return np.stack([extract_features(c) for c in chips])


# --------------------------------------------------------------------------- #
#  Localisation within positive chip                                           #
# --------------------------------------------------------------------------- #

def localise_within_chip(vv_chip: np.ndarray,
                          min_area: int = 2) -> list[tuple[int, int, float]]:
    """
    Find ship pixel locations within an SVM-positive chip.

    Uses a simple local threshold (mean + 2*std) and connected components.

    Returns list of (row_local, col_local, score).
    """
    mu   = vv_chip.mean()
    sig  = vv_chip.std() + 1e-9
    thr  = mu + 2.0 * sig

    det_map = (vv_chip > thr).astype(np.uint8)
    labeled, n_comp = nd_label(det_map, structure=np.ones((3, 3), int))

    results = []
    for cid in range(1, n_comp + 1):
        mask = labeled == cid
        if mask.sum() < min_area:
            continue
        rows, cols = np.where(mask)
        peak       = vv_chip[mask].argmax()
        score      = float(vv_chip[rows[peak], cols[peak]] / (abs(thr) + 1e-9))
        results.append((int(rows[peak]), int(cols[peak]), score))

    # Fallback: if threshold missed everything, use peak pixel
    if not results:
        r, c = np.unravel_index(vv_chip.argmax(), vv_chip.shape)
        results.append((int(r), int(c), 1.0))

    return results


# --------------------------------------------------------------------------- #
#  Inference on tiled scene                                                    #
# --------------------------------------------------------------------------- #

def run_svm_on_tiles(tiles: list[dict],
                     model,
                     min_area: int = 2) -> list[dict]:
    """
    Run SVM on tiled chips and return scene-level detections.

    Parameters
    ----------
    tiles : list of dicts  {chip: (2,H,W), row: int, col: int}
            (same format as tile_scene() in inference.py)
    model : fitted sklearn classifier (loaded from .pkl)

    Returns
    -------
    list of dicts: {row, col, score, source="svm"}
    """
    if not tiles:
        return []

    chip_size = tiles[0]["chip"].shape[-1]
    half      = chip_size // 2

    chips  = np.stack([t["chip"] for t in tiles])   # (N, 2, H, W)
    X      = extract_features_batch(chips)           # (N, F)
    # Use decision function margin to filter low-confidence positives
    # (predict() gives 0/1; decision_function gives signed distance to boundary)
    try:
        margins = model.decision_function(X)         # (N,)
        preds   = (margins > 0.5).astype(int)        # higher margin = more confident
    except AttributeError:
        preds   = model.predict(X)

    detections = []
    for tile, pred in zip(tiles, preds):
        if pred != 1:
            continue

        vv_chip = tile["chip"][0]   # VV channel only for localisation
        local   = localise_within_chip(vv_chip, min_area)

        for r_local, c_local, score in local:
            det_row = tile["row"] - half + r_local
            det_col = tile["col"] - half + c_local
            detections.append({
                "row":    det_row,
                "col":    det_col,
                "score":  score,
                "source": "svm",
            })

    return detections
