"""
CA-CFAR (Cell Averaging Constant False Alarm Rate) detector.

CFAR is the classical baseline for radar ship detection. It was developed in
the 1970s and is still used in operational maritime surveillance today.

How it works
------------
For every pixel (the Cell Under Test, CUT):
  1. A sliding window surrounds the CUT.
  2. The innermost ring (guard cells) is ignored — ship energy spills here.
  3. The outer ring (training cells) estimates local ocean clutter statistics.
  4. The CUT is flagged as a ship if it exceeds  mean + k * std  of the
     training cells, where k controls the false alarm rate.

Why this matters for comparison
--------------------------------
CFAR has no learned parameters — it adapts its threshold to local clutter,
which is why it handles varying sea states better than a fixed global
threshold.  Its main weakness: it cannot distinguish ships from SAR
artefacts (azimuth ambiguities, sidelobes, offshore platforms).
YOLOv8 closes this gap by learning artefact signatures from examples.
"""

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as nd_label
import yaml


@dataclass
class CFARConfig:
    guard_size: int = 2       # half-width of guard window (pixels)
    train_size: int = 8       # half-width of training window (pixels)
    false_alarm_rate: float = 1e-4  # drives threshold multiplier k


def _cfar_threshold_map(vv_linear: np.ndarray, cfg: CFARConfig) -> np.ndarray:
    """
    Compute pixel-wise CFAR threshold using 2D sliding window.

    Parameters
    ----------
    vv_linear : np.ndarray
        VV channel in LINEAR (power) scale — NOT dB.
        CFAR statistics assume a Gamma-distributed clutter model
        which is multiplicative, so we work in linear scale.
    cfg : CFARConfig

    Returns
    -------
    threshold : np.ndarray  same shape as vv_linear
    """
    g = cfg.guard_size
    t = cfg.train_size
    outer = g + t          # outer window half-width

    H, W = vv_linear.shape
    threshold = np.full((H, W), np.inf, dtype=np.float32)

    # Pad to handle borders — use reflect padding to avoid edge artefacts
    pad = outer
    padded = np.pad(vv_linear, pad, mode="reflect")

    for r in range(H):
        for c in range(W):
            pr, pc = r + pad, c + pad  # padded coords

            # Extract outer window
            outer_win = padded[pr - outer: pr + outer + 1,
                               pc - outer: pc + outer + 1]
            # Mask out guard region
            guard_win = np.zeros_like(outer_win, dtype=bool)
            guard_win[outer - g: outer + g + 1,
                      outer - g: outer + g + 1] = True
            # Mask out CUT itself
            guard_win[outer, outer] = True

            training_cells = outer_win[~guard_win]
            mu  = training_cells.mean()
            std = training_cells.std() + 1e-8

            # Gaussian approximation of Gamma tail → k from false alarm rate
            # P_fa = Q(k) → k ≈ -log(P_fa) for exponential model (common approx)
            k = -np.log(cfg.false_alarm_rate)
            threshold[r, c] = mu + k * std

    return threshold


def _cfar_threshold_map_fast(vv_linear: np.ndarray,
                              cfg: CFARConfig) -> np.ndarray:
    """
    Fully vectorised CFAR using 2D integral images, processed in row strips
    to keep memory usage bounded regardless of scene size.

    Each strip loads only (strip_h + 2*outer) rows into the integral image,
    so peak memory per strip is roughly 2 * strip_h * W * 8 bytes (float64).
    """
    g = cfg.guard_size
    t = cfg.train_size
    outer = g + t

    H, W = vv_linear.shape
    # Pad the whole image once (float32 to save memory)
    padded = np.pad(vv_linear, outer, mode="reflect").astype(np.float32)

    n_outer = (2 * outer + 1) ** 2
    n_guard = (2 * g + 1) ** 2
    n_train = n_outer - n_guard
    k       = -np.log(cfg.false_alarm_rate)

    threshold = np.empty((H, W), dtype=np.float32)

    C = np.arange(W, dtype=np.int32)   # column indices (shape W)

    def _boxsum(ig, r0, r1, c0, c1):
        """Box sum using 1-indexed integral image.
        r0/r1 shape (S,), c0/c1 shape (W,) — result shape (S, W)."""
        return (ig[(r1 + 1)[:, None], (c1 + 1)[None, :]]
                - ig[r0[:, None],     (c1 + 1)[None, :]]
                - ig[(r1 + 1)[:, None], c0[None, :]]
                + ig[r0[:, None],       c0[None, :]])

    strip_h = 256   # output rows per strip — tune down if still OOM

    for r_start in range(0, H, strip_h):
        r_end   = min(r_start + strip_h, H)
        S       = r_end - r_start          # rows in this strip

        # Slice of padded that covers all boxes for output rows [r_start, r_end)
        # Outer box for row r needs padded rows [r, r+2*outer]
        ps = padded[r_start : r_end + 2 * outer, :]   # shape (S+2*outer, W+2*outer)

        Ps, Pw = ps.shape
        ig    = np.zeros((Ps + 1, Pw + 1), dtype=np.float64)
        ig_sq = np.zeros((Ps + 1, Pw + 1), dtype=np.float64)
        ps64  = ps.astype(np.float64)
        ig[1:, 1:]    = ps64.cumsum(axis=0).cumsum(axis=1)
        ig_sq[1:, 1:] = (ps64 ** 2).cumsum(axis=0).cumsum(axis=1)
        del ps64

        R = np.arange(S, dtype=np.int32)   # local row indices 0..S-1

        s_o  = _boxsum(ig,    R,     R + 2*outer, C,     C + 2*outer)
        s_osq= _boxsum(ig_sq, R,     R + 2*outer, C,     C + 2*outer)
        s_g  = _boxsum(ig,    R + t, R + t + 2*g, C + t, C + t + 2*g)
        s_gsq= _boxsum(ig_sq, R + t, R + t + 2*g, C + t, C + t + 2*g)

        st    = s_o   - s_g
        st_sq = s_osq - s_gsq

        mu  = st / n_train
        var = np.maximum(st_sq / n_train - mu ** 2, 0.0)
        std = np.sqrt(var) + 1e-8

        threshold[r_start:r_end] = (mu + k * std).astype(np.float32)

        if r_start % (strip_h * 10) == 0:
            print(f"  CFAR {r_start}/{H} rows done ...", flush=True)

    return threshold


def detect(vv_linear: np.ndarray,
           cfg: CFARConfig,
           min_area: int = 2) -> list[dict]:
    """
    Run CFAR detection on a VV channel (linear scale).

    Parameters
    ----------
    vv_linear : np.ndarray  (H, W)  linear power values
    cfg       : CFARConfig
    min_area  : int  minimum connected component size (pixels) to keep

    Returns
    -------
    detections : list of dicts
        Each dict has:
            row, col   : int    centre of detection in scene pixel coords
            score      : float  peak-to-threshold ratio (proxy for confidence)
    """
    threshold = _cfar_threshold_map_fast(vv_linear, cfg)

    # Binary detection map
    det_map = (vv_linear > threshold).astype(np.uint8)

    # Connected component labelling — merge adjacent pixels into one detection
    struct = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, n_components = nd_label(det_map, structure=struct)

    detections = []
    for comp_id in range(1, n_components + 1):
        mask = labeled == comp_id
        area = mask.sum()
        if area < min_area:
            continue

        rows, cols = np.where(mask)
        # Use peak-value pixel as detection centre
        peak_idx   = vv_linear[mask].argmax()
        row = int(rows[peak_idx])
        col = int(cols[peak_idx])

        thr_val = threshold[row, col]
        score   = float(vv_linear[row, col] / (thr_val + 1e-9))

        detections.append({"row": row, "col": col, "score": score})

    return detections


# --------------------------------------------------------------------------- #
#  Coordinate conversion helpers                                               #
# --------------------------------------------------------------------------- #

def pixel_to_latlon(row: int,
                    col: int,
                    scene_meta: dict) -> tuple[float, float]:
    """
    Convert scene pixel (row, col) to (lat, lon) using linear interpolation.

    scene_meta must contain:
        lat_min, lat_max, lon_min, lon_max  (geographic bounding box)
        height, width                        (scene dimensions in pixels)
    """
    lat = scene_meta["lat_max"] - row * (
        (scene_meta["lat_max"] - scene_meta["lat_min"]) / scene_meta["height"])
    lon = scene_meta["lon_min"] + col * (
        (scene_meta["lon_max"] - scene_meta["lon_min"]) / scene_meta["width"])
    return lat, lon


# --------------------------------------------------------------------------- #
#  CLI entry point (run CFAR on a single scene for debugging)                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import h5py
    import json
    import rasterio
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run CFAR on one xView3 scene folder")
    parser.add_argument("--scene", required=True, help="Path to scene folder (contains VV_dB.tif)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out", default="cfar_detections.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfar_cfg = CFARConfig(
        guard_size=cfg_dict["cfar"]["guard_size"],
        train_size=cfg_dict["cfar"]["train_size"],
        false_alarm_rate=cfg_dict["cfar"]["false_alarm_rate"],
    )

    print(f"Loading scene: {args.scene}")
    scene_dir = Path(args.scene)
    with rasterio.open(scene_dir / "VV_dB.tif") as src:
        vv_db  = src.read(1).astype(np.float32)
        bounds = src.bounds
        meta = {
            "lat_min": float(bounds.bottom),
            "lat_max": float(bounds.top),
            "lon_min": float(bounds.left),
            "lon_max": float(bounds.right),
            "height":  src.height,
            "width":   src.width,
        }

    # CFAR works on relative contrast so normalised dB values are fine
    print(f"Running CFAR on scene {vv_db.shape} ...")
    dets = detect(vv_db, cfar_cfg)
    print(f"Found {len(dets)} detections")

    for d in dets:
        d["lat"], d["lon"] = pixel_to_latlon(d["row"], d["col"], meta)

    with open(args.out, "w") as f:
        json.dump(dets, f, indent=2)
    print(f"Saved -> {args.out}")
