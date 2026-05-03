"""
Full-scene inference pipeline.

Given a Sentinel-1 HDF5 scene and an optional AIS CSV, this script:
  1. Loads and normalises VV + VH channels
  2. Tiles the scene into overlapping 512x512 chips
  3. Runs CFAR baseline on the full VV channel
  4. Runs the fine-tuned 2-channel YOLOv8 on all chips
  5. Stitches chip detections back to scene coordinates (lat/lon)
  6. Cross-references with AIS to classify dark vs legitimate vessels
  7. Writes a GeoJSON file consumable by the frontend

Usage:
    python inference.py --scene data/xview3/scenes/abc123.h5 \\
                        --checkpoint checkpoints/best.pt \\
                        --ais data/xview3/ais/abc123.csv \\
                        --scene-time "2021-06-01T12:00:00Z" \\
                        --out backend/precomputed/abc123.geojson
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from pathlib import Path

from data.preprocess import normalise_db, extract_chip, load_scene_meta
from models.cfar import CFARConfig, detect as cfar_detect, pixel_to_latlon
from models.yolo_sar import build_sar_yolo, load_checkpoint
from models.ais_matcher import load_ais_for_scene, classify_detections, to_geojson
from models.otsu_detector import detect as otsu_detect
from models.svm_detector import run_svm_on_tiles
from evaluate import predict_batch


# --------------------------------------------------------------------------- #
#  Scene loading                                                               #
# --------------------------------------------------------------------------- #

def load_scene_normalised(scene_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load VV and VH channels from an xView3 scene folder.

    Scenes are folders containing VV_dB.tif and VH_dB.tif (already in dB).

    Returns
    -------
    vv_db : np.ndarray  float32 (H, W)  normalised to [-1, 1]
    vh_db : np.ndarray  float32 (H, W)  normalised to [-1, 1]
    meta  : dict        geographic metadata (lat/lon bounds, dimensions)
    """
    scene_dir = Path(scene_dir)
    with rasterio.open(scene_dir / "VV_dB.tif") as src:
        vv_db = normalise_db(src.read(1).astype(np.float32))
        from rasterio.warp import transform_bounds
        left, bottom, right, top = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds)
        meta = {
            "lat_min": float(bottom),
            "lat_max": float(top),
            "lon_min": float(left),
            "lon_max": float(right),
            "height":  src.height,
            "width":   src.width,
        }
    with rasterio.open(scene_dir / "VH_dB.tif") as src:
        vh_db = normalise_db(src.read(1).astype(np.float32))

    return vv_db, vh_db, meta


# --------------------------------------------------------------------------- #
#  Chip tiling                                                                 #
# --------------------------------------------------------------------------- #

def tile_scene(vv: np.ndarray,
               vh: np.ndarray,
               chip_size: int = 512,
               overlap: int = 64) -> list[dict]:
    """
    Tile a full scene into overlapping chips for inference.

    Returns list of dicts:
        chip   : np.ndarray (2, chip_size, chip_size)
        row    : int  centre row in scene coords
        col    : int  centre col in scene coords
    """
    H, W = vv.shape
    stride = chip_size - overlap
    half   = chip_size // 2
    tiles  = []

    for r in range(half, H - half, stride):
        for c in range(half, W - half, stride):
            chip = extract_chip(vv, vh, r, c, chip_size)
            if chip is not None:
                tiles.append({"chip": chip, "row": r, "col": c})

    # Ensure we cover the bottom-right corner
    r_last = H - half - 1
    c_last = W - half - 1
    chip = extract_chip(vv, vh, r_last, c_last, chip_size)
    if chip is not None:
        tiles.append({"chip": chip, "row": r_last, "col": c_last})

    return tiles


# --------------------------------------------------------------------------- #
#  YOLOv8 inference on tiles                                                   #
# --------------------------------------------------------------------------- #

def run_yolo_on_tiles(tiles: list[dict],
                      model,
                      device: torch.device,
                      conf_thresh: float = 0.30,
                      nms_iou: float = 0.50,
                      batch_size: int = 8) -> list[dict]:
    """
    Run YOLOv8 on all chips, convert detections to scene pixel coords.

    Returns list of dicts: {row, col, score}  in scene pixel coordinates.
    """
    all_detections = []
    chip_size = tiles[0]["chip"].shape[-1]

    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i: i + batch_size]
        chips = np.stack([t["chip"] for t in batch_tiles])  # (B, 2, H, W)
        imgs  = torch.from_numpy(chips).to(device)

        preds = predict_batch(model, imgs, conf_thresh, nms_iou)

        for b, (tile, pred) in enumerate(zip(batch_tiles, preds)):
            boxes  = pred["boxes"]   # (M, 4) normalised [xc, yc, w, h]
            scores = pred["scores"]  # (M,)

            half = chip_size // 2
            for box, score in zip(boxes, scores):
                # Convert normalised chip coords to scene pixel coords
                det_row = tile["row"] - half + int(box[1] * chip_size)
                det_col = tile["col"] - half + int(box[0] * chip_size)
                all_detections.append({
                    "row":   det_row,
                    "col":   det_col,
                    "score": float(score),
                    "source": "yolo",
                })

    return all_detections


# --------------------------------------------------------------------------- #
#  NMS on scene-level detections (removes duplicates from overlapping tiles)   #
# --------------------------------------------------------------------------- #

def scene_nms(detections: list[dict],
              radius_px: int = 20) -> list[dict]:
    """
    Simple greedy NMS: suppress detections within `radius_px` of a
    higher-scoring detection.

    This is needed because overlapping tiles produce duplicate detections
    for the same physical vessel.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in detections:
        too_close = False
        for k in kept:
            dist = np.hypot(det["row"] - k["row"], det["col"] - k["col"])
            if dist < radius_px:
                too_close = True
                break
        if not too_close:
            kept.append(det)

    return kept


# --------------------------------------------------------------------------- #
#  Main inference pipeline                                                     #
# --------------------------------------------------------------------------- #

def run_inference(scene_dir: str,
                  checkpoint: str,
                  svm_checkpoint: str | None,
                  ais_csv: str | None,
                  scene_time: datetime,
                  cfg: dict,
                  device: torch.device,
                  run_cfar: bool = True,
                  run_otsu: bool = True,
                  run_svm: bool = True) -> dict:
    """
    Full inference pipeline for one scene — runs all four detectors.

    Returns a dict with geojson for each detector + stats.
    """
    print(f"\nLoading scene: {scene_dir}")
    vv_db, vh_db, meta = load_scene_normalised(scene_dir)
    print(f"  Scene shape: {vv_db.shape}")

    # ---- Tile scene once — shared by YOLO and SVM ---- #
    print("\nTiling scene ...")
    tiles = tile_scene(vv_db, vh_db,
                       chip_size=cfg["data"]["chip_size"],
                       overlap=cfg["data"]["overlap"])
    print(f"  {len(tiles)} chips")

    # ---- YOLO inference ---- #
    print("\nBuilding YOLOv8 model ...")
    model = build_sar_yolo(
        pretrained_path=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
    ).to(device)
    load_checkpoint(model, None, checkpoint, str(device))
    model.eval()

    print("Running YOLOv8 ...")
    yolo_dets = run_yolo_on_tiles(
        tiles, model, device,
        conf_thresh=cfg["inference"]["confidence_threshold"],
        nms_iou=cfg["inference"]["nms_iou_threshold"],
    )
    yolo_dets = scene_nms(yolo_dets, radius_px=20)
    print(f"  {len(yolo_dets)} YOLO detections after NMS")
    for det in yolo_dets:
        det["lat"], det["lon"] = pixel_to_latlon(det["row"], det["col"], meta)

    # ---- CFAR inference ---- #
    cfar_dets = []
    if run_cfar:
        print("\nRunning CFAR ...")
        cfar_cfg = CFARConfig(
            guard_size=cfg["cfar"]["guard_size"],
            train_size=cfg["cfar"]["train_size"],
            false_alarm_rate=cfg["cfar"]["false_alarm_rate"],
        )
        cfar_dets = cfar_detect(vv_db, cfar_cfg)
        for det in cfar_dets:
            det["lat"], det["lon"] = pixel_to_latlon(det["row"], det["col"], meta)
            det["source"] = "cfar"
        print(f"  {len(cfar_dets)} CFAR detections")

    # ---- Otsu inference ---- #
    otsu_dets = []
    if run_otsu:
        print("\nRunning Otsu ...")
        otsu_dets = otsu_detect(vv_db)
        otsu_dets = scene_nms(otsu_dets, radius_px=20)
        for det in otsu_dets:
            det["lat"], det["lon"] = pixel_to_latlon(det["row"], det["col"], meta)
            det["source"] = "otsu"
        print(f"  {len(otsu_dets)} Otsu detections after NMS")

    # ---- SVM inference ---- #
    svm_dets = []
    if run_svm and svm_checkpoint and Path(svm_checkpoint).exists():
        print("\nRunning SVM ...")
        import joblib
        svm_model = joblib.load(svm_checkpoint)
        svm_dets  = run_svm_on_tiles(tiles, svm_model)
        svm_dets  = scene_nms(svm_dets, radius_px=20)
        for det in svm_dets:
            det["lat"], det["lon"] = pixel_to_latlon(det["row"], det["col"], meta)
        print(f"  {len(svm_dets)} SVM detections after NMS")
    elif run_svm:
        print("\n[skip] SVM checkpoint not found — run train_svm.py first")

    # ---- AIS cross-reference ---- #
    ais_df = None
    if ais_csv:
        scene_id = Path(scene_dir).stem
        ais_df   = load_ais_for_scene(ais_csv, scene_id)
        print(f"\nAIS pings loaded: {len(ais_df)}")

    all_det_groups = [
        ("yolo", yolo_dets),
        ("cfar", cfar_dets),
        ("otsu", otsu_dets),
        ("svm",  svm_dets),
    ]

    for name, dets in all_det_groups:
        if ais_df is not None and not ais_df.empty and dets:
            classify_detections(
                dets, ais_df, scene_time,
                match_radius_m=cfg["inference"]["ais_match_radius_m"],
                time_window_min=cfg["inference"]["ais_time_window_min"],
            )
        else:
            for det in dets:
                det["is_dark"] = None

    stats = {
        "yolo_total": len(yolo_dets),
        "cfar_total": len(cfar_dets),
        "otsu_total": len(otsu_dets),
        "svm_total":  len(svm_dets),
        "scene_shape": list(vv_db.shape),
    }

    return {
        "yolo_geojson": to_geojson(yolo_dets),
        "cfar_geojson": to_geojson(cfar_dets),
        "otsu_geojson": to_geojson(otsu_dets),
        "svm_geojson":  to_geojson(svm_dets),
        "stats":        stats,
        "meta":         meta,
    }


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",          required=True)
    parser.add_argument("--checkpoint",     default="checkpoints/best.pt")
    parser.add_argument("--svm-checkpoint", default="checkpoints/svm_model.pkl")
    parser.add_argument("--ais",            default=None)
    parser.add_argument("--scene-time",     default=None)
    parser.add_argument("--config",         default="config/config.yaml")
    parser.add_argument("--out",            default="output.geojson")
    parser.add_argument("--no-cfar",        action="store_true")
    parser.add_argument("--no-otsu",        action="store_true")
    parser.add_argument("--no-svm",         action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.scene_time:
        scene_time = datetime.fromisoformat(
            args.scene_time.replace("Z", "+00:00"))
    else:
        scene_time = datetime.now(tz=timezone.utc)
        print("[warn] No scene-time provided — using current UTC time for AIS matching")

    result = run_inference(
        scene_dir=args.scene,
        checkpoint=args.checkpoint,
        svm_checkpoint=args.svm_checkpoint,
        ais_csv=args.ais,
        scene_time=scene_time,
        cfg=cfg,
        device=device,
        run_cfar=not args.no_cfar,
        run_otsu=not args.no_otsu,
        run_svm=not args.no_svm,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = [
        ("yolo_geojson", out_path,                                              True),
        ("cfar_geojson", out_path.parent / (out_path.stem + "_cfar.geojson"),  not args.no_cfar),
        ("otsu_geojson", out_path.parent / (out_path.stem + "_otsu.geojson"),  not args.no_otsu),
        ("svm_geojson",  out_path.parent / (out_path.stem + "_svm.geojson"),   not args.no_svm),
    ]
    for key, path, was_run in outputs:
        if not was_run and Path(path).exists():
            print(f"{key.split('_')[0].upper()} skipped — keeping existing {path}")
            continue
        with open(path, "w") as f:
            json.dump(result[key], f, indent=2)
        print(f"{key.split('_')[0].upper()} detections -> {path}")

    print(f"\nStats: {result['stats']}")


if __name__ == "__main__":
    main()
