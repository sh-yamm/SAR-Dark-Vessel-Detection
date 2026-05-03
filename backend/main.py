"""
FastAPI backend for the SAR Dark Vessel Detection web UI.

Endpoints
---------
GET  /health
    Liveness check.

POST /detect
    Body: { "bbox": [lon_min, lat_min, lon_max, lat_max] }
    Returns: GeoJSON FeatureCollection with ship detections.
    Each feature has properties:
        score       : float   detection confidence
        is_dark     : bool    True = no AIS match (dark/suspicious vessel)
        matched_mmsi: str     AIS vessel ID (null if dark)
        source      : str     "yolo" or "cfar"

GET  /scenes
    Returns list of available precomputed scene IDs and their bounding boxes.

Strategy
--------
Live inference on a full Sentinel-1 scene takes 2-5 minutes.
For the web demo we use PRECOMPUTED GeoJSON files produced by inference.py.
The backend finds which precomputed scene overlaps the requested AOI bbox
and returns the detections that fall inside it.

If you want live inference, set LIVE_INFERENCE=1 in environment and
provide a scenes directory. Be aware of the latency.

Usage:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
#  App setup                                                                   #
# --------------------------------------------------------------------------- #

app = FastAPI(title="SAR Dark Vessel Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
_cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

PRECOMPUTED_DIR = Path(__file__).parent / "precomputed"
PRECOMPUTED_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
#  Precomputed scene index                                                     #
# --------------------------------------------------------------------------- #

def _load_scene_index() -> list[dict]:
    """
    Build an index of available precomputed GeoJSON files.

    Each GeoJSON produced by inference.py has a 'meta' sidecar JSON
    with geographic bounds, or we read bounds from the GeoJSON itself.
    """
    index = []
    for gj_path in PRECOMPUTED_DIR.glob("*.geojson"):
        if "_cfar" in gj_path.stem:
            continue  # skip CFAR files; they're paired with main files

        meta_path = gj_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            index.append({
                "scene_id": gj_path.stem,
                "path":     str(gj_path),
                "lat_min":  meta["lat_min"],
                "lat_max":  meta["lat_max"],
                "lon_min":  meta["lon_min"],
                "lon_max":  meta["lon_max"],
            })
        else:
            # Fall back: compute bounds from feature coordinates
            with open(gj_path) as f:
                gj = json.load(f)
            coords = [
                feat["geometry"]["coordinates"]
                for feat in gj.get("features", [])
                if feat["geometry"]["type"] == "Point"
            ]
            if not coords:
                continue
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            index.append({
                "scene_id": gj_path.stem,
                "path":     str(gj_path),
                "lat_min":  min(lats),
                "lat_max":  max(lats),
                "lon_min":  min(lons),
                "lon_max":  max(lons),
            })

    return index


SCENE_INDEX = _load_scene_index()


# --------------------------------------------------------------------------- #
#  Request / Response models                                                   #
# --------------------------------------------------------------------------- #

class DetectRequest(BaseModel):
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[lon_min, lat_min, lon_max, lat_max]",
    )
    include_cfar: bool = Field(
        False,
        description="Also return CFAR baseline detections for comparison",
    )
    model_source: str = Field(
        "yolo",
        description="'yolo' or 'cfar' — which model's detections to return",
    )


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _bbox_overlaps(scene: dict, bbox: list[float]) -> bool:
    """Check if scene bounding box overlaps the request bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return not (
        scene["lon_max"] < lon_min or
        scene["lon_min"] > lon_max or
        scene["lat_max"] < lat_min or
        scene["lat_min"] > lat_max
    )


def _filter_features_by_bbox(features: list[dict],
                               bbox: list[float]) -> list[dict]:
    """Return only features whose point falls inside bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    result = []
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"]
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            result.append(feat)
    return result


def _load_geojson(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Routes                                                                      #
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {"status": "ok", "scenes_available": len(SCENE_INDEX)}


@app.get("/scenes")
def list_scenes():
    """Return available precomputed scenes and their geographic extents."""
    return {
        "scenes": [
            {
                "scene_id": s["scene_id"],
                "bbox":     [s["lon_min"], s["lat_min"],
                             s["lon_max"], s["lat_max"]],
            }
            for s in SCENE_INDEX
        ]
    }


@app.post("/detect")
def detect(req: DetectRequest):
    """
    Return ship detections within the requested bounding box.

    Looks up precomputed results — no live inference in the demo.
    """
    lon_min, lat_min, lon_max, lat_max = req.bbox

    # Basic validation
    if lon_min >= lon_max or lat_min >= lat_max:
        raise HTTPException(status_code=400,
                            detail="Invalid bbox: min must be less than max")

    # Find overlapping scenes
    matching_scenes = [s for s in SCENE_INDEX if _bbox_overlaps(s, req.bbox)]

    if not matching_scenes:
        # Return empty FeatureCollection rather than 404 —
        # the user drew a bbox over open ocean with no precomputed scene
        return {
            "type": "FeatureCollection",
            "features": [],
            "meta": {
                "message": "No precomputed scene overlaps this area. "
                           "Try one of the available scene bounding boxes.",
                "available_scenes": [
                    {"scene_id": s["scene_id"],
                     "bbox": [s["lon_min"], s["lat_min"],
                              s["lon_max"], s["lat_max"]]}
                    for s in SCENE_INDEX
                ],
            },
        }

    # Use the first (best) overlapping scene
    scene = matching_scenes[0]

    # Choose the right GeoJSON file based on model_source
    stem = Path(scene["path"]).stem
    parent = Path(scene["path"]).parent
    suffix_map = {
        "yolo": scene["path"],
        "cfar": str(parent / f"{stem}_cfar.geojson"),
        "otsu": str(parent / f"{stem}_otsu.geojson"),
        "svm":  str(parent / f"{stem}_svm.geojson"),
    }
    candidate = suffix_map.get(req.model_source, scene["path"])
    gj_path = candidate if Path(candidate).exists() else scene["path"]

    gj = _load_geojson(gj_path)
    filtered = _filter_features_by_bbox(gj.get("features", []), req.bbox)

    # Optionally append CFAR detections for comparison
    extra_features = []
    if req.include_cfar and req.model_source != "cfar":
        cfar_path = Path(scene["path"]).parent / \
                    (Path(scene["path"]).stem + "_cfar.geojson")
        if cfar_path.exists():
            cfar_gj   = _load_geojson(str(cfar_path))
            cfar_feats = _filter_features_by_bbox(
                cfar_gj.get("features", []), req.bbox)
            for feat in cfar_feats:
                feat["properties"]["source"] = "cfar"
            extra_features = cfar_feats

    # Stats
    all_features = filtered + extra_features
    n_dark    = sum(1 for f in all_features
                    if f["properties"].get("is_dark") is True)
    n_matched = sum(1 for f in all_features
                    if f["properties"].get("is_dark") is False)

    return {
        "type":     "FeatureCollection",
        "features": all_features,
        "meta": {
            "scene_id":       scene["scene_id"],
            "total_vessels":  len(all_features),
            "dark_vessels":   n_dark,
            "ais_matched":    n_matched,
            "model_source":   req.model_source,
            "bbox_requested": req.bbox,
        },
    }


@app.get("/scene/{scene_id}")
def get_scene(scene_id: str, source: str = "yolo"):
    """Return all detections for a specific scene (ignores bbox filter)."""
    scene = next((s for s in SCENE_INDEX if s["scene_id"] == scene_id), None)
    if scene is None:
        raise HTTPException(status_code=404,
                            detail=f"Scene '{scene_id}' not found")

    if source == "cfar":
        cfar_path = Path(scene["path"]).parent / \
                    (Path(scene["path"]).stem + "_cfar.geojson")
        path = str(cfar_path) if cfar_path.exists() else scene["path"]
    else:
        path = scene["path"]

    return _load_geojson(path)


# --------------------------------------------------------------------------- #
#  Dev entrypoint                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host=CFG["backend"]["host"],
                port=CFG["backend"]["port"],
                reload=True)
