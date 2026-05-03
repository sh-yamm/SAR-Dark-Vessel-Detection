"""
AIS spatial join — cross-reference SAR detections with AIS vessel reports.

A detection is "dark" if no AIS ping falls within match_radius_m metres
and ais_time_window_min minutes of the scene acquisition timestamp.

Why this is not ML
------------------
The AIS cross-reference is pure geometry + database join. There is no model.
The interesting part is that the *absence* of a match is the signal — we are
detecting what is NOT in the AIS record rather than what IS in the SAR image.

xView3 AIS data format
-----------------------
The xView3 dataset ships with an AIS CSV per scene containing:
    scene_id, vessel_id (MMSI), timestamp (UTC), lat, lon, vessel_length_m

If you use raw Sentinel-1 data with a live AIS feed (e.g. AISHub,
MarineTraffic) the same logic applies — just adapt the loader.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Haversine distance                                                          #
# --------------------------------------------------------------------------- #

_R_EARTH = 6_371_000.0  # metres


def haversine_m(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlam       = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _R_EARTH * math.asin(math.sqrt(a))


def haversine_vectorised(lat: float, lon: float,
                         lats: np.ndarray,
                         lons: np.ndarray) -> np.ndarray:
    """Vectorised haversine: distance from (lat, lon) to each point in arrays."""
    phi1  = math.radians(lat)
    phi2  = np.radians(lats)
    dphi  = np.radians(lats - lat)
    dlam  = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * _R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------- #
#  AIS loader                                                                  #
# --------------------------------------------------------------------------- #

def load_ais_for_scene(ais_csv: str, scene_id: str) -> pd.DataFrame:
    """
    Load AIS pings for a specific scene from the xView3 AIS CSV.

    Returns DataFrame with columns: vessel_id, timestamp, lat, lon
    """
    df = pd.read_csv(ais_csv)
    df = df[df["scene_id"] == scene_id].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Matching                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class MatchResult:
    is_dark: bool           # True = no AIS match found
    matched_mmsi: str | None = None
    distance_m: float | None = None
    time_delta_min: float | None = None


def match_detection(det_lat: float,
                    det_lon: float,
                    scene_time: datetime,
                    ais_df: pd.DataFrame,
                    match_radius_m: float = 500.0,
                    time_window_min: float = 20.0) -> MatchResult:
    """
    Attempt to match one SAR detection to an AIS ping.

    Parameters
    ----------
    det_lat, det_lon : float
        Geographic coordinates of the SAR detection.
    scene_time : datetime (timezone-aware UTC)
        Acquisition time of the SAR scene.
    ais_df : pd.DataFrame
        AIS pings for the scene (output of load_ais_for_scene).
    match_radius_m : float
        Maximum allowed distance (metres) for a match.
    time_window_min : float
        Maximum allowed time difference (minutes) for a match.

    Returns
    -------
    MatchResult
    """
    if ais_df.empty:
        return MatchResult(is_dark=True)

    # ---- Time filter first (cheap) ---- #
    window = timedelta(minutes=time_window_min)
    time_mask = (ais_df["timestamp"] >= scene_time - window) & \
                (ais_df["timestamp"] <= scene_time + window)
    candidates = ais_df[time_mask]

    if candidates.empty:
        return MatchResult(is_dark=True)

    # ---- Spatial filter (vectorised haversine) ---- #
    dists = haversine_vectorised(det_lat, det_lon,
                                  candidates["lat"].values,
                                  candidates["lon"].values)
    within_radius = dists <= match_radius_m

    if not within_radius.any():
        return MatchResult(is_dark=True)

    # Best match = closest AIS ping within constraints
    best_idx  = dists[within_radius].argmin()
    best_row  = candidates[within_radius].iloc[best_idx]
    best_dist = float(dists[within_radius][best_idx])
    dt_min    = abs((best_row["timestamp"] - scene_time).total_seconds()) / 60.0

    return MatchResult(
        is_dark=False,
        matched_mmsi=str(best_row.get("vessel_id", "unknown")),
        distance_m=best_dist,
        time_delta_min=dt_min,
    )


# --------------------------------------------------------------------------- #
#  Batch matching                                                              #
# --------------------------------------------------------------------------- #

def classify_detections(detections: list[dict],
                        ais_df: pd.DataFrame,
                        scene_time: datetime,
                        match_radius_m: float = 500.0,
                        time_window_min: float = 20.0) -> list[dict]:
    """
    Label every detection as dark or AIS-matched.

    Parameters
    ----------
    detections : list of dicts
        Each dict must have 'lat', 'lon' keys.
        Additional keys (score, row, col) are preserved.

    Returns
    -------
    Same list with 'is_dark', 'matched_mmsi', 'distance_m' added to each dict.
    """
    results = []
    for det in detections:
        match = match_detection(
            det["lat"], det["lon"],
            scene_time, ais_df,
            match_radius_m, time_window_min,
        )
        det_out = det.copy()
        det_out["is_dark"]      = match.is_dark
        det_out["matched_mmsi"] = match.matched_mmsi
        det_out["distance_m"]   = match.distance_m
        results.append(det_out)

    n_dark = sum(1 for d in results if d["is_dark"])
    print(f"  AIS match: {len(results) - n_dark} matched, "
          f"{n_dark} dark vessels ({100*n_dark/max(len(results),1):.1f}%)")
    return results


# --------------------------------------------------------------------------- #
#  GeoJSON export                                                              #
# --------------------------------------------------------------------------- #

def to_geojson(detections: list[dict]) -> dict:
    """
    Convert classified detections to GeoJSON FeatureCollection.

    Each feature has properties:
        score       : float   detection confidence
        is_dark     : bool    dark vessel flag
        matched_mmsi: str     AIS vessel ID (null if dark)
    """
    features = []
    for det in detections:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [det["lon"], det["lat"]],
            },
            "properties": {
                "score":        round(det.get("score", 0.0), 4),
                "is_dark":      bool(det.get("is_dark", True)),
                "matched_mmsi": det.get("matched_mmsi"),
                "distance_m":   det.get("distance_m"),
                "source":       det.get("source", "yolo"),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


# --------------------------------------------------------------------------- #
#  Quick test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Synthetic test: 3 detections, 1 has a nearby AIS ping
    scene_time = datetime(2021, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    ais_data = pd.DataFrame({
        "scene_id":  ["test", "test"],
        "vessel_id": ["123456789", "987654321"],
        "timestamp": [
            datetime(2021, 6, 1, 12, 5, 0, tzinfo=timezone.utc),
            datetime(2021, 6, 1, 11, 0, 0, tzinfo=timezone.utc),  # outside window
        ],
        "lat": [10.001, 10.5],
        "lon": [20.001, 20.5],
    })

    detections = [
        {"lat": 10.0,   "lon": 20.0,  "score": 0.95},   # close to AIS ping → matched
        {"lat": 10.5,   "lon": 20.5,  "score": 0.80},   # AIS ping outside time window → dark
        {"lat": 11.0,   "lon": 21.0,  "score": 0.70},   # no AIS nearby → dark
    ]

    result = classify_detections(detections, ais_data, scene_time)
    for r in result:
        status = "DARK" if r["is_dark"] else f"MATCHED ({r['matched_mmsi']})"
        print(f"  ({r['lat']:.3f}, {r['lon']:.3f})  score={r['score']:.2f}  -> {status}")

    geojson = to_geojson(result)
    print(f"\nGeoJSON features: {len(geojson['features'])}")
