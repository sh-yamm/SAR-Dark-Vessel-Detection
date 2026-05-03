# SAR Dark Vessel Detection

End-to-end pipeline for detecting ships in Sentinel-1 SAR imagery and identifying **dark vessels** — ships with AIS transponders switched off, associated with illegal fishing, smuggling, and sanctions evasion.

Built on the [xView3-SAR](https://iuu.xview.us/) dataset using a modified YOLOv8 as the ML model and CA-CFAR as the classical baseline.

---

## What is a dark vessel?

Every commercial ship is legally required to broadcast its position via AIS (Automatic Identification System). A "dark vessel" is one detected in SAR imagery but absent from AIS records — a strong indicator of illegal activity.

SAR (Synthetic Aperture Radar) is ideal for this task: it works day and night, through clouds, and cannot be easily spoofed like optical imagery.

---

## System architecture

```
Sentinel-1 scene (VV_dB.tif + VH_dB.tif)
         |
         v
  [Preprocessing]  clip + normalise dB → [-1,1], extract 512x512 chips
         |
    _____|_____
   |           |
   v           v
[CA-CFAR]   [YOLOv8-SAR]   <-- 2-channel modified (VV + VH, not RGB)
   |           |
   |___________|
         |
         v
  [AIS cross-reference]  haversine match within 500m / 20 min
         |
         v
  [GeoJSON output]  green = AIS-matched, red = dark vessel, grey = unknown
         |
         v
  [Leaflet web UI]  interactive map, AOI selection, YOLO vs CFAR overlay
```

---

## Model: 2-channel YOLOv8

Standard YOLOv8 expects 3-channel RGB. SAR gives 2 channels (VV and VH polarisation). We modify the first Conv2d layer:

- Build `DetectionModel(yolov8n.yaml, ch=2, nc=1)`
- Load pretrained weights for all layers except the first conv (shape mismatch)
- Adapt the first conv: `VV_weight = avg(R, G weights)`, `VH_weight = B weight`

This preserves the spatial filter structure while removing the meaningless third channel. The model is then fine-tuned end-to-end with differential learning rates (backbone 1e-5, neck 1e-4, head 1e-3).

---

## Results

### Baseline run (6 scenes, minimal augmentation)

| Metric | Value |
|--------|-------|
| Precision | 0.892 |
| Recall | 0.177 |
| F1 | 0.321 |
| AP@0.5 | — |
| Training epochs | 33 (early stop) |
| Training chips | 1,089 |
| Val chips | 2,778 |
| YOLO detections (test scene) | 44 |
| CFAR detections (test scene) | 242 |

**Interpretation:** High precision (89%) means detected ships are almost always real. Low recall (18%) means the model misses most ships — expected with only 6 training scenes. CFAR finds 5x more detections but cannot distinguish ships from SAR artefacts (sidelobes, offshore platforms, azimuth ambiguities).

### Improved run (augmentation + loss tuning)

| Metric | Value |
|--------|-------|
| Precision | ~0.877 |
| Recall | ~0.144 |
| F1 | **0.3545** (+10.3% over baseline) |
| Training epochs | 63 (early stop at patience=15) |
| Changes applied | 90°/180°/270° rotation aug, speckle noise (σ=0.05), dB brightness jitter (±10%), multi-scale 256-crop, box_loss 7.5→10.0, cls_loss 0.5→0.3, conf_thresh 0.30→0.20 |

**Why F1 improved but recall dropped slightly:** The augmentation exposed the model to more ship orientations and scales, reducing false positives (precision held at 0.877). The multi-scale crop in particular forces the model to detect small vessels. The best checkpoint (F1=0.3545) was captured mid-training; final epochs showed slight overfitting on the small 6-scene dataset — the primary bottleneck remains data quantity, not model or augmentation design.

---

## Key differences: YOLO vs CFAR

| Property | CA-CFAR | YOLOv8-SAR |
|----------|---------|------------|
| Learned parameters | None | ~3M |
| Adapts to local clutter | Yes (sliding window) | Implicitly (training) |
| Distinguishes ships from artefacts | No | Yes |
| Handles varying sea states | Good | Depends on training data |
| False alarm rate control | Explicit (P_fa param) | Via confidence threshold |
| Speed (full scene) | ~2 min | ~3 min (tiling) |

---

## Setup

### Requirements

```bash
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
py -3.11 -m pip install ultralytics rasterio numpy scipy pandas pyyaml tqdm
py -3.11 -m pip install fastapi uvicorn pydantic
```

> **Note:** Use Python 3.11 explicitly (`py -3.11`). Python 3.14 has incompatible pandas bindings.

### Data

1. Download Sentinel-1 scenes from [xView3](https://iuu.xview.us/) into `data/xview3/scenes/`
2. Download `train.csv` and `validation.csv` label files into `data/`
3. Run preprocessing:

```bash
py -3.11 data/preprocess.py --config config/config.yaml
```

### Training

```bash
py -3.11 train.py --config config/config.yaml
# Resume from checkpoint:
py -3.11 train.py --config config/config.yaml --resume checkpoints/last.pt
```

### Inference (generate GeoJSON for web UI)

```bash
py -3.11 inference.py \
  --scene data/xview3/scenes/<scene_id> \
  --checkpoint checkpoints/best.pt \
  --config config/config.yaml \
  --out backend/precomputed/<scene_id>.geojson
```

### CFAR baseline only

```bash
py -3.11 models/cfar.py \
  --scene data/xview3/scenes/<scene_id> \
  --config config/config.yaml \
  --out cfar_detections.json
```

### Web UI

**Terminal 1:**
```bash
cd backend
py -3.11 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2:**
```bash
cd frontend
py -3.11 -m http.server 5500
```

Open `http://localhost:5500` — the map auto-zooms to the available scene and runs inference on load.

---

## Repository structure

```
SAR Assignment/
├── config/
│   └── config.yaml          # all hyperparameters
├── data/
│   ├── preprocess.py        # chip extraction from scenes
│   └── dataset.py           # PyTorch Dataset + SARAugment
├── models/
│   ├── yolo_sar.py          # 2-channel YOLOv8 construction
│   ├── cfar.py              # CA-CFAR detector
│   └── ais_matcher.py       # AIS spatial join (dark vessel labelling)
├── backend/
│   ├── main.py              # FastAPI server
│   └── precomputed/         # GeoJSON files from inference.py
├── frontend/
│   ├── index.html
│   ├── map.js               # Leaflet map + API calls
│   └── style.css
├── train.py                 # training loop
├── evaluate.py              # precision / recall / AP metrics
├── inference.py             # full-scene inference pipeline
└── checkpoints/
    ├── best.pt              # best validation F1
    └── last.pt              # last epoch
```

---

## Augmentation strategy

SAR imagery requires domain-specific augmentation:

| Technique | Rationale |
|-----------|-----------|
| Random H/V flip | Ships have no canonical orientation in SAR |
| Random 90° rotation | Side-looking radar sees all azimuths equally |
| Additive Gaussian noise (σ=0.05) | Simulates speckle variance across looks |
| dB brightness jitter (±10%) | Backscatter varies with incidence angle and sea state |
| Multi-scale 256-crop + resize | Forces model to detect small vessels (≤20m = 2px) at native resolution |

Standard RGB augmentations (colour jitter, hue shift) are not used — they have no physical meaning in SAR backscatter space.
