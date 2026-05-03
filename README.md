# SAR Dark Vessel Detection

End-to-end pipeline for detecting ships in Sentinel-1 SAR imagery and identifying **dark vessels** — ships with AIS transponders switched off, associated with illegal fishing, smuggling, and sanctions evasion.

Built on the [xView3-SAR](https://iuu.xview.us/) dataset. The pipeline implements four detectors spanning three eras of SAR ship detection — from classical thresholding (1970s–2000s), to handcrafted ML (2005–2016), to modern fine-tuned deep learning — plus AIS fusion for dark vessel identification.

---

## SAR notes

A technical primer on SAR and why it works for this task is in [`SAR_notes.md`](SAR_notes.md). Short version:

SAR is active radar — the satellite transmits its own pulse and records the echo. It works day/night through clouds, and ocean surface backscatter is very dark (specular reflection away from sensor) while ships are bright point targets. The contrast makes ship detection tractable even with simple methods. The hard part is distinguishing real ships from SAR artefacts (azimuth ambiguities, sidelobes, offshore platforms) that look identical in raw backscatter.

Sentinel-1 gives two polarisation channels — **VV** (surface roughness, ocean) and **VH** (volume scattering, structure) — which together give complementary information. We use both as a 2-channel input throughout.

---

## What is a dark vessel?

Every commercial ship is legally required to broadcast its position via AIS (Automatic Identification System). A "dark vessel" is one detected in SAR imagery but absent from AIS records — a strong indicator of illegal activity (illegal fishing, sanctions evasion, smuggling).

The signal is an *absence* rather than a visual feature, making this a data fusion problem: detect all ships in SAR, then cross-reference against AIS broadcasts. Whatever doesn't match is a dark vessel candidate.

---

## Use case selection

After surveying the SAR ML landscape (see notes), I chose dark vessel detection for three reasons:

- The xView3-SAR dataset provides clean Sentinel-1 scenes with matched AIS labels — no data engineering needed to get started.
- It's a meaningful real-world problem, not a toy task.
- The AIS cross-referencing step makes it genuinely distinct from a standard object detection problem — it requires multi-source data fusion to produce actionable output.

I considered flood detection (simpler, Sen1Floods11 dataset) and land cover mapping (U-Net on multi-temporal stacks), but both would have required more custom data prep for less interesting output.

**Datasets referenced during exploration:**

| Dataset | Task | Link |
|---------|------|-------|
| xView3-SAR | Ship detection + dark vessel | https://iuu.xview.us/ |
| Sen1Floods11 | Flood/water detection | https://github.com/cloudtostreet/Sen1Floods11 |
| BigEarthNet | Land cover (multi-label) | https://bigearth.net/ |
| SAR-Ship | Ship detection (simpler) | https://github.com/CAESAR-Radi/SAR-Ship-Dataset |

**Foundation/pretrained models considered:**

- [Prithvi-EO](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) — IBM/NASA geospatial foundation model, HLS data (optical + SAR). Decided against it: it expects a different input format and fine-tuning it properly would require more time than available.
- [SARDet-100K pretrained weights](https://github.com/zcablii/SARDet-100K) — SAR-specific detection backbone. Noted for future use.

---

## System architecture

```
Sentinel-1 scene (VV_dB.tif + VH_dB.tif)
         |
         v
  [Preprocessing]  clip + normalise dB → [-1,1], extract 512x512 chips
         |
    _____|_____________________________
   |           |          |           |
   v           v          v           v
[Otsu]     [CA-CFAR]   [SVM]    [YOLOv8-SAR]
   |           |          |           |
   |___________|__________|___________|
                     |
                     v
  [AIS cross-reference]  haversine match within 500m / 20 min
                     |
                     v
  [GeoJSON output]  green = AIS-matched, red = dark vessel, grey = unknown
                     |
                     v
  [Leaflet web UI]  interactive map, AOI selection, detector overlay toggle
```

---

## Models

Four detectors are implemented, representing different eras and philosophies of SAR ship detection.

### 1. Otsu global thresholding (`models/otsu_detector.py`)

The simplest possible baseline. Otsu's method finds a single global intensity threshold that maximises between-class variance (ship vs ocean). No parameters to tune, no training required.

**Weakness:** Cannot adapt to spatially varying clutter. A bright storm cell dominates the histogram and pushes the threshold too high everywhere else. Treats every bright pixel as a ship candidate regardless of shape or context.

**Why it's useful:** Upper bound on false alarms from pure intensity thresholding. Shows how much spatial context matters.

### 2. CA-CFAR (`models/cfar.py`)

Cell Averaging Constant False Alarm Rate — the operational maritime radar standard since the 1970s, still used in real surveillance systems today.

For every pixel (Cell Under Test), a sliding window computes the mean and std of surrounding training cells (excluding a guard ring around the CUT). The CUT is flagged if it exceeds `mean + k·std`, where `k` is set by a target false alarm rate `P_fa`. This adapts the threshold to local clutter — calm sea vs rough sea get different thresholds automatically.

**Weakness:** Cannot distinguish ships from SAR artefacts (azimuth ambiguities, sidelobes, offshore platforms). Finds 5× more candidates than YOLO but with far more false positives.

### 3. SVM with handcrafted features (`models/svm_detector.py`, `train_svm.py`)

Represents the pre-deep-learning era of SAR ship detection (~2005–2016). A `LinearSVC` is trained on a 20-feature vector extracted from each 512×512 chip:

- Per-channel (VV, VH): mean, std, skewness, kurtosis, p95, p99, max, peak-to-mean ratio, bright pixel fraction
- Cross-channel polarimetric: VV−VH difference, VV/VH ratio

Ships are localised within positive chips by thresholding at `mean + 2·std` and running connected-component labelling.

**Why this matters:** Shows how much learned feature extraction improves over manual engineering. SVM gets chip-level presence/absence right but the localisations within positive chips are coarse.

Training:
```bash
py -3.11 train_svm.py --config config/config.yaml
```

### 4. YOLOv8-SAR (`models/yolo_sar.py`, `train.py`)

The primary model. Standard YOLOv8 expects 3-channel RGB; SAR gives 2 channels. The first Conv2d is adapted:

- Build `DetectionModel(yolov8n.yaml, ch=2, nc=1)`
- Load pretrained weights for all layers except the first conv (shape mismatch)
- Adapt first conv: `VV_weight = avg(R, G weights)`, `VH_weight = B weight`

This preserves the spatial filter structure while adapting to 2-channel input. The model is fine-tuned end-to-end with differential learning rates (backbone 1e-5, neck 1e-4, head 1e-3).

The 2-channel model has 33% fewer first-layer parameters than the RGB version — a meaningful regulariser when training on only 6 scenes.

---

## Results

### Model comparison (qualitative)

| Property | Otsu | CA-CFAR | SVM | YOLOv8-SAR |
|----------|------|---------|-----|------------|
| Learned parameters | None | None | ~few K | ~3M |
| Adapts to local clutter | No | Yes | Implicitly | Implicitly |
| Distinguishes artefacts from ships | No | No | Partially | Yes |
| Handles varying sea states | Poor | Good | Moderate | Depends on training data |
| False alarm rate control | None | Explicit (P_fa) | Confidence threshold | Confidence threshold |
| Speed (full scene) | <1 min | ~2 min | ~1 min | ~3 min (tiling) |

### YOLO quantitative results

#### Baseline run (6 scenes, minimal augmentation)

| Metric | Value |
|--------|-------|
| Precision | 0.892 |
| Recall | 0.177 |
| F1 | 0.321 |
| Training epochs | 33 (early stop) |
| Training chips | 1,089 |
| Val chips | 2,778 |
| YOLO detections (test scene) | 44 |
| CFAR detections (test scene) | 242 |

High precision (89%) means detected ships are almost always real. Low recall (18%) means the model misses most ships — expected with only 6 training scenes. CFAR finds 5× more detections but cannot distinguish ships from artefacts.

#### Improved run (augmentation + loss tuning)

| Metric | Value |
|--------|-------|
| Precision | ~0.877 |
| Recall | ~0.144 |
| F1 | **0.3545** (+10.3% over baseline) |
| Training epochs | 63 (early stop at patience=15) |
| Changes applied | 90°/180°/270° rotation aug, speckle noise (σ=0.05), dB brightness jitter (±10%), multi-scale 256-crop, box_loss 7.5→10.0, cls_loss 0.5→0.3, conf_thresh 0.30→0.20 |

F1 improved while recall dropped slightly. Augmentation exposed the model to more orientations and scales, reducing false positives (precision held at 0.877). The best checkpoint (F1=0.3545) was captured mid-training; final epochs showed slight overfitting on the small 6-scene dataset. **The primary bottleneck is data quantity, not model or augmentation design.**

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

---

## Web UI

A minimal map-based demo connecting the pipeline to a Leaflet frontend via FastAPI.

The user can:
- Pan/zoom to any area within the available scene
- Draw a bounding box AOI
- Click "Run inference" — the backend returns a GeoJSON with detection results
- Toggle between YOLO and CFAR overlays
- See colour-coded markers: green = AIS-matched vessel, red = dark vessel, grey = unknown

The backend loads pre-computed GeoJSON results (inference is too slow to run live on a full scene in a demo context). This is called out explicitly: the AOI → backend → model → map flow is fully wired, but the "model" step at demo time is a lookup into pre-computed results, not a fresh forward pass.

---

## Setup

### Requirements

```bash
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
py -3.11 -m pip install ultralytics rasterio numpy scipy pandas pyyaml tqdm
py -3.11 -m pip install fastapi uvicorn pydantic scikit-learn joblib
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
# YOLO
py -3.11 train.py --config config/config.yaml
# Resume from checkpoint:
py -3.11 train.py --config config/config.yaml --resume checkpoints/last.pt

# SVM
py -3.11 train_svm.py --config config/config.yaml
```

### Inference (generate GeoJSON for web UI)

```bash
py -3.11 inference.py \
  --scene data/xview3/scenes/<scene_id> \
  --checkpoint checkpoints/best.pt \
  --config config/config.yaml \
  --out backend/precomputed/<scene_id>.geojson
```

### Run individual detectors

```bash
# CFAR only
py -3.11 models/cfar.py \
  --scene data/xview3/scenes/<scene_id> \
  --config config/config.yaml \
  --out cfar_detections.json

# Otsu only
py -3.11 models/otsu_detector.py --scene data/xview3/scenes/<scene_id>
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

Open `http://localhost:5500` — the map auto-zooms to the available scene.

---

## Repository structure

```
SAR-Dark-Vessel-Detection/
├── config/
│   └── config.yaml             # all hyperparameters
├── data/
│   ├── preprocess.py           # chip extraction from scenes
│   └── dataset.py              # PyTorch Dataset + SARAugment
├── models/
│   ├── yolo_sar.py             # 2-channel YOLOv8 construction
│   ├── cfar.py                 # CA-CFAR detector
│   ├── otsu_detector.py        # Otsu global threshold detector
│   ├── svm_detector.py         # SVM with handcrafted SAR features
│   └── ais_matcher.py          # AIS spatial join (dark vessel labelling)
├── backend/
│   ├── main.py                 # FastAPI server
│   └── precomputed/            # GeoJSON files from inference.py
├── frontend/
│   ├── index.html
│   ├── map.js                  # Leaflet map + API calls
│   └── style.css
├── train.py                    # YOLO training loop
├── train_svm.py                # SVM training
├── evaluate.py                 # precision / recall / AP metrics
├── inference.py                # full-scene inference pipeline (all detectors)
├── SAR_notes.md                # SAR technical primer
└── checkpoints/
    ├── best.pt                 # best YOLO validation F1
    ├── last.pt                 # last YOLO epoch
    └── svm_model.pkl           # trained SVM pipeline
```

---

## What's incomplete / future work

- **Quantitative comparison across all 4 detectors** — YOLO is evaluated (F1=0.35), but Otsu and SVM results are not yet tabulated against the same xView3 ground truth. The evaluation script (`evaluate.py`) supports all detectors; just needs a run.
- **Data scale** — 6 training scenes is the main bottleneck. Top xView3 entries use hundreds. Semi-supervised training using CFAR pseudo-labels on unlabelled scenes is the most promising next step.
- **Live inference in the web UI** — currently pre-computed GeoJSON lookup. A GPU backend would make this live.
- **Ensemble** — CFAR + YOLO union (with NMS) would likely improve recall without hurting precision significantly.
- **Prithvi-EO / SARDet pretrained backbone** — swapping in a SAR-specific pretrained encoder instead of ImageNet weights is the cleanest architectural improvement.

---

## LLM usage

The thinking side of this project, understanding the SAR landscape, surveying ML approaches, selecting the use case, designing the architecture, the 2-channel weight adaptation scheme, the feature set for the SVM, and all trade-off decisions all were done by me.

 LLM assistance was used for the implementation side: boilerplate, translating design decisions into working code, and routine engineering (FastAPI setup, dataset class structure, training loop scaffolding). I used it more like to speedup implementation and I used it to make work efficient as there is no merit in merely writing the boilerplate code by myself but in planning and architectural decisions I played the main part.