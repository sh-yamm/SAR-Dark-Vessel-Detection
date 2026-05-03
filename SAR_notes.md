# SAR Technical Notes

*Written as part of the take-home assignment. Sources: NASA ARSET "Introduction to SAR", ESA EO College "Echoes in Space", xView3 dataset paper.*

---

## 1. What is SAR and how is it different from optical imagery?

### Optical imaging

Optical satellites work passively as they wait for sunlight and measure how much of it reflects off the Earth's surface, and record that reflectance in visible and near-infrared bands. The result looks more like a photograph. The fundamental limitation is that no sun means no image (polar winters, night-time), and clouds block the signal almost entirely. Large parts of the tropics are cloud-covered more than 70% of the time.

### SAR (Synthetic Aperture Radar)

SAR is active - the satellite carries its own radar transmitter. It emits a microwave pulse, waits for the echo to return, and records both the amplitude (how strong the echo is) and the phase (the exact timing of the wave cycle). Because it uses its own signal:

- Works day and night (no sunlight dependency)
- Microwaves pass through clouds, rain, smoke — all-weather imaging and it can penetrate vegetation canopy and soil to some depth (depends on wavelength)

The "Synthetic Aperture" part: a real antenna large enough to give useful resolution would have to be kilometres long — physically impossible. Instead, the satellite moves along its orbit and combines echoes received across a long flight path to *synthesise* a large virtual antenna. This is done via range-Doppler signal processing.

**Key difference in what is recorded:** Optical records reflectance — how much light bounces back. SAR records backscatter — how much radar energy returns, which depends on surface roughness, dielectric properties (moisture), and geometric structure. Water looks dark in SAR (specular reflection goes away from the sensor), rough urban surfaces look very bright (corner reflectors), and vegetation shows moderate intermediate values.

---

## 2. Key SAR concepts relevant to ML

### Backscatter (σ⁰ — sigma naught)

The fundamental measurement: how much radar energy per unit area returns to the sensor, expressed in decibels (dB). Water: −20 to −30 dB (very low). Urban: +5 to +15 dB (very high). This contrast is what makes ship detection on ocean so tractable — ships appear as bright points on dark background.

**Why dB matters for ML:** Raw linear scale values span many orders of magnitude. Converting to dB compresses this range and makes the distribution closer to Gaussian — better for neural network training. We normalise dB values to [−1, 1] before feeding to the model.

### Polarisation (VV, VH)

The radar pulse oscillates in a plane. You can transmit vertical (V) or horizontal (H) and receive in either polarisation:

- **VV** (transmit V, receive V): sensitive to surface roughness. Best for water/ocean, smooth surfaces.
- **VH** (transmit V, receive H): cross-polarised. Sensitive to volume scattering — vegetation, random structure.
- Ships create **double-bounce** (building wall + ground reflection) which is strong in both VV and VH.

Using both VV and VH as a 2-channel input (rather than one channel) gives the model complementary information — similar to how RGB gives complementary spectral information in optical.

### Speckle

SAR images look grainy. This is not sensor noise — it is a physical phenomenon from coherent interference. When many scatterers exist within one resolution cell, their echoes interfere constructively and destructively in a pattern that looks random even over a homogeneous surface.

**For ML:** Speckle is multiplicative noise. A pixel-wise classifier will see high variance even for the same material class. Two solutions: (1) spatial averaging / filtering before the model, or (2) use a CNN with a large enough receptive field that it implicitly averages speckle. We use the latter — YOLOv8's receptive field covers 512×512 pixels, averaging out most speckle.

### Incidence angle

The angle at which the radar beam hits the surface. Sentinel-1 IW mode varies from ~29° to ~46° across the swath. The same surface imaged at different incidence angles gives different backscatter values — a known systematic bias. In practice, for a fixed sensor like Sentinel-1, the variation within a scene is small enough that we don't normalise per-incidence-angle. For cross-sensor work, it would matter.

### SAR-specific artefacts (critical for ship detection)

Ships are bright — but so are other things that are NOT ships:

- **Azimuth ambiguities:** Due to how the SAR focusing works, very bright targets produce "ghost" copies displaced ~10 km in the along-track direction. These look identical to ships.
- **Sidelobes:** Energy leaking from strong point targets creates bright streaks adjacent to them.
- **Offshore platforms, buoys:** Also appear as bright point targets.

CFAR (our baseline) cannot distinguish these from real ships — it only knows "bright relative to local background." YOLOv8 learns the spatial signature of each from training examples and suppresses them.

### Product types

- **GRD (Ground Range Detected):** Amplitude only, multi-looked (averaged over several pulses to reduce speckle), projected to ground range. Standard product for detection tasks. This is what xView3 uses.
- **SLC (Single Look Complex):** Full complex signal (amplitude + phase). Required for interferometry. Not used here.

---

## 3. Application areas

### Ship detection and dark vessel identification

My chosen use case. Ships appear as bright point targets on dark ocean. The physically interesting extension: cross-reference detections with AIS (Automatic Identification System) transponder broadcasts. Vessels that appear in SAR but not in AIS have switched off their transponders — dark vessels. This is associated with illegal fishing, sanctions evasion, and smuggling. The signal is an *absence* (no AIS record) rather than a feature in the image, making it a data fusion problem rather than a pure vision problem.

### Flood / water detection

Water is very dark in SAR (smooth surface → specular reflection away from sensor). During floods, normally dry land becomes dark. Simple comparison between pre-flood and post-flood acquisitions highlights inundated areas. Even a simple pixel-level threshold on VV backscatter works reasonably well. More sophisticated approaches use U-Nets trained on labelled flood events (Sen1Floods11 dataset). The result is immediately interpretable on a map — blue overlay = water.

### Land cover mapping

Different surfaces have characteristic backscatter signatures across VV and VH, and these evolve seasonally. Urban areas: high backscatter, temporally stable. Forests: moderate backscatter, seasonal variation. Cropland: strong seasonal signal as crops grow and are harvested. A stack of 12 monthly Sentinel-1 acquisitions captures this temporal pattern. Classical approach: GLCM texture features + Random Forest. Deep approach: U-Net with multi-temporal input channels.

---

**Key insight:** The distinction between classical (CFAR, RF) and deep (YOLO, U-Net) is not just accuracy — it is what the model can learn. CFAR can only threshold relative to local statistics. YOLOv8, after fine-tuning on xView3, learns to recognise the *shape* of azimuth ambiguities and sidelobes and suppress them. This is the core improvement we expect to see in the evaluation.

---

## 5. Why 2-channel input matters

Standard YOLOv8 expects 3-channel RGB. We modified the first convolutional layer to accept 2 channels (VV, VH). The alternative — faking a third channel as VV−VH — introduces a derived feature that is not independent information. Our approach:

1. Creates a 2-channel model via `DetectionModel(cfg, ch=2)`
2. Initialises from pretrained RGB weights: VV ← average(R, G), VH ← B
3. Fine-tunes all layers with differential learning rates

The 2-channel model has 33% fewer parameters in the first layer. With limited training data (~20K xView3 labels), this reduces overfitting risk in the first layer while all other pretrained features are preserved.
