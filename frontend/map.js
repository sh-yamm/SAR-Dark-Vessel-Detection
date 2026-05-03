/**
 * map.js — Leaflet frontend for SAR Dark Vessel Detection
 *
 * Flow:
 *   1. User draws a rectangle AOI on the map
 *   2. Clicks "Run Inference"
 *   3. POST /detect with bbox → receives GeoJSON
 *   4. Renders markers: red = dark vessel, green = AIS-matched, yellow = CFAR
 *   5. Clicking a marker shows a popup with vessel details
 *
 * LLM note: this file was written with Claude assistance.
 * Domain logic (colour coding, popup content) was written manually.
 */

const API_BASE = "http://localhost:8000";

// ── Map initialisation ─────────────────────────────────────────────────── //

const map = L.map("map", {
  center: [20, 0],
  zoom: 3,
  zoomControl: true,
});

// Dark-themed basemap
L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' +
      ' &copy; <a href="https://carto.com/attributions">CARTO</a>',
    maxZoom: 18,
  }
).addTo(map);

// ── State ──────────────────────────────────────────────────────────────── //

let currentRect   = null;   // Leaflet rectangle drawn by user
let detectionLayer = null;  // LayerGroup holding ship markers

// ── Leaflet Draw setup ─────────────────────────────────────────────────── //

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

const drawControl = new L.Control.Draw({
  draw: {
    rectangle:  { shapeOptions: { color: "#58a6ff", weight: 2 } },
    polygon:    false,
    polyline:   false,
    circle:     false,
    circlemarker: false,
    marker:     false,
  },
  edit: { featureGroup: drawnItems },
});
map.addControl(drawControl);

map.on(L.Draw.Event.CREATED, (e) => {
  drawnItems.clearLayers();
  drawnItems.addLayer(e.layer);
  currentRect = e.layer;
  enableDetectButton();
});

map.on(L.Draw.Event.DELETED, () => {
  currentRect = null;
  disableDetectButton("Draw an AOI first");
});

// ── UI helpers ─────────────────────────────────────────────────────────── //

const btnDetect    = document.getElementById("btn-detect");
const btnClear     = document.getElementById("btn-clear");
const statsPanel   = document.getElementById("stats");
const loadingEl    = document.getElementById("loading");
const errorEl      = document.getElementById("error-msg");

function enableDetectButton() {
  btnDetect.disabled   = false;
  btnDetect.textContent = "Run Inference";
}

function disableDetectButton(msg) {
  btnDetect.disabled    = true;
  btnDetect.textContent = msg;
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
}

function showLoading()  { loadingEl.classList.remove("hidden"); }
function hideLoading()  { loadingEl.classList.add("hidden"); }

function updateStats(meta) {
  document.getElementById("stat-total").textContent   = meta.total_vessels   ?? "—";
  document.getElementById("stat-dark").textContent    = meta.dark_vessels    ?? "—";
  document.getElementById("stat-matched").textContent = meta.ais_matched     ?? "—";
  document.getElementById("stat-scene").textContent   = meta.scene_id        ?? "—";
  document.getElementById("stat-model").textContent   = meta.model_source    ?? "—";
  statsPanel.classList.remove("hidden");
}

// ── Marker factory ─────────────────────────────────────────────────────── //

function markerColour(props) {
  if (props.source === "cfar")     return "#d29922";   // yellow
  if (props.source === "otsu")     return "#a371f7";   // purple
  if (props.source === "svm")      return "#39d0d8";   // teal
  if (props.is_dark === true)      return "#f85149";   // red
  if (props.is_dark === false)     return "#3fb950";   // green
  return "#8b949e";                                     // grey (unknown AIS)
}

function makeCircleMarker(latlng, props) {
  return L.circleMarker(latlng, {
    radius:      props.source === "cfar" ? 4 : 6,
    color:       markerColour(props),
    fillColor:   markerColour(props),
    fillOpacity: 0.85,
    weight:      2,
    opacity:     1,
  });
}

function popupContent(props) {
  const isDark    = props.is_dark;
  const titleCls  = isDark === true  ? "dark-title"
                  : isDark === false ? "matched-title"
                  : "unknown-title";
  const titleText = isDark === true  ? "Dark Vessel (No AIS)"
                  : isDark === false ? "AIS-Matched Vessel"
                  : props.source === "cfar" ? "CFAR Detection" : "Unknown";

  const mmsi     = props.matched_mmsi ?? "—";
  const score    = props.score != null ? (props.score * 100).toFixed(1) + "%" : "—";
  const distM    = props.distance_m != null
                    ? props.distance_m.toFixed(0) + " m"
                    : "—";
  const src      = props.source ?? "yolo";

  return `
    <div class="popup-content">
      <h3 class="${titleCls}">${titleText}</h3>
      <table>
        <tr><td>Confidence</td><td>${score}</td></tr>
        <tr><td>AIS MMSI</td><td>${mmsi}</td></tr>
        <tr><td>AIS dist</td><td>${distM}</td></tr>
        <tr><td>Detector</td><td>${src.toUpperCase()}</td></tr>
      </table>
    </div>
  `;
}

// ── Render GeoJSON features as markers ─────────────────────────────────── //

function renderDetections(geojson) {
  if (detectionLayer) {
    map.removeLayer(detectionLayer);
  }
  detectionLayer = L.layerGroup();

  const features = geojson.features ?? [];
  features.forEach((feat) => {
    const [lon, lat] = feat.geometry.coordinates;
    const props      = feat.properties;
    const marker     = makeCircleMarker([lat, lon], props);
    marker.bindPopup(popupContent(props), { maxWidth: 220 });
    detectionLayer.addLayer(marker);
  });

  detectionLayer.addTo(map);

  // Zoom to detections if there are any
  if (features.length > 0) {
    const latlngs = features.map((f) => [
      f.geometry.coordinates[1],
      f.geometry.coordinates[0],
    ]);
    map.fitBounds(L.latLngBounds(latlngs), { padding: [40, 40], maxZoom: 10 });
  }
}

// ── Inference call ─────────────────────────────────────────────────────── //

async function runInference() {
  clearError();

  if (!currentRect) {
    showError("Please draw a bounding box on the map first.");
    return;
  }

  const bounds    = currentRect.getBounds();
  const bbox      = [
    bounds.getWest(),
    bounds.getSouth(),
    bounds.getEast(),
    bounds.getNorth(),
  ];

  const modelSource  = document.querySelector('input[name="model"]:checked').value;
  const includeCfar  = document.getElementById("include-cfar").checked;

  showLoading();
  disableDetectButton("Running …");

  try {
    const resp = await fetch(`${API_BASE}/detect`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        bbox:          bbox,
        model_source:  modelSource,
        include_cfar:  includeCfar,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail ?? `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    renderDetections(data);
    updateStats(data.meta ?? {});

    if ((data.features ?? []).length === 0) {
      showError(
        data.meta?.message ??
        "No vessels detected in this area. Try a different region or check that a precomputed scene covers this bbox."
      );
    }
  } catch (err) {
    showError(`Request failed: ${err.message}`);
  } finally {
    hideLoading();
    enableDetectButton();
  }
}

// ── Clear ──────────────────────────────────────────────────────────────── //

function clearAll() {
  drawnItems.clearLayers();
  if (detectionLayer) {
    map.removeLayer(detectionLayer);
    detectionLayer = null;
  }
  statsPanel.classList.add("hidden");
  clearError();
  currentRect = null;
  disableDetectButton("Draw an AOI first");
}

// ── Event listeners ────────────────────────────────────────────────────── //

btnDetect.addEventListener("click", runInference);
btnClear.addEventListener("click",  clearAll);

// ── Load available scenes on startup, auto-select first scene ──────────── //

(async () => {
  try {
    const resp = await fetch(`${API_BASE}/scenes`);
    if (!resp.ok) return;
    const data   = await resp.json();
    const scenes = data.scenes ?? [];

    if (scenes.length === 0) return;

    // Draw dashed outlines for all scenes
    scenes.forEach((scene) => {
      const [lonMin, latMin, lonMax, latMax] = scene.bbox;
      L.rectangle(
        [[latMin, lonMin], [latMax, lonMax]],
        { color: "#58a6ff", weight: 1, fillOpacity: 0.04, dashArray: "4 4" }
      ).addTo(map);
    });

    // Auto-select the first scene as the AOI and run inference immediately
    const [lonMin, latMin, lonMax, latMax] = scenes[0].bbox;
    map.fitBounds([[latMin, lonMin], [latMax, lonMax]], { padding: [40, 40] });

    // Place the AOI rectangle programmatically
    const aoiRect = L.rectangle([[latMin, lonMin], [latMax, lonMax]], {
      color: "#58a6ff", weight: 2, fillOpacity: 0.0,
    });
    drawnItems.addLayer(aoiRect);
    currentRect = aoiRect;
    enableDetectButton();

    // Auto-run inference after a short delay (let map settle)
    setTimeout(runInference, 600);
  } catch (_) {
    console.warn("Could not reach backend. Start uvicorn first.");
  }
})();
