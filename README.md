# TerraSpectra — Hyperspectral Crop Disease Forecasting

## What this proves

Before building any model, the data pipeline answers the essential question:
**is there actually a detectable signal here?** It generates a mock
1,000-acre farm as a hyperspectral cube (200 bands per zone), plants a
hidden early-stage disease outbreak that's invisible in normal RGB light
but chemically different in the red-edge/near-infrared bands, then runs
PCA to prove that signal is real and separable — catching every stressed
zone with zero false positives in testing.

On top of that foundation, a real 3D-CNN is trained (using PyTorch) to
classify healthy vs. stressed zones directly from the raw spectral cube,
with its predictions rendered as a GPS-aligned risk heatmap on the
dashboard.

## What's inside

```
terraspectra/
├── backend/
│   ├── data.py           # mock hyperspectral farm generator + GPS coordinates per zone
│   ├── pca_pipeline.py   # PCA dimensionality reduction + anomaly scoring
│   ├── model.py           # 3D-CNN (PyTorch) trained on spectral patches to classify healthy vs. stressed zones
│   ├── main.py            # FastAPI app (runs on port 8001, so it can run alongside EconoCausal's 8000)
│   └── requirements.txt
└── frontend/
    └── index.html         # dashboard (vanilla HTML/JS + Plotly — no React/Babel)
```

## Run it

**Backend:**
```bash
cd backend
py -3.11 -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

**Frontend** — in a second terminal:
```bash
cd frontend
python -m http.server 5501
```
Then open `http://localhost:5501` in your browser.

## Walk through the demo

1. **Generate farm dataset** — builds the 10×20 grid of 5-acre zones, each
   with a full 200-band spectrum and real GPS coordinates.
2. **RGB view** — deliberately unremarkable: the hidden outbreak looks like
   every other healthy zone here, which is the whole point of the problem
   TerraSpectra solves.
3. **Run PCA risk analysis** — the Week 1 proof-of-signal step. Flags
   anomalous zones without ever using ground-truth labels.
4. **Train 3D-CNN** — trains the actual classifier on the spectral data
   (using the ground-truth labels this time — this is a supervised step,
   unlike PCA). Reports training accuracy, and accuracy on zones the model
   never saw during training (held-out validation).
5. **Click any zone** on the field view to see its GPS coordinates, PCA
   score, and the CNN's predicted risk side by side.
6. **Model vs. PCA agreement** — shows how often the two independent
   methods agree, a useful sanity check.

## Implementation notes worth being able to explain

- **Why "3D" CNN**: a normal 2D image CNN convolves over (height, width)
  with a few color channels. Here, each zone carries a 200-value spectrum
  instead of 3 color values — a 3D convolution lets the network learn
  patterns spanning both space (neighboring zones) and the spectral axis
  (nearby wavelengths) simultaneously.
- **Why a small 3×3 spatial patch per zone, not the whole farm at once**:
  disease evidence is local — real outbreaks cluster, so a zone's immediate
  neighbors are informative but a zone across the farm isn't. This also
  keeps the network small enough to train on a laptop CPU in seconds, and
  mirrors how a real tiled inference pipeline (Week 4) would process a
  large satellite image in chunks.
- **Why PCA uses the median point (Week 1) but the 3D-CNN uses true labels
  (Week 2)**: PCA's job was to prove a signal exists *without* assuming any
  labeled training data — a fair test for an early-stage problem. The
  3D-CNN is a supervised model and legitimately uses labels to learn to
  classify, the same way any real deployment would train on historical
  outbreak records.
- **Why GPS coordinates are synthetic**: real deployment would pull these
  from the actual satellite imagery's georeferencing metadata (what
  Rasterio/GDAL are for) — here they're generated as a plausible grid
  anchored near Maharashtra, India, since there's no real satellite feed
  behind this demo.

## What's next (Week 3)

Integrating a Vision Transformer's self-attention mechanism on top of the
CNN (to let the model directly learn correlations across distant spectral
bands, not just local convolution windows), plus a dynamic risk-analytics
sidebar.
