# TerraSpectra — Hyperspectral Crop Disease Forecasting

**Week 1 of 4** — Data Pipeline + Map Scaffolding, per the project plan.

## What Week 1 proves

Before building any model, Week 1 answers the essential question: **is there
actually a detectable signal here?** It generates a mock 1,000-acre farm as a
hyperspectral cube (200 bands per zone), plants a hidden early-stage disease
outbreak that's invisible in normal RGB light but chemically different in the
red-edge/near-infrared bands, then runs PCA to prove that signal is real and
separable — catching all 9 stressed zones with zero false positives in
testing.

## What's inside

```
terraspectra/
├── backend/
│   ├── data.py           # mock hyperspectral farm generator (the disease is invisible in RGB, real in the spectrum)
│   ├── pca_pipeline.py   # PCA dimensionality reduction + anomaly scoring
│   ├── main.py           # FastAPI app (runs on port 8001, so it can run alongside EconoCausal's 8000)
│   └── requirements.txt
└── frontend/
    └── index.html        # plain HTML/JS dashboard (no React/Babel — learned from EconoCausal's issues)
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

(Using ports 8001/5501 instead of EconoCausal's 8000/5500 so both projects
can run side by side if you need to demo them together.)

## Walk through the demo

1. **Generate farm dataset** — builds the 10×20 grid of 5-acre zones, each
   with a full 200-band spectrum.
2. **See the RGB view** — this is deliberately unremarkable: the hidden
   outbreak looks like every other healthy zone here, which is the whole
   point of the problem TerraSpectra solves.
3. **Run PCA** — reduces the 200 bands down to a few principal components
   and flags zones that sit far from the healthy majority in that reduced
   space. The flagged zones line up with the hidden outbreak, proving there's
   a real signal before any model gets built.

## What's next (Week 2)

Week 2 replaces PCA's crude "distance from centroid" heuristic with the real
3D-CNN + Vision Transformer hybrid described in the project brief — trained
to classify healthy vs. chemically-stressed zones directly from the raw
spectral cube, plus the heatmap rendering on the frontend to visualize its
predictions.

## Implementation notes worth being able to explain

- **Why no real Rasterio/GDAL**: those parse real satellite file formats
  (GeoTIFF, HDF5) — since we don't have real satellite data, plain numpy
  generates the same *shape* of data (a spatial grid × spectral bands cube)
  without the native-dependency headache GDAL is notorious for.
- **Why the disease is a contiguous 3×3-ish cluster, not scattered pixels**:
  real fungal outbreaks spread locally from a point of infection — scattered
  random "stressed" pixels wouldn't be a realistic disease pattern.
- **Why PCA uses the median point, not the true labels, to score anomalies**:
  a real pipeline doesn't get to see the ground truth. Using the median
  assumes most zones are healthy (a fair assumption for an early, localized
  outbreak) — this is what makes it a legitimate proof of signal, not a
  rigged demo.
