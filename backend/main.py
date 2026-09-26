"""
main.py
--------
TerraSpectra REST API.

Endpoints:
    GET  /api/farm/generate   -> generate the mock farm dataset, return metadata
    GET  /api/farm/rgb        -> RGB composite grid, for the "what a camera sees" view
    GET  /api/farm/gps        -> per-zone latitude/longitude, for GPS-aligned map rendering
    POST /api/farm/pca        -> run PCA, return explained variance + per-zone scores
    POST /api/model/train     -> train the 3D-CNN, return its predicted risk grid + accuracy

Run with:
    uvicorn main:app --reload --port 8001
    (using 8001, not 8000, so this can run alongside EconoCausal's backend if needed)
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data import generate_farm_dataset, N_BANDS, WAVELENGTH_RANGE_NM
from pca_pipeline import run_pca

app = FastAPI(title="TerraSpectra API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"dataset": None, "model_result": None}


@app.get("/api/farm/generate")
def generate(rows: int = 10, cols: int = 20, seed: int = 7):
    ds = generate_farm_dataset(rows=rows, cols=cols, seed=seed)
    STATE["dataset"] = ds

    cube_size_mb = ds.cube.nbytes / (1024 * 1024)

    return {
        "rows": ds.rows,
        "cols": ds.cols,
        "n_bands": ds.cube.shape[2],
        "total_acres": ds.total_acres,
        "acres_per_zone": ds.acres_per_zone,
        "cube_size_mb": round(cube_size_mb, 2),
        "n_zones": ds.rows * ds.cols,
    }


@app.get("/api/farm/rgb")
def get_rgb_composite():
    ds = STATE["dataset"]
    if ds is None:
        raise HTTPException(400, "No farm dataset yet — call /api/farm/generate first")
    return {"rgb_grid": ds.rgb_composite.tolist()}  # (rows, cols, 3) nested list


@app.get("/api/farm/gps")
def get_gps_grid():
    ds = STATE["dataset"]
    if ds is None:
        raise HTTPException(400, "No farm dataset yet — call /api/farm/generate first")
    return {"lat_grid": ds.lat_grid.tolist(), "lon_grid": ds.lon_grid.tolist()}


@app.post("/api/farm/pca")
def compute_pca(n_components: int = 5):
    ds = STATE["dataset"]
    if ds is None:
        raise HTTPException(400, "No farm dataset yet — call /api/farm/generate first")

    result = run_pca(ds.cube, n_components=n_components)

    rows, cols = ds.rows, ds.cols
    anomaly_grid = result.anomaly_score.reshape(rows, cols)

    # flag the highest-scoring zones as "suspected stress" — using the
    # SAME simple heuristic a Week-1 pipeline would (no ML model yet):
    # anything notably above the median score.
    threshold = np.median(result.anomaly_score) + 3 * np.std(result.anomaly_score)
    suspected_mask = anomaly_grid > threshold

    # Average spectrum of flagged vs. unflagged zones -- using PCA's OWN
    # output here, not the hidden ground-truth mask, so this genuinely
    # reflects what the pipeline detected rather than what we know is true.
    flat_cube = ds.cube.reshape(rows * cols, ds.cube.shape[2])
    flat_suspected = suspected_mask.flatten()
    wavelengths = np.linspace(*WAVELENGTH_RANGE_NM, N_BANDS)

    healthy_mean = flat_cube[~flat_suspected].mean(axis=0) if (~flat_suspected).any() else np.zeros(N_BANDS)
    suspected_mean = flat_cube[flat_suspected].mean(axis=0) if flat_suspected.any() else np.zeros(N_BANDS)

    return {
        "explained_variance_ratio": result.explained_variance_ratio.tolist(),
        "pc1_pc2": result.components_2d.tolist(),
        "anomaly_grid": anomaly_grid.tolist(),
        "suspected_stress_grid": suspected_mask.tolist(),
        "suspected_zone_count": int(suspected_mask.sum()),
        "suspected_acres": float(suspected_mask.sum() * ds.acres_per_zone),
        "wavelengths": wavelengths.tolist(),
        "healthy_mean_spectrum": healthy_mean.tolist(),
        "suspected_mean_spectrum": suspected_mean.tolist(),
    }


@app.post("/api/model/train")
def train_cnn(epochs: int = 60):
    ds = STATE["dataset"]
    if ds is None:
        raise HTTPException(400, "No farm dataset yet — call /api/farm/generate first")

    from model import train_model  # imported here so the API still starts up
                                     # even before torch is installed

    result = train_model(ds.cube, ds.stressed_mask, epochs=epochs)
    STATE["model_result"] = result

    rows, cols = ds.rows, ds.cols
    threshold = 0.5
    flagged = result.predicted_prob_grid > threshold

    return {
        "predicted_prob_grid": result.predicted_prob_grid.tolist(),
        "flagged_grid": flagged.tolist(),
        "flagged_zone_count": int(flagged.sum()),
        "flagged_acres": float(flagged.sum() * ds.acres_per_zone),
        "train_accuracy": result.train_accuracy,
        "val_accuracy": result.val_accuracy,
        "val_zone_count": result.val_zone_count,
        "loss_history": result.loss_history,
        "lat_grid": ds.lat_grid.tolist(),
        "lon_grid": ds.lon_grid.tolist(),
    }


@app.get("/")
def root():
    return {"status": "TerraSpectra API running", "docs": "/docs"}
