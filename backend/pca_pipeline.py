"""
pca_pipeline.py
----------------
Week 1's "prove the pipeline can handle the data" step.

Real hyperspectral cubes are huge (200+ bands per pixel), so the very
first thing any pipeline does is dimensionality reduction -- typically
PCA -- before anything else (visualization, modeling) is computationally
feasible.

Here we:
  1. Flatten the (rows, cols, bands) cube into (n_zones, bands).
  2. Run PCA down to a handful of components.
  3. Score each zone by how far it sits from the "healthy cluster" in
     PCA space -- an early, crude anomaly signal that previews what the
     real 3D-CNN + Vision Transformer model (Week 2) will formalize
     properly. This is NOT the final model, just proof the signal exists.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    explained_variance_ratio: np.ndarray  # how much signal each component captures
    components_2d: np.ndarray             # (n_zones, 2) -- PC1/PC2 coordinates per zone, for plotting
    anomaly_score: np.ndarray             # (n_zones,) -- distance from the healthy centroid in PCA space


def run_pca(cube: np.ndarray, n_components: int = 5) -> PCAResult:
    rows, cols, n_bands = cube.shape
    flattened = cube.reshape(rows * cols, n_bands)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flattened)

    # crude anomaly score: distance from the majority ("healthy") centroid.
    # We don't get to use the ground-truth mask here -- that would be
    # cheating -- so we assume the majority of zones are healthy (a fair
    # assumption for an early-stage, localized outbreak) and measure
    # distance from THEIR centroid in PC1/PC2 space.
    pc1_pc2 = transformed[:, :2]
    median_point = np.median(pc1_pc2, axis=0)  # robust to a small outlier cluster
    distances = np.linalg.norm(pc1_pc2 - median_point, axis=1)

    return PCAResult(
        explained_variance_ratio=pca.explained_variance_ratio_,
        components_2d=pc1_pc2,
        anomaly_score=distances,
    )
