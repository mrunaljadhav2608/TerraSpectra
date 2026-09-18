"""
data.py
--------
Generates a mock hyperspectral data cube for a farm, standing in for what
Rasterio/GDAL would normally parse out of a real NASA Hyperion or ESA
Sentinel-2 file (HDF5 / GeoTIFF). We skip GDAL entirely here since it's a
notoriously painful native dependency to install, and since this is
synthetic data anyway, plain numpy does the same job.

Farm layout:
    A 1,000-acre farm divided into 5-acre zones -> a grid of zones
    (default 10 rows x 20 cols = 200 zones = 1,000 acres).

Each zone has a full reflectance SPECTRUM across `N_BANDS` wavelengths
(200 bands, standing in for a real hyperspectral sensor's 200+ channels
across visible + near-infrared + short-wave infrared).

The key idea the whole project hinges on:
    A healthy plant and an EARLY-STAGE infected plant look nearly
    IDENTICAL in the 3 visible-light bands (red/green/blue) -- which is
    why RGB satellite imagery can't catch it early. But a real fungal
    infection changes the plant's internal chemistry (chlorophyll
    breakdown) well before any visible yellowing, and that shows up as a
    subtle shift in the NEAR-INFRARED "red edge" region of the spectrum.

So: we bake a hidden "infected" cluster of zones into the data whose
visible-band reflectance looks normal, but whose red-edge/NIR bands are
measurably shifted. That's the signal a real hyperspectral pipeline (and
our future 3D-CNN + ViT model in Week 2) is built to catch.
"""

from dataclasses import dataclass

import numpy as np

N_BANDS = 200
WAVELENGTH_RANGE_NM = (400, 2400)  # visible through short-wave infrared

# Rough index ranges within our 200 bands standing in for real-sensor regions
VISIBLE_BAND_RANGE = (0, 45)      # ~400-830nm: what RGB cameras see
RED_EDGE_BAND_RANGE = (45, 70)    # ~830-1050nm: where stress shows up first
NIR_BAND_RANGE = (70, 130)        # near-infrared
SWIR_BAND_RANGE = (130, 200)      # short-wave infrared


@dataclass
class FarmDataset:
    cube: np.ndarray            # shape (rows, cols, N_BANDS) -- the full hyperspectral cube
    rgb_composite: np.ndarray   # shape (rows, cols, 3) uint8 -- what a normal camera would see
    stressed_mask: np.ndarray   # shape (rows, cols) bool -- ground truth, hidden from the "model"
    rows: int
    cols: int
    acres_per_zone: float = 5.0

    @property
    def total_acres(self) -> float:
        return self.rows * self.cols * self.acres_per_zone


def _wavelengths() -> np.ndarray:
    return np.linspace(*WAVELENGTH_RANGE_NM, N_BANDS)


def _healthy_spectrum(wavelengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    A simplified healthy-vegetation reflectance curve: low in visible
    (absorbed by chlorophyll for photosynthesis), a sharp jump at the
    'red edge' (~700-750nm), high in near-infrared (leaf structure
    reflects strongly), gently declining through SWIR (water absorption).
    """
    red_edge_center = 720
    base = (
        0.08
        + 0.35 / (1 + np.exp(-(wavelengths - red_edge_center) / 15))  # the red-edge rise
        + 0.05 * np.exp(-((wavelengths - 550) ** 2) / (2 * 40 ** 2))   # slight green bump
    )
    swir_decay = np.clip(1 - (wavelengths - 1400) / 3000, 0.5, 1.0)
    base *= swir_decay
    noise = rng.normal(0, 0.01, size=wavelengths.shape)
    return np.clip(base + noise, 0.01, 0.95)


def _stressed_spectrum(wavelengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Early fungal-blight stress: chlorophyll starts breaking down, which
    shifts the red-edge transition EARLIER (toward shorter wavelengths)
    and slightly dampens the NIR plateau -- but critically, the VISIBLE
    bands barely move yet, so it still looks green to the naked eye / RGB
    camera.
    """
    shifted_red_edge_center = 695  # shifted left vs. healthy's 720
    base = (
        0.08
        + 0.30 / (1 + np.exp(-(wavelengths - shifted_red_edge_center) / 15))  # earlier, slightly weaker rise
        + 0.05 * np.exp(-((wavelengths - 550) ** 2) / (2 * 40 ** 2))
    )
    swir_decay = np.clip(1 - (wavelengths - 1400) / 3000, 0.5, 1.0)
    base *= swir_decay
    noise = rng.normal(0, 0.01, size=wavelengths.shape)
    return np.clip(base + noise, 0.01, 0.95)


def generate_farm_dataset(rows: int = 10, cols: int = 20, seed: int = 7) -> FarmDataset:
    rng = np.random.default_rng(seed)
    wavelengths = _wavelengths()

    cube = np.zeros((rows, cols, N_BANDS), dtype=np.float32)
    stressed_mask = np.zeros((rows, cols), dtype=bool)

    # Plant a single contiguous "outbreak" cluster of zones (a real fungal
    # blight spreads locally, it doesn't appear as random scattered pixels).
    center_r, center_c = rng.integers(2, rows - 2), rng.integers(3, cols - 3)
    outbreak_radius = 1  # a 3x3-ish cluster -> roughly matches the "5-acre zone" scale in the brief

    for r in range(rows):
        for c in range(cols):
            is_stressed = abs(r - center_r) <= outbreak_radius and abs(c - center_c) <= outbreak_radius
            spectrum = _stressed_spectrum(wavelengths, rng) if is_stressed else _healthy_spectrum(wavelengths, rng)
            cube[r, c, :] = spectrum
            stressed_mask[r, c] = is_stressed

    rgb_composite = _to_rgb_composite(cube)

    return FarmDataset(
        cube=cube,
        rgb_composite=rgb_composite,
        stressed_mask=stressed_mask,
        rows=rows,
        cols=cols,
    )


def _to_rgb_composite(cube: np.ndarray) -> np.ndarray:
    """
    Builds a 'true color' composite the way a normal camera would see the
    field, by sampling 3 bands roughly corresponding to red/green/blue
    wavelengths out of the visible range. This is deliberately similar for
    healthy vs. early-stressed zones -- proving the point that RGB alone
    can't catch this yet.
    """
    wavelengths = _wavelengths()
    red_idx = int(np.argmin(np.abs(wavelengths - 660)))
    green_idx = int(np.argmin(np.abs(wavelengths - 560)))
    blue_idx = int(np.argmin(np.abs(wavelengths - 470)))

    r = cube[:, :, red_idx]
    g = cube[:, :, green_idx]
    b = cube[:, :, blue_idx]

    # boost green channel a bit for a natural-looking vegetation composite
    rgb = np.stack([r * 0.9, g * 1.6, b * 0.8], axis=-1)
    rgb = np.clip(rgb / rgb.max(), 0, 1)
    return (rgb * 255).astype(np.uint8)


if __name__ == "__main__":
    ds = generate_farm_dataset()
    print("cube shape:", ds.cube.shape)
    print("total acres:", ds.total_acres)
    print("stressed zones:", ds.stressed_mask.sum(), "of", ds.rows * ds.cols)
