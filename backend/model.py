"""
model.py
---------
Week 2 — the real 3D-CNN, replacing PCA's rough anomaly-distance stand-in
with an actual trained classifier.

Why "3D" and not a normal 2D CNN:
    A normal image CNN convolves over (height, width) with a handful of
    color channels. Here, each "pixel" (zone) carries a full 200-value
    SPECTRUM, not 3 color values. A 3D convolution lets the network learn
    patterns that span BOTH space (neighboring zones) AND the spectral
    dimension (nearby wavelengths) at once -- e.g. "a dip across bands
    60-75 that also shows up in the 2 zones next to this one" is a
    genuinely 3D pattern (2 spatial axes + 1 spectral axis).

Why a small spatial patch (k=3) per zone, not the whole farm at once:
    Disease evidence is local -- a zone's neighbors are informative (real
    outbreaks cluster), but a zone on the other side of a 1,000-acre farm
    isn't. Feeding the model a small neighborhood window keeps the network
    tiny and fast to train on a laptop CPU, and mirrors how a real tiled
    inference pipeline (Week 4) would process a huge satellite image in
    chunks anyway.
"""

from dataclasses import dataclass

import numpy as np

PATCH_SIZE = 3  # k x k spatial neighborhood per zone


def extract_patches(cube: np.ndarray, k: int = PATCH_SIZE) -> np.ndarray:
    """
    For every zone (r, c) in the cube, cuts out a (k, k, n_bands) neighborhood
    centered on it, using reflect-padding at the farm's edges (so edge zones
    still get a full k x k window instead of a smaller one).

    Returns shape (rows * cols, 1, n_bands, k, k) -- ready for a PyTorch
    Conv3d, which expects (batch, channels, depth, height, width).
    """
    rows, cols, n_bands = cube.shape
    pad = k // 2
    padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    patches = np.zeros((rows * cols, 1, n_bands, k, k), dtype=np.float32)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            window = padded[r:r + k, c:c + k, :]          # (k, k, n_bands)
            window = np.transpose(window, (2, 0, 1))       # (n_bands, k, k)
            patches[idx, 0] = window
            idx += 1
    return patches


@dataclass
class TrainResult:
    predicted_prob_grid: np.ndarray   # (rows, cols) -- model's predicted P(stressed) per zone
    train_accuracy: float
    val_accuracy: float
    val_zone_count: int
    loss_history: list


def build_model(n_bands: int, k: int = PATCH_SIZE):
    """
    A deliberately small 3D-CNN: two Conv3d blocks that downsample mainly
    along the spectral axis (stride 4) while keeping the small spatial
    window mostly intact, then a global pool and a 2-class head.
    """
    import torch.nn as nn

    class TerraCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=(9, 3, 3), stride=(4, 1, 1), padding=(4, 1, 1)),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 16, kernel_size=(9, 3, 3), stride=(4, 1, 1), padding=(4, 1, 1)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
            self.head = nn.Linear(16, 2)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.head(x)

    return TerraCNN()


def _oversample_minority(patches: np.ndarray, labels: np.ndarray, rng: np.random.Generator, noise_std: float = 0.01) -> tuple:
    """
    Duplicates minority-class (stressed) training patches, with a little
    Gaussian noise added to each duplicate, until classes are balanced.

    This is used INSTEAD OF an extreme loss-reweighting factor. Reweighting
    the loss by a large ratio (this dataset's true imbalance is roughly
    20:1) turned out to destabilize training badly enough that the network
    collapsed to predicting the minority class for every input -- an easy
    way to satisfy a heavily-weighted loss term without learning anything.
    Oversampling changes the actual data distribution the network sees
    instead of distorting the loss landscape, which is the more robust
    fix for a dataset this small and this imbalanced. The noise keeps
    duplicated samples from being bit-for-bit identical (which would let
    the network trivially memorize rather than generalize).
    """
    minority_idx = np.where(labels == 1)[0]
    majority_idx = np.where(labels == 0)[0]
    if len(minority_idx) == 0 or len(minority_idx) >= len(majority_idx):
        return patches, labels

    n_needed = len(majority_idx) - len(minority_idx)
    extra_idx = rng.choice(minority_idx, size=n_needed, replace=True)
    extra_patches = patches[extra_idx] + rng.normal(0, noise_std, size=(n_needed,) + patches.shape[1:]).astype(np.float32)

    balanced_patches = np.concatenate([patches, extra_patches], axis=0)
    balanced_labels = np.concatenate([labels, np.ones(n_needed, dtype=labels.dtype)])

    shuffle = rng.permutation(len(balanced_labels))
    return balanced_patches[shuffle], balanced_labels[shuffle]


def train_model(
    cube: np.ndarray,
    stressed_mask: np.ndarray,
    epochs: int = 30,
    val_fraction: float = 0.2,
    seed: int = 0,
    batch_size: int = 16,
) -> TrainResult:
    """
    Trains the 3D-CNN on ALL zones' patches, holding out a random
    `val_fraction` of zones (not used in training OR oversampling at all)
    to report an honest accuracy number rather than one measured on data
    the model has seen.

    Note on labels: this step DOES use the ground-truth stressed_mask to
    train (that's what "supervised learning" means -- Week 1's PCA step
    was unsupervised specifically to prove the signal existed BEFORE
    committing to a labeled-training approach). In a real deployment, these
    labels would come from historical ground-truth outbreak records.
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    rows, cols, n_bands = cube.shape
    patches = extract_patches(cube)
    labels = stressed_mask.flatten().astype(np.int64)
    n_zones = patches.shape[0]

    indices = rng.permutation(n_zones)
    n_val = max(1, int(n_zones * val_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_patches_bal, train_labels_bal = _oversample_minority(patches[train_idx], labels[train_idx], rng)

    X_train = torch.from_numpy(train_patches_bal)
    y_train = torch.from_numpy(train_labels_bal)
    X_val = torch.from_numpy(patches[val_idx])
    y_val = torch.from_numpy(labels[val_idx])

    model = build_model(n_bands)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_train = len(y_train)
    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_order = rng.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, batch_size):
            batch_idx = epoch_order[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X_train[batch_idx])
            loss = criterion(logits, y_train[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        loss_history.append(float(np.mean(epoch_losses)))

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).argmax(dim=1)
        train_acc = float((train_preds == y_train).float().mean())

        val_preds_logits = model(X_val)
        val_preds = val_preds_logits.argmax(dim=1)
        val_acc = float((val_preds == y_val).float().mean()) if len(y_val) else float("nan")

        all_probs = torch.softmax(model(torch.from_numpy(patches)), dim=1)[:, 1].numpy()

    predicted_prob_grid = all_probs.reshape(rows, cols)

    return TrainResult(
        predicted_prob_grid=predicted_prob_grid,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        val_zone_count=len(y_val),
        loss_history=loss_history,
    )


if __name__ == "__main__":
    from data import generate_farm_dataset

    ds = generate_farm_dataset()
    patches = extract_patches(ds.cube)
    print("patches shape:", patches.shape)

    result = train_model(ds.cube, ds.stressed_mask, epochs=60)
    print("train accuracy:", result.train_accuracy)
    print("val accuracy:", result.val_accuracy, "on", result.val_zone_count, "held-out zones")
    print("final loss:", result.loss_history[-1])
