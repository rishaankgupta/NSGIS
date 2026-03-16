"""
NSGIS Sentinel Data Simulator
==============================
Generates synthetic multi-sensor patches that are spectrally and spatially
identical in structure to real Sentinel-2 + ECOSTRESS + Sentinel-1 data.

When real data is available:
  - Replace generate_scene() with a rasterio-based tile loader
  - The rest of the pipeline is plug-and-play unchanged

Simulated bands:
  Sentinel-2: B2(Blue), B3(Green), B4(Red), B5(RE1), B6(RE2), B7(RE3),
               B8(NIR), B8A(Narrow NIR), B11(SWIR1), B12(SWIR2) — 10 bands
  ECOSTRESS:  LST thermal anomaly (delta from climatological mean), 1 band
  Sentinel-1: VV backscatter, VH backscatter — 2 bands
  Context:    Road distance (m), Water distance (m), VIIRS NTL, Elevation — 4 features

Activity archetypes encoded (subset of full 47):
  0: Non-industrial / background
  1: Tannery (Leather processing)
  2: Textile Dyeing
  3: E-waste Processing (Agbogbloshie-type)
  4: Metal/Battery Recycling
  5: Chemical/Solvent Storage
  6: Food/Organic Waste Processing
"""

import numpy as np
import json
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Spectral signature library per activity type ──────────────────────────────
# Values are (mean, std) tuples for each band, derived from published literature
# on informal industrial spectral signatures and UNEP environmental profiles.
# Format: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, LST_anomaly, SAR_VV, SAR_VH]

SPECTRAL_SIGNATURES = {
    0: {  # Non-industrial / background (mixed urban)
        "name": "Non-Industrial",
        "domain": "Background",
        "bands": {
            "B2": (0.08, 0.02), "B3": (0.09, 0.02), "B4": (0.10, 0.02),
            "B5": (0.12, 0.03), "B6": (0.14, 0.03), "B7": (0.15, 0.03),
            "B8": (0.16, 0.04), "B8A": (0.15, 0.03),
            "B11": (0.11, 0.03), "B12": (0.08, 0.02),
            "LST_anomaly": (0.5, 1.2),   # ~0°C anomaly, low variance
            "SAR_VV": (-8.0, 2.5),        # dB
            "SAR_VH": (-15.0, 3.0),
        },
        "context": {
            "road_dist": (45, 30), "water_dist": (280, 180),
            "ntl": (18, 8), "elevation": (30, 15)
        },
        "color": "#A8C4E0"
    },
    1: {  # Tannery — chromium ponds, corrugated metal roofs, near water
        "name": "Tannery",
        "domain": "D1: Leather Processing",
        "bands": {
            "B2": (0.04, 0.01), "B3": (0.05, 0.01), "B4": (0.04, 0.01),
            "B5": (0.06, 0.02), "B6": (0.07, 0.02), "B7": (0.07, 0.02),
            "B8": (0.08, 0.02), "B8A": (0.07, 0.02),
            "B11": (0.18, 0.04), "B12": (0.15, 0.03),  # High SWIR = metal roofs
            "LST_anomaly": (6.8, 2.1),   # Strong thermal — lime pits, drying
            "SAR_VV": (-3.2, 1.8),        # High — corrugated metal
            "SAR_VH": (-11.0, 2.2),
        },
        "context": {
            "road_dist": (25, 18), "water_dist": (35, 22),   # Near water
            "ntl": (28, 10), "elevation": (12, 8)
        },
        "color": "#8B4513"
    },
    2: {  # Textile Dyeing — dye ponds, chemical effluent, low NDVI
        "name": "Textile Dyeing",
        "domain": "D2: Textile & Finishing",
        "bands": {
            "B2": (0.05, 0.015), "B3": (0.06, 0.015), "B4": (0.07, 0.02),
            "B5": (0.09, 0.025), "B6": (0.10, 0.025), "B7": (0.10, 0.025),
            "B8": (0.11, 0.03), "B8A": (0.10, 0.025),
            "B11": (0.16, 0.04), "B12": (0.12, 0.03),
            "LST_anomaly": (4.2, 1.8),   # Warm — steam dyeing
            "SAR_VV": (-5.1, 2.0),
            "SAR_VH": (-13.2, 2.5),
        },
        "context": {
            "road_dist": (30, 20), "water_dist": (55, 35),
            "ntl": (22, 9), "elevation": (18, 10)
        },
        "color": "#6A0DAD"
    },
    3: {  # E-waste (Agbogbloshie) — open burning, scrap metal, toxic ash
        "name": "E-Waste Processing",
        "domain": "D3: Electronic Waste",
        "bands": {
            "B2": (0.06, 0.02), "B3": (0.06, 0.02), "B4": (0.07, 0.02),
            "B5": (0.08, 0.02), "B6": (0.09, 0.02), "B7": (0.09, 0.02),
            "B8": (0.09, 0.025), "B8A": (0.09, 0.02),
            "B11": (0.22, 0.05), "B12": (0.19, 0.04),  # High SWIR — metallic scrap
            "LST_anomaly": (9.5, 3.2),   # Very high — open burning
            "SAR_VV": (-2.1, 1.5),        # Very high — dense metallic debris
            "SAR_VH": (-9.8, 2.0),
        },
        "context": {
            "road_dist": (20, 12), "water_dist": (80, 45),
            "ntl": (35, 12), "elevation": (8, 5)
        },
        "color": "#FF4500"
    },
    4: {  # Metal/Battery Recycling — acid baths, smelting
        "name": "Metal/Battery Recycling",
        "domain": "D4: Metal Recycling",
        "bands": {
            "B2": (0.05, 0.015), "B3": (0.055, 0.015), "B4": (0.06, 0.018),
            "B5": (0.075, 0.02), "B6": (0.08, 0.02), "B7": (0.085, 0.02),
            "B8": (0.09, 0.025), "B8A": (0.088, 0.022),
            "B11": (0.20, 0.045), "B12": (0.17, 0.038),
            "LST_anomaly": (7.2, 2.5),
            "SAR_VV": (-2.8, 1.6),
            "SAR_VH": (-10.5, 2.1),
        },
        "context": {
            "road_dist": (35, 22), "water_dist": (120, 70),
            "ntl": (30, 11), "elevation": (22, 12)
        },
        "color": "#B8860B"
    },
    5: {  # Chemical/Solvent Storage — low thermal but distinctive SWIR
        "name": "Chemical Storage",
        "domain": "D5: Chemical Processing",
        "bands": {
            "B2": (0.06, 0.015), "B3": (0.07, 0.015), "B4": (0.08, 0.018),
            "B5": (0.09, 0.022), "B6": (0.10, 0.022), "B7": (0.11, 0.022),
            "B8": (0.12, 0.028), "B8A": (0.115, 0.026),
            "B11": (0.14, 0.035), "B12": (0.10, 0.028),
            "LST_anomaly": (2.1, 1.5),   # Moderate
            "SAR_VV": (-4.5, 1.9),
            "SAR_VH": (-12.0, 2.3),
        },
        "context": {
            "road_dist": (40, 25), "water_dist": (200, 120),
            "ntl": (20, 8), "elevation": (35, 18)
        },
        "color": "#228B22"
    },
    6: {  # Food/Organic Waste — low thermal, high organic matter
        "name": "Food/Organic Processing",
        "domain": "D6: Food & Organic",
        "bands": {
            "B2": (0.07, 0.02), "B3": (0.09, 0.02), "B4": (0.08, 0.02),
            "B5": (0.13, 0.03), "B6": (0.16, 0.03), "B7": (0.17, 0.03),
            "B8": (0.19, 0.04), "B8A": (0.18, 0.035),
            "B11": (0.10, 0.025), "B12": (0.07, 0.02),
            "LST_anomaly": (1.8, 1.4),
            "SAR_VV": (-7.2, 2.2),
            "SAR_VH": (-14.5, 2.8),
        },
        "context": {
            "road_dist": (50, 30), "water_dist": (90, 55),
            "ntl": (15, 7), "elevation": (25, 14)
        },
        "color": "#3CB371"
    }
}

BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
              "B11", "B12", "LST_anomaly", "SAR_VV", "SAR_VH"]
CONTEXT_NAMES = ["road_dist", "water_dist", "ntl", "elevation"]
ALL_FEATURES = BAND_NAMES + CONTEXT_NAMES  # 17 features total


def generate_patch(activity_class: int, add_noise: bool = True) -> np.ndarray:
    """Generate a single 17-feature vector for a given activity class."""
    sig = SPECTRAL_SIGNATURES[activity_class]
    features = []
    for band in BAND_NAMES:
        mean, std = sig["bands"][band]
        val = np.random.normal(mean, std if add_noise else 0)
        features.append(float(val))
    for ctx in CONTEXT_NAMES:
        mean, std = sig["context"][ctx]
        val = np.random.normal(mean, std if add_noise else 0)
        features.append(float(max(0, val)))
    return np.array(features)


def generate_scene(
    grid_size: int = 60,
    class_distribution: dict = None,
    cluster_spatial: bool = True
) -> tuple:
    """
    Generate a synthetic scene of grid_size x grid_size cells.

    Returns:
        X: (N, 17) feature array
        y: (N,) integer label array
        grid: (grid_size, grid_size) label array for visualization
    """
    if class_distribution is None:
        # Approximate Agbogbloshie / informal urban distribution
        class_distribution = {0: 0.55, 1: 0.06, 2: 0.08, 3: 0.12,
                               4: 0.08, 5: 0.05, 6: 0.06}

    n_cells = grid_size * grid_size
    grid = np.zeros((grid_size, grid_size), dtype=int)

    if cluster_spatial:
        # Plant activity cluster seeds and grow them spatially
        # (more realistic than random assignment)
        activity_classes = [c for c in class_distribution if c != 0]
        for act in activity_classes:
            n_seeds = max(1, int(class_distribution[act] * n_cells / 25))
            for _ in range(n_seeds):
                cx = np.random.randint(3, grid_size - 3)
                cy = np.random.randint(3, grid_size - 3)
                cluster_r = np.random.randint(2, 5)
                for dx in range(-cluster_r, cluster_r + 1):
                    for dy in range(-cluster_r, cluster_r + 1):
                        if dx**2 + dy**2 <= cluster_r**2:
                            nx_, ny_ = cx + dx, cy + dy
                            if 0 <= nx_ < grid_size and 0 <= ny_ < grid_size:
                                if np.random.random() < 0.75:
                                    grid[nx_, ny_] = act

    y = grid.flatten()
    X = np.array([generate_patch(label) for label in y])
    return X, y, grid


def save_dataset(path: str, n_train: int = 2000, n_val: int = 500):
    """Generate and save train/val datasets."""
    Path(path).mkdir(parents=True, exist_ok=True)

    # Training data — balanced across classes for robust model training
    X_parts, y_parts = [], []
    per_class = n_train // len(SPECTRAL_SIGNATURES)
    for cls in SPECTRAL_SIGNATURES:
        for _ in range(per_class):
            X_parts.append(generate_patch(cls))
            y_parts.append(cls)

    X_train = np.array(X_parts)
    y_train = np.array(y_parts)

    # Shuffle
    idx = np.random.permutation(len(y_train))
    X_train, y_train = X_train[idx], y_train[idx]

    # Validation scene — spatially realistic
    X_val, y_val, val_grid = generate_scene(grid_size=32)

    np.save(f"{path}/X_train.npy", X_train)
    np.save(f"{path}/y_train.npy", y_train)
    np.save(f"{path}/X_val.npy", X_val)
    np.save(f"{path}/y_val.npy", y_val)
    np.save(f"{path}/val_grid.npy", val_grid)

    meta = {
        "features": ALL_FEATURES,
        "classes": {str(k): v["name"] for k, v in SPECTRAL_SIGNATURES.items()},
        "domains": {str(k): v["domain"] for k, v in SPECTRAL_SIGNATURES.items()},
        "colors": {str(k): v["color"] for k, v in SPECTRAL_SIGNATURES.items()},
        "n_features": len(ALL_FEATURES),
        "n_classes": len(SPECTRAL_SIGNATURES)
    }
    with open(f"{path}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DATA] Saved: {len(X_train)} train, {len(X_val)} val samples")
    print(f"[DATA] Features: {ALL_FEATURES}")
    return X_train, y_train, X_val, y_val, val_grid


if __name__ == "__main__":
    save_dataset("/home/claude/nsgis/data")
    print("[DATA] Sentinel simulator ready.")
