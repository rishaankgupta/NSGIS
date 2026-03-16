"""
NSGIS Real Data Loader — Fixed version with cloud masking
Place in: nsgis/data/real_data_loader.py
"""

import numpy as np
import sys
from pathlib import Path

# ── ONLY CHANGE THIS LINE ──────────────────────────────────────────────────────
SAFE_PATH = r"C:\Users\siter\Downloads\S2B_MSIL2A_20260201T101139_N0511_R022_T30NZM_20260201T153614.SAFE\S2B_MSIL2A_20260201T101139_N0511_R022_T30NZM_20260201T153614.SAFE"
# ──────────────────────────────────────────────────────────────────────────────

AGBOGBLOSHIE_LAT = 5.5322
AGBOGBLOSHIE_LON = -0.2210
CROP_SIZE        = 300

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def find_granule_path(safe_path):
    safe = Path(safe_path)
    granule_dir = safe / "GRANULE"
    if not granule_dir.exists():
        raise FileNotFoundError(f"GRANULE folder not found in: {safe_path}")
    l2a_folders = [f for f in granule_dir.iterdir() if f.is_dir()]
    if not l2a_folders:
        raise FileNotFoundError("No L2A subfolder found in GRANULE/")
    print(f"[LOADER] Found granule: {l2a_folders[0].name}")
    return l2a_folders[0] / "IMG_DATA"


def load_band(img_data_path, resolution, band_suffix):
    import rasterio
    r_folder = Path(img_data_path) / resolution
    matches  = list(r_folder.glob(f"*_{band_suffix}_{resolution[1:]}.jp2"))
    if not matches:
        raise FileNotFoundError(f"Band not found: {band_suffix} in {r_folder}")
    with rasterio.open(matches[0]) as src:
        data      = src.read(1).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        profile   = src.profile
    data = np.clip(data / 10000.0, 0, 1)
    return data, transform, crs, profile


def load_scl(img_data_path):
    try:
        import rasterio
        r_folder = Path(img_data_path) / "R20m"
        matches  = list(r_folder.glob("*_SCL_20m.jp2"))
        if not matches:
            print("[LOADER] SCL not found — no cloud masking")
            return None
        with rasterio.open(matches[0]) as src:
            scl = src.read(1).astype(np.uint8)
        print(f"[LOADER] SCL loaded: {scl.shape}")
        return scl
    except Exception as e:
        print(f"[LOADER] SCL load failed ({e})")
        return None


def upsample(band_20m, target_shape):
    from scipy.ndimage import zoom
    zf = (target_shape[0] / band_20m.shape[0],
          target_shape[1] / band_20m.shape[1])
    return zoom(band_20m, zf, order=1).astype(np.float32)


def compute_water_distance(b03, b08):
    from scipy.ndimage import distance_transform_edt
    denom = b03 + b08
    denom[denom == 0] = 1e-6
    ndwi       = (b03 - b08) / denom
    water_mask = (ndwi > 0.15).astype(np.uint8)
    if water_mask.sum() > 0:
        return (distance_transform_edt(1 - water_mask) * 10.0).astype(np.float32)
    return np.full(b03.shape, 300.0, dtype=np.float32)


def compute_lst_proxy(b11, b04):
    denom = b11 + b04
    denom[denom == 0] = 1e-6
    return ((b11 - b04) / denom * 15.0).astype(np.float32)


def compute_sar_proxy(b11, b08):
    sar_vv = ((b11 - 0.10) / 0.10 * 4.0 - 6.0).astype(np.float32)
    sar_vh = (sar_vv - 7.0).astype(np.float32)
    return sar_vv, sar_vh


def load_real_scene(safe_path=SAFE_PATH, crop_size=CROP_SIZE,
                    center_lat=AGBOGBLOSHIE_LAT, center_lon=AGBOGBLOSHIE_LON):
    print(f"[LOADER] Loading: {safe_path}")
    img_data = find_granule_path(safe_path)

    # 10m bands
    print("[LOADER] Reading 10m bands...")
    b02, transform, _, _ = load_band(img_data, "R10m", "B02")
    b03, _, _, _          = load_band(img_data, "R10m", "B03")
    b04, _, _, _          = load_band(img_data, "R10m", "B04")
    b08, _, _, _          = load_band(img_data, "R10m", "B08")
    full_shape = b02.shape
    print(f"[LOADER] Tile: {full_shape[0]}x{full_shape[1]} = "
          f"{full_shape[0]*10/1000:.0f}km x {full_shape[1]*10/1000:.0f}km")

    # Cloud mask from SCL
    print("[LOADER] Loading cloud mask...")
    scl_20m = load_scl(img_data)
    if scl_20m is not None:
        cloud_20m   = np.isin(scl_20m, [8, 9, 10, 11]).astype(np.float32)
        cloud_full  = upsample(cloud_20m, full_shape) > 0.5
        print(f"[LOADER] {cloud_full.mean()*100:.1f}% of tile is cloud")
    else:
        cloud_full = np.zeros(full_shape, dtype=bool)

    # Locate Agbogbloshie
    print(f"[LOADER] Locating Agbogbloshie ({center_lat}N, {center_lon}E)...")
    try:
        from rasterio.transform import rowcol
        from pyproj import Transformer
        tr  = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
        x,y = tr.transform(center_lon, center_lat)
        center_row, center_col = [int(v) for v in rowcol(transform, x, y)]
        print(f"[LOADER] Pixel: row={center_row}, col={center_col}")
    except Exception as e:
        print(f"[LOADER] Fallback coords ({e})")
        center_row = int(full_shape[0] * 0.47)
        center_col = int(full_shape[1] * 0.38)
        print(f"[LOADER] Fallback: row={center_row}, col={center_col}")

    # Crop
    half      = crop_size // 2
    row_start = max(0, center_row - half)
    row_end   = min(full_shape[0], row_start + crop_size)
    col_start = max(0, center_col - half)
    col_end   = min(full_shape[1], col_start + crop_size)
    actual_rows = row_end - row_start
    actual_cols = col_end - col_start
    ts = (actual_rows, actual_cols)

    print(f"[LOADER] Crop: rows {row_start}-{row_end}, cols {col_start}-{col_end}")
    print(f"         {actual_rows*10/1000:.1f}km x {actual_cols*10/1000:.1f}km "
          f"({actual_rows*actual_cols:,} cells)")

    def c10(band):
        return band[row_start:row_end, col_start:col_end]

    b02_c = c10(b02); b03_c = c10(b03)
    b04_c = c10(b04); b08_c = c10(b08)
    cloud_crop = c10(cloud_full)

    # 20m bands
    print("[LOADER] Reading 20m bands...")
    def load20(suffix):
        d, _, _, _ = load_band(img_data, "R20m", suffix)
        hs = row_start//2; hc = col_start//2
        hr = actual_rows//2; hco = actual_cols//2
        return upsample(d[hs:hs+hr, hc:hc+hco], ts)

    b05 = load20("B05"); b06 = load20("B06")
    b07 = load20("B07"); b8a = load20("B8A")
    b11 = load20("B11"); b12 = load20("B12")

    # Derived features
    print("[LOADER] Computing proxies...")
    lst_anomaly    = compute_lst_proxy(b11, b04_c)
    sar_vv, sar_vh = compute_sar_proxy(b11, b08_c)
    water_dist     = compute_water_distance(b03_c, b08_c)
    road_dist      = np.full(ts, 35.0,  dtype=np.float32)
    ntl            = np.full(ts, 28.0,  dtype=np.float32)
    elevation      = np.full(ts, 20.0,  dtype=np.float32)

    # Stack — order must match ALL_FEATURES in sentinel_simulator.py
    feature_stack = np.stack([
        b02_c, b03_c, b04_c, b05, b06, b07, b08_c, b8a,
        b11,   b12,
        lst_anomaly, sar_vv, sar_vh,
        road_dist, water_dist, ntl, elevation
    ], axis=-1)

    X          = feature_stack.reshape(actual_rows * actual_cols, 17)
    cloud_flat = cloud_crop.flatten()

    # Mask clouds
    n_cloudy = int(cloud_flat.sum())
    if n_cloudy > 0:
        X[cloud_flat] = -9999.0
        print(f"[LOADER] Masked {n_cloudy:,} cloudy cells "
              f"({n_cloudy/len(cloud_flat)*100:.1f}%)")
    else:
        print("[LOADER] No cloudy cells in crop")

    # Print feature ranges for clear cells
    print("[LOADER] Feature ranges (clear cells):")
    names = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12",
             "LST_anomaly","SAR_VV","SAR_VH",
             "road_dist","water_dist","ntl","elevation"]
    clear = ~cloud_flat
    for i, name in enumerate(names):
        v = X[clear, i]
        if len(v) > 0:
            print(f"  {name:<15} {v.min():.3f} / {v.max():.3f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_DIR / "real_X_scene.npy"),    X)
    np.save(str(OUTPUT_DIR / "real_cloud_mask.npy"), cloud_flat)
    np.save(str(OUTPUT_DIR / "real_grid_shape.npy"), np.array([actual_rows, actual_cols]))

    _save_preview(b04_c, b03_c, b02_c, cloud_crop,
                  OUTPUT_DIR / "agbogbloshie_preview.png",
                  actual_rows, actual_cols)

    print(f"\n[LOADER] Saved to: {OUTPUT_DIR}")
    return X, None, np.zeros((actual_rows, actual_cols), dtype=int), (actual_rows, actual_cols)


def _save_preview(r, g, b, cloud_mask, path, rows, cols):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        rgb  = np.clip(np.dstack([r, g, b]) * 3.5, 0, 1)
        rgba = np.zeros((rows, cols, 4), dtype=np.float32)
        rgba[cloud_mask, :3] = 1.0
        rgba[cloud_mask,  3] = 0.65

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb, interpolation='nearest')
        ax.imshow(rgba, interpolation='nearest')

        ax.annotate("Korle Lagoon\n(Agbogbloshie adjacent)",
                    xy=(145, 120), xytext=(20, 40),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
                    color="yellow", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

        clear_pct = (~cloud_mask).sum() / cloud_mask.size * 100
        ax.set_title(f"Agbogbloshie Study Area — Sentinel-2 True Color\n"
                     f"S2B L2A 2026-02-01  |  T30NZM  |  10m  |  {clear_pct:.0f}% clear",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Easting (10m pixels)")
        ax.set_ylabel("Northing (10m pixels)")
        ax.plot([20, 120], [rows-20, rows-20], 'w-', linewidth=3)
        ax.text(70, rows-30, '1 km', ha='center',
                color='white', fontsize=9, fontweight='bold')

        patches = [mpatches.Patch(color='white',   alpha=0.65, label='Cloud (masked)'),
                   mpatches.Patch(color='#C8A882',             label='Clear (processed)')]
        ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.8)

        plt.tight_layout()
        plt.savefig(str(path), dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[LOADER] Preview saved: {path.name}")
    except Exception as e:
        print(f"[LOADER] Preview failed ({e})")


if __name__ == "__main__":
    print("=" * 60)
    print("  NSGIS Real Data Loader  |  Agbogbloshie  |  T30NZM")
    print("=" * 60)
    for pkg in ["rasterio", "scipy", "numpy", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[ERROR] Missing: {pkg}  →  pip install {pkg}")
            sys.exit(1)
    X, _, _, shape = load_real_scene()
    print(f"\n{'='*60}")
    print(f"  LOADED: {shape[0]}x{shape[1]} = {shape[0]*shape[1]:,} cells")
    print(f"  Area:   {shape[0]*10/1000:.1f}km x {shape[1]*10/1000:.1f}km")
    print(f"{'='*60}")
