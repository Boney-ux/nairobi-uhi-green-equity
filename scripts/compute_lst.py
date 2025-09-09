import os
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import mapping
import matplotlib.pyplot as plt

# -------------------------------
# Generic helpers
# -------------------------------

def nan_min_max(arr: np.ndarray):
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return (np.nan, np.nan)
    return (float(np.nanmin(arr)), float(np.nanmax(arr)))


def assert_aoi_ok(aoi: gpd.GeoDataFrame, aoi_path: str):
    if aoi.empty:
        raise ValueError(f"AOI has no features: {aoi_path}")
    if aoi.crs is None:
        raise ValueError(
            "AOI has no CRS. Provide a .prj for shapefile or use GeoJSON with a defined CRS."
        )



def read_aoi(aoi_path: str) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(aoi_path)
    assert_aoi_ok(aoi, aoi_path)
    return aoi


def read_clip_band(raster_path: str, aoi: gpd.GeoDataFrame):

    with rasterio.open(raster_path) as src:
        if aoi.crs != src.crs:
            aoi = aoi.to_crs(src.crs)
        shapes = [mapping(geom) for geom in aoi.geometry]
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, nodata=src.nodata, filled=False
        )
        meta = src.meta.copy()

    meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "count": 1,
        }
    )

    band_masked = out_image[0]
    band = band_masked.astype(np.float32).filled(np.nan)
    band_min, band_max = nan_min_max(band)
    return band, meta, {"min": band_min, "max": band_max}

def meta_as_float(meta):
    m = meta.copy()
    m.update({"dtype": "float32"})
    return m


def save_raster(path: str, arr2d: np.ndarray, meta):
    meta_w = meta.copy()
    meta_w.update({"dtype": "float32", "count": 1})
    with rasterio.open(path, "w", **meta_w) as dst:
        dst.write(arr2d.astype(np.float32), 1)




def step_toa_radiance(b10: np.ndarray, ML: float, AL: float):
    """TOA radiance = ML * DN + AL"""
    L = (ML * b10) + AL
    return L, {"min": float(np.nanmin(L)), "max": float(np.nanmax(L))}

def step_bt_kelvin(toa_L: np.ndarray, K1: float, K2: float):

    eps = 1e-6
    T = K2 / np.log((K1 / (toa_L + eps)) + 1.0)
    return T, {"min": float(np.nanmin(T)), "max": float(np.nanmax(T))}


def step_ndvi(b5: np.ndarray, b4: np.ndarray):
    eps = 1e-6
    ndvi = (b5 - b4) / (b5 + b4 + eps)
    return ndvi, {"min": float(np.nanmin(ndvi)), "max": float(np.nanmax(ndvi))}


def step_pv(ndvi: np.ndarray):

    ndvi_valid = ndvi[np.isfinite(ndvi)]
    if ndvi_valid.size == 0:
        raise ValueError("NDVI has no valid pixels within the AOI.")

    ndvi_min = float(np.nanmin(ndvi_valid))
    ndvi_max = float(np.nanmax(ndvi_valid))

    denom = (ndvi_max - ndvi_min) if (ndvi_max - ndvi_min) != 0 else 1e-6
    pv = ((ndvi - ndvi_min) / denom) ** 2
    pv = np.clip(pv, 0, 1)
    return pv, {"min": float(np.nanmin(pv)), "max": float(np.nanmax(pv))}


def step_emissivity(pv: np.ndarray):
    emiss = 0.986 + 0.004 * pv
    return emiss, {"min": float(np.nanmin(emiss)), "max": float(np.nanmax(emiss))}


def step_lst_celsius(bt_kelvin: np.ndarray, emissivity: np.ndarray):
    """Corrected LST (°C) using NDVI-based emissivity.
    Uses lambda = 10.895e-6 m, rho = 1.438e-2 m*K
    """
    lambda_m = 10.895e-6
    rho = 1.438e-2
    eps = 1e-12
    lst_K = bt_kelvin / (1.0 + (lambda_m * bt_kelvin / rho) * np.log(np.clip(emissivity, eps, None)))
    lst_C = lst_K - 273.15
    return lst_C, {"min": float(np.nanmin(lst_C)), "max": float(np.nanmax(lst_C))}



if __name__ == "__main__":

    band10_path = input("Enter path to Band 10 (thermal): ").strip().strip('"')
    band5_path = input("Enter path to Band 5 (NIR): ").strip().strip('"')
    band4_path = input("Enter path to Band 4 (Red): ").strip().strip('"')
    aoi_path = input("Enter path to AOI shapefile/geojson: ").strip().strip('"')

    ML = float(input("Enter ML (Radiance Multiplier for Band 10): "))
    AL = float(input("Enter AL (Radiance Add for Band 10): "))
    K1 = float(input("Enter K1 constant for Band 10: "))
    K2 = float(input("Enter K2 constant for Band 10: "))


    aoi = read_aoi(aoi_path)
    b10, meta, stats_b10 = read_clip_band(band10_path, aoi)
    b5, _, stats_b5 = read_clip_band(band5_path, aoi)
    b4, _, stats_b4 = read_clip_band(band4_path, aoi)

    print(f"Band10 (clipped) min/max: {stats_b10}")
    print(f"Band5  (clipped) min/max: {stats_b5}")
    print(f"Band4  (clipped) min/max: {stats_b4}")


    toa_L, stats_toa = step_toa_radiance(b10, ML, AL)
    print(f"TOA radiance min/max: {stats_toa}")

    bt_K, stats_bt = step_bt_kelvin(toa_L, K1, K2)
    print(f"Brightness temp (K) min/max: {stats_bt}")

    ndvi, stats_ndvi = step_ndvi(b5, b4)
    print(f"NDVI min/max: {stats_ndvi}")

    pv, stats_pv = step_pv(ndvi)  # FULL-range scaling
    print(f"PV min/max: {stats_pv}")

    emiss, stats_em = step_emissivity(pv)
    print(f"Emissivity min/max: {stats_em}")

    lst_C, stats_lst = step_lst_celsius(bt_K, emiss)
    print(f"LST (°C) min/max: {stats_lst}")


    outdir = Path("outputs1")
    outdir.mkdir(parents=True, exist_ok=True)

    save_raster(str(outdir / "B10_clipped.tif"), b10, meta_as_float(meta))
    save_raster(str(outdir / "B5_clipped.tif"), b5, meta_as_float(meta))
    save_raster(str(outdir / "B4_clipped.tif"), b4, meta_as_float(meta))

    save_raster(str(outdir / "TOA_Radiance.tif"), toa_L, meta_as_float(meta))
    save_raster(str(outdir / "BT_Kelvin.tif"), bt_K, meta_as_float(meta))
    save_raster(str(outdir / "NDVI.tif"), ndvi, meta_as_float(meta))
    save_raster(str(outdir / "PV.tif"), pv, meta_as_float(meta))
    save_raster(str(outdir / "Emissivity.tif"), emiss, meta_as_float(meta))
    save_raster(str(outdir / "LST_Celsius.tif"), lst_C, meta_as_float(meta))

    print(f"Saved outputs to: {outdir.resolve()}")


    plt.figure(figsize=(8, 6))
    im = plt.imshow(lst_C, cmap="inferno")
    plt.title("Land Surface Temperature (°C)")
    plt.colorbar(im, label="°C")
    plt.axis("off")
    plt.show()
