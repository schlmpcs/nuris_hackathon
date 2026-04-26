from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box


@dataclass(frozen=True)
class ClippedRaster:
    path: Path
    width: int
    height: int
    crs: str


def clip_raster_to_aoi(raster_path: str | Path, aoi: gpd.GeoDataFrame, out_path: str | Path) -> ClippedRaster:
    raster_path = Path(raster_path)
    out_path = Path(out_path)
    with rasterio.open(raster_path) as src:
        aoi_in_raster_crs = aoi.to_crs(src.crs)
        raster_extent = box(*src.bounds)
        if not aoi_in_raster_crs.geometry.intersects(raster_extent).any():
            raise ValueError("AOI does not intersect raster extent")

        clipped, transform = mask(src, aoi_in_raster_crs.geometry, crop=True)
        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)

    return ClippedRaster(path=out_path, width=clipped.shape[2], height=clipped.shape[1], crs=str(profile["crs"]))
