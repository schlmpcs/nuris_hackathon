from __future__ import annotations

from math import floor

import geopandas as gpd
from pyproj import CRS


def reproject_gdf(gdf: gpd.GeoDataFrame, target_crs: str | int | CRS) -> gpd.GeoDataFrame:
    return gdf.to_crs(target_crs)


def ensure_projected_crs(crs_like: str | int | CRS) -> CRS:
    crs = CRS.from_user_input(crs_like)
    if not crs.is_projected:
        raise ValueError(f"Projected CRS required, got {crs.to_string()}")
    return crs


def choose_working_crs(gdf: gpd.GeoDataFrame) -> CRS:
    lonlat = gdf.to_crs("EPSG:4326")
    centroid = lonlat.geometry.union_all().centroid
    lon = centroid.x
    lat = centroid.y
    zone = floor((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)
