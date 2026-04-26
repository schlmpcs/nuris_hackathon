from __future__ import annotations

import geopandas as gpd
from shapely.ops import linemerge, unary_union


def merge_polygon_features(gdf: gpd.GeoDataFrame, class_name: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    merged = unary_union(gdf.geometry.tolist())
    geometries = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return gpd.GeoDataFrame(
        {
            "class": [class_name] * len(geometries),
            "confidence": [float(gdf["confidence"].mean())] * len(geometries),
            "source_id": [",".join(sorted(set(gdf["source_id"].astype(str))))] * len(geometries),
        },
        geometry=geometries,
        crs=gdf.crs,
    )


def merge_road_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    merged = linemerge(unary_union(gdf.geometry.tolist()))
    geometries = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return gpd.GeoDataFrame(
        {
            "class": ["road"] * len(geometries),
            "confidence": [float(gdf["confidence"].mean())] * len(geometries),
            "source_id": [",".join(sorted(set(gdf["source_id"].astype(str))))] * len(geometries),
        },
        geometry=geometries,
        crs=gdf.crs,
    )
