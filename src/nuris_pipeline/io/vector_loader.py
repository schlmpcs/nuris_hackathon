from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def load_polygon_layer(path: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Vector layer is empty: {path}")
    if not gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"Expected polygon geometries in {path}")
    return gdf
