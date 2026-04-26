from __future__ import annotations

from pathlib import Path

import geopandas as gpd


REQUIRED_COLUMNS = ["id", "class", "confidence", "source_id", "geometry"]


def write_geojson(features: gpd.GeoDataFrame, output_path: str | Path, epsg: int = 4326) -> Path:
    missing = [column for column in REQUIRED_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Missing required GeoJSON columns: {missing}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_gdf = features.to_crs(epsg=epsg)
    export_gdf.to_file(output_path, driver="GeoJSON")
    return output_path
