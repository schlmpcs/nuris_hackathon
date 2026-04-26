from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import LineString, shape
from skimage.morphology import skeletonize


def _polygonize_mask(mask: np.ndarray, transform: Affine, class_name: str, source_id: str, tile_id: str) -> gpd.GeoDataFrame:
    records = []
    for geometry, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if value != 1:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        records.append(
            {
                "class": class_name,
                "confidence": 100,
                "source_id": source_id,
                "tile_id": tile_id,
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:3857" if records else None)


def vectorize_buildings(mask: np.ndarray, transform: Affine, source_id: str, tile_id: str) -> gpd.GeoDataFrame:
    return _polygonize_mask(mask, transform, "building", source_id, tile_id)


def vectorize_water(mask: np.ndarray, transform: Affine, source_id: str, tile_id: str) -> gpd.GeoDataFrame:
    return _polygonize_mask(mask, transform, "water", source_id, tile_id)


def vectorize_roads(mask: np.ndarray, transform: Affine, source_id: str, tile_id: str) -> gpd.GeoDataFrame:
    skeleton = skeletonize(mask.astype(bool))
    points_by_row: dict[int, list[tuple[float, float]]] = defaultdict(list)
    rows, cols = np.where(skeleton)
    for row, col in zip(rows.tolist(), cols.tolist()):
        x, y = transform * (col + 0.5, row + 0.5)
        points_by_row[row].append((x, y))

    lines = []
    for row_points in points_by_row.values():
        row_points.sort(key=lambda pt: pt[0])
        if len(row_points) >= 2:
            lines.append(LineString(row_points))

    records = [
        {
            "class": "road",
            "confidence": 100,
            "source_id": source_id,
            "tile_id": tile_id,
            "geometry": line,
        }
        for line in lines
    ]

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:3857" if records else None)
