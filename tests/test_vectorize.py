import numpy as np
from affine import Affine

from nuris_pipeline.postprocess.vectorize import vectorize_buildings, vectorize_roads, vectorize_water


def test_vectorize_buildings_returns_polygon_features():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:4, 1:4] = 1

    gdf = vectorize_buildings(mask, Affine.translation(0, 6) * Affine.scale(1, -1), "scene_1", "tile_1")

    assert len(gdf) == 1
    assert gdf.iloc[0]["class"] == "building"
    assert gdf.iloc[0].geometry.geom_type == "Polygon"


def test_vectorize_water_returns_polygon_features():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:5, 2:5] = 1

    gdf = vectorize_water(mask, Affine.translation(0, 6) * Affine.scale(1, -1), "scene_1", "tile_1")

    assert len(gdf) == 1
    assert gdf.iloc[0]["class"] == "water"


def test_vectorize_roads_returns_line_features():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2, 1:5] = 1

    gdf = vectorize_roads(mask, Affine.translation(0, 6) * Affine.scale(1, -1), "scene_1", "tile_1")

    assert len(gdf) == 1
    assert gdf.iloc[0]["class"] == "road"
    assert gdf.iloc[0].geometry.geom_type in {"LineString", "MultiLineString"}
