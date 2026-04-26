from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from nuris_pipeline.preprocess.aoi import clip_raster_to_aoi


def test_clip_raster_to_aoi_masks_pixels(tmp_path: Path):
    raster_path = tmp_path / "scene.tif"
    out_path = tmp_path / "clipped.tif"

    data = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)
    transform = from_origin(0, 4, 1, 1)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data)

    aoi = gpd.GeoDataFrame({"id": [1]}, geometry=[box(1, 1, 3, 3)], crs="EPSG:3857")

    result = clip_raster_to_aoi(raster_path, aoi, out_path)

    assert result.width == 2
    assert result.height == 2

    with rasterio.open(out_path) as clipped:
        arr = clipped.read(1)
        assert arr.shape == (2, 2)
        assert arr.min() >= 0


def test_clip_raster_to_aoi_rejects_non_intersection(tmp_path: Path):
    raster_path = tmp_path / "scene.tif"
    out_path = tmp_path / "clipped.tif"

    data = np.ones((1, 4, 4), dtype=np.uint8)
    transform = from_origin(0, 4, 1, 1)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data)

    aoi = gpd.GeoDataFrame({"id": [1]}, geometry=[box(100, 100, 101, 101)], crs="EPSG:3857")

    with pytest.raises(ValueError):
        clip_raster_to_aoi(raster_path, aoi, out_path)
