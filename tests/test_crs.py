from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from nuris_pipeline.preprocess.crs import choose_working_crs, ensure_projected_crs, reproject_gdf


def test_reproject_gdf_changes_crs():
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(71.43, 51.13)], crs="EPSG:4326")

    projected = reproject_gdf(gdf, "EPSG:32643")

    assert projected.crs.to_epsg() == 32643
    assert projected.geometry.iloc[0].x != pytest.approx(71.43)


def test_choose_working_crs_prefers_utm_for_lon_lat_geometry():
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(71.43, 51.13)], crs="EPSG:4326")

    crs = choose_working_crs(gdf)

    assert crs.to_epsg() == 32642


def test_ensure_projected_crs_rejects_geographic():
    with pytest.raises(ValueError):
        ensure_projected_crs("EPSG:4326")
