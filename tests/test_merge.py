import geopandas as gpd
from shapely.geometry import LineString, Polygon

from nuris_pipeline.postprocess.merge import merge_polygon_features, merge_road_features


def test_merge_polygon_features_dissolves_overlaps():
    gdf = gpd.GeoDataFrame(
        {"class": ["building", "building"], "confidence": [90, 80], "source_id": ["s1", "s1"]},
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]),
        ],
        crs="EPSG:3857",
    )

    merged = merge_polygon_features(gdf, "building")

    assert len(merged) == 1
    assert merged.iloc[0]["class"] == "building"


def test_merge_road_features_merges_connected_lines():
    gdf = gpd.GeoDataFrame(
        {"class": ["road", "road"], "confidence": [90, 90], "source_id": ["s1", "s1"]},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs="EPSG:3857",
    )

    merged = merge_road_features(gdf)

    assert len(merged) == 1
