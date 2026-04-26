import geopandas as gpd
from shapely.geometry import LineString, Polygon

from nuris_pipeline.postprocess.filtering import filter_features


def test_filter_features_removes_tiny_buildings_and_short_roads():
    gdf = gpd.GeoDataFrame(
        {
            "class": ["building", "building", "road", "road", "water"],
            "confidence": [90, 90, 90, 90, 90],
            "source_id": ["s1"] * 5,
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
            LineString([(0, 0), (2, 0)]),
            LineString([(0, 0), (20, 0)]),
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ],
        crs="EPSG:3857",
    )

    filtered = filter_features(
        gdf,
        building_min_area_m2=12.0,
        road_min_length_m=10.0,
        water_min_area_m2=25.0,
    )

    assert len(filtered) == 3
    assert set(filtered["class"]) == {"building", "road", "water"}
