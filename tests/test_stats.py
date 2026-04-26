import geopandas as gpd
from shapely.geometry import LineString, Polygon

from nuris_pipeline.export.stats_writer import summarize_by_zone


def test_summarize_by_zone_computes_counts_and_metrics():
    zones = gpd.GeoDataFrame(
        {"zone_id": ["zone_a"]},
        geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])],
        crs="EPSG:3857",
    )
    features = gpd.GeoDataFrame(
        {
            "class": ["building", "road", "water"],
            "confidence": [90, 80, 70],
            "source_id": ["s1", "s1", "s1"],
        },
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            LineString([(0, 0), (20, 0)]),
            Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
        ],
        crs="EPSG:3857",
    )

    summary = summarize_by_zone(features, zones)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["zone_id"] == "zone_a"
    assert row["building_count"] == 1
    assert row["road_count"] == 1
    assert row["water_count"] == 1
    assert row["building_area_m2"] == 100.0
    assert row["road_length_m"] == 20.0
