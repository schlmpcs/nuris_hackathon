from __future__ import annotations

import geopandas as gpd
import pandas as pd


def summarize_by_zone(features: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _, zone in zones.iterrows():
        zone_geom = zone.geometry
        zone_features = features[features.geometry.intersects(zone_geom)].copy()

        building_features = zone_features[zone_features["class"] == "building"]
        road_features = zone_features[zone_features["class"] == "road"]
        water_features = zone_features[zone_features["class"] == "water"]

        rows.append(
            {
                "zone_id": zone["zone_id"],
                "zone_area_m2": zone_geom.area,
                "building_count": int(len(building_features)),
                "road_count": int(len(road_features)),
                "water_count": int(len(water_features)),
                "building_area_m2": float(building_features.geometry.area.sum()),
                "water_area_m2": float(water_features.geometry.area.sum()),
                "road_length_m": float(road_features.geometry.length.sum()),
                "building_density_per_km2": float(len(building_features) / (zone_geom.area / 1_000_000)),
                "road_density_per_km2": float(len(road_features) / (zone_geom.area / 1_000_000)),
                "water_density_per_km2": float(len(water_features) / (zone_geom.area / 1_000_000)),
            }
        )

    return pd.DataFrame(rows)
