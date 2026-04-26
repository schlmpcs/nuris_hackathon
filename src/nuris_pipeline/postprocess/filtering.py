from __future__ import annotations

import geopandas as gpd


def filter_features(
    gdf: gpd.GeoDataFrame,
    building_min_area_m2: float,
    road_min_length_m: float,
    water_min_area_m2: float,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    cleaned = gdf[gdf.geometry.notnull()].copy()
    cleaned["geometry"] = cleaned.geometry.make_valid()
    cleaned = cleaned[cleaned.geometry.is_valid]

    keep_mask = []
    for _, row in cleaned.iterrows():
        class_name = row["class"]
        geom = row.geometry
        if class_name == "building":
            keep_mask.append(geom.area >= building_min_area_m2)
        elif class_name == "water":
            keep_mask.append(geom.area >= water_min_area_m2)
        elif class_name == "road":
            keep_mask.append(geom.length >= road_min_length_m)
        else:
            keep_mask.append(True)

    return cleaned.loc[keep_mask].reset_index(drop=True)
