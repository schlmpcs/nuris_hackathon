from __future__ import annotations

import geopandas as gpd
import pandas as pd


def build_control_sample(features: gpd.GeoDataFrame, sample_size: int, confidence_bins: int) -> gpd.GeoDataFrame:
    if features.empty:
        return features.copy()

    sampled = features.copy()
    sampled["confidence_bin"] = pd.cut(sampled["confidence"], bins=confidence_bins, labels=False, include_lowest=True)
    grouped = sampled.groupby(["class", "confidence_bin"], group_keys=False)
    per_group = max(1, sample_size // max(1, grouped.ngroups))
    return grouped.head(per_group).reset_index(drop=True)
