from __future__ import annotations

import geopandas as gpd
import pandas as pd


def compute_detection_metrics(predicted: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, iou_threshold: float = 0.5) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for class_name in sorted(set(predicted["class"]).union(set(truth["class"]))):
        pred_class = predicted[predicted["class"] == class_name]
        truth_class = truth[truth["class"] == class_name]

        matches = 0
        used_truth: set[int] = set()
        for _, pred_row in pred_class.iterrows():
            best_iou = 0.0
            best_index = None
            for truth_index, truth_row in truth_class.iterrows():
                if truth_index in used_truth:
                    continue
                intersection = pred_row.geometry.intersection(truth_row.geometry).area
                union = pred_row.geometry.union(truth_row.geometry).area or 1.0
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_index = truth_index
            if best_iou >= iou_threshold and best_index is not None:
                matches += 1
                used_truth.add(best_index)

        precision = matches / len(pred_class) if len(pred_class) else 0.0
        recall = matches / len(truth_class) if len(truth_class) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_objects": matches,
                "predicted_objects": int(len(pred_class)),
                "truth_objects": int(len(truth_class)),
            }
        )

    return pd.DataFrame(rows)
