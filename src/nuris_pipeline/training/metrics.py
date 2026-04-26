from __future__ import annotations

import numpy as np


def update_confusion_matrix(
    confusion: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    flat_predictions = predictions.reshape(-1)
    flat_targets = targets.reshape(-1)
    valid = (flat_targets >= 0) & (flat_targets < num_classes)
    encoded = num_classes * flat_targets[valid] + flat_predictions[valid]
    confusion += np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confusion


def summarize_confusion_matrix(confusion: np.ndarray, class_names: list[str]) -> dict[str, object]:
    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    union = true_positive + false_positive + false_negative

    per_class_iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=union > 0,
    )
    pixel_accuracy = float(true_positive.sum() / confusion.sum()) if confusion.sum() else 0.0
    mean_iou = float(np.nanmean(per_class_iou)) if not np.isnan(per_class_iou).all() else 0.0

    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou": {
            class_name: (None if np.isnan(score) else float(score))
            for class_name, score in zip(class_names, per_class_iou, strict=True)
        },
    }
