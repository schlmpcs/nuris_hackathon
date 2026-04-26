from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nuris_pipeline.models.model_registry import SegmentationModel


@dataclass(frozen=True)
class InferenceResult:
    source_id: str
    tile_id: str
    probability_maps: dict[str, np.ndarray]


def run_model_inference(model: SegmentationModel, tile_array: np.ndarray, source_id: str, tile_id: str) -> InferenceResult:
    probability_maps = model.predict(tile_array)
    return InferenceResult(source_id=source_id, tile_id=tile_id, probability_maps=probability_maps)
