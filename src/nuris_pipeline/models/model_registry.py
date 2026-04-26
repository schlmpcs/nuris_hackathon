from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SegmentationModel(Protocol):
    class_names: tuple[str, ...]

    def predict(self, tile_array: np.ndarray) -> dict[str, np.ndarray]:
        ...


@dataclass
class HeuristicSegmentationModel:
    class_names: tuple[str, ...] = ("building", "road", "water")

    def predict(self, tile_array: np.ndarray) -> dict[str, np.ndarray]:
        arr = tile_array.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        rgb = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, axis=0)
        red, green, blue = rgb[0], rgb[1], rgb[2]
        nir = arr[3] if arr.shape[0] >= 4 else green

        brightness = np.clip((red + green + blue) / 3.0, 0.0, 1.0)
        ndvi = np.divide(nir - red, nir + red + 1e-6)
        ndwi = np.divide(green - nir, green + nir + 1e-6)
        grayness = 1.0 - (np.abs(red - green) + np.abs(green - blue) + np.abs(red - blue)) / 3.0

        water = np.clip((ndwi + 1.0) / 2.0, 0.0, 1.0)
        building = np.clip(0.6 * brightness + 0.4 * grayness - 0.3 * np.clip(ndvi, 0.0, 1.0), 0.0, 1.0)
        road = np.clip(0.7 * grayness + 0.3 * brightness - 0.2 * water, 0.0, 1.0)

        return {
            "building": building,
            "road": road,
            "water": water,
        }


def load_model(backend: str, checkpoint: str | None = None, device: str = "cpu") -> SegmentationModel:
    if backend == "heuristic":
        return HeuristicSegmentationModel()

    raise ValueError(
        f"Unsupported backend '{backend}'. v1 ships with the 'heuristic' baseline; "
        "pretrained checkpoints can be integrated later through this registry."
    )
