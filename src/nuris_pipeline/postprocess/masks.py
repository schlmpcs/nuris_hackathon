from __future__ import annotations

import numpy as np
from skimage.morphology import binary_closing, binary_opening, disk


def threshold_probability_map(probability_map: np.ndarray, threshold: float, smooth_radius: int = 1) -> np.ndarray:
    mask = probability_map >= threshold
    if smooth_radius > 0:
        footprint = disk(smooth_radius)
        mask = binary_opening(mask, footprint)
        mask = binary_closing(mask, footprint)
    return mask.astype(np.uint8)
