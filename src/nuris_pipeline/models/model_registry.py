from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class TorchCheckpointSegmentationModel:
    model: object
    class_names: tuple[str, ...]
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    device: str

    def predict(self, tile_array: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        arr = tile_array.astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected CHW tile array, got shape {arr.shape}")
        if arr.max() > 1.0:
            arr = arr / 255.0

        rgb = arr[:3]
        normalized = (rgb - np.asarray(self.image_mean, dtype=np.float32)[:, None, None]) / np.asarray(
            self.image_std,
            dtype=np.float32,
        )[:, None, None]
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        output: dict[str, np.ndarray] = {}
        for index, class_name in enumerate(self.class_names):
            if class_name == "background":
                continue
            output[class_name] = probabilities[index]
        return output


def load_model(backend: str, checkpoint: str | None = None, device: str = "cpu") -> SegmentationModel:
    if backend == "heuristic":
        return HeuristicSegmentationModel()
    if backend == "torch_unet":
        if not checkpoint:
            raise ValueError("The 'torch_unet' backend requires a checkpoint path")

        import torch

        from nuris_pipeline.training.models import create_segmentation_model

        checkpoint_path = Path(checkpoint)
        payload = torch.load(checkpoint_path, map_location=device)
        model = create_segmentation_model(payload["model_name"], num_classes=payload["num_classes"])
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        class_names = tuple(payload["class_names"])
        return TorchCheckpointSegmentationModel(
            model=model,
            class_names=class_names,
            image_mean=tuple(payload["image_mean"]),
            image_std=tuple(payload["image_std"]),
            device=device,
        )

    raise ValueError(
        f"Unsupported backend '{backend}'. Supported backends: heuristic, torch_unet"
    )
