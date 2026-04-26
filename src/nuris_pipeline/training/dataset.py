from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset

from nuris_pipeline.training.manifest import load_training_manifest


@dataclass(frozen=True)
class PatchSample:
    patch_id: str
    image_path: Path
    mask_path: Path
    split: str


class SegmentationAugmentation:
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotate_90: bool = True,
        color_jitter: float = 0.1,
    ) -> None:
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate_90 = rotate_90
        self.color_jitter = color_jitter

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.horizontal_flip and random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if self.vertical_flip and random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        if self.rotate_90:
            k = random.randint(0, 3)
            if k:
                image = np.rot90(image, k=k, axes=(0, 1))
                mask = np.rot90(mask, k=k, axes=(0, 1))
        if self.color_jitter > 0:
            image = _apply_color_jitter(image, self.color_jitter)
        return image.copy(), mask.copy()


def _apply_color_jitter(image: np.ndarray, strength: float) -> np.ndarray:
    brightness = random.uniform(1.0 - strength, 1.0 + strength)
    contrast = random.uniform(1.0 - strength, 1.0 + strength)
    saturation = random.uniform(1.0 - strength, 1.0 + strength)

    jittered = np.clip(image * brightness, 0.0, 1.0)
    mean = jittered.mean(axis=(0, 1), keepdims=True)
    jittered = np.clip((jittered - mean) * contrast + mean, 0.0, 1.0)
    gray = jittered.mean(axis=2, keepdims=True)
    jittered = np.clip((jittered - gray) * saturation + gray, 0.0, 1.0)
    return jittered


class LandCoverPatchDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
        augmentation: SegmentationAugmentation | None = None,
    ) -> None:
        records = load_training_manifest(manifest_path)
        self.samples = [
            PatchSample(
                patch_id=record["patch_id"],
                image_path=Path(record["image_path"]),
                mask_path=Path(record["mask_path"]),
                split=record["split"],
            )
            for record in records
            if record["split"] == split
        ]
        self.image_mean = np.asarray(image_mean, dtype=np.float32)
        self.image_std = np.asarray(image_std, dtype=np.float32)
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = iio.imread(sample.image_path).astype(np.float32)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Expected RGB image for patch {sample.patch_id}, got shape {image.shape}")
        image = image[..., :3] / 255.0
        mask = iio.imread(sample.mask_path).astype(np.int64)
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        image = (image - self.image_mean) / self.image_std
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).to(torch.float32)
        mask_tensor = torch.from_numpy(mask).to(torch.long)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patch_id": sample.patch_id,
        }
