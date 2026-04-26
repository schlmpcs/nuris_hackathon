from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import rasterio


LANDCOVER_AI_CRS = "EPSG:2180"
LANDCOVER_AI_LABELS = {
    0: "background",
    1: "building",
    2: "woodland",
    3: "water",
    4: "road",
}
NURIS_TARGET_LABELS = {
    0: "background",
    1: "building",
    2: "water",
    3: "road",
}
LANDCOVER_TO_NURIS = {
    0: 0,
    1: 1,
    2: 0,
    3: 2,
    4: 3,
}


@dataclass(frozen=True)
class LandCoverAiSample:
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str


@dataclass(frozen=True)
class LandCoverAiPatch:
    patch_id: str
    image_path: Path
    mask_path: Path
    source_scene_id: str


def _read_split_names(root: Path, filename: str, split_name: str) -> dict[str, str]:
    path = root / filename
    if not path.exists():
        return {}
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {name: split_name for name in names}


def discover_landcover_ai_samples(dataset_root: str | Path) -> list[LandCoverAiSample]:
    root = Path(dataset_root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("LandCover.ai expects 'images/' and 'masks/' directories")

    split_lookup = {}
    split_lookup.update(_read_split_names(root, "train.txt", "train"))
    split_lookup.update(_read_split_names(root, "val.txt", "validation"))
    split_lookup.update(_read_split_names(root, "test.txt", "test"))

    samples: list[LandCoverAiSample] = []
    for image_path in sorted(images_dir.glob("*.tif")):
        sample_id = image_path.stem
        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image {image_path.name}")
        samples.append(
            LandCoverAiSample(
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_path,
                split=split_lookup.get(sample_id, "unspecified"),
            )
        )
    return samples


def remap_landcover_mask(mask: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for source_value, target_value in LANDCOVER_TO_NURIS.items():
        remapped[mask == source_value] = target_value
    return remapped


def get_landcover_ai_download_spec() -> dict[str, object]:
    return {
        "repo_id": "dragon7/LandCover.ai",
        "repo_type": "dataset",
        "allow_patterns": [
            "images/*",
            "masks/*",
            "train.txt",
            "val.txt",
            "test.txt",
            "README.md",
            "split.py",
        ],
    }


def prepare_landcover_ai_patches(
    dataset_root: str | Path,
    output_root: str | Path,
    tile_size: int = 512,
) -> list[LandCoverAiPatch]:
    samples = discover_landcover_ai_samples(dataset_root)
    output_root = Path(output_root)
    images_out = output_root / "images"
    masks_out = output_root / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    patches: list[LandCoverAiPatch] = []
    for sample in samples:
        with rasterio.open(sample.image_path) as src:
            image = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        with rasterio.open(sample.mask_path) as src:
            mask = src.read(1)

        patch_index = 0
        for y in range(0, image.shape[0], tile_size):
            for x in range(0, image.shape[1], tile_size):
                image_tile = image[y : y + tile_size, x : x + tile_size]
                mask_tile = mask[y : y + tile_size, x : x + tile_size]
                patch_id = f"{sample.sample_id}_{patch_index}"
                if image_tile.shape[0] != tile_size or image_tile.shape[1] != tile_size:
                    patch_index += 1
                    continue

                image_path = images_out / f"{patch_id}.jpg"
                mask_path = masks_out / f"{patch_id}_m.png"
                iio.imwrite(image_path, image_tile)
                iio.imwrite(mask_path, remap_landcover_mask(mask_tile))
                patches.append(
                    LandCoverAiPatch(
                        patch_id=patch_id,
                        image_path=image_path,
                        mask_path=mask_path,
                        source_scene_id=sample.sample_id,
                    )
                )
                patch_index += 1

    return patches


def build_landcover_patch_manifest(dataset_root: str | Path, patches_root: str | Path) -> list[dict[str, str]]:
    dataset_root = Path(dataset_root)
    patches_root = Path(patches_root)
    split_lookup = {}
    split_lookup.update(_read_split_names(dataset_root, "train.txt", "train"))
    split_lookup.update(_read_split_names(dataset_root, "val.txt", "validation"))
    split_lookup.update(_read_split_names(dataset_root, "test.txt", "test"))

    manifest: list[dict[str, str]] = []
    for image_path in sorted((patches_root / "images").glob("*.jpg")):
        patch_id = image_path.stem
        mask_path = patches_root / "masks" / f"{patch_id}_m.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask patch for {patch_id}")
        manifest.append(
            {
                "patch_id": patch_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "split": split_lookup.get(patch_id, "unspecified"),
            }
        )
    return manifest


def write_landcover_patch_manifest(
    dataset_root: str | Path,
    patches_root: str | Path,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_landcover_patch_manifest(dataset_root, patches_root)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def write_landcover_ai_manifest(samples: list[LandCoverAiSample], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            **asdict(sample),
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }
        for sample in samples
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
