from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputConfig:
    scene_paths: list[str]
    aoi_path: str
    zones_path: str | None


@dataclass(frozen=True)
class CrsConfig:
    working_crs_strategy: str
    export_epsg: int


@dataclass(frozen=True)
class TilingConfig:
    tile_size: int
    overlap: int


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    checkpoint: str | None
    device: str
    class_thresholds: dict[str, float]


@dataclass(frozen=True)
class FilteringConfig:
    building_min_area_m2: float
    road_min_length_m: float
    water_min_area_m2: float


@dataclass(frozen=True)
class ExportConfig:
    output_dir: str
    geojson_epsg: int
    write_geopackage: bool


@dataclass(frozen=True)
class QaConfig:
    sample_size: int
    confidence_bins: int


@dataclass(frozen=True)
class TrainingDatasetConfig:
    name: str
    manifest_path: str
    tile_size: int
    classes: dict[int, str]
    dataset_root: str | None = None
    patches_root: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainingRunConfig:
    model_name: str
    batch_size: int
    num_workers: int
    learning_rate: float
    epochs: int
    device: str
    encoder_weights: str | None = None
    seed: int = 42


@dataclass(frozen=True)
class TrainingAugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate_90: bool = True
    color_jitter: float = 0.1


@dataclass(frozen=True)
class TrainingExportConfig:
    checkpoint_dir: str


@dataclass(frozen=True)
class TrainingConfig:
    dataset: TrainingDatasetConfig
    training: TrainingRunConfig
    augmentation: TrainingAugmentationConfig
    export: TrainingExportConfig


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    crs: CrsConfig
    tiling: TilingConfig
    model: ModelConfig
    filtering: FilteringConfig
    export: ExportConfig
    qa: QaConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path) -> AppConfig:
    payload = _read_yaml(Path(path))

    return AppConfig(
        input=InputConfig(**payload["input"]),
        crs=CrsConfig(**payload["crs"]),
        tiling=TilingConfig(**payload["tiling"]),
        model=ModelConfig(**payload["model"]),
        filtering=FilteringConfig(**payload["filtering"]),
        export=ExportConfig(**payload["export"]),
        qa=QaConfig(**payload["qa"]),
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    payload = _read_yaml(Path(path))
    dataset_payload = payload["dataset"]
    augmentation_payload = payload.get("augmentation", {})

    return TrainingConfig(
        dataset=TrainingDatasetConfig(
            name=dataset_payload["name"],
            dataset_root=dataset_payload.get("dataset_root"),
            patches_root=dataset_payload.get("patches_root"),
            manifest_path=dataset_payload["manifest_path"],
            tile_size=dataset_payload["tile_size"],
            classes={int(key): value for key, value in dataset_payload["classes"].items()},
            train_split=dataset_payload.get("train_split", "train"),
            validation_split=dataset_payload.get("validation_split", "validation"),
            test_split=dataset_payload.get("test_split", "test"),
            image_mean=tuple(dataset_payload.get("image_mean", (0.485, 0.456, 0.406))),
            image_std=tuple(dataset_payload.get("image_std", (0.229, 0.224, 0.225))),
        ),
        training=TrainingRunConfig(**payload["training"]),
        augmentation=TrainingAugmentationConfig(**augmentation_payload),
        export=TrainingExportConfig(**payload["export"]),
    )
