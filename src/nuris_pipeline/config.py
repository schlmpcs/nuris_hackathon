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
