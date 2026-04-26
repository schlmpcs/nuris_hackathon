from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path
import json


@dataclass(frozen=True)
class SceneManifest:
    source_id: str
    path: str
    crs: str
    width: int
    height: int
    resolution_x: float
    resolution_y: float


@dataclass(frozen=True)
class RunManifest:
    created_at: str
    config_path: str
    aoi_path: str
    zones_path: str | None
    scenes: list[SceneManifest] = field(default_factory=list)


def build_run_manifest(config_path: str, aoi_path: str, zones_path: str | None, scenes: list[SceneManifest]) -> RunManifest:
    return RunManifest(
        created_at=datetime.now(UTC).isoformat(),
        config_path=config_path,
        aoi_path=aoi_path,
        zones_path=zones_path,
        scenes=scenes,
    )


def write_manifest(manifest: RunManifest, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return output_path
