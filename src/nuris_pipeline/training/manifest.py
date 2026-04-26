from __future__ import annotations

from pathlib import Path
import json


def load_training_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
