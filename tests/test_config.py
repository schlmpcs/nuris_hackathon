from pathlib import Path

import pytest


from nuris_pipeline.config import load_config


def test_config_loads_defaults():
    cfg = load_config(Path("configs/base.yaml"))

    assert cfg.tiling.tile_size == 1024
    assert cfg.tiling.overlap == 128
    assert cfg.export.geojson_epsg == 4326
    assert cfg.crs.working_crs_strategy == "auto_utm"


def test_config_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("configs/missing.yaml"))
