from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


@dataclass(frozen=True)
class RasterMetadata:
    path: Path
    width: int
    height: int
    count: int
    dtype: str
    crs: str | None
    transform: Affine
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    nodata: float | None


def read_raster_metadata(path: str | Path) -> RasterMetadata:
    path = Path(path)
    with rasterio.open(path) as src:
        return RasterMetadata(
            path=path,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=src.dtypes[0],
            crs=src.crs.to_string() if src.crs else None,
            transform=src.transform,
            bounds=src.bounds,
            resolution=src.res,
            nodata=src.nodata,
        )


def read_raster_window(path: str | Path, window: Window | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(window=window)
