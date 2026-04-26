from __future__ import annotations

from dataclasses import dataclass

from affine import Affine
from rasterio.windows import Window


@dataclass(frozen=True)
class Tile:
    tile_id: str
    row: int
    col: int
    window: Window
    core_window: Window
    transform: Affine


def _core_window(window: Window, width: int, height: int, tile_size: int, overlap: int) -> Window:
    left = overlap if window.col_off > 0 else 0
    top = overlap if window.row_off > 0 else 0
    right_trim = overlap if window.col_off + window.width < width else 0
    bottom_trim = overlap if window.row_off + window.height < height else 0
    return Window(
        col_off=window.col_off + left,
        row_off=window.row_off + top,
        width=window.width - left - right_trim,
        height=window.height - top - bottom_trim,
    )


def _starts(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]

    starts: list[int] = []
    current = 0
    while True:
        clamped = min(current, size - tile_size)
        if not starts or starts[-1] != clamped:
            starts.append(clamped)
        if clamped >= size - tile_size:
            break
        current += stride
    return starts


def generate_tiles(width: int, height: int, tile_size: int, overlap: int, transform: Affine) -> list[Tile]:
    stride = tile_size - overlap
    tiles: list[Tile] = []
    row_offsets = _starts(height, tile_size, stride)
    col_offsets = _starts(width, tile_size, stride)

    for row_index, row_off in enumerate(row_offsets):
        for col_index, col_off in enumerate(col_offsets):
            window = Window(
                col_off=col_off,
                row_off=row_off,
                width=min(tile_size, width - col_off),
                height=min(tile_size, height - row_off),
            )
            tile_transform = transform * Affine.translation(window.col_off, window.row_off)
            tiles.append(
                Tile(
                    tile_id=f"r{row_index}_c{col_index}",
                    row=row_index,
                    col=col_index,
                    window=window,
                    core_window=_core_window(window, width, height, tile_size, overlap),
                    transform=tile_transform,
                )
            )
    return tiles
