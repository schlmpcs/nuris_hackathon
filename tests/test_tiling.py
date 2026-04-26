from affine import Affine

from nuris_pipeline.preprocess.tiling import generate_tiles


def test_generate_tiles_builds_overlap_and_core_windows():
    tiles = generate_tiles(width=2048, height=1536, tile_size=1024, overlap=128, transform=Affine.identity())

    assert len(tiles) == 6
    assert tiles[0].window.width == 1024
    assert tiles[0].core_window.width == 896
    assert tiles[-1].window.width <= 1024
    assert tiles[-1].window.height <= 1024
