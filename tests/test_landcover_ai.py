from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from nuris_pipeline.data.landcover_ai import (
    build_landcover_patch_manifest,
    discover_landcover_ai_samples,
    get_landcover_ai_download_spec,
    prepare_landcover_ai_patches,
    remap_landcover_mask,
)


def test_discover_landcover_ai_samples_reads_pairs_and_splits(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    (images_dir / "tile_a.tif").write_bytes(b"img-a")
    (images_dir / "tile_b.tif").write_bytes(b"img-b")
    (masks_dir / "tile_a.tif").write_bytes(b"mask-a")
    (masks_dir / "tile_b.tif").write_bytes(b"mask-b")
    (tmp_path / "train.txt").write_text("tile_a\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("tile_b\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("", encoding="utf-8")

    samples = discover_landcover_ai_samples(tmp_path)

    assert len(samples) == 2
    assert samples[0].image_path.name == "tile_a.tif"
    assert samples[0].mask_path.name == "tile_a.tif"
    assert samples[0].split == "train"
    assert samples[1].split == "validation"


def test_remap_landcover_mask_keeps_nuris_target_classes():
    mask = np.array(
        [
            [0, 1, 2],
            [3, 4, 2],
        ],
        dtype=np.uint8,
    )

    remapped = remap_landcover_mask(mask)

    assert remapped.tolist() == [
        [0, 1, 0],
        [2, 3, 0],
    ]


def test_landcover_ai_download_spec_points_to_expected_repo():
    spec = get_landcover_ai_download_spec()

    assert spec["repo_id"] == "dragon7/LandCover.ai"
    assert spec["repo_type"] == "dataset"
    assert "images/*" in spec["allow_patterns"]
    assert "masks/*" in spec["allow_patterns"]


def test_prepare_landcover_ai_patches_matches_upstream_naming(tmp_path: Path):
    dataset_root = tmp_path / "landcover_ai"
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()

    image = np.zeros((3, 1024, 1024), dtype=np.uint8)
    image[0, :, :] = 10
    image[1, :, :] = 20
    image[2, :, :] = 30
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[:512, :512] = 1
    mask[:512, 512:] = 3
    mask[512:, :512] = 4
    mask[512:, 512:] = 2

    transform = from_origin(0, 1024, 1, 1)
    with rasterio.open(
        images_dir / "scene_a.tif",
        "w",
        driver="GTiff",
        height=1024,
        width=1024,
        count=3,
        dtype=image.dtype,
        crs="EPSG:2180",
        transform=transform,
    ) as dst:
        dst.write(image)
    with rasterio.open(
        masks_dir / "scene_a.tif",
        "w",
        driver="GTiff",
        height=1024,
        width=1024,
        count=1,
        dtype=mask.dtype,
        crs="EPSG:2180",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    output_root = tmp_path / "prepared"
    patches = prepare_landcover_ai_patches(dataset_root, output_root)

    assert len(patches) == 4
    assert (output_root / "images" / "scene_a_0.jpg").exists()
    assert (output_root / "masks" / "scene_a_0_m.png").exists()
    assert patches[0].patch_id == "scene_a_0"
    assert patches[-1].patch_id == "scene_a_3"


def test_build_landcover_patch_manifest_assigns_splits(tmp_path: Path):
    patches_dir = tmp_path / "prepared"
    images_dir = patches_dir / "images"
    masks_dir = patches_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()

    for patch_id in ("scene_a_0", "scene_a_1", "scene_a_2"):
        (images_dir / f"{patch_id}.jpg").write_bytes(b"img")
        (masks_dir / f"{patch_id}_m.png").write_bytes(b"mask")

    (tmp_path / "train.txt").write_text("scene_a_0\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("scene_a_1\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("scene_a_2\n", encoding="utf-8")

    manifest = build_landcover_patch_manifest(tmp_path, patches_dir)

    assert len(manifest) == 3
    assert manifest[0]["split"] == "train"
    assert manifest[1]["split"] == "validation"
    assert manifest[2]["split"] == "test"


def test_prepare_landcover_ai_patches_matches_upstream_counter_gaps(tmp_path: Path):
    dataset_root = tmp_path / "landcover_ai"
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()

    image = np.zeros((3, 1024, 1300), dtype=np.uint8)
    mask = np.zeros((1024, 1300), dtype=np.uint8)
    transform = from_origin(0, 1024, 1, 1)
    with rasterio.open(
        images_dir / "scene_b.tif",
        "w",
        driver="GTiff",
        height=1024,
        width=1300,
        count=3,
        dtype=image.dtype,
        crs="EPSG:2180",
        transform=transform,
    ) as dst:
        dst.write(image)
    with rasterio.open(
        masks_dir / "scene_b.tif",
        "w",
        driver="GTiff",
        height=1024,
        width=1300,
        count=1,
        dtype=mask.dtype,
        crs="EPSG:2180",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    patches = prepare_landcover_ai_patches(dataset_root, tmp_path / "prepared")

    assert [patch.patch_id for patch in patches] == ["scene_b_0", "scene_b_1", "scene_b_3", "scene_b_4"]
