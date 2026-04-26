from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import yaml

from nuris_pipeline.config import load_training_config
from nuris_pipeline.training.metrics import summarize_confusion_matrix, update_confusion_matrix


def _write_patch_dataset(root: Path) -> Path:
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()

    manifest: list[dict[str, str]] = []
    split_specs = [
        ("train_patch_0", "train", 1),
        ("train_patch_1", "train", 2),
        ("validation_patch_0", "validation", 3),
        ("test_patch_0", "test", 1),
    ]
    for patch_id, split, label in split_specs:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = label * 40
        image[..., 1] = label * 30
        image[..., 2] = label * 20
        mask = np.full((32, 32), label, dtype=np.uint8)
        image_path = images_dir / f"{patch_id}.jpg"
        mask_path = masks_dir / f"{patch_id}_m.png"
        iio.imwrite(image_path, image)
        iio.imwrite(mask_path, mask)
        manifest.append(
            {
                "patch_id": patch_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "split": split,
            }
        )

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def test_load_training_config_reads_defaults(tmp_path: Path):
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "landcover_ai",
                    "manifest_path": "data/manifest.json",
                    "tile_size": 512,
                    "classes": {0: "background", 1: "building", 2: "water", 3: "road"},
                },
                "training": {
                    "model_name": "unet",
                    "batch_size": 2,
                    "num_workers": 0,
                    "learning_rate": 1e-4,
                    "epochs": 1,
                    "device": "cpu",
                },
                "export": {"checkpoint_dir": "outputs/checkpoints"},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_training_config(config_path)

    assert cfg.dataset.train_split == "train"
    assert cfg.dataset.validation_split == "validation"
    assert cfg.dataset.image_mean == (0.485, 0.456, 0.406)
    assert cfg.augmentation.color_jitter == 0.1
    assert cfg.training.seed == 42


def test_segmentation_metrics_report_per_class_iou():
    confusion = np.zeros((4, 4), dtype=np.int64)
    update_confusion_matrix(
        confusion,
        predictions=np.array([[0, 1], [1, 3]], dtype=np.int64),
        targets=np.array([[0, 1], [2, 3]], dtype=np.int64),
        num_classes=4,
    )

    summary = summarize_confusion_matrix(confusion, ["background", "building", "water", "road"])

    assert summary["pixel_accuracy"] == pytest.approx(0.75)
    assert summary["per_class_iou"]["background"] == pytest.approx(1.0)
    assert summary["per_class_iou"]["building"] == pytest.approx(0.5)
    assert summary["per_class_iou"]["water"] == pytest.approx(0.0)
    assert summary["per_class_iou"]["road"] == pytest.approx(1.0)


def test_landcover_patch_dataset_filters_split_and_normalizes(tmp_path: Path):
    torch = pytest.importorskip("torch")

    from nuris_pipeline.training.dataset import LandCoverPatchDataset

    manifest_path = _write_patch_dataset(tmp_path)
    dataset = LandCoverPatchDataset(
        manifest_path=manifest_path,
        split="train",
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.25, 0.25, 0.25),
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["patch_id"] == "train_patch_0"
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)


def test_train_segmentation_writes_checkpoints_and_metrics(tmp_path: Path):
    pytest.importorskip("torch")

    from nuris_pipeline.training.trainer import train_segmentation

    manifest_path = _write_patch_dataset(tmp_path / "patches")
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "landcover_ai",
                    "manifest_path": str(manifest_path),
                    "tile_size": 32,
                    "classes": {0: "background", 1: "building", 2: "water", 3: "road"},
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                },
                "training": {
                    "model_name": "unet",
                    "batch_size": 1,
                    "num_workers": 0,
                    "learning_rate": 1e-3,
                    "epochs": 1,
                    "device": "cpu",
                    "seed": 7,
                },
                "augmentation": {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                    "rotate_90": False,
                    "color_jitter": 0.0,
                },
                "export": {"checkpoint_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )

    exit_code = train_segmentation(config_path)

    assert exit_code == 0
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "final.pt").exists()
    assert (output_dir / "metrics.csv").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "history" in metrics
    assert "test" in metrics
    assert (output_dir / "training.yaml").exists()
