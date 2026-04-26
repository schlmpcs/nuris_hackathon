from __future__ import annotations

from dataclasses import asdict
import csv
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nuris_pipeline.config import TrainingConfig, load_training_config
from nuris_pipeline.training.dataset import LandCoverPatchDataset, SegmentationAugmentation
from nuris_pipeline.training.metrics import summarize_confusion_matrix, update_confusion_matrix
from nuris_pipeline.training.models import create_segmentation_model


LOGGER = logging.getLogger(__name__)


def train_segmentation(config_path: str | Path) -> int:
    distributed = initialize_distributed_training()
    try:
        config_path = Path(config_path)
        cfg = load_training_config(config_path)
        manifest_path = Path(cfg.dataset.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        set_training_seed(cfg.training.seed + distributed.rank)
        device = resolve_training_device(cfg.training.device, distributed)
        class_names = [cfg.dataset.classes[index] for index in sorted(cfg.dataset.classes)]
        num_classes = len(class_names)
        output_dir = Path(cfg.export.checkpoint_dir)
        if distributed.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, output_dir / config_path.name)
        if distributed.enabled:
            dist.barrier()

        train_dataset = LandCoverPatchDataset(
            manifest_path=manifest_path,
            split=cfg.dataset.train_split,
            image_mean=cfg.dataset.image_mean,
            image_std=cfg.dataset.image_std,
            augmentation=SegmentationAugmentation(
                horizontal_flip=cfg.augmentation.horizontal_flip,
                vertical_flip=cfg.augmentation.vertical_flip,
                rotate_90=cfg.augmentation.rotate_90,
                color_jitter=cfg.augmentation.color_jitter,
            ),
        )
        validation_dataset = LandCoverPatchDataset(
            manifest_path=manifest_path,
            split=cfg.dataset.validation_split,
            image_mean=cfg.dataset.image_mean,
            image_std=cfg.dataset.image_std,
        )
        test_dataset = LandCoverPatchDataset(
            manifest_path=manifest_path,
            split=cfg.dataset.test_split,
            image_mean=cfg.dataset.image_mean,
            image_std=cfg.dataset.image_std,
        )

        if not train_dataset:
            raise ValueError(f"No training samples found for split '{cfg.dataset.train_split}'")
        if not validation_dataset:
            raise ValueError(f"No validation samples found for split '{cfg.dataset.validation_split}'")
        if not test_dataset:
            raise ValueError(f"No test samples found for split '{cfg.dataset.test_split}'")

        LOGGER.info(
            "Loaded dataset splits from %s: train=%s validation=%s test=%s",
            manifest_path,
            len(train_dataset),
            len(validation_dataset),
            len(test_dataset),
        )

        train_sampler = create_sampler(train_dataset, distributed, shuffle=True)
        validation_sampler = create_sampler(validation_dataset, distributed, shuffle=False)
        test_sampler = create_sampler(test_dataset, distributed, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=device.type == "cuda",
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=device.type == "cuda",
        )

        base_model = create_segmentation_model(cfg.training.model_name, num_classes=num_classes).to(device)
        model = wrap_model_for_distributed_training(base_model, device, distributed)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_validation_iou = float("-inf")
        history: list[dict[str, Any]] = []

        for epoch in range(1, cfg.training.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_metrics = run_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
                training=True,
                distributed=distributed,
            )
            validation_metrics = run_epoch(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
                training=False,
                distributed=distributed,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mean_iou": train_metrics["mean_iou"],
                "train_pixel_accuracy": train_metrics["pixel_accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_mean_iou": validation_metrics["mean_iou"],
                "validation_pixel_accuracy": validation_metrics["pixel_accuracy"],
            }
            for class_name, score in train_metrics["per_class_iou"].items():
                row[f"train_iou_{class_name}"] = score
            for class_name, score in validation_metrics["per_class_iou"].items():
                row[f"validation_iou_{class_name}"] = score
            if distributed.is_main_process:
                history.append(row)

            LOGGER.info(
                "Epoch %s/%s train_loss=%.4f train_mIoU=%.4f val_loss=%.4f val_mIoU=%.4f",
                epoch,
                cfg.training.epochs,
                train_metrics["loss"],
                train_metrics["mean_iou"],
                validation_metrics["loss"],
                validation_metrics["mean_iou"],
            )

            if validation_metrics["mean_iou"] > best_validation_iou:
                best_validation_iou = validation_metrics["mean_iou"]
                if distributed.is_main_process:
                    save_checkpoint(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        epoch=epoch,
                        best_validation_iou=best_validation_iou,
                        class_names=class_names,
                    )

        test_metrics = run_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            training=False,
            distributed=distributed,
        )
        if distributed.is_main_process:
            save_checkpoint(
                output_dir / "final.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=cfg.training.epochs,
                best_validation_iou=best_validation_iou,
                class_names=class_names,
            )
            write_history_csv(output_dir / "metrics.csv", history)
            write_metrics_json(
                output_dir / "metrics.json",
                history=history,
                test_metrics=test_metrics,
                best_validation_iou=best_validation_iou,
                class_names=class_names,
            )
            LOGGER.info("Saved checkpoints and metrics to %s", output_dir)
        return 0
    finally:
        finalize_distributed_training(distributed)


def run_epoch(
    model: nn.Module | DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    training: bool,
    distributed: "DistributedTrainingContext",
) -> dict[str, Any]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total_batches = 0

    if training:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            update_confusion_matrix(
                confusion,
                predictions.detach().cpu().numpy(),
                masks.detach().cpu().numpy(),
                num_classes=num_classes,
            )
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

    if distributed.enabled:
        confusion = reduce_confusion_matrix(confusion, device)
        total_loss, total_batches = reduce_loss_totals(total_loss, total_batches, device)

    summary = summarize_confusion_matrix(confusion, class_names)
    summary["loss"] = total_loss / max(total_batches, 1)
    return summary


def save_checkpoint(
    path: Path,
    model: nn.Module | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    epoch: int,
    best_validation_iou: float,
    class_names: list[str],
) -> None:
    model_state = unwrap_model(model).state_dict()
    payload = {
        "epoch": epoch,
        "model_name": cfg.training.model_name,
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_mean": list(cfg.dataset.image_mean),
        "image_std": list(cfg.dataset.image_std),
        "tile_size": cfg.dataset.tile_size,
        "best_validation_mean_iou": best_validation_iou,
        "training_config": asdict(cfg),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_metrics_json(
    path: Path,
    history: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    best_validation_iou: float,
    class_names: list[str],
) -> None:
    payload = {
        "history": history,
        "test": test_metrics,
        "best_validation_mean_iou": best_validation_iou,
        "class_names": class_names,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_training_device(requested_device: str, distributed: "DistributedTrainingContext") -> torch.device:
    normalized = requested_device.lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if normalized.startswith("cuda") and distributed.enabled:
        if torch.cuda.device_count() <= distributed.local_rank:
            raise ValueError(
                f"LOCAL_RANK={distributed.local_rank} is unavailable on this machine; "
                f"visible CUDA devices={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(distributed.local_rank)
        return torch.device(f"cuda:{distributed.local_rank}")
    return torch.device(requested_device)


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DistributedTrainingContext:
    def __init__(self, enabled: bool, rank: int, world_size: int, local_rank: int) -> None:
        self.enabled = enabled
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed_training() -> DistributedTrainingContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        backend = resolve_distributed_backend()
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        if not master_addr or not master_port:
            raise ValueError("Distributed training requires MASTER_ADDR and MASTER_PORT to be set")
        configure_distributed_network_interface(backend)
        store_host = "0.0.0.0" if rank == 0 else master_addr
        store = distributed.TCPStore(
            host_name=store_host,
            port=int(master_port),
            world_size=world_size,
            is_master=rank == 0,
            timeout=timedelta(seconds=300),
            wait_for_workers=True,
            multi_tenant=False,
            use_libuv=False,
        )
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            store=store,
        )
    return DistributedTrainingContext(enabled=enabled, rank=rank, world_size=world_size, local_rank=local_rank)


def finalize_distributed_training(distributed: DistributedTrainingContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def resolve_distributed_backend() -> str:
    if platform.system() == "Windows":
        return "gloo"
    return "nccl" if torch.cuda.is_available() else "gloo"


def configure_distributed_network_interface(backend: str) -> None:
    if backend != "gloo":
        return
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        return
    if platform.system() == "Windows":
        os.environ["GLOO_SOCKET_IFNAME"] = "Ethernet"


def create_sampler(
    dataset: LandCoverPatchDataset,
    distributed: DistributedTrainingContext,
    shuffle: bool,
) -> DistributedSampler | None:
    if not distributed.enabled:
        return None
    return DistributedSampler(dataset, num_replicas=distributed.world_size, rank=distributed.rank, shuffle=shuffle)


def wrap_model_for_distributed_training(
    model: nn.Module,
    device: torch.device,
    distributed: DistributedTrainingContext,
) -> nn.Module | DistributedDataParallel:
    if not distributed.enabled:
        return model
    if device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
    return DistributedDataParallel(model)


def unwrap_model(model: nn.Module | DistributedDataParallel) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def reduce_confusion_matrix(confusion: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(confusion).to(device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


def reduce_loss_totals(total_loss: float, total_batches: int, device: torch.device) -> tuple[float, int]:
    totals = torch.tensor([total_loss, float(total_batches)], device=device, dtype=torch.float64)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item()), int(totals[1].item())
