"""
Script: Train face recognition model.

Đây là entry point chính cho training. Script này:
1. Load config (với _base_ inheritance)
2. Build datasets + dataloaders
3. Build model + loss + optimizer + scheduler
4. Initialize Trainer + run

Usage:
    python scripts/train.py --config configs/train/exp_001_baseline.yaml

    # Resume từ checkpoint:
    python scripts/train.py --config configs/train/exp_001_baseline.yaml \\
        --resume experiments/exp_001/checkpoints/last.pth

    # Override experiment name:
    python scripts/train.py --config configs/train/exp_001_baseline.yaml \\
        --experiment-name exp_001_test
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Cho phép import từ src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FaceDataset, VerificationDataset, build_transforms
from src.data.mask_augment import MaskAugmenter
from src.losses.arcface_loss import ArcFaceLoss
from src.models.face_recognizer import build_model
from src.training.trainer import Trainer, build_optimizer, build_scheduler
from src.utils.config import load_config, save_config
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path tới training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path tới checkpoint để resume training",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment folder name (default từ config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override device (default: auto-detect)",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster training, slight non-determinism
        torch.backends.cudnn.benchmark = True


def build_dataloaders(config: dict) -> Tuple:
    """Build train + val dataloaders từ config."""
    import torch
    from torch.utils.data import DataLoader

    dataset_cfg = config["dataset"]
    aug_cfg = config.get("augmentation", {})
    train_cfg = config["training"]

    data_dir = Path(dataset_cfg["processed_dir"])
    splits_dir = Path(dataset_cfg["splits_dir"])

    # Mask augmenter (optional)
    mask_augmenter = None
    syn_mask_cfg = aug_cfg.get("training", {}).get("synthetic_mask", {})
    if syn_mask_cfg.get("probability", 0) > 0:
        mask_augmenter = MaskAugmenter(
            probability=syn_mask_cfg["probability"],
        )
        logger.info(
            f"Enabled synthetic mask augmentation (p={syn_mask_cfg['probability']})"
        )

    # Build train dataset
    train_transform = build_transforms(aug_cfg, is_training=True)
    train_dataset = FaceDataset(
        data_dir=data_dir,
        split_file=splits_dir / "train.txt",
        transform=train_transform,
        mask_augmenter=mask_augmenter,
    )

    # Build val dataset (verification pairs)
    val_transform = build_transforms(aug_cfg, is_training=False)
    val_dataset = VerificationDataset(
        data_dir=data_dir,
        pairs_file=splits_dir / "val_pairs.txt",
        transform=val_transform,
    )

    # DataLoaders
    batch_size = train_cfg.get("batch_size", 128)
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Tránh batch cuối size khác
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset.num_classes


def main() -> int:
    args = parse_args()

    # === Load config ===
    config = load_config(args.config)

    # === Determine experiment dir ===
    experiment_name = args.experiment_name or config.get(
        "experiment_name", "default_experiment"
    )
    experiment_dir = PROJECT_ROOT / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # === Setup logger (log vào file của experiment) ===
    global logger
    logger = setup_logger(
        name="train",
        log_file=experiment_dir / "train.log",
        level=logging.INFO,
    )

    # === Save resolved config (cho reproducibility) ===
    save_config(config, experiment_dir / "config_resolved.yaml")

    # === Set seed ===
    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", False)
    set_seed(seed, deterministic=deterministic)
    logger.info(f"Set seed={seed}, deterministic={deterministic}")

    # === Determine device ===
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # === Build dataloaders ===
    logger.info("Building dataloaders...")
    train_loader, val_loader, num_classes = build_dataloaders(config)
    logger.info(
        f"Train: {len(train_loader.dataset)} samples ({num_classes} classes), "
        f"Val: {len(val_loader.dataset)} pairs"
    )

    # === Build model ===
    logger.info("Building model...")
    model = build_model(config, num_classes=num_classes)

    # === Build loss ===
    loss_cfg = config.get("training", {}).get("loss", {})
    criterion = ArcFaceLoss(
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )
    logger.info(
        f"Loss: ArcFace (label_smoothing={loss_cfg.get('label_smoothing', 0.0)})"
    )

    # === Build optimizer + scheduler ===
    train_cfg = config["training"]
    optimizer = build_optimizer(model, train_cfg)
    logger.info(f"Optimizer: {type(optimizer).__name__}")

    num_epochs = train_cfg.get("num_epochs", 30)
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs=num_epochs)
    if scheduler:
        logger.info(f"Scheduler: {type(scheduler).__name__}")

    # === Init Trainer ===
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_dir=experiment_dir,
        config=config,
    )

    # === Resume từ checkpoint nếu có ===
    if args.resume:
        if not args.resume.exists():
            logger.error(f"❌ Resume checkpoint không tồn tại: {args.resume}")
            return 1
        trainer.load_checkpoint(args.resume)

    # === Train! ===
    logger.info(f"Starting training: {num_epochs} epochs")
    try:
        trainer.fit(num_epochs=num_epochs)
    except KeyboardInterrupt:
        logger.warning("[WARNING] Training bị ngắt bởi user (Ctrl+C)")
        logger.info("Last checkpoint đã được save trước khi dừng.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}", exc_info=True)
        return 1

    logger.info(f"[SUCCESS] Training hoàn tất. Checkpoints tại: {experiment_dir}/checkpoints/")
    return 0


if __name__ == "__main__":
    sys.exit(main())