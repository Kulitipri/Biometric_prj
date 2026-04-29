"""
Trainer cho Face Recognition.

Module này là "orchestrator" kết nối tất cả components:
- Model (FaceRecognizer)
- Loss (ArcFaceLoss)
- DataLoaders (train/val)
- Metrics (verification)
- EmbeddingExtractor

Features:
- 2-stage training: freeze backbone → full fine-tune
- Mixed precision (fp16) với GradScaler
- Linear warmup + cosine annealing LR
- TensorBoard logging
- Best/last checkpoint management
- Early stopping
- Resume from checkpoint

Usage:
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cuda',
        experiment_dir='experiments/exp_001',
        config=config,
    )
    trainer.fit(num_epochs=30)
"""

import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.inference.embedding_extractor import EmbeddingExtractor
from src.metrics.verification import evaluate_verification

logger = logging.getLogger(__name__)


# ====================================================================== #
# Helper classes
# ====================================================================== #


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup + cosine annealing LR scheduler.

    Schedule:
        - Epoch [0, warmup_epochs): LR tăng tuyến tính từ ~0 đến base_lr
        - Epoch [warmup_epochs, total_epochs): LR giảm theo cosine
                                              từ base_lr xuống min_lr

    Tại sao cần warmup? Với ArcFace, loss có thể explode ở đầu training
    do margin penalty + scale lớn (s=64). Warmup giúp model "ổn định" trước.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Số epoch warmup tuyến tính.
        total_epochs: Tổng số epoch (warmup + cosine).
        min_lr: LR cuối cùng sau cosine annealing.
        last_epoch: Index epoch bắt đầu (cho resume).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        if warmup_epochs >= total_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) phải < total_epochs ({total_epochs})"
            )
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup từ 0 → base_lr
            # +1 trong numerator để epoch=0 không cho LR=0
            scale = (epoch + 1) / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing từ base_lr → min_lr
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class EarlyStopping:
    """
    Track validation metric và signal stop khi không cải thiện.

    Args:
        patience: Số epoch không cải thiện trước khi stop.
        min_delta: Mức cải thiện tối thiểu để được tính là "tốt hơn".
        mode: 'max' (e.g. accuracy) hoặc 'min' (e.g. loss).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode phải là 'max' hoặc 'min', nhận {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Update tracker với value mới.

        Returns:
            True nếu nên stop training.
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        # Check cải thiện theo mode
        if self.mode == "max":
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ====================================================================== #
# Builders cho optimizer và scheduler
# ====================================================================== #


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    Build optimizer từ config.

    Hỗ trợ: sgd, adamw.
    """
    opt_cfg = config.get("optimizer", {})
    opt_type = opt_cfg.get("type", "sgd").lower()
    lr = opt_cfg.get("lr", 0.1)
    weight_decay = opt_cfg.get("weight_decay", 5e-4)

    # Chỉ optimize parameters có requires_grad=True
    # (quan trọng khi backbone đang freeze)
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_type == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=opt_cfg.get("nesterov", True),
        )
    elif opt_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_cfg.get("betas", (0.9, 0.999)),
        )
    else:
        raise ValueError(f"Optimizer không hỗ trợ: {opt_type}")


def build_scheduler(
    optimizer: Optimizer, config: Dict[str, Any], total_epochs: int
) -> Optional[_LRScheduler]:
    """
    Build LR scheduler từ config.

    Hỗ trợ: cosine_warmup, cosine, step, none.
    """
    sched_cfg = config.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine_warmup").lower()

    if sched_type == "cosine_warmup":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=sched_cfg.get("warmup_epochs", 3),
            total_epochs=total_epochs,
            min_lr=sched_cfg.get("min_lr", 1e-5),
        )
    elif sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=sched_cfg.get("min_lr", 1e-5),
        )
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif sched_type in ("none", "null"):
        return None
    else:
        raise ValueError(f"Scheduler không hỗ trợ: {sched_type}")


# ====================================================================== #
# Main: Trainer class
# ====================================================================== #


class Trainer:
    """
    Trainer cho Face Recognition model.

    Args:
        model: FaceRecognizer instance.
        train_loader: DataLoader cho training (yields (images, labels)).
        val_loader: DataLoader cho validation (VerificationDataset).
        criterion: Loss function (e.g. ArcFaceLoss).
        optimizer: PyTorch optimizer.
        scheduler: LR scheduler (None = không scheduler).
        device: 'cuda' / 'cpu'.
        experiment_dir: Thư mục lưu checkpoints + logs.
        config: Full config dict (cần các keys: training, checkpoint,
                early_stopping, logging).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        device: str,
        experiment_dir: Path,
        config: Dict[str, Any],
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_dir = Path(experiment_dir)
        self.config = config

        # Parse config
        self.train_cfg = config.get("training", {})
        self.ckpt_cfg = config.get("checkpoint", {})
        self.es_cfg = config.get("early_stopping", {})
        self.log_cfg = config.get("logging", {})

        # Stages (cho 2-stage training: freeze head_only → full_finetune)
        self.stages: List[Dict] = self.train_cfg.get("stages", [])
        self._current_stage_idx = -1

        # Mixed precision
        self.use_amp = self.train_cfg.get("mixed_precision", False) and (
            self.device.type == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision training (fp16) enabled")

        # Checkpointing
        self.ckpt_dir = self.experiment_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_metric = self.ckpt_cfg.get(
            "monitor_metric", "val/tar_at_far_1e-3"
        )
        self.monitor_mode = self.ckpt_cfg.get("monitor_mode", "max")
        self.best_metric: float = (
            -float("inf") if self.monitor_mode == "max" else float("inf")
        )

        # Early stopping
        self.early_stopping: Optional[EarlyStopping] = None
        if self.es_cfg.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=self.es_cfg.get("patience", 10),
                min_delta=self.es_cfg.get("min_delta", 0.0),
                mode=self.monitor_mode,
            )

        # TensorBoard
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # State
        self.current_epoch = 0
        self.global_step = 0

        # EmbeddingExtractor cho validation (lazy init)
        self._extractor: Optional[EmbeddingExtractor] = None

        logger.info(
            f"Trainer initialized: device={device}, "
            f"experiment_dir={self.experiment_dir}, "
            f"monitor={self.monitor_metric} ({self.monitor_mode})"
        )

    # ------------------------------------------------------------------ #
    # Stage management (cho 2-stage training)
    # ------------------------------------------------------------------ #

    def _switch_stage(self, stage_idx: int) -> None:
        """
        Chuyển sang stage mới: freeze/unfreeze backbone, scale LR.

        Stage config format:
            {
                'name': 'head_only',
                'epochs': 3,                  # số epoch cho stage này
                'freeze_backbone': True,
                'lr_multiplier': 1.0,         # nhân với base_lr
            }
        """
        if stage_idx >= len(self.stages):
            return

        stage = self.stages[stage_idx]
        self._current_stage_idx = stage_idx

        logger.info(f"=== Bắt đầu stage {stage_idx + 1}: '{stage.get('name', '?')}' ===")

        # Freeze/unfreeze backbone
        if stage.get("freeze_backbone", False):
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()
        else:
            if hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()

        # Scale LR (nếu có)
        lr_mult = stage.get("lr_multiplier", 1.0)
        if lr_mult != 1.0:
            for param_group in self.optimizer.param_groups:
                # initial_lr được set bởi LRScheduler khi init
                base_lr = param_group.get("initial_lr", param_group["lr"])
                param_group["lr"] = base_lr * lr_mult
            logger.info(f"  LR scaled by {lr_mult}")

    def _get_current_stage(self, epoch: int) -> int:
        """Tìm stage hiện tại dựa trên epoch."""
        if not self.stages:
            return -1

        cumulative = 0
        for idx, stage in enumerate(self.stages):
            cumulative += stage.get("epochs", 0)
            if epoch < cumulative:
                return idx
        # Nếu epoch vượt total stages, ở stage cuối
        return len(self.stages) - 1

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train_one_epoch(self) -> Dict[str, float]:
        """
        Train 1 epoch.

        Returns:
            Dict metrics: {'loss': avg_loss, 'lr': current_lr, ...}
        """
        self.model.train()

        running_loss = 0.0
        n_samples = 0
        n_correct = 0
        log_every = self.log_cfg.get("log_every_n_steps", 50)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [train]",
            leave=False,
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward + loss với mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    embeddings, logits = self.model(images, labels=labels)
                    loss = self.criterion(logits, labels)
                # Backward với GradScaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings, logits = self.model(images, labels=labels)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            # Track metrics
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            # Train accuracy approximation từ logits (chỉ tham khảo)
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                n_correct += (preds == labels).sum().item()

            self.global_step += 1

            # Log mỗi N steps
            if (batch_idx + 1) % log_every == 0:
                avg_loss = running_loss / n_samples
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", current_lr, self.global_step)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

        # Epoch summary
        avg_loss = running_loss / n_samples if n_samples > 0 else 0.0
        train_acc = n_correct / n_samples if n_samples > 0 else 0.0
        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "loss": avg_loss,
            "train_acc": train_acc,
            "lr": current_lr,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate trên val_loader (VerificationDataset).

        Returns:
            Dict metrics từ evaluate_verification + prefix 'val/'.
        """
        self.model.eval()

        # Lazy init extractor (sau khi model đã trên đúng device)
        if self._extractor is None:
            self._extractor = EmbeddingExtractor(self.model, device=str(self.device))

        # Extract embeddings cho tất cả pairs
        emb1, emb2, labels = self._extractor.extract_pairs(
            self.val_loader,
            show_progress=True,
            desc=f"Epoch {self.current_epoch + 1} [val]",
        )

        # Tính metrics
        metrics = evaluate_verification(emb1, emb2, labels)

        # Prefix 'val/' cho TensorBoard
        return {f"val/{k}": v for k, v in metrics.items()}

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(
        self, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """
        Save checkpoint. Luôn save 'last.pth', optional save 'best.pth'.

        Save mỗi N epochs (config: save_every_n_epochs) với tên 'epoch_N.pth'.
        Tự cleanup giữ lại N checkpoint gần nhất (config: keep_last_n).
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.scaler else None
            ),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "config": self.config,
        }

        # Save 'last.pth' mỗi epoch (overwrite)
        last_path = self.ckpt_dir / "last.pth"
        torch.save(checkpoint, last_path)

        # Save 'best.pth' nếu best
        if is_best:
            best_path = self.ckpt_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(
                f"  ✓ Best checkpoint saved (epoch {self.current_epoch + 1}, "
                f"{self.monitor_metric}={self.best_metric:.4f})"
            )

        # Save periodic checkpoint
        save_every = self.ckpt_cfg.get("save_every_n_epochs", 0)
        if save_every > 0 and (self.current_epoch + 1) % save_every == 0:
            epoch_path = self.ckpt_dir / f"epoch_{self.current_epoch + 1}.pth"
            shutil.copy2(last_path, epoch_path)
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Giữ chỉ N epoch checkpoints gần nhất."""
        keep_last_n = self.ckpt_cfg.get("keep_last_n", 3)
        if keep_last_n <= 0:
            return

        # Tìm tất cả epoch_*.pth
        epoch_ckpts = sorted(
            self.ckpt_dir.glob("epoch_*.pth"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        # Xóa các checkpoint cũ
        if len(epoch_ckpts) > keep_last_n:
            for old in epoch_ckpts[:-keep_last_n]:
                old.unlink()
                logger.debug(f"Removed old checkpoint: {old.name}")

    def load_checkpoint(self, path: Path) -> None:
        """
        Load checkpoint để resume training.

        Restore: model, optimizer, scheduler, scaler, epoch, best_metric.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {path}")

        logger.info(f"Loading checkpoint: {path}")
        # weights_only=False vì checkpoint có config dict
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0) + 1  # epoch tiếp theo
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", self.best_metric)

        logger.info(
            f"Resumed from epoch {self.current_epoch}, "
            f"best_metric={self.best_metric:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def fit(self, num_epochs: int) -> None:
        """
        Main training loop.

        Args:
            num_epochs: Tổng số epochs.
        """
        logger.info(f"Bắt đầu training: {num_epochs} epochs")
        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Check stage switch
            target_stage = self._get_current_stage(epoch)
            if target_stage != self._current_stage_idx and target_stage >= 0:
                self._switch_stage(target_stage)

            epoch_start = time.time()

            # === Train ===
            train_metrics = self.train_one_epoch()

            # === Validate ===
            val_metrics = self.validate()

            # === Step scheduler ===
            if self.scheduler is not None:
                self.scheduler.step()

            # === Log to TensorBoard ===
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(k, v, epoch)

            # === Determine if best ===
            current_value = val_metrics.get(self.monitor_metric)
            if current_value is None:
                logger.warning(
                    f"Monitor metric '{self.monitor_metric}' không tìm thấy "
                    f"trong val_metrics. Available: {list(val_metrics.keys())}"
                )
                is_best = False
            else:
                if self.monitor_mode == "max":
                    is_best = current_value > self.best_metric
                else:
                    is_best = current_value < self.best_metric
                if is_best:
                    self.best_metric = current_value

            # === Save checkpoint ===
            all_metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(all_metrics, is_best=is_best)

            # === Log epoch summary ===
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"{self.monitor_metric}="
                f"{val_metrics.get(self.monitor_metric, float('nan')):.4f} | "
                f"lr={train_metrics['lr']:.2e} | "
                f"time={epoch_time:.1f}s"
            )

            # === Early stopping check ===
            if self.early_stopping and current_value is not None:
                if self.early_stopping(current_value):
                    logger.info(
                        f"Early stopping triggered (no improvement in "
                        f"{self.early_stopping.patience} epochs)"
                    )
                    break

        # Training xong
        total_time = time.time() - start_time
        logger.info(
            f"Training hoàn tất sau {total_time / 60:.1f} phút. "
            f"Best {self.monitor_metric}: {self.best_metric:.4f}"
        )
        self.writer.close()