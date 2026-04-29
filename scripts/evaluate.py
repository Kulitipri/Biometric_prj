"""
Script: Evaluate trained model trên test set.

Chạy full evaluation:
- Load model từ checkpoint
- Extract embeddings cho tất cả test pairs
- Tính accuracy, TAR@FAR, EER, ROC AUC
- Save metrics + plots vào experiment folder

Usage:
    python scripts/evaluate.py --exp experiments/exp_001_arcface_mobilenetv2

    # Dùng last checkpoint thay vì best:
    python scripts/evaluate.py --exp experiments/exp_001 --checkpoint last

    # Override test pairs file:
    python scripts/evaluate.py --exp experiments/exp_001 \\
        --test-pairs data/splits/lfw/test_pairs.txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Cho phép import từ src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import VerificationDataset, build_transforms
from src.inference.embedding_extractor import EmbeddingExtractor
from src.metrics.verification import compute_roc_curve, evaluate_verification
from src.models.face_recognizer import build_model
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.visualization import (
    plot_roc_curve,
    plot_similarity_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate face recognition model")
    parser.add_argument(
        "--exp",
        type=Path,
        required=True,
        help="Path tới experiment folder (e.g. experiments/exp_001)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Checkpoint nào để load (default: best)",
    )
    parser.add_argument(
        "--test-pairs",
        type=Path,
        default=None,
        help="Override test pairs file (default từ config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override device (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size cho extraction",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.exp.exists():
        print(f"[ERROR] Experiment folder does not exist: {args.exp}")
        return 1

    # Setup paths
    config_path = args.exp / "config_resolved.yaml"
    checkpoint_path = args.exp / "checkpoints" / f"{args.checkpoint}.pth"
    results_dir = args.exp / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        name="evaluate",
        log_file=results_dir / "evaluate.log",
        level=logging.INFO,
    )

    # Validate paths
    if not config_path.exists():
        logger.error(f"[ERROR] Config does not exist: {config_path}")
        logger.error("   Please ensure experiment has been trained.")
        return 1
    if not checkpoint_path.exists():
        logger.error(f"[ERROR] Checkpoint does not exist: {checkpoint_path}")
        return 1

    # Load config
    logger.info(f"Loading config: {config_path}")
    config = load_config(config_path)
    dataset_cfg = config["dataset"]

    # Determine device
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # === Load model ===
    logger.info(f"Loading model from: {checkpoint_path}")
    # Build model in inference mode (no need num_classes if only using embedding)
    # But to load correct state_dict (with ArcFace weights), num_classes must match.
    # Trick: read num_classes from checkpoint state_dict.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Find num_classes from shape of ArcFace weight matrix
    # Pattern: 'head.weight' shape = (num_classes, embedding_dim)
    head_weight_key = "head.weight"
    if head_weight_key in state_dict:
        num_classes = state_dict[head_weight_key].shape[0]
        logger.info(f"Detected num_classes={num_classes} from checkpoint")
    else:
        logger.warning(
            "Could not find head.weight in checkpoint, building model "
            "without ArcFace head (inference only)"
        )
        num_classes = None

    model = build_model(config, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    logger.info("[SUCCESS] Model loaded")

    # === Load test pairs ===
    test_pairs_file = args.test_pairs or (
        Path(dataset_cfg["splits_dir"]) / "test_pairs.txt"
    )
    if not test_pairs_file.exists():
        logger.error(f"[ERROR] Test pairs do not exist: {test_pairs_file}")
        return 1

    aug_cfg = config.get("augmentation", {})
    val_transform = build_transforms(aug_cfg, is_training=False)
    test_dataset = VerificationDataset(
        data_dir=Path(dataset_cfg["processed_dir"]),
        pairs_file=test_pairs_file,
        transform=val_transform,
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    logger.info(f"Test set: {len(test_dataset)} pairs")

    # === Extract embeddings ===
    logger.info("Extracting embeddings...")
    extractor = EmbeddingExtractor(model, device=device)
    emb1, emb2, labels = extractor.extract_pairs(
        test_loader, desc="Test extraction"
    )

    # === Compute metrics ===
    logger.info("Computing metrics...")
    metrics = evaluate_verification(emb1, emb2, labels)

    # === Print metrics ===
    logger.info("=" * 60)
    logger.info("Test Results:")
    for k, v in metrics.items():
        logger.info(f"  {k:25s}: {v:.4f}")
    logger.info("=" * 60)

    # === Save metrics JSON ===
    metrics_file = results_dir / f"metrics_{args.checkpoint}.json"
    metrics_with_meta = {
        "checkpoint": str(checkpoint_path),
        "test_pairs": str(test_pairs_file),
        "num_pairs": len(labels),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_with_meta, f, indent=2)
    logger.info(f"[SUCCESS] Saved metrics: {metrics_file}")

    # === Compute similarities cho plots ===
    similarities = np.sum(emb1 * emb2, axis=1)

    # === Plot ROC curve ===
    fars, tars, _ = compute_roc_curve(similarities, labels)
    roc_path = results_dir / f"roc_{args.checkpoint}.png"
    plot_roc_curve(
        fars, tars,
        save_path=roc_path,
        title=f"ROC Curve - {args.exp.name} ({args.checkpoint})",
        label="Test set",
        auc_value=metrics["auc"],
    )

    # === Plot similarity distribution ===
    sim_path = results_dir / f"similarity_dist_{args.checkpoint}.png"
    plot_similarity_distribution(
        similarities, labels,
        save_path=sim_path,
        title=f"Similarity Distribution - {args.exp.name}",
        threshold=metrics["threshold_acc"],
    )

    logger.info(f"\n[SUCCESS] Evaluation completed. Results at: {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())