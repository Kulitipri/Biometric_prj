"""
Script: Preprocess raw dataset.

Chạy preprocessing.py trên toàn bộ dataset:
- Detect khuôn mặt bằng MTCNN
- Align về 112x112
- Save vào data/processed/

Usage:
    python scripts/preprocess_data.py --config configs/data/lfw.yaml

    # Override device hoặc input/output dirs:
    python scripts/preprocess_data.py --config configs/data/lfw.yaml --device cpu
    python scripts/preprocess_data.py --config configs/data/lfw.yaml \\
        --input-dir data/raw/custom \\
        --output-dir data/processed/custom

Yêu cầu input:
    Dataset đã được tải về data/raw/<dataset_name>/ với cấu trúc:
        identity_1/
            img1.jpg
            img2.jpg
        identity_2/
            ...
"""

import argparse
import logging
import sys
from pathlib import Path

# Cho phép import từ src/ khi chạy script trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import preprocess_dataset
from src.utils.config import load_config
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess face dataset (detect + align)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path tới data config YAML (e.g. configs/data/lfw.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override device. Default: cuda nếu có, không thì cpu",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override input dir (mặc định lấy từ config.dataset.raw_dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir (mặc định lấy từ config.dataset.processed_dir)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại tất cả ảnh (không skip ảnh đã có trong output_dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Setup logger
    logger = setup_logger(name="preprocess", level=logging.INFO)

    # Load config
    logger.info(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Resolve paths
    dataset_cfg = config.get("dataset", {})
    preprocessing_cfg = config.get("preprocessing", {})

    input_dir = args.input_dir or Path(dataset_cfg.get("raw_dir", "data/raw"))
    output_dir = args.output_dir or Path(
        dataset_cfg.get("processed_dir", "data/processed")
    )

    # Resolve device: CLI > auto-detect
    if args.device:
        device = args.device
    else:
        # Lazy check torch để không phải import nếu không cần
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = preprocessing_cfg.get("image_size", 112)
    min_face_size = preprocessing_cfg.get("min_face_size", 40)

    # Print summary
    logger.info("=" * 60)
    logger.info("Preprocessing Configuration:")
    logger.info(f"  Dataset:        {dataset_cfg.get('name', '?')}")
    logger.info(f"  Input:          {input_dir}")
    logger.info(f"  Output:         {output_dir}")
    logger.info(f"  Image size:     {image_size}x{image_size}")
    logger.info(f"  Min face size:  {min_face_size}")
    logger.info(f"  Device:         {device}")
    logger.info(f"  Skip existing:  {not args.force}")
    logger.info("=" * 60)

    # Validate input dir
    if not input_dir.exists():
        logger.error(f"❌ Input dir không tồn tại: {input_dir}")
        logger.error("   Hãy tải dataset trước khi chạy preprocessing.")
        return 1

    # Run preprocessing
    try:
        stats = preprocess_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            image_size=image_size,
            device=device,
            min_face_size=min_face_size,
            skip_existing=not args.force,
        )
    except Exception as e:
        logger.error(f"[ERROR] Preprocessing failed: {e}", exc_info=True)
        return 1

    # Check kết quả
    if stats["total"] == 0:
        logger.error("[ERROR] Không có ảnh nào trong input dir!")
        return 1

    success_rate = stats["success"] / stats["total"] * 100
    if success_rate < 50:
        logger.warning(
            f"⚠️  Success rate thấp ({success_rate:.1f}%). "
            f"Kiểm tra failed.txt trong {output_dir}"
        )

    logger.info(f"✓ Preprocessing hoàn tất: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())