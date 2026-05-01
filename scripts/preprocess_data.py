"""
Script: Preprocess raw dataset.

Chạy preprocessing.py trên toàn bộ dataset:
- Detect khuôn mặt bằng MTCNN
- Align về 112x112
- Save vào data/processed/

Usage:
    # Auto-detect device (ưu tiên CUDA nếu có)
    python scripts/preprocess_data.py --config configs/data/lfw.yaml

    # Ép dùng GPU
    python scripts/preprocess_data.py --config configs/data/lfw.yaml --device cuda

    # Ép dùng CPU
    python scripts/preprocess_data.py --config configs/data/lfw.yaml --device cpu

    # Chạy lại từ đầu, không skip ảnh đã có
    python scripts/preprocess_data.py --config configs/data/lfw.yaml --force

    # Override input/output dirs:
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
import warnings
from pathlib import Path

# Tắt FutureWarning của facenet-pytorch (về torch.load weights_only).
# Đây là warning vô hại, chỉ làm rối log. PHẢI gọi TRƯỚC khi import facenet_pytorch.
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

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
        help="Chạy lại tất cả ảnh (không skip ảnh đã có trong output_dir). "
             "Dùng khi muốn re-process từ đầu.",
    )
    return parser.parse_args()


def resolve_device(requested, logger: logging.Logger) -> str:
    """
    Xác định device sử dụng (cuda/cpu) một cách rõ ràng và minh bạch.

    Logic:
    - Nếu user truyền --device cuda nhưng CUDA không khả dụng:
        cảnh báo lớn (KHÔNG fallback im lặng) và fallback về CPU.
    - Nếu user không truyền --device: tự chọn cuda nếu có, không thì cpu.
    - Nếu user truyền --device cpu: tôn trọng lựa chọn.
    """
    import torch

    cuda_available = torch.cuda.is_available()

    # Trường hợp 1: User ép dùng CUDA
    if requested == "cuda":
        if not cuda_available:
            logger.error("=" * 60)
            logger.error("❌ Bạn yêu cầu --device cuda NHƯNG CUDA không khả dụng!")
            logger.error("   Lý do có thể:")
            logger.error("   1. PyTorch cài là bản CPU-only.")
            logger.error("      → Cài lại: pip install torch torchvision \\")
            logger.error("                  --index-url https://download.pytorch.org/whl/cu121")
            logger.error("   2. NVIDIA driver chưa cài hoặc quá cũ.")
            logger.error("      → Kiểm tra: nvidia-smi")
            logger.error("   3. Không có GPU NVIDIA trên máy.")
            logger.error("")
            logger.error(f"   PyTorch version: {torch.__version__}")
            logger.error(f"   PyTorch CUDA build: {torch.version.cuda}")
            logger.error("=" * 60)
            logger.warning("⚠️  Tự fallback về CPU.")
            return "cpu"
        return "cuda"

    # Trường hợp 2: User ép dùng CPU
    if requested == "cpu":
        if cuda_available:
            logger.info(
                "ℹ️  CUDA khả dụng nhưng bạn chọn --device cpu. "
                "Bỏ flag --device để tự dùng GPU."
            )
        return "cpu"

    # Trường hợp 3: Auto-detect
    if cuda_available:
        return "cuda"
    else:
        logger.warning(
            "⚠️  CUDA không khả dụng → dùng CPU. "
            f"PyTorch version: {torch.__version__}, "
            f"CUDA build: {torch.version.cuda}. "
            "Nếu bạn có GPU NVIDIA, hãy cài bản PyTorch có CUDA "
            "(https://pytorch.org/get-started/locally/)."
        )
        return "cpu"


def log_device_info(device: str, logger: logging.Logger) -> None:
    """Log chi tiết về device sẽ dùng (tên GPU, VRAM)."""
    import torch

    if device == "cuda":
        gpu_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
        logger.info(f"  GPU:            {gpu_name} ({vram_gb:.1f} GB VRAM)")
        logger.info(f"  CUDA version:   {torch.version.cuda}")
    else:
        logger.info(f"  CPU mode (PyTorch {torch.__version__})")


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

    # Resolve device với logic minh bạch hơn
    device = resolve_device(args.device, logger)

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
    log_device_info(device, logger)
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

    # === Check kết quả với logic mới (phân biệt skip vs fail) ===
    if stats["total"] == 0:
        logger.error("[ERROR] Không có ảnh nào trong input dir!")
        return 1

    # Số ảnh thực sự được xử lý lần này (không tính skipped)
    processed = stats["total"] - stats["skipped"]

    if processed == 0:
        # Toàn bộ bị skip → output đã đầy đủ từ lần chạy trước
        logger.info(
            f"✓ Tất cả {stats['skipped']} ảnh đã có sẵn trong {output_dir}. "
            f"Không cần xử lý thêm."
        )
        logger.info(
            "   Nếu muốn chạy lại từ đầu, dùng flag: --force"
        )
    else:
        # Có ảnh được xử lý → tính success rate trên số ảnh thực sự xử lý
        success_rate = stats["success"] / processed * 100
        logger.info(
            f"Đã xử lý {processed} ảnh (skip {stats['skipped']} ảnh đã tồn tại): "
            f"thành công {stats['success']} ({success_rate:.1f}%), fail {stats['failed']}"
        )

        if success_rate < 50:
            logger.warning(
                f"⚠️  Success rate thấp ({success_rate:.1f}% trên "
                f"{processed} ảnh được xử lý mới). "
                f"Kiểm tra failed.txt trong {output_dir} để xem chi tiết."
            )
        elif success_rate < 90:
            logger.warning(
                f"⚠️  Success rate vừa phải ({success_rate:.1f}%). "
                f"Một số ảnh detect không ra mặt — bình thường với LFW "
                f"(chiếm ~5-10% do ảnh nhiễu/chất lượng kém)."
            )

    logger.info(f"✓ Preprocessing hoàn tất: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())