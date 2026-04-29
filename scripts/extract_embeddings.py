"""
Script: Extract embeddings cho 1 folder ảnh, save ra .npz file.

Dùng để:
- Build gallery database cho identification (1 lần, dùng nhiều)
- Visualize embedding space (t-SNE/UMAP)
- Phân tích kết quả model (find hard cases, cluster identities)
- Export cho deployment

Format output (.npz):
    embeddings : (N, D) - L2-normalized embeddings
    paths      : (N,)   - relative path tới ảnh gốc
    labels     : (N,)   - integer labels (nếu input có cấu trúc identity)
    label_names: (M,)   - tên identity tương ứng với label index

Usage:
    # Extract từ folder dataset (cấu trúc identity/img.jpg)
    python scripts/extract_embeddings.py \\
        --exp experiments/exp_001_arcface_mobilenetv2 \\
        --images data/processed/lfw \\
        --output data/embeddings/lfw_embeddings.npz

    # Dùng last checkpoint thay vì best
    python scripts/extract_embeddings.py \\
        --exp experiments/exp_001 \\
        --images data/processed/lfw \\
        --output data/embeddings/lfw.npz \\
        --checkpoint last
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Cho phép import từ src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FaceDataset, build_transforms
from src.inference.embedding_extractor import EmbeddingExtractor
from src.models.face_recognizer import build_model
from src.utils.config import load_config
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings from images")
    parser.add_argument(
        "--exp",
        type=Path,
        required=True,
        help="Path tới experiment folder",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path tới folder ảnh (cấu trúc identity/img.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "last"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
    )
    return parser.parse_args()


def _build_temp_split_file(
    images_dir: Path,
) -> Tuple[Path, List[str], List[str]]:
    """
    Tạo split file tạm cho FaceDataset từ folder ảnh.

    Quét folder theo cấu trúc identity/img.jpg, tạo file split:
        identity_1/img1.jpg 0
        identity_1/img2.jpg 0
        identity_2/img1.jpg 1
        ...

    Returns:
        (temp_file_path, label_names, paths)
        - temp_file_path: Path tới file split tạm (caller phải xóa sau)
        - label_names: List tên identities (theo thứ tự label index)
        - paths: List relative paths theo thứ tự xuất hiện trong file
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Quét folder
    label_names: List[str] = []
    paths: List[str] = []
    labels: List[int] = []

    for label_idx, identity_dir in enumerate(sorted(images_dir.iterdir())):
        if not identity_dir.is_dir():
            continue

        identity = identity_dir.name
        identity_images = sorted(
            [
                str(img.relative_to(images_dir).as_posix())
                for img in identity_dir.iterdir()
                if img.is_file() and img.suffix.lower() in image_extensions
            ]
        )

        if not identity_images:
            continue

        label_names.append(identity)
        for img_path in identity_images:
            paths.append(img_path)
            labels.append(len(label_names) - 1)

    if not paths:
        raise ValueError(f"Không có ảnh nào trong {images_dir}")

    # Write tạm vào file
    tmp_fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    try:
        for path, label in zip(paths, labels):
            tmp_fd.write(f"{path} {label}\n")
        tmp_fd.flush()
    finally:
        tmp_fd.close()

    return Path(tmp_fd.name), label_names, paths


def main() -> int:
    args = parse_args()

    logger = setup_logger(name="extract_embeddings", level=logging.INFO)

    # === Validate paths ===
    if not args.exp.exists():
        logger.error(f"❌ Experiment folder không tồn tại: {args.exp}")
        return 1

    if not args.images.exists():
        logger.error(f"❌ Images folder không tồn tại: {args.images}")
        return 1

    config_path = args.exp / "config_resolved.yaml"
    checkpoint_path = args.exp / "checkpoints" / f"{args.checkpoint}.pth"

    if not config_path.exists():
        logger.error(f"❌ Config không tồn tại: {config_path}")
        return 1
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint không tồn tại: {checkpoint_path}")
        return 1

    # === Load config + determine device ===
    config = load_config(config_path)

    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("Extract Embeddings Configuration:")
    logger.info(f"  Experiment:    {args.exp}")
    logger.info(f"  Checkpoint:    {args.checkpoint}")
    logger.info(f"  Images:        {args.images}")
    logger.info(f"  Output:        {args.output}")
    logger.info(f"  Device:        {device}")
    logger.info("=" * 60)

    # === Load model ===
    logger.info(f"Loading model từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Auto-detect num_classes từ checkpoint
    num_classes = (
        state_dict["head.weight"].shape[0]
        if "head.weight" in state_dict
        else None
    )

    model = build_model(config, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    logger.info(f"[SUCCESS] Model loaded (embedding_dim={model.backbone.embedding_dim})")

    # === Build dataset từ folder ảnh ===
    logger.info(f"Scanning images trong {args.images}...")
    temp_split_file, label_names, paths = _build_temp_split_file(args.images)
    logger.info(f"  Tìm được {len(paths)} ảnh, {len(label_names)} identities")

    try:
        # Build transform (no augmentation cho extraction)
        aug_cfg = config.get("augmentation", {})
        transform = build_transforms(aug_cfg, is_training=False)

        dataset = FaceDataset(
            data_dir=args.images,
            split_file=temp_split_file,
            transform=transform,
            mask_augmenter=None,
        )

        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,  # quan trọng: giữ thứ tự để map với paths
            num_workers=4,
            pin_memory=(device == "cuda"),
        )

        # === Extract ===
        extractor = EmbeddingExtractor(model, device=device)
        embeddings, labels = extractor.extract_from_loader(
            loader, desc="Extracting"
        )

    finally:
        # Cleanup temp file
        temp_split_file.unlink(missing_ok=True)

    logger.info(f"[SUCCESS] Extracted: {embeddings.shape}")

    # === Save ra .npz ===
    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        paths=np.array(paths),
        labels=labels,
        label_names=np.array(label_names),
    )

    file_size = args.output.stat().st_size / (1024 * 1024)
    logger.info(f"[SUCCESS] Saved {file_size:.2f}MB → {args.output}")

    # Print summary của file đã save
    logger.info("\nFile contents:")
    logger.info(f"  embeddings : ({len(embeddings)}, {embeddings.shape[1]})")
    logger.info(f"  paths      : ({len(paths)},)")
    logger.info(f"  labels     : ({len(labels)},)")
    logger.info(f"  label_names: ({len(label_names)},)")

    return 0


if __name__ == "__main__":
    sys.exit(main())