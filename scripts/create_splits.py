"""
Script: Create train/val/test splits.

QUAN TRỌNG: Chia theo IDENTITY, không phải theo ảnh!
- Train identities → dùng để train classifier
- Val/Test identities → đánh giá verification trên pairs

Output files:
    data/splits/<dataset>/train.txt        - format: 'rel_path label' mỗi dòng
    data/splits/<dataset>/val_pairs.txt    - format: 'img1 img2 label' mỗi dòng
    data/splits/<dataset>/test_pairs.txt   - tương tự val_pairs

Usage:
    python scripts/create_splits.py --config configs/data/lfw.yaml
    python scripts/create_splits.py --config configs/data/lfw.yaml --seed 42
"""

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Cho phép import từ src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = logging.getLogger(__name__)


# ====================================================================== #
# Helpers
# ====================================================================== #


def _scan_dataset(data_dir: Path) -> Dict[str, List[str]]:
    """
    Scan thư mục dataset, return dict {identity: [list relative img paths]}.

    Cấu trúc mong đợi:
        data_dir/
            identity_1/
                img1.jpg
                img2.jpg
            identity_2/
                ...
    """
    identity_to_images: Dict[str, List[str]] = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for identity_dir in sorted(data_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        identity = identity_dir.name
        images = sorted(
            [
                str(img.relative_to(data_dir).as_posix())
                for img in identity_dir.iterdir()
                if img.is_file() and img.suffix.lower() in image_extensions
            ]
        )
        if images:
            identity_to_images[identity] = images

    return identity_to_images


def _filter_min_images(
    identity_to_images: Dict[str, List[str]], min_images: int
) -> Dict[str, List[str]]:
    """Loại các identity có ít hơn min_images ảnh."""
    filtered = {
        identity: imgs
        for identity, imgs in identity_to_images.items()
        if len(imgs) >= min_images
    }
    return filtered


# ====================================================================== #
# Identity splits
# ====================================================================== #


def create_identity_splits(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_images_per_identity: int = 2,
) -> Tuple[List[str], List[str], List[str], Path]:
    """
    Chia identities thành 3 splits.

    Output:
        output_dir/train.txt - 'rel_path label' mỗi dòng (cho FaceDataset)
        output_dir/val_identities.txt - list val identities (để tạo pairs)
        output_dir/test_identities.txt - list test identities (để tạo pairs)

    Returns:
        (train_identities, val_identities, test_identities, actual_data_dir)
    """
    logger.info(f"Scanning dataset tại: {data_dir}")
    
    # Auto-detect subdirectory if root dir has no identities
    actual_data_dir = data_dir
    identity_to_images = _scan_dataset(data_dir)
    
    if len(identity_to_images) == 0:
        logger.info(f"Không tìm thấy identities trong {data_dir}, kiểm tra subdirectories...")
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            actual_data_dir = subdirs[0]
            logger.info(f"Tự động dùng subdirectory: {actual_data_dir.name}")
            identity_to_images = _scan_dataset(actual_data_dir)
        elif len(subdirs) > 1:
            logger.info(f"Tìm thấy {len(subdirs)} subdirectories, cố gắng scan tất cả...")
            for subdir in subdirs:
                test_identities = _scan_dataset(subdir)
                if len(test_identities) > 0:
                    actual_data_dir = subdir
                    identity_to_images = test_identities
                    logger.info(f"Tìm được identities trong: {actual_data_dir.name}")
                    break
    
    logger.info(f"Tìm được {len(identity_to_images)} identities")

    # Filter các identity quá ít ảnh
    identity_to_images = _filter_min_images(identity_to_images, min_images_per_identity)
    logger.info(
        f"Sau filter (>={min_images_per_identity} ảnh/identity): "
        f"{len(identity_to_images)} identities"
    )

    if len(identity_to_images) == 0:
        raise ValueError(
            f"Không có identity nào đủ điều kiện trong {data_dir}"
        )

    # Validate ratios
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio phải trong (0, 1), nhận {train_ratio}")
    if not 0 < val_ratio < 1 - train_ratio:
        raise ValueError(
            f"val_ratio + train_ratio phải < 1, nhận {train_ratio + val_ratio}"
        )

    # Shuffle deterministic
    rng = random.Random(seed)
    identities = sorted(identity_to_images.keys())
    rng.shuffle(identities)

    # Split
    n_total = len(identities)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_identities = identities[:n_train]
    val_identities = identities[n_train : n_train + n_val]
    test_identities = identities[n_train + n_val :]

    logger.info(
        f"Split: train={len(train_identities)}, "
        f"val={len(val_identities)}, test={len(test_identities)}"
    )

    # Tính relative path prefix nếu data_dir khác actual_data_dir
    # (khi auto-detect subdirectory)
    path_prefix = ""
    if actual_data_dir != data_dir:
        try:
            path_prefix = str(actual_data_dir.relative_to(data_dir)) + "/"
            logger.info(f"  Using path prefix: {path_prefix}")
        except ValueError:
            # actual_data_dir không phải subdirectory của data_dir
            # Cố dùng actual_data_dir.name
            path_prefix = actual_data_dir.name + "/"

    # Generate train.txt với format 'rel_path label_idx'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for label_idx, identity in enumerate(sorted(train_identities)):
            for img_path in identity_to_images[identity]:
                full_rel_path = path_prefix + img_path
                f.write(f"{full_rel_path} {label_idx}\n")

    n_train_samples = sum(len(identity_to_images[i]) for i in train_identities)
    logger.info(
        f"  Train: {n_train_samples} samples "
        f"({len(train_identities)} identities) → {train_file}"
    )

    # Save danh sách val/test identities (để tạo pairs sau)
    for split_name, ids in [("val", val_identities), ("test", test_identities)]:
        ids_file = output_dir / f"{split_name}_identities.txt"
        with open(ids_file, "w", encoding="utf-8") as f:
            for identity in sorted(ids):
                f.write(f"{identity}\n")

    return train_identities, val_identities, test_identities, actual_data_dir


# ====================================================================== #
# Verification pairs
# ====================================================================== #


def create_verification_pairs(
    data_dir: Path,
    actual_data_dir: Path,
    identities: List[str],
    output_file: Path,
    num_positive: int = 3000,
    num_negative: int = 3000,
    seed: int = 42,
) -> int:
    """
    Tạo pairs (img1, img2, label) cho verification task.

    Strategy:
        - Positive pair: 2 ảnh khác nhau của CÙNG 1 person
        - Negative pair: 1 ảnh từ person A + 1 ảnh từ person B (A ≠ B)

    Args:
        data_dir: Thư mục root (có thể chứa subdirectories).
        actual_data_dir: Thư mục thực sự chứa ảnh (sau khi auto-detect).
        identities: List identities được phép dùng (val hoặc test).
        output_file: Path output file.
        num_positive: Số positive pairs muốn tạo.
        num_negative: Số negative pairs muốn tạo.
        seed: Random seed.

    Returns:
        Tổng số pairs đã tạo.
    """
    rng = random.Random(seed)
    
    # Auto-detect subdirectory if root dir has no identities
    identity_to_images = _scan_dataset(data_dir)
    if len(identity_to_images) == 0:
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            identity_to_images = _scan_dataset(subdirs[0])
        elif len(subdirs) > 1:
            for subdir in subdirs:
                test_identities = _scan_dataset(subdir)
                if len(test_identities) > 0:
                    identity_to_images = test_identities
                    break

    # Tính relative path prefix nếu data_dir khác actual_data_dir
    path_prefix = ""
    if actual_data_dir != data_dir:
        try:
            path_prefix = str(actual_data_dir.relative_to(data_dir)) + "/"
        except ValueError:
            path_prefix = actual_data_dir.name + "/"

    # Filter chỉ giữ identities được phép
    identity_to_images = {
        identity: imgs
        for identity, imgs in identity_to_images.items()
        if identity in set(identities)
    }

    # Cần ít nhất 2 ảnh để tạo positive pair
    eligible_identities = [
        identity for identity, imgs in identity_to_images.items() if len(imgs) >= 2
    ]

    if len(eligible_identities) < 2:
        logger.warning(
            f"Chỉ có {len(eligible_identities)} identity đủ ảnh để tạo pair, "
            f"không đủ để tạo negative pairs."
        )

    # === Tạo positive pairs ===
    # Tạo tất cả possible pairs rồi sample, thay vì random 30k lần
    all_possible_positive = []
    for identity in eligible_identities:
        imgs = identity_to_images[identity]
        # Tạo tất cả pair từ ảnh của 1 identity
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                all_possible_positive.append((imgs[i], imgs[j], 1))
    
    logger.info(
        f"  Total possible positive pairs: {len(all_possible_positive)}"
    )
    
    # Shuffle & sample
    rng.shuffle(all_possible_positive)
    positive_pairs = all_possible_positive[:num_positive]
    
    if len(positive_pairs) < num_positive:
        logger.warning(
            f"Chỉ tạo được {len(positive_pairs)}/{num_positive} positive pairs. "
            f"Max possible: {len(all_possible_positive)}"
        )

    # === Tạo negative pairs ===
    negative_pairs = []
    all_identities = list(identity_to_images.keys())
    if len(all_identities) >= 2:
        max_attempts = num_negative * 10
        seen_neg = set()
        attempts = 0
        while len(negative_pairs) < num_negative and attempts < max_attempts:
            attempts += 1
            id1, id2 = rng.sample(all_identities, 2)
            img1 = rng.choice(identity_to_images[id1])
            img2 = rng.choice(identity_to_images[id2])
            key = tuple(sorted([img1, img2]))
            if key in seen_neg:
                continue
            seen_neg.add(key)
            negative_pairs.append((img1, img2, 0))

    # === Shuffle & write ===
    all_pairs = positive_pairs + negative_pairs
    rng.shuffle(all_pairs)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for img1, img2, label in all_pairs:
            # Apply path prefix if needed
            full_img1 = path_prefix + img1 if path_prefix else img1
            full_img2 = path_prefix + img2 if path_prefix else img2
            f.write(f"{full_img1} {full_img2} {label}\n")

    logger.info(
        f"  Saved {len(all_pairs)} pairs "
        f"(pos={len(positive_pairs)}, neg={len(negative_pairs)}) → {output_file}"
    )
    return len(all_pairs)


# ====================================================================== #
# Main
# ====================================================================== #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset splits")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (default từ config)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override processed data dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output splits dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(name="create_splits", level=logging.INFO)

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    splits_cfg = config.get("splits", {})
    pairs_cfg = config.get("verification_pairs", {})

    data_dir = args.data_dir or Path(dataset_cfg.get("processed_dir", "data/processed"))
    output_dir = args.output_dir or Path(dataset_cfg.get("splits_dir", "data/splits"))

    seed = args.seed if args.seed is not None else splits_cfg.get("seed", 42)
    train_ratio = splits_cfg.get("train_ratio", 0.7)
    val_ratio = splits_cfg.get("val_ratio", 0.15)

    num_positive = pairs_cfg.get("num_positive", 3000)
    num_negative = pairs_cfg.get("num_negative", 3000)

    logger.info("=" * 60)
    logger.info("Create Splits Configuration:")
    logger.info(f"  Data dir:    {data_dir}")
    logger.info(f"  Output dir:  {output_dir}")
    logger.info(f"  Seed:        {seed}")
    logger.info(f"  Train/Val:   {train_ratio:.2f} / {val_ratio:.2f}")
    logger.info(f"  Pairs:       pos={num_positive}, neg={num_negative}")
    logger.info("=" * 60)

    if not data_dir.exists():
        logger.error(f"[ERROR] Data dir không tồn tại: {data_dir}")
        logger.error("   Hãy chạy preprocess_data.py trước.")
        return 1

    # 1. Tạo identity splits + train.txt
    try:
        train_ids, val_ids, test_ids, actual_data_dir = create_identity_splits(
            data_dir=data_dir,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"[ERROR] Lỗi tạo identity splits: {e}", exc_info=True)
        return 1

    # 2. Tạo val_pairs.txt và test_pairs.txt
    logger.info("\nTạo verification pairs...")
    try:
        create_verification_pairs(
            data_dir=data_dir,
            actual_data_dir=actual_data_dir,
            identities=val_ids,
            output_file=output_dir / "val_pairs.txt",
            num_positive=num_positive,
            num_negative=num_negative,
            seed=seed,
        )
        create_verification_pairs(
            data_dir=data_dir,
            actual_data_dir=actual_data_dir,
            identities=test_ids,
            output_file=output_dir / "test_pairs.txt",
            num_positive=num_positive,
            num_negative=num_negative,
            seed=seed + 1,  # khác seed để pairs khác val
        )
    except Exception as e:
        logger.error(f"[ERROR] Lỗi tạo pairs: {e}", exc_info=True)
        return 1

    logger.info(f"\n[SUCCESS] Splits hoàn tất tại: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())