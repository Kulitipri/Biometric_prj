"""
Dataset Classes cho Face Recognition.

Chứa 2 class chính:
- FaceDataset: cho training (return image + identity label)
- VerificationDataset: cho evaluation (return pair + same/different label)

Format split file (cho FaceDataset):
    relative/path/to/image.jpg label
    Aaron_Peirsol/Aaron_Peirsol_0001.jpg 0
    Aaron_Peirsol/Aaron_Peirsol_0002.jpg 0
    Aaron_Sorkin/Aaron_Sorkin_0001.jpg 1
    ...

Format pairs file (cho VerificationDataset):
    img1_path img2_path label
    label=1: cùng người, label=0: khác người

Lưu ý paths trong split files đều RELATIVE so với data_dir.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ====================================================================== #
# Helper: load image
# ====================================================================== #


def _load_image_rgb(path: Path) -> np.ndarray:
    """
    Load ảnh từ disk -> RGB numpy array uint8.

    Dùng cv2 vì nhanh hơn PIL với Albumentations pipeline.
    cv2 đọc BGR nên cần convert sang RGB.

    Raises:
        FileNotFoundError: nếu file không tồn tại
        ValueError: nếu cv2 không decode được
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        # cv2.imread trả None thay vì raise, nên check thủ công
        if not Path(path).exists():
            raise FileNotFoundError(f"Ảnh không tồn tại: {path}")
        raise ValueError(f"Không decode được ảnh: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ====================================================================== #
# Split file parsing
# ====================================================================== #


def _parse_split_file(split_file: Path) -> List[Tuple[str, int]]:
    """
    Parse split file format: 'relative_path label' mỗi dòng.

    Returns:
        List of (relative_path, label) tuples.

    Skip dòng trống và dòng comment (bắt đầu bằng #).
    """
    samples: List[Tuple[str, int]] = []

    with open(split_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # Skip dòng trống / comment
            if not line or line.startswith("#"):
                continue

            # Split tối đa 1 lần để xử lý path có space
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                logger.warning(
                    f"{split_file}:{line_num} sai format, skip: {line!r}"
                )
                continue

            rel_path, label_str = parts
            try:
                label = int(label_str)
            except ValueError:
                logger.warning(
                    f"{split_file}:{line_num} label không phải int, skip: {line!r}"
                )
                continue

            samples.append((rel_path, label))

    return samples


def _parse_pairs_file(pairs_file: Path) -> List[Tuple[str, str, int]]:
    """
    Parse pairs file format: 'img1_path img2_path label' mỗi dòng.

    Returns:
        List of (img1_path, img2_path, label) tuples. Label 1=same, 0=diff.
    """
    pairs: List[Tuple[str, str, int]] = []

    with open(pairs_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 3:
                logger.warning(
                    f"{pairs_file}:{line_num} sai format (cần 3 phần), skip"
                )
                continue

            img1, img2, label_str = parts
            try:
                label = int(label_str)
                if label not in (0, 1):
                    raise ValueError(f"label phải là 0 hoặc 1, nhận {label}")
            except ValueError as e:
                logger.warning(f"{pairs_file}:{line_num} label sai: {e}")
                continue

            pairs.append((img1, img2, label))

    return pairs


# ====================================================================== #
# Transforms
# ====================================================================== #


# Chuẩn InsightFace cho ArcFace: pixel range [-1, 1]
# (pixel - 127.5) / 128 ≈ pixel/128 - ~1
ARCFACE_MEAN = (127.5, 127.5, 127.5)
ARCFACE_STD = (128.0, 128.0, 128.0)


def build_transforms(
    config: Optional[Dict] = None,
    is_training: bool = True,
    image_size: int = 112,
) -> A.Compose:
    """
    Xây dựng Albumentations transform pipeline.

    Args:
        config: Dict chứa augmentation settings (xem configs/data/lfw.yaml).
                Nếu None, dùng default sensible values.
        is_training: True = apply augmentation, False = chỉ normalize + ToTensor.
        image_size: Output size (default 112 cho ArcFace).

    Returns:
        Albumentations Compose pipeline.

    Pipeline output: tensor (3, H, W) đã normalize, ready cho model.
    """
    # Default config nếu không truyền
    if config is None:
        config = {}

    # Lấy aug config tùy theo mode
    aug_key = "training" if is_training else "validation"
    aug_cfg = config.get(aug_key, {}) if config else {}

    # Lấy normalize params từ config hoặc dùng default ArcFace
    norm_cfg = config.get("normalization", {}) if config else {}
    mean = tuple(norm_cfg.get("mean", ARCFACE_MEAN))
    std = tuple(norm_cfg.get("std", ARCFACE_STD))

    transforms_list: List[A.BasicTransform] = []

    # Resize an toàn (đề phòng ảnh không đúng size do bug preprocessing)
    transforms_list.append(A.Resize(image_size, image_size))

    if is_training:
        # === Augmentation cho training ===

        # Horizontal flip - augmentation cơ bản, hầu như luôn có lợi
        flip_p = aug_cfg.get("horizontal_flip", 0.5)
        if flip_p > 0:
            transforms_list.append(A.HorizontalFlip(p=flip_p))

        # Color jitter - tăng robust với điều kiện ánh sáng khác nhau
        cj_cfg = aug_cfg.get("color_jitter", {})
        if cj_cfg:
            transforms_list.append(
                A.ColorJitter(
                    brightness=cj_cfg.get("brightness", 0.2),
                    contrast=cj_cfg.get("contrast", 0.2),
                    saturation=cj_cfg.get("saturation", 0.2),
                    hue=cj_cfg.get("hue", 0.0),
                    p=cj_cfg.get("probability", 0.5),
                )
            )

        # Random occlusion (CutOut) - mô phỏng che ngẫu nhiên không đặc thù
        # Bổ trợ cho synthetic mask, làm model robust với occlusion bất kỳ
        occ_cfg = aug_cfg.get("random_occlusion", {})
        if occ_cfg.get("probability", 0) > 0:
            max_ratio = occ_cfg.get("max_size_ratio", 0.3)
            max_size = int(image_size * max_ratio)
            transforms_list.append(
                A.CoarseDropout(
                    max_holes=1,
                    max_height=max_size,
                    max_width=max_size,
                    min_holes=1,
                    min_height=max_size // 2,
                    min_width=max_size // 2,
                    fill_value=0,
                    p=occ_cfg["probability"],
                )
            )

    # === Normalize + ToTensor (cho cả train và val) ===
    transforms_list.append(
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0)
    )
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


# ====================================================================== #
# FaceDataset
# ====================================================================== #


class FaceDataset(Dataset):
    """
    Dataset cho training với classification loss (ArcFace).

    Mỗi sample = (image_tensor, identity_label).

    Args:
        data_dir: Thư mục chứa ảnh đã align (e.g. data/processed/lfw/).
        split_file: File .txt format 'relative_path label' mỗi dòng.
        transform: Albumentations Compose. Có thể None nếu không augment.
        mask_augmenter: Optional callable(image, landmarks) -> image.
            Nếu cung cấp, sẽ apply trước khi transform.
            Lưu ý: hiện tại FaceDataset chưa lưu landmarks nên
            mask_augmenter sẽ được gọi với landmarks=None.
        skip_missing: True = skip ảnh không load được khi __getitem__.
            Mặc định False để fail fast khi có vấn đề.
    """

    def __init__(
        self,
        data_dir: Path,
        split_file: Path,
        transform: Optional[A.Compose] = None,
        mask_augmenter: Optional[Callable] = None,
        skip_missing: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split_file = Path(split_file)
        self.transform = transform
        self.mask_augmenter = mask_augmenter
        self.skip_missing = skip_missing

        # Validation
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir không tồn tại: {self.data_dir}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"split_file không tồn tại: {self.split_file}")

        # Parse split file
        self.samples: List[Tuple[str, int]] = _parse_split_file(self.split_file)

        if len(self.samples) == 0:
            raise ValueError(f"Split file rỗng: {self.split_file}")

        # Tính num_classes (cần cho ArcFace head)
        labels = [label for _, label in self.samples]
        self.num_classes = max(labels) + 1

        # Sanity check: label phải là 0..num_classes-1 liên tục
        unique_labels = set(labels)
        expected = set(range(self.num_classes))
        if unique_labels != expected:
            missing = expected - unique_labels
            extra = unique_labels - expected
            logger.warning(
                f"Labels không liên tục từ 0 đến {self.num_classes - 1}. "
                f"Missing: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}, "
                f"Extra: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}"
            )

        logger.info(
            f"FaceDataset: {len(self.samples)} samples, "
            f"{self.num_classes} identities từ {self.split_file.name}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load 1 sample. Tách ra để dễ override / test."""
        rel_path, label = self.samples[idx]
        full_path = self.data_dir / rel_path
        image = _load_image_rgb(full_path)
        return image, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (image_tensor, label)
            - image_tensor: (3, H, W) float, đã normalize
            - label: int identity label
        """
        try:
            image, label = self._load_sample(idx)
        except (FileNotFoundError, ValueError) as e:
            if self.skip_missing:
                # Random một sample khác để thay thế
                logger.debug(f"Skip idx={idx}: {e}")
                next_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(next_idx)
            raise

        # Apply mask augmentation (nếu có)
        # Note: mask_augmenter cần landmarks để place mask đúng vị trí.
        # Hiện tại pass None, mask_augmenter cần handle case này
        # (e.g., dùng default position cho ảnh đã align 112x112).
        if self.mask_augmenter is not None:
            image = self.mask_augmenter(image, landmarks=None)

        # Apply transform (Albumentations)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            # Fallback nếu không có transform: chỉ convert sang tensor
            # (HWC uint8 -> CHW float)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


# ====================================================================== #
# VerificationDataset
# ====================================================================== #


class VerificationDataset(Dataset):
    """
    Dataset cho evaluation với verification protocol.

    Mỗi sample = (img1_tensor, img2_tensor, label).
    Label 1 = cùng người, 0 = khác người.

    KHÔNG dùng augmentation (chỉ normalize), để evaluation deterministic.

    Args:
        data_dir: Thư mục chứa ảnh đã align.
        pairs_file: File .txt format 'img1_path img2_path label' mỗi dòng.
        transform: Transform cho cả 2 ảnh. Khuyến nghị dùng
            build_transforms(is_training=False) để chỉ normalize.
    """

    def __init__(
        self,
        data_dir: Path,
        pairs_file: Path,
        transform: Optional[A.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.pairs_file = Path(pairs_file)
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir không tồn tại: {self.data_dir}")
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"pairs_file không tồn tại: {self.pairs_file}")

        self.pairs: List[Tuple[str, str, int]] = _parse_pairs_file(self.pairs_file)
        if len(self.pairs) == 0:
            raise ValueError(f"Pairs file rỗng: {self.pairs_file}")

        n_positive = sum(1 for _, _, lbl in self.pairs if lbl == 1)
        n_negative = len(self.pairs) - n_positive
        logger.info(
            f"VerificationDataset: {len(self.pairs)} pairs "
            f"({n_positive} positive, {n_negative} negative) "
            f"từ {self.pairs_file.name}"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        """Load + transform 1 ảnh."""
        image = _load_image_rgb(self.data_dir / rel_path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            (img1_tensor, img2_tensor, label)
        """
        img1_path, img2_path, label = self.pairs[idx]
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        return img1, img2, label