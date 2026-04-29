"""
Face Detection and Alignment Module.

Chịu trách nhiệm:
- Detect khuôn mặt trong ảnh bằng MTCNN
- Extract 5 landmarks (mắt trái, mắt phải, mũi, 2 khóe miệng)
- Align + crop về kích thước chuẩn 112x112 cho ArcFace

Usage cơ bản:
    from src.data.preprocessing import FaceAligner

    aligner = FaceAligner(image_size=112, device='cuda')
    aligned_face = aligner.align_from_path('path/to/image.jpg')

Usage batch:
    from src.data.preprocessing import preprocess_dataset

    stats = preprocess_dataset(
        input_dir='data/raw/lfw',
        output_dir='data/processed/lfw',
        device='cuda'
    )
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


# === ArcFace 5-point template cho 112x112 ===
# Đây là template CHUẨN của InsightFace. Pretrained weights của ArcFace
# được train với template này, nên BẮT BUỘC phải dùng đúng.
# Thứ tự: [mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải]
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


class FaceAligner:
    """
    Detect và align khuôn mặt về kích thước chuẩn (112x112 mặc định).

    Dùng MTCNN để detect bbox + 5 landmarks, sau đó tính affine transform
    để warp khuôn mặt về template 5 điểm chuẩn của ArcFace.
    """

    def __init__(
        self,
        image_size: int = 112,
        device: str = "cpu",
        min_face_size: int = 40,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        select_largest: bool = True,
    ):
        """
        Args:
            image_size: Kích thước output vuông (112 cho ArcFace).
            device: 'cuda' hoặc 'cpu'. Với 'cuda' MTCNN sẽ chạy GPU.
            min_face_size: Bỏ qua các mặt nhỏ hơn ngưỡng này (pixels).
            thresholds: Confidence thresholds cho 3 stages của MTCNN (P/R/O-Net).
                        Giá trị mặc định (0.6, 0.7, 0.7) là chuẩn của paper.
            select_largest: True = chọn mặt có bbox lớn nhất khi nhiều mặt.
                            False = chọn mặt có confidence cao nhất.
        """
        if image_size != 112:
            logger.warning(
                "image_size != 112 - template hiện tại chỉ tính cho 112. "
                "Sẽ scale template tương ứng, nhưng khuyến nghị dùng 112 "
                "để tương thích với pretrained ArcFace weights."
            )

        self.image_size = image_size
        self.device = torch.device(device)
        self.min_face_size = min_face_size

        # Scale template theo image_size nếu khác 112
        scale = image_size / 112.0
        self.template = ARCFACE_TEMPLATE_112 * scale

        # Khởi tạo MTCNN từ facenet-pytorch.
        # keep_all=True để lấy tất cả mặt, sau đó tự chọn theo select_largest.
        # post_process=False để nhận raw pixel values (không normalize).
        self.detector = MTCNN(
            image_size=image_size,
            margin=0,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=0.709,
            post_process=False,
            keep_all=True,
            device=self.device,
        )
        self.select_largest = select_largest

    # ------------------------------------------------------------------ #
    # Core methods
    # ------------------------------------------------------------------ #

    def detect(
        self, image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Detect khuôn mặt chính trong ảnh.

        Args:
            image: numpy array RGB, shape (H, W, 3), dtype uint8.

        Returns:
            (bbox, landmarks, confidence) hoặc None nếu không detect được.
            - bbox: (4,) array [x1, y1, x2, y2]
            - landmarks: (5, 2) array, thứ tự theo ARCFACE_TEMPLATE
            - confidence: float [0, 1]
        """
        # facenet-pytorch MTCNN nhận PIL Image hoặc numpy array
        # Input phải là RGB (không phải BGR từ cv2)
        if image.ndim != 3 or image.shape[2] != 3:
            logger.debug("Ảnh không phải RGB 3-channel, skip")
            return None

        pil_image = Image.fromarray(image)

        # detect() trả về: (boxes, probs, landmarks)
        # Nếu không detect được: tất cả đều None
        try:
            boxes, probs, landmarks = self.detector.detect(
                pil_image, landmarks=True
            )
        except Exception as e:
            logger.debug(f"MTCNN detect failed: {e}")
            return None

        if boxes is None or len(boxes) == 0:
            return None

        # Chọn mặt tốt nhất khi có nhiều mặt
        if len(boxes) > 1:
            if self.select_largest:
                # Bbox area = (x2-x1) * (y2-y1)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = int(np.argmax(areas))
            else:
                idx = int(np.argmax(probs))
        else:
            idx = 0

        bbox = boxes[idx].astype(np.float32)
        landmark = landmarks[idx].astype(np.float32)  # shape (5, 2)
        confidence = float(probs[idx])

        # Kiểm tra bbox hợp lệ
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        if face_width < self.min_face_size or face_height < self.min_face_size:
            logger.debug(
                f"Face quá nhỏ ({face_width:.0f}x{face_height:.0f}), skip"
            )
            return None

        return bbox, landmark, confidence

    def align_from_landmarks(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Align ảnh dựa trên 5 landmarks cho sẵn (dùng cho advanced use case).

        Tính affine transform (chỉ rotation + translation + uniform scale,
        KHÔNG shear) từ landmarks -> template, rồi warp ảnh.

        Args:
            image: numpy array RGB (H, W, 3).
            landmarks: (5, 2) array theo thứ tự ARCFACE_TEMPLATE.

        Returns:
            aligned: (image_size, image_size, 3) RGB uint8.
        """
        # estimateAffinePartial2D: tìm similarity transform (4 DOF)
        # Ổn định hơn estimateAffine2D (6 DOF) vì không bị shear
        # LMEDS method để robust với outliers
        src = landmarks.astype(np.float32)
        dst = self.template.astype(np.float32)

        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

        if matrix is None:
            # Fallback: thử method khác
            matrix, _ = cv2.estimateAffinePartial2D(src, dst)
            if matrix is None:
                raise RuntimeError(
                    "Không tính được affine transform từ landmarks"
                )

        aligned = cv2.warpAffine(
            image,
            matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return aligned

    def align(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect + align trong 1 bước.

        Args:
            image: numpy array RGB (H, W, 3), uint8.

        Returns:
            aligned: (image_size, image_size, 3) hoặc None nếu detect fail.
        """
        detection = self.detect(image)
        if detection is None:
            return None
        _, landmarks, _ = detection
        return self.align_from_landmarks(image, landmarks)

    # ------------------------------------------------------------------ #
    # Convenience methods
    # ------------------------------------------------------------------ #

    def align_from_path(self, image_path: Path) -> Optional[np.ndarray]:
        """Load ảnh từ file và align. Return None nếu fail."""
        image = _load_image_rgb(Path(image_path))
        if image is None:
            return None
        return self.align(image)


# ---------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------- #


def _load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """
    Load ảnh từ disk, trả về RGB numpy array uint8.

    Dùng PIL thay vì cv2 để tránh xử lý BGR/RGB lộn xộn, và PIL
    xử lý nhiều format hơn (gif, webp, etc.).
    """
    try:
        with Image.open(path) as img:
            # Convert về RGB (xử lý các mode khác: RGBA, L, P, ...)
            img = img.convert("RGB")
            return np.array(img, dtype=np.uint8)
    except Exception as e:
        logger.debug(f"Không load được ảnh {path}: {e}")
        return None


def _save_image_rgb(image: np.ndarray, path: Path, quality: int = 95) -> None:
    """Save RGB numpy array ra file JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=quality, optimize=True)


# ---------------------------------------------------------------------- #
# Batch processing
# ---------------------------------------------------------------------- #


def _collect_image_paths(
    root: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> List[Path]:
    """
    Thu thập tất cả file ảnh trong root (recursive).

    Cấu trúc mong đợi (LFW-style):
        root/
            identity_1/
                img1.jpg
                img2.jpg
            identity_2/
                img1.jpg
            ...

    Return: list các Path đã sort để đảm bảo reproducibility.
    """
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(root.rglob(f"*{ext}"))
        paths.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    image_size: int = 112,
    device: str = "cuda",
    min_face_size: int = 40,
    failed_log_path: Optional[Path] = None,
    skip_existing: bool = True,
) -> dict:
    """
    Preprocess toàn bộ dataset: detect + align + save.

    Giữ nguyên cấu trúc thư mục identity:
        input_dir/Aaron_Peirsol/Aaron_Peirsol_0001.jpg
        -> output_dir/Aaron_Peirsol/Aaron_Peirsol_0001.jpg

    Args:
        input_dir: Thư mục dataset gốc.
        output_dir: Nơi save ảnh đã align.
        image_size: Kích thước output (default 112).
        device: 'cuda' hoặc 'cpu'.
        min_face_size: Bỏ qua mặt nhỏ hơn ngưỡng.
        failed_log_path: File log các ảnh fail. Default: output_dir/failed.txt.
        skip_existing: True = skip ảnh đã có trong output_dir (resume support).

    Returns:
        stats dict: {total, success, failed, skipped}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory không tồn tại: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if failed_log_path is None:
        failed_log_path = output_dir / "failed.txt"

    # Device handling: tự fallback về CPU nếu không có CUDA
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA không khả dụng, fallback về CPU")
        device = "cpu"

    logger.info(f"Khởi tạo FaceAligner (device={device}, size={image_size})")
    aligner = FaceAligner(
        image_size=image_size,
        device=device,
        min_face_size=min_face_size,
    )

    # Thu thập ảnh input
    logger.info(f"Scan ảnh trong {input_dir}...")
    image_paths = _collect_image_paths(input_dir)
    logger.info(f"Tìm được {len(image_paths)} ảnh")

    if len(image_paths) == 0:
        logger.warning("Không có ảnh nào trong input_dir!")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    # Process từng ảnh
    stats = {"total": len(image_paths), "success": 0, "failed": 0, "skipped": 0}
    failed_paths: List[Path] = []

    pbar = tqdm(image_paths, desc="Aligning faces", unit="img")
    for src_path in pbar:
        # Tạo output path: giữ nguyên relative structure
        rel_path = src_path.relative_to(input_dir)
        dst_path = output_dir / rel_path.with_suffix(".jpg")

        # Skip nếu đã tồn tại (hỗ trợ resume)
        if skip_existing and dst_path.exists():
            stats["skipped"] += 1
            continue

        # Load + align
        image = _load_image_rgb(src_path)
        if image is None:
            failed_paths.append(src_path)
            stats["failed"] += 1
            continue

        try:
            aligned = aligner.align(image)
        except Exception as e:
            logger.debug(f"Align exception {src_path}: {e}")
            aligned = None

        if aligned is None:
            failed_paths.append(src_path)
            stats["failed"] += 1
            continue

        # Save
        try:
            _save_image_rgb(aligned, dst_path)
            stats["success"] += 1
        except Exception as e:
            logger.warning(f"Save failed {dst_path}: {e}")
            failed_paths.append(src_path)
            stats["failed"] += 1

        # Update progress bar với stats
        pbar.set_postfix(
            ok=stats["success"],
            fail=stats["failed"],
            skip=stats["skipped"],
        )

    # Log failed paths ra file để debug
    if failed_paths:
        failed_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_log_path, "w", encoding="utf-8") as f:
            for p in failed_paths:
                f.write(f"{p}\n")
        logger.info(
            f"Đã lưu danh sách {len(failed_paths)} ảnh fail vào {failed_log_path}"
        )

    # Summary
    logger.info("=" * 50)
    logger.info("Preprocessing hoàn tất:")
    logger.info(f"  Tổng:       {stats['total']}")
    if stats["total"] > 0:
        logger.info(
            f"  Thành công: {stats['success']} "
            f"({stats['success'] / stats['total'] * 100:.1f}%)"
        )
    logger.info(f"  Fail:       {stats['failed']}")
    logger.info(f"  Skip:       {stats['skipped']}")
    logger.info("=" * 50)

    return stats