"""
Synthetic Mask Augmentation Module.

Tạo ảnh đeo khẩu trang synthetic bằng cách VẼ HÌNH HỌC lên ảnh đã align.

Tại sao vẽ hình học (không overlay PNG)?
- Ảnh input đã được align về template ArcFace 112x112 cố định
  → biết chính xác vị trí mũi/miệng mà không cần re-detect landmarks
- Không cần asset PNG, không thêm dependency
- Mục đích là TRAIN model học pattern "vùng dưới bị che" -
  không cần mask trông thật 100%

Các module:
- MaskAugmenter: vẽ mask theo nhiều style (surgical, n95, cloth, black)
- RandomOcclusion: CutOut - vẽ rectangle ngẫu nhiên (mô phỏng occlusion bất kỳ)

Usage:
    from src.data.mask_augment import MaskAugmenter

    augmenter = MaskAugmenter(probability=0.5)
    masked_image = augmenter(image, landmarks=None)  # landmarks không cần
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ====================================================================== #
# Geometry constants (cho ảnh đã align về ArcFace template 112x112)
# ====================================================================== #
# Đây là các điểm tham chiếu đã biết trước, không cần detect lại.
# Tham khảo: ARCFACE_TEMPLATE_112 trong preprocessing.py

# Landmarks gốc (đã round)
_NOSE = (56, 72)
_MOUTH_LEFT = (42, 92)
_MOUTH_RIGHT = (71, 92)

# Bounding box vùng mask phủ (top, bottom, left, right)
# Phủ từ giữa mũi xuống hết cạnh dưới, rộng hơn 2 khóe miệng 1 chút
_MASK_TOP_Y = 60       # qua sống mũi
_MASK_BOTTOM_Y = 112   # cạnh dưới ảnh
_MASK_LEFT_X = 22
_MASK_RIGHT_X = 90

# Image size mặc định (project dùng 112x112)
_DEFAULT_IMAGE_SIZE = 112


# Color palette cho các loại mask phổ biến (RGB)
_PRESET_COLORS = {
    "surgical": [(168, 213, 186), (140, 200, 220), (220, 220, 230)],  # xanh y tế
    "n95":      [(245, 245, 245), (220, 220, 220), (200, 200, 200)],  # trắng/xám
    "cloth":    [(50, 50, 50), (80, 60, 100), (130, 60, 60),          # đa dạng
                 (60, 100, 130), (200, 150, 100)],
    "black":    [(15, 15, 15), (30, 30, 30)],
}

# Các style mask available
_AVAILABLE_STYLES = list(_PRESET_COLORS.keys())


# ====================================================================== #
# MaskAugmenter
# ====================================================================== #


class MaskAugmenter:
    """
    Vẽ mask synthetic lên ảnh khuôn mặt đã align.

    Lợi dụng việc ảnh đã được align về template ArcFace cố định nên
    không cần landmarks: vị trí mũi/miệng đã biết trước.

    Args:
        probability: Xác suất apply mask (cho random augmentation trong DataLoader).
            Default 0.5 = 50% ảnh sẽ có mask.
        styles: List style mask được phép random chọn. Default = tất cả.
            Available: 'surgical', 'n95', 'cloth', 'black'.
        random_color: True = random màu trong palette của style.
            False = lấy màu đầu tiên của style (deterministic).
        add_noise: True = thêm Gaussian noise nhẹ lên vùng mask cho realistic.
        add_strings: True = vẽ dây đeo mask ở 2 bên tai.
        image_size: Kích thước ảnh (default 112). Nếu khác sẽ scale geometry.
        random_seed: Set seed cho deterministic output (cho debug/test).
            None = mỗi lần gọi sẽ random khác nhau.
    """

    def __init__(
        self,
        probability: float = 0.5,
        styles: Optional[List[str]] = None,
        random_color: bool = True,
        add_noise: bool = True,
        add_strings: bool = True,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        random_seed: Optional[int] = None,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability phải trong [0, 1], nhận {probability}")

        # Validate styles
        if styles is None:
            styles = _AVAILABLE_STYLES.copy()
        invalid = set(styles) - set(_AVAILABLE_STYLES)
        if invalid:
            raise ValueError(
                f"Style không hợp lệ: {invalid}. "
                f"Available: {_AVAILABLE_STYLES}"
            )

        self.probability = probability
        self.styles = styles
        self.random_color = random_color
        self.add_noise = add_noise
        self.add_strings = add_strings
        self.image_size = image_size

        # Scale các geometry constants theo image_size
        self._scale = image_size / _DEFAULT_IMAGE_SIZE
        self._nose = self._scale_point(_NOSE)
        self._mouth_left = self._scale_point(_MOUTH_LEFT)
        self._mouth_right = self._scale_point(_MOUTH_RIGHT)
        self._mask_top = int(_MASK_TOP_Y * self._scale)
        self._mask_bottom = int(_MASK_BOTTOM_Y * self._scale)
        self._mask_left = int(_MASK_LEFT_X * self._scale)
        self._mask_right = int(_MASK_RIGHT_X * self._scale)

        # Random state riêng để không ảnh hưởng global numpy random
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Random apply mask với xác suất self.probability.

        Args:
            image: RGB numpy array (H, W, 3), uint8.
            landmarks: Không sử dụng (giữ tương thích với interface chung).
                Vì ảnh đã align, landmarks luôn ở vị trí cố định.

        Returns:
            Ảnh có hoặc không có mask (uint8).
        """
        if self._rng.random() >= self.probability:
            return image

        style = self._rng.choice(self.styles)
        return self.apply_mask(image, style=style)

    def apply_mask(
        self,
        image: np.ndarray,
        style: Optional[str] = None,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Vẽ mask lên ảnh (deterministic nếu style và color được chỉ định).

        Args:
            image: RGB array (H, W, 3), uint8.
            style: 'surgical', 'n95', 'cloth', 'black'. None = random.
            color: RGB tuple. None = lấy từ palette của style.

        Returns:
            Ảnh đã apply mask (copy, không modify input).
        """
        if image.dtype != np.uint8:
            raise ValueError(f"image phải là uint8, nhận {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image phải là RGB (H, W, 3), nhận {image.shape}")

        # Pick style
        if style is None:
            style = self._rng.choice(self.styles)
        elif style not in _AVAILABLE_STYLES:
            raise ValueError(f"Style không hợp lệ: {style}")

        # Pick color
        if color is None:
            palette = _PRESET_COLORS[style]
            if self.random_color:
                color = tuple(palette[self._rng.integers(0, len(palette))])
            else:
                color = palette[0]

        # Dispatch theo style
        # Mỗi style có hình dạng polygon hơi khác
        output = image.copy()
        if style == "surgical":
            output = self._draw_surgical(output, color)
        elif style == "n95":
            output = self._draw_n95(output, color)
        elif style == "cloth":
            output = self._draw_cloth(output, color)
        elif style == "black":
            output = self._draw_black(output, color)

        # Optional: dây đeo
        if self.add_strings:
            output = self._draw_strings(output, color)

        return output

    # ------------------------------------------------------------------ #
    # Private: draw functions cho từng style
    # ------------------------------------------------------------------ #

    def _draw_surgical(
        self, image: np.ndarray, color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Surgical mask: hình thang cong nhẹ, phủ mũi-cằm.

        Polygon 6 điểm tạo hình thang với cạnh trên cong xuống ở giữa
        (mô phỏng phần ôm sống mũi).
        """
        # Polygon: top-left, top-mid (lõm xuống), top-right, bottom-right, bottom-left
        nose_x, nose_y = self._nose
        polygon = np.array([
            [self._mask_left, self._mask_top + 5],          # top-left
            [nose_x, self._mask_top - 2],                   # top-mid (cao nhất)
            [self._mask_right, self._mask_top + 5],         # top-right
            [self._mask_right + 2, self._mask_bottom],      # bottom-right
            [self._mask_left - 2, self._mask_bottom],       # bottom-left
        ], dtype=np.int32)

        return self._fill_and_decorate(image, polygon, color, add_pleats=True)

    def _draw_n95(
        self, image: np.ndarray, color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        N95 mask: hình lồi 3D, đỉnh nhô ra ở giữa.

        Vẽ ellipse + polygon để tạo cảm giác phồng.
        """
        nose_x, nose_y = self._nose
        polygon = np.array([
            [self._mask_left + 5, self._mask_top + 8],
            [nose_x - 8, self._mask_top - 3],
            [nose_x, self._mask_top - 5],
            [nose_x + 8, self._mask_top - 3],
            [self._mask_right - 5, self._mask_top + 8],
            [self._mask_right - 2, self._mask_bottom - 5],
            [nose_x, self._mask_bottom + 2],
            [self._mask_left + 2, self._mask_bottom - 5],
        ], dtype=np.int32)

        output = self._fill_and_decorate(image, polygon, color, add_pleats=False)

        # Vẽ vạch ngang ở giữa mask (đặc trưng N95)
        line_y = (self._mask_top + self._mask_bottom) // 2
        cv2.line(
            output,
            (self._mask_left + 8, line_y),
            (self._mask_right - 8, line_y),
            self._darken(color, 0.85),
            1,
        )
        return output

    def _draw_cloth(
        self, image: np.ndarray, color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Cloth mask: hình rộng hơn, ít nếp gấp.
        """
        nose_x, _ = self._nose
        polygon = np.array([
            [self._mask_left - 3, self._mask_top + 8],
            [nose_x, self._mask_top + 2],
            [self._mask_right + 3, self._mask_top + 8],
            [self._mask_right, self._mask_bottom],
            [self._mask_left, self._mask_bottom],
        ], dtype=np.int32)

        return self._fill_and_decorate(image, polygon, color, add_pleats=False)

    def _draw_black(
        self, image: np.ndarray, color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Black mask: polygon đơn giản nhất, không decoration."""
        nose_x, _ = self._nose
        polygon = np.array([
            [self._mask_left, self._mask_top + 3],
            [nose_x, self._mask_top - 2],
            [self._mask_right, self._mask_top + 3],
            [self._mask_right, self._mask_bottom],
            [self._mask_left, self._mask_bottom],
        ], dtype=np.int32)

        return self._fill_and_decorate(image, polygon, color, add_pleats=False)

    # ------------------------------------------------------------------ #
    # Private: helpers
    # ------------------------------------------------------------------ #

    def _fill_and_decorate(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
        color: Tuple[int, int, int],
        add_pleats: bool = False,
    ) -> np.ndarray:
        """Fill polygon + optional decoration (noise, pleats)."""
        # Tạo mask binary để biết vùng cần vẽ
        mask_binary = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_binary, [polygon], 255)

        # Fill màu
        cv2.fillPoly(image, [polygon], color)

        # Optional: thêm Gaussian noise lên vùng mask cho realistic
        if self.add_noise:
            noise = self._rng.normal(0, 8, image.shape).astype(np.int16)
            # Chỉ apply noise lên vùng mask (mask_binary > 0)
            mask_3ch = np.repeat(mask_binary[:, :, None], 3, axis=2) > 0
            image_int = image.astype(np.int16)
            image_int[mask_3ch] = np.clip(
                image_int[mask_3ch] + noise[mask_3ch], 0, 255
            )
            image = image_int.astype(np.uint8)

        # Optional: vẽ pleats (nếp gấp) cho surgical mask
        if add_pleats:
            self._draw_pleats(image, polygon, color)

        # Vẽ viền nhẹ để mask rõ ràng hơn
        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=self._darken(color, 0.7),
            thickness=1,
        )

        return image

    def _draw_pleats(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
        color: Tuple[int, int, int],
    ) -> None:
        """Vẽ 2-3 đường ngang mô phỏng nếp gấp surgical mask (in-place)."""
        # Lấy bbox của polygon
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)

        # Vẽ 3 đường ngang ở 1/4, 2/4, 3/4 chiều cao
        darker = self._darken(color, 0.85)
        height = y_max - y_min
        for i in range(1, 4):
            y = y_min + height * i // 4
            cv2.line(
                image,
                (x_min + 4, y),
                (x_max - 4, y),
                darker,
                thickness=1,
            )

    def _draw_strings(
        self, image: np.ndarray, color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Vẽ dây đeo mask ở 2 bên tai.

        Đường cong nhẹ từ góc mask ra cạnh ảnh.
        """
        # Dùng màu hơi tối hơn mask cho dây đeo
        string_color = self._darken(color, 0.6)
        thickness = max(1, int(self._scale))

        # Dây trái: từ mask trên-trái cong ra phía trái cạnh ảnh
        cv2.line(
            image,
            (self._mask_left, self._mask_top + 8),
            (0, self._mask_top - 5),
            string_color,
            thickness,
        )
        # Dây dưới-trái
        cv2.line(
            image,
            (self._mask_left, self._mask_bottom - 8),
            (0, self._mask_bottom - 15),
            string_color,
            thickness,
        )

        # Dây phải: tương tự ở phía phải
        cv2.line(
            image,
            (self._mask_right, self._mask_top + 8),
            (self.image_size - 1, self._mask_top - 5),
            string_color,
            thickness,
        )
        cv2.line(
            image,
            (self._mask_right, self._mask_bottom - 8),
            (self.image_size - 1, self._mask_bottom - 15),
            string_color,
            thickness,
        )

        return image

    @staticmethod
    def _darken(
        color: Tuple[int, int, int], factor: float
    ) -> Tuple[int, int, int]:
        """Làm tối màu đi factor lần (0.5 = nửa tối)."""
        return tuple(int(c * factor) for c in color)

    def _scale_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Scale 1 điểm từ template 112 sang image_size hiện tại."""
        return (int(point[0] * self._scale), int(point[1] * self._scale))


# ====================================================================== #
# RandomOcclusion (CutOut)
# ====================================================================== #


class RandomOcclusion:
    """
    Random vẽ rectangle đặc lên ảnh để mô phỏng occlusion bất kỳ.

    Bổ trợ cho MaskAugmenter: trong khi mask chỉ che vùng dưới khuôn mặt,
    RandomOcclusion mô phỏng các occlusion khác (kính râm, tay, vật cản).

    Reference:
        DeVries & Taylor, "Improved Regularization of Convolutional
        Neural Networks with Cutout" (2017).

    Args:
        probability: Xác suất apply.
        max_size_ratio: Kích thước max của rectangle so với cạnh ảnh (0-1).
        min_size_ratio: Kích thước min.
        fill_value: Giá trị fill (0 = đen, hoặc tuple RGB, hoặc 'random').
        random_seed: Seed cho reproducibility.
    """

    def __init__(
        self,
        probability: float = 0.3,
        max_size_ratio: float = 0.3,
        min_size_ratio: float = 0.1,
        fill_value: object = 0,
        random_seed: Optional[int] = None,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability phải trong [0, 1], nhận {probability}")
        if not 0.0 < min_size_ratio <= max_size_ratio <= 1.0:
            raise ValueError(
                f"size ratios phải thỏa: 0 < min={min_size_ratio} "
                f"<= max={max_size_ratio} <= 1"
            )

        self.probability = probability
        self.max_size_ratio = max_size_ratio
        self.min_size_ratio = min_size_ratio
        self.fill_value = fill_value
        self._rng = np.random.default_rng(random_seed)

    def __call__(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Random apply CutOut."""
        if self._rng.random() >= self.probability:
            return image

        h, w = image.shape[:2]
        # Random size
        size_ratio = self._rng.uniform(self.min_size_ratio, self.max_size_ratio)
        rect_h = int(h * size_ratio)
        rect_w = int(w * size_ratio)

        # Random position (đảm bảo rectangle nằm trong ảnh)
        y = self._rng.integers(0, max(1, h - rect_h))
        x = self._rng.integers(0, max(1, w - rect_w))

        # Determine fill
        if self.fill_value == "random":
            fill = self._rng.integers(0, 256, size=3).tolist()
        elif isinstance(self.fill_value, (int, float)):
            fill = [int(self.fill_value)] * 3
        else:
            fill = list(self.fill_value)

        # Apply (copy để không modify input)
        output = image.copy()
        output[y:y + rect_h, x:x + rect_w] = fill

        return output