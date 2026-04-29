"""
Backbone Networks cho Face Recognition.

Hỗ trợ:
- MobileNetV2 (primary, nhẹ, ~3.5M params)
- MobileNetV3-Small (~2.5M params, nhẹ nhất)
- MobileNetV3-Large (~5.5M params, có SE blocks - tốt cho occluded faces)
- ResNet50 (baseline để so sánh, ~25M params)

Tất cả backbone đều output embedding vector chưa normalize, kích thước embedding_dim.
L2 normalization được làm ở chỗ khác (trong ArcFace head hoặc inference).

Pattern theo InsightFace:
    Conv layers → Flatten → FC → BN1d → Embedding
                                    └─> output cuối, KHÔNG có activation
"""

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


# Type alias cho backbone names được hỗ trợ
BackboneName = Literal[
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet50",
]


# Mapping: backbone name -> (constructor, weights_enum, last_channel_dim)
# last_channel_dim = số channels output của feature extractor (trước classifier)
_BACKBONE_REGISTRY = {
    "mobilenet_v2": {
        "constructor": models.mobilenet_v2,
        "weights": models.MobileNet_V2_Weights.IMAGENET1K_V1,
        "last_channel": 1280,
    },
    "mobilenet_v3_small": {
        "constructor": models.mobilenet_v3_small,
        "weights": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "last_channel": 576,
    },
    "mobilenet_v3_large": {
        "constructor": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "last_channel": 960,
    },
    "resnet50": {
        "constructor": models.resnet50,
        "weights": models.ResNet50_Weights.IMAGENET1K_V2,
        "last_channel": 2048,
    },
}


class FaceEmbeddingNet(nn.Module):
    """
    Wrapper đồng bộ các backbone thành embedding network.

    Pipeline:
        image (B, 3, 112, 112)
            -> [Backbone feature extractor]
            -> features (B, C, H', W')
            -> [Adaptive Pool to 7x7] (đảm bảo size cố định)
            -> [Flatten + Dropout]
            -> [FC: C*7*7 -> embedding_dim]
            -> [BatchNorm1d]
            -> embedding (B, embedding_dim) -- CHƯA normalize

    Lưu ý: BN1d ở cuối là chuẩn của InsightFace, không thay bằng activation.
    L2 normalize sẽ được làm ở chỗ khác (ArcFace head hoặc inference).

    Args:
        backbone_name: Tên backbone. Xem BackboneName.
        embedding_dim: Chiều embedding (thường 128/256/512). Default 512.
        pretrained: True = load ImageNet pretrained weights.
        dropout: Dropout rate trước FC layer (default 0.2).
        feature_size: Spatial size sau adaptive pool. Default 7.
            Với input 112x112: backbone output ~ 4x4 (mobilenet) hoặc 4x4 (resnet),
            adaptive pool sẽ pool về feature_size x feature_size.
    """

    def __init__(
        self,
        backbone_name: BackboneName = "mobilenet_v2",
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.2,
        feature_size: int = 7,
    ):
        super().__init__()

        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Backbone không hỗ trợ: {backbone_name}. "
                f"Available: {list(_BACKBONE_REGISTRY.keys())}"
            )

        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.feature_size = feature_size

        spec = _BACKBONE_REGISTRY[backbone_name]
        last_channel = spec["last_channel"]

        # === Build backbone (loại bỏ classifier cuối) ===
        weights = spec["weights"] if pretrained else None
        backbone_full = spec["constructor"](weights=weights)

        # Mỗi family có cách lấy feature extractor khác nhau
        if backbone_name.startswith("mobilenet"):
            # MobileNet: backbone.features là Sequential conv layers
            # backbone.classifier là FC head -> ta bỏ đi
            self.features = backbone_full.features
        elif backbone_name == "resnet50":
            # ResNet: lấy tất cả layers trừ avgpool và fc cuối
            # children() returns: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
            modules = list(backbone_full.children())[:-2]  # bỏ avgpool và fc
            self.features = nn.Sequential(*modules)
        else:
            # Defensive: branch không nên bao giờ hit nếu registry đúng
            raise NotImplementedError(f"Chưa handle backbone: {backbone_name}")

        # === Adaptive pool về spatial size cố định ===
        # Đảm bảo embedding head luôn nhận shape giống nhau bất kể input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((feature_size, feature_size))

        # === Embedding head ===
        # Pattern InsightFace: Flatten -> Dropout -> FC -> BN1d
        # KHÔNG có activation cuối (để FC + BN làm projection thuần)
        flat_dim = last_channel * feature_size * feature_size
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(flat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Init weights cho FC layer (pretrained backbone đã có weights)
        self._init_embedding_head()

        logger.info(
            f"FaceEmbeddingNet: {backbone_name} "
            f"(pretrained={pretrained}, embedding_dim={embedding_dim}, "
            f"flat_dim={flat_dim}, params={self._count_params():,})"
        )

    def _init_embedding_head(self) -> None:
        """Khởi tạo weights cho FC layer mới (pretrained backbone giữ nguyên)."""
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                # Xavier init giống InsightFace
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _count_params(self) -> int:
        """Đếm số trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - khuyến nghị 112x112 cho ArcFace.

        Returns:
            embedding: (B, embedding_dim) - CHƯA L2 normalize.
                Normalize được làm ở ArcFace head (training)
                hoặc extract_embedding() (inference).
        """
        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Input phải 4D (B,C,H,W), nhận shape {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Input phải 3 channels (RGB), nhận {x.shape[1]}")

        # Feature extraction
        features = self.features(x)         # (B, last_channel, H', W')
        features = self.adaptive_pool(features)  # (B, last_channel, 7, 7)

        # Embedding projection
        embedding = self.embedding_head(features)  # (B, embedding_dim)
        return embedding

    def freeze_backbone(self) -> None:
        """Đóng băng backbone (chỉ train embedding head). Dùng cho stage 1."""
        for param in self.features.parameters():
            param.requires_grad = False
        # adaptive_pool không có params nên không cần freeze
        logger.info(f"Đã freeze backbone {self.backbone_name}")

    def unfreeze_backbone(self) -> None:
        """Mở khóa backbone để fine-tune toàn bộ. Dùng cho stage 2."""
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info(f"Đã unfreeze backbone {self.backbone_name}")


def build_backbone(config: dict) -> FaceEmbeddingNet:
    """
    Factory function tạo backbone từ config dict.

    Args:
        config: Dict với keys (xem configs/model/*.yaml):
            - backbone (str): tên backbone
            - embedding_dim (int): chiều embedding
            - pretrained (bool): load ImageNet weights
            - dropout (float): dropout rate

    Returns:
        FaceEmbeddingNet đã khởi tạo.
    """
    return FaceEmbeddingNet(
        backbone_name=config.get("backbone", "mobilenet_v2"),
        embedding_dim=config.get("embedding_dim", 512),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.2),
        feature_size=config.get("feature_size", 7),
    )