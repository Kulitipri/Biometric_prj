"""
Full Face Recognition Model.

Kết hợp backbone + ArcFace head thành 1 model hoàn chỉnh.

Khác biệt training vs inference:
    Training:  cần cả backbone + head + labels → tính loss
    Inference: chỉ cần backbone + L2 normalize → embedding để verify/identify

Usage:
    # Training
    model = build_model(config, num_classes=10000)
    embeddings, logits = model(images, labels=labels)
    loss = criterion(logits, labels)

    # Inference
    model = build_model(config, num_classes=None)  # không cần head
    embeddings = model.extract_embedding(images)   # đã L2-normalized
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.arcface_head import ArcFaceHead
from src.models.backbone import FaceEmbeddingNet, build_backbone

logger = logging.getLogger(__name__)


class FaceRecognizer(nn.Module):
    """
    Full pipeline: backbone + ArcFace head.

    Forward behavior:
        Có labels và có head → trả về (embedding, logits) cho training
        Không có labels HOẶC không có head → trả về (embedding, None)

    Note: embedding trong forward() chưa được L2-normalized.
    Để lấy embedding cho inference, dùng extract_embedding() để có normalized.

    Args:
        backbone: Embedding network.
        head: ArcFace head. None nếu chỉ dùng cho inference.
    """

    def __init__(
        self,
        backbone: FaceEmbeddingNet,
        head: Optional[ArcFaceHead] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

        # Sanity check: nếu có head, embedding_dim phải khớp
        if head is not None and backbone.embedding_dim != head.embedding_dim:
            raise ValueError(
                f"Mismatch embedding_dim: backbone={backbone.embedding_dim} "
                f"vs head={head.embedding_dim}"
            )

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 3, 112, 112).
            labels: (B,) cho training. None khi inference.

        Returns:
            (embeddings, logits):
                - embeddings: (B, embedding_dim) CHƯA normalize
                - logits: (B, num_classes) hoặc None
        """
        # 1. Extract embedding từ backbone
        embeddings = self.backbone(x)

        # 2. Tính logits qua head (nếu có)
        logits: Optional[torch.Tensor] = None
        if self.head is not None and labels is not None:
            logits = self.head(embeddings, labels)

        return embeddings, logits

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference helper: trả về L2-normalized embedding.

        Tự động set eval mode và disable gradient. Đây là method nên dùng
        cho verification/identification - embedding đã sẵn sàng để
        cosine similarity = dot product.

        Args:
            x: (B, 3, 112, 112).

        Returns:
            embeddings: (B, embedding_dim) đã L2-normalized.
        """
        # Note: với @torch.no_grad() decorator + tự switch eval,
        # method này an toàn để gọi mọi lúc
        was_training = self.training
        self.eval()
        try:
            embeddings = self.backbone(x)
            # L2 normalize cho cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
        finally:
            # Restore mode cũ (quan trọng nếu đang trong training loop)
            if was_training:
                self.train()
        return embeddings

    # ------------------------------------------------------------------ #
    # Convenience methods cho training pipeline
    # ------------------------------------------------------------------ #

    def freeze_backbone(self) -> None:
        """Đóng băng backbone, chỉ train head + embedding layer."""
        self.backbone.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Mở khóa toàn bộ backbone."""
        self.backbone.unfreeze_backbone()

    # ------------------------------------------------------------------ #
    # Checkpoint helpers
    # ------------------------------------------------------------------ #

    def save_for_inference(self, path: Path) -> None:
        """
        Save chỉ backbone weights cho inference (không kèm head).

        Useful cho deployment - file nhỏ hơn, dễ load mà không cần
        biết num_classes của training set.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "backbone_state_dict": self.backbone.state_dict(),
            "backbone_name": self.backbone.backbone_name,
            "embedding_dim": self.backbone.embedding_dim,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved inference checkpoint: {path}")


def build_model(
    config: dict,
    num_classes: Optional[int] = None,
) -> FaceRecognizer:
    """
    Factory function build full model từ config.

    Args:
        config: Dict với 2 sections (xem configs/model/*.yaml):
            - model: backbone config
            - head: ArcFace head config (chỉ cần khi training)
        num_classes: Số identities. Bắt buộc cho training, None cho inference.

    Returns:
        FaceRecognizer đã khởi tạo.

    Examples:
        # Training
        config = yaml.safe_load(open('configs/model/mobilenet_v2_arcface.yaml'))
        model = build_model(config, num_classes=10000)

        # Inference
        model = build_model(config, num_classes=None)
    """
    # 1. Build backbone
    if "model" not in config:
        raise KeyError("Config thiếu key 'model'")
    backbone = build_backbone(config["model"])

    # 2. Build head (nếu có num_classes)
    head: Optional[ArcFaceHead] = None
    if num_classes is not None:
        if "head" not in config:
            raise KeyError("Config thiếu key 'head' khi training (num_classes given)")

        head_cfg = config["head"]
        head_type = head_cfg.get("type", "arcface").lower()
        if head_type != "arcface":
            raise ValueError(
                f"Hiện chỉ hỗ trợ head 'arcface', nhận '{head_type}'"
            )

        head = ArcFaceHead(
            embedding_dim=backbone.embedding_dim,
            num_classes=num_classes,
            s=head_cfg.get("s", 64.0),
            m=head_cfg.get("m", 0.5),
            easy_margin=head_cfg.get("easy_margin", False),
        )

    return FaceRecognizer(backbone=backbone, head=head)