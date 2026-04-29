"""
Unit tests cho models.

Chạy:
    pytest tests/
"""

import pytest
import torch


class TestBackbone:
    """Tests cho backbone networks."""

    @pytest.mark.skip(reason="TODO: Implement khi backbone xong")
    def test_mobilenet_v2_output_shape(self):
        """Kiểm tra output shape của MobileNetV2."""
        from src.models.backbone import FaceEmbeddingNet

        model = FaceEmbeddingNet(backbone_name="mobilenet_v2", embedding_dim=512)
        x = torch.randn(2, 3, 112, 112)
        out = model(x)
        assert out.shape == (2, 512)

    @pytest.mark.skip(reason="TODO: Implement khi backbone xong")
    def test_different_embedding_dims(self):
        """Thử với các embedding_dim khác nhau."""
        from src.models.backbone import FaceEmbeddingNet

        for dim in [128, 256, 512]:
            model = FaceEmbeddingNet(embedding_dim=dim)
            x = torch.randn(1, 3, 112, 112)
            assert model(x).shape == (1, dim)


class TestArcFaceHead:
    """Tests cho ArcFace head."""

    @pytest.mark.skip(reason="TODO: Implement khi head xong")
    def test_output_shape(self):
        from src.models.arcface_head import ArcFaceHead

        head = ArcFaceHead(embedding_dim=512, num_classes=100)
        embeddings = torch.randn(4, 512)
        labels = torch.tensor([0, 1, 2, 3])
        logits = head(embeddings, labels)
        assert logits.shape == (4, 100)


# TODO: Thêm tests cho dataset, metrics, losses
