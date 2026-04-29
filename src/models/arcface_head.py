"""
ArcFace Classification Head.

Reference:
    Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    CVPR 2019. https://arxiv.org/abs/1801.07698

Ý tưởng:
    1. Normalize cả weight matrix W và embedding x → mọi thứ trên hypersphere
    2. cos(θ) = W^T x (vì cả 2 đều unit-norm)
    3. Thêm additive angular margin m vào target class:
        cos(θ + m) cho ground-truth class
        cos(θ)     cho các class khác
    4. Scale bằng s (radius hypersphere, thường 64):
        logits = s * cos(...)
    5. Apply CrossEntropyLoss như bình thường

Tại sao thêm margin?
    Margin tạo "khoảng cách an toàn" giữa các class trên hypersphere,
    ép model học embedding tách biệt rõ ràng hơn so với softmax thường.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ArcFaceHead(nn.Module):
    """
    ArcFace head với additive angular margin.

    Training:
        input  : embedding (B, embedding_dim) chưa normalize
                 labels    (B,) ground truth class indices
        output : logits    (B, num_classes) cho CrossEntropyLoss

    Inference:
        Không cần dùng head này. Chỉ cần embedding L2-normalized.

    Args:
        embedding_dim: Chiều embedding từ backbone.
        num_classes: Số identities trong training set.
        s: Scale factor (radius hypersphere). Paper dùng 64.
        m: Angular margin (radians). Paper dùng 0.5 (≈28.6 độ).
        easy_margin: Variant ổn định hơn ở đầu training.
            False (default) = công thức gốc, có check θ + m vượt π
            True = chỉ apply margin khi cos(θ) > 0
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()

        if embedding_dim <= 0 or num_classes <= 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) và num_classes ({num_classes}) "
                f"phải > 0"
            )
        if s <= 0:
            raise ValueError(f"s phải > 0, nhận {s}")
        if not 0 < m < math.pi / 2:
            # m phải thuộc (0, π/2) để cos(θ+m) còn đơn điệu
            raise ValueError(f"m phải trong (0, π/2), nhận {m}")

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # === Weight matrix W (embedding_dim, num_classes) ===
        # Mỗi class có 1 vector trong embedding space.
        # Sẽ được L2-normalized trong forward để mỗi class nằm trên hypersphere.
        self.weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        nn.init.xavier_normal_(self.weight)

        # === Pre-compute constants ===
        # Tránh tính lặp lại trong mỗi forward pass.
        # cos(θ + m) = cos θ * cos m - sin θ * sin m
        self.register_buffer("cos_m", torch.tensor(math.cos(m)))
        self.register_buffer("sin_m", torch.tensor(math.sin(m)))

        # Threshold cho non-easy margin: cos(π - m)
        # Khi cos(θ) < cos(π - m), có nghĩa θ + m > π → cos(θ+m) tăng theo θ
        # → margin không còn tác dụng đẩy θ lớn hơn → cần fallback
        self.register_buffer("threshold", torch.tensor(math.cos(math.pi - m)))

        # Penalty fallback khi vượt threshold: cos(π-m) - sin(π-m) * m
        # = cos(π-m) - m*sin(π-m), giữ tính đơn điệu của loss
        self.register_buffer("penalty", torch.tensor(math.sin(math.pi - m) * m))

        logger.info(
            f"ArcFaceHead: dim={embedding_dim}, classes={num_classes}, "
            f"s={s}, m={m} ({math.degrees(m):.1f}°), easy_margin={easy_margin}"
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, embedding_dim) chưa normalize.
            labels: (B,) ground truth class indices, mỗi giá trị trong [0, num_classes).
                Có thể None khi inference - return cosine similarities scaled.

        Returns:
            logits: (B, num_classes) ready cho CrossEntropyLoss.
        """
        # === 1. L2 normalize embedding và weight ===
        # Sau normalize, mỗi vector có norm = 1 → nằm trên hypersphere
        # F.normalize default p=2, dim=1 cho 2D input
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # === 2. Tính cosine similarity matrix ===
        # cos(θ_ij) = <embedding_i, weight_j> (vì cả 2 đều unit-norm)
        # cosine: (B, num_classes)
        cosine = F.linear(embeddings_norm, weight_norm)

        # Clamp để tránh NaN khi tính sqrt(1 - cos^2)
        # cos có thể out-of-range [-1, 1] do floating point error
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Nếu không có labels (inference), return cosine scaled
        if labels is None:
            return cosine * self.s

        # === 3. Tính cos(θ + m) cho TARGET class ===
        # Chỉ thay đổi cosine ở vị trí target, các class khác giữ nguyên cos(θ)
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(θ + m) = cos θ * cos m - sin θ * sin m
        cos_theta_plus_m = cosine * self.cos_m - sine * self.sin_m

        # === 4. Handle edge case: θ + m > π ===
        # Khi đó cos(θ+m) không còn monotonic theo θ → loss không còn chuẩn
        if self.easy_margin:
            # Easy margin: chỉ apply margin khi cos(θ) > 0 (tức θ < π/2)
            # Khi cos(θ) <= 0, fall back về cos(θ) gốc (không apply margin)
            cos_theta_plus_m = torch.where(
                cosine > 0, cos_theta_plus_m, cosine
            )
        else:
            # Non-easy margin (paper version):
            # Khi cos(θ) < cos(π-m), apply linear penalty thay vì cos(θ+m)
            cos_theta_plus_m = torch.where(
                cosine > self.threshold,
                cos_theta_plus_m,
                cosine - self.penalty,
            )

        # === 5. Build output: thay cos(θ) bằng cos(θ+m) chỉ ở target class ===
        # one_hot mask để select target positions
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # output = one_hot * cos(θ+m) + (1 - one_hot) * cos(θ)
        output = (one_hot * cos_theta_plus_m) + ((1.0 - one_hot) * cosine)

        # === 6. Scale by s ===
        return output * self.s

    def extra_repr(self) -> str:
        """Cho repr() đẹp hơn khi print model."""
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"num_classes={self.num_classes}, "
            f"s={self.s}, m={self.m}, easy_margin={self.easy_margin}"
        )