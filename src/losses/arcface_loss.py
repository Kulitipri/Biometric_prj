"""
ArcFace Loss.

Về bản chất là CrossEntropyLoss áp lên logits từ ArcFaceHead.
File này tách riêng để dễ mở rộng sang CosFace, SphereFace sau.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    Wrapper cho CrossEntropy áp lên ArcFace logits.

    Có thể thêm label smoothing để regularize.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) từ ArcFaceHead
            labels: (B,) ground truth

        Returns:
            scalar loss
        """
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)


# TODO (optional): Thêm CosFaceLoss, TripletLoss ở đây nếu muốn experiment
