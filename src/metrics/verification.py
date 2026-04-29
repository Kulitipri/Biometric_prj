"""
Verification Metrics.

Bài toán face verification: cho 2 ảnh, quyết định cùng người hay không.
Đầu ra của model là embeddings → tính cosine similarity → so với threshold.

Các metrics chính:
- Accuracy tại best threshold
- TAR@FAR (True Accept Rate tại False Accept Rate cố định) - chuẩn LFW
- EER (Equal Error Rate) - điểm FAR = FRR
- ROC curve + AUC

Định nghĩa terminology:
    Positive pair = cùng người (label=1)
    Negative pair = khác người (label=0)

    TP (True Positive)  = pair cùng người, predict cùng    (đúng accept)
    TN (True Negative)  = pair khác người, predict khác    (đúng reject)
    FP (False Positive) = pair khác người, predict cùng    (sai accept)
    FN (False Negative) = pair cùng người, predict khác    (sai reject)

    TAR (True Accept Rate)  = TP / (TP + FN) = recall trên positive
    FAR (False Accept Rate) = FP / (FP + TN) = false positive rate
    FRR (False Reject Rate) = FN / (TP + FN) = 1 - TAR

    Predict cùng người khi cosine_similarity >= threshold.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)


# ====================================================================== #
# Helpers
# ====================================================================== #


def compute_cosine_similarity(
    embeddings1: np.ndarray, embeddings2: np.ndarray
) -> np.ndarray:
    """
    Tính cosine similarity giữa 2 bộ embeddings.

    Quan trọng: cả 2 đều phải là L2-normalized vectors. Khi đó:
        cosine(a, b) = a · b   (dot product)

    Args:
        embeddings1: (N, D) đã L2 normalized.
        embeddings2: (N, D) đã L2 normalized.

    Returns:
        similarities: (N,) giá trị trong [-1, 1].

    Raises:
        ValueError: nếu shape mismatch.
    """
    if embeddings1.shape != embeddings2.shape:
        raise ValueError(
            f"Shape mismatch: {embeddings1.shape} vs {embeddings2.shape}"
        )
    if embeddings1.ndim != 2:
        raise ValueError(
            f"Embeddings phải là 2D (N, D), nhận {embeddings1.shape}"
        )

    # Element-wise multiply rồi sum theo dim=1 = dot product per row
    return np.sum(embeddings1 * embeddings2, axis=1)


def _validate_inputs(
    similarities: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate + chuẩn hóa inputs.

    Returns:
        (similarities, labels) đã đảm bảo shape và type đúng.
    """
    similarities = np.asarray(similarities, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()

    if similarities.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: similarities={similarities.shape}, "
            f"labels={labels.shape}"
        )
    if similarities.size == 0:
        raise ValueError("Inputs rỗng")

    # Verify labels chỉ chứa 0/1
    unique = np.unique(labels)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"Labels phải là 0/1, nhận {unique}")

    # Cảnh báo nếu chỉ có 1 class
    if len(unique) < 2:
        logger.warning(
            f"Chỉ có 1 class trong labels ({unique}). "
            f"Một số metrics sẽ không tính được."
        )

    return similarities, labels


# ====================================================================== #
# Accuracy at best threshold
# ====================================================================== #


def compute_accuracy_at_best_threshold(
    similarities: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """
    Tìm threshold tối ưu và accuracy tại đó.

    Strategy: Sort similarities, sweep qua từng candidate threshold (giữa
    các giá trị liên tiếp), tính accuracy. Giữ threshold cho accuracy max.

    Args:
        similarities: (N,) cosine similarities.
        labels: (N,) 1=same, 0=different.

    Returns:
        (best_accuracy, best_threshold)

    Examples:
        >>> sims = np.array([0.9, 0.8, 0.3, 0.2])
        >>> labels = np.array([1, 1, 0, 0])
        >>> acc, thr = compute_accuracy_at_best_threshold(sims, labels)
        >>> # threshold giữa 0.3 và 0.8 → accuracy = 1.0
    """
    similarities, labels = _validate_inputs(similarities, labels)

    # Tạo candidate thresholds: trung điểm giữa các similarity unique
    # Cách này đảm bảo bao phủ mọi điểm phân tách possible
    sorted_sims = np.unique(similarities)

    # Edge case: chỉ có 1 unique value → 2 threshold candidates
    if len(sorted_sims) == 1:
        # Threshold ngay dưới và trên giá trị đó
        thresholds = np.array([sorted_sims[0] - 1e-6, sorted_sims[0] + 1e-6])
    else:
        # Trung điểm + 2 giá trị biên
        midpoints = (sorted_sims[:-1] + sorted_sims[1:]) / 2.0
        thresholds = np.concatenate([
            [sorted_sims[0] - 1e-6],   # tất cả accept
            midpoints,
            [sorted_sims[-1] + 1e-6],  # tất cả reject
        ])

    # Sweep tất cả threshold (vectorized cho nhanh)
    # predictions[i, j] = 1 nếu similarities[j] >= thresholds[i]
    predictions = similarities[None, :] >= thresholds[:, None]  # (T, N)
    correct = predictions == labels[None, :].astype(bool)        # (T, N)
    accuracies = correct.mean(axis=1)                            # (T,)

    # Lấy best
    best_idx = int(np.argmax(accuracies))
    return float(accuracies[best_idx]), float(thresholds[best_idx])


# ====================================================================== #
# TAR @ FAR (chuẩn của face recognition benchmarks)
# ====================================================================== #


def compute_tar_at_far(
    similarities: np.ndarray,
    labels: np.ndarray,
    target_far: float = 1e-3,
) -> float:
    """
    True Accept Rate tại FAR cố định.

    Đây là metric CHUẨN cho face recognition benchmarks (LFW, MegaFace, IJB-C).
    Ý nghĩa: với tỷ lệ false accept ≤ target_far (e.g. 0.1%), accept được
    bao nhiêu % positive pairs?

    Strategy:
        1. Lấy negative similarities → tìm threshold sao cho FAR ≤ target_far.
        2. Tính TAR (recall trên positive pairs) tại threshold đó.

    Args:
        similarities: (N,) cosine similarities.
        labels: (N,) 1=positive, 0=negative.
        target_far: FAR mục tiêu (e.g. 1e-3 = 0.1%).

    Returns:
        TAR tại target_far. Trả về 0.0 nếu không đạt được FAR đó
        (do quá ít negative samples).
    """
    similarities, labels = _validate_inputs(similarities, labels)

    if not 0 < target_far < 1:
        raise ValueError(f"target_far phải trong (0, 1), nhận {target_far}")

    # Tách positive và negative
    positive_sims = similarities[labels == 1]
    negative_sims = similarities[labels == 0]

    if len(positive_sims) == 0 or len(negative_sims) == 0:
        logger.warning("Không có đủ positive hoặc negative pairs cho TAR@FAR")
        return 0.0

    # Tìm threshold sao cho FAR ≤ target_far
    # FAR = (số negative >= threshold) / total_negative
    # → cần: số negative >= threshold ≤ target_far * total_negative
    # → threshold = quantile thứ (1 - target_far) của negative_sims

    # np.quantile dùng linear interpolation, ta cần "ceil" để đảm bảo FAR ≤ target
    # → dùng phương pháp 'higher' từ percentile
    n_negative = len(negative_sims)
    n_allowed_fp = int(np.floor(target_far * n_negative))

    if n_allowed_fp == 0:
        # Không cho phép false positive nào
        # → threshold = max(negative) + epsilon
        threshold = float(negative_sims.max()) + 1e-9
    else:
        # Sort descending và lấy threshold ở vị trí n_allowed_fp
        # Threshold > giá trị này → có đúng n_allowed_fp negative bị accept
        sorted_neg_desc = np.sort(negative_sims)[::-1]
        # threshold = giá trị thứ (n_allowed_fp + 1) - 1, tức index n_allowed_fp
        # Nhưng cần threshold > sorted_neg_desc[n_allowed_fp - 1] để không vượt quota
        threshold = float(sorted_neg_desc[n_allowed_fp])

    # Tính TAR tại threshold này
    # Note: dùng > vì ta muốn strictly greater để tránh edge case ties
    tar = float((positive_sims >= threshold).mean())
    return tar


# ====================================================================== #
# EER (Equal Error Rate)
# ====================================================================== #


def compute_eer(
    similarities: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """
    Equal Error Rate: điểm FAR == FRR.

    EER thấp = model tốt. Là 1 chỉ số gọn để so sánh nhanh giữa các model.

    Strategy:
        1. Tính ROC curve để có FAR (=FPR) và TAR (=TPR) tại nhiều threshold.
        2. FRR = 1 - TAR.
        3. Tìm threshold sao cho FAR và FRR gần bằng nhau nhất.

    Args:
        similarities: (N,) cosine similarities.
        labels: (N,) 1=positive, 0=negative.

    Returns:
        (eer, threshold_at_eer)

    Note: EER thực tế hiếm khi đạt FAR == FRR chính xác do discrete data.
    Ta lấy điểm có |FAR - FRR| nhỏ nhất.
    """
    similarities, labels = _validate_inputs(similarities, labels)

    # sklearn.roc_curve trả về fpr=FAR, tpr=TAR theo threshold giảm dần
    fars, tars, thresholds = roc_curve(labels, similarities)
    frrs = 1 - tars

    # Tìm điểm |FAR - FRR| min
    diffs = np.abs(fars - frrs)
    eer_idx = int(np.argmin(diffs))

    # EER = trung bình FAR và FRR tại điểm đó (gần bằng nhau)
    eer = float((fars[eer_idx] + frrs[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx])

    return eer, threshold


# ====================================================================== #
# ROC curve
# ====================================================================== #


def compute_roc_curve(
    similarities: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ROC curve data.

    Args:
        similarities: (N,) cosine similarities.
        labels: (N,) 1=positive, 0=negative.

    Returns:
        (fars, tars, thresholds)
        - fars: False Accept Rates (= FPR), monotonically increasing.
        - tars: True Accept Rates (= TPR).
        - thresholds: similarity thresholds tương ứng (giảm dần).
    """
    similarities, labels = _validate_inputs(similarities, labels)
    fars, tars, thresholds = roc_curve(labels, similarities)
    return fars, tars, thresholds


def compute_auc(similarities: np.ndarray, labels: np.ndarray) -> float:
    """
    Area Under ROC Curve.

    AUC ∈ [0, 1]:
        - 1.0 = perfect classifier
        - 0.5 = random
        - < 0.5 = worse than random (model bị flip dấu)

    Args:
        similarities: (N,) cosine similarities.
        labels: (N,) 1=positive, 0=negative.

    Returns:
        AUC value.
    """
    fars, tars, _ = compute_roc_curve(similarities, labels)
    return float(auc(fars, tars))


# ====================================================================== #
# Main: full evaluation
# ====================================================================== #


def evaluate_verification(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    labels: np.ndarray,
    far_targets: Tuple[float, ...] = (1e-4, 1e-3, 1e-2),
) -> Dict[str, float]:
    """
    Full evaluation pipeline cho verification task.

    Tính tất cả metrics quan trọng từ embeddings và labels.

    Args:
        embeddings1: (N, D) L2-normalized embeddings của ảnh 1.
        embeddings2: (N, D) L2-normalized embeddings của ảnh 2.
        labels: (N,) 1=same person, 0=different.
        far_targets: Các giá trị FAR muốn report TAR.

    Returns:
        Dict với keys:
            - accuracy: best accuracy
            - threshold_acc: threshold cho best accuracy
            - eer: Equal Error Rate
            - threshold_eer: threshold tại EER
            - auc: Area Under ROC Curve
            - tar_at_far_<value>: TAR tại từng target_far
              (e.g. 'tar_at_far_1e-3')

    Examples:
        >>> emb1 = np.random.randn(100, 512)
        >>> emb1 /= np.linalg.norm(emb1, axis=1, keepdims=True)
        >>> emb2 = np.random.randn(100, 512)
        >>> emb2 /= np.linalg.norm(emb2, axis=1, keepdims=True)
        >>> labels = np.random.randint(0, 2, 100)
        >>> metrics = evaluate_verification(emb1, emb2, labels)
        >>> metrics['accuracy']  # ~0.5 cho random
    """
    # Tính similarities 1 lần, dùng cho tất cả metrics
    similarities = compute_cosine_similarity(embeddings1, embeddings2)

    metrics: Dict[str, float] = {}

    # Accuracy
    acc, thr_acc = compute_accuracy_at_best_threshold(similarities, labels)
    metrics["accuracy"] = acc
    metrics["threshold_acc"] = thr_acc

    # EER
    eer, thr_eer = compute_eer(similarities, labels)
    metrics["eer"] = eer
    metrics["threshold_eer"] = thr_eer

    # AUC
    metrics["auc"] = compute_auc(similarities, labels)

    # TAR @ multiple FAR targets
    for far in far_targets:
        # Format key: 1e-3 → '1e-3' (gọn, dễ đọc)
        far_str = f"{far:.0e}"  # '1e-04', '1e-03', '1e-02'
        # Loại bỏ leading zero: '1e-04' → '1e-4'
        far_str = far_str.replace("e-0", "e-")
        key = f"tar_at_far_{far_str}"
        metrics[key] = compute_tar_at_far(similarities, labels, target_far=far)

    return metrics