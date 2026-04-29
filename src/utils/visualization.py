"""
Visualization Utilities.

Helper functions để visualize kết quả face recognition:
- ROC curves (FAR vs TAR)
- Confusion matrices
- Embedding space (t-SNE / UMAP)
- Training curves (loss + metrics theo epoch)
- Similarity distribution (positive vs negative pairs)
- Metrics comparison giữa scenarios

Tất cả functions:
- Save ra file nếu save_path được set
- Return matplotlib Figure để caller có thể custom thêm
- KHÔNG gọi plt.show() để chạy được headless trên server
- Dùng dpi=150 cho chất lượng cao
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# Default style settings
_DEFAULT_DPI = 150
_DEFAULT_FIGSIZE = (8, 6)


def _save_figure(fig: Figure, save_path: Optional[Path]) -> None:
    """Helper: save figure ra file với mkdir parent + dpi cao."""
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved figure: {save_path}")


# ====================================================================== #
# ROC curve
# ====================================================================== #


def plot_roc_curve(
    fars: np.ndarray,
    tars: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve",
    label: Optional[str] = None,
    auc_value: Optional[float] = None,
    figsize: tuple = _DEFAULT_FIGSIZE,
) -> Figure:
    """
    Plot ROC curve với log-scale x-axis (chuẩn của face recognition papers).

    Args:
        fars: False Accept Rates (= FPR), từ compute_roc_curve.
        tars: True Accept Rates (= TPR).
        save_path: Output file (.png/.pdf). None = không save.
        title: Title của plot.
        label: Legend label cho curve. None = không hiển thị legend.
        auc_value: Nếu cung cấp, sẽ show trong legend.
        figsize: Size figure.

    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Build label với AUC
    curve_label = label or "ROC"
    if auc_value is not None:
        curve_label = f"{curve_label} (AUC={auc_value:.4f})"

    ax.plot(fars, tars, linewidth=2, label=curve_label)

    # Diagonal reference (random classifier)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    # Log-scale FAR (chuẩn FR papers - quan tâm vùng FAR thấp)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(0.0, 1.05)

    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_roc_curves_comparison(
    curves: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[Path] = None,
    title: str = "ROC Curves Comparison",
    figsize: tuple = _DEFAULT_FIGSIZE,
) -> Figure:
    """
    Plot nhiều ROC curves trên cùng 1 figure (so sánh models/scenarios).

    Args:
        curves: Dict {label: {'fars': ..., 'tars': ..., 'auc': ...}}
            'auc' là optional.
        save_path: Output file.
        title: Title.
        figsize: Size.

    Returns:
        Figure object.

    Example:
        curves = {
            'Full vs Full': {'fars': [...], 'tars': [...], 'auc': 0.99},
            'Full vs Masked': {'fars': [...], 'tars': [...], 'auc': 0.92},
            'Masked vs Masked': {'fars': [...], 'tars': [...], 'auc': 0.85},
        }
        plot_roc_curves_comparison(curves)
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, data in curves.items():
        fars = data["fars"]
        tars = data["tars"]
        auc_val = data.get("auc")

        curve_label = f"{label} (AUC={auc_val:.4f})" if auc_val is not None else label
        ax.plot(fars, tars, linewidth=2, label=curve_label)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ====================================================================== #
# Confusion matrix
# ====================================================================== #


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: tuple = (10, 8),
) -> Figure:
    """
    Plot confusion matrix bằng seaborn heatmap.

    Args:
        cm: (n_classes, n_classes) confusion matrix.
        class_names: List tên các class. None = dùng index.
        save_path: Output file.
        title: Title.
        normalize: True = normalize theo row (mỗi row sum = 1).
        figsize: Size figure.

    Returns:
        Figure object.

    Note: Cho identification task. Cho verification (binary same/diff),
    có thể dùng với cm 2x2.
    """
    # Lazy import seaborn (chỉ cần ở function này)
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "seaborn không được cài. Cài bằng: pip install seaborn"
        )

    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm phải là square matrix, nhận shape {cm.shape}")

    # Normalize nếu cần
    if normalize:
        cm_display = cm.astype(np.float64)
        row_sums = cm_display.sum(axis=1, keepdims=True)
        # Tránh chia cho 0 nếu row sum = 0
        row_sums[row_sums == 0] = 1
        cm_display = cm_display / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    # Default class names = index
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    elif len(class_names) != cm.shape[0]:
        raise ValueError(
            f"len(class_names)={len(class_names)} không match cm shape {cm.shape}"
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Xoay xticklabels nếu nhiều class
    if len(class_names) > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ====================================================================== #
# Embedding space visualization
# ====================================================================== #


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    label_names: Optional[Dict[int, str]] = None,
    max_classes: int = 20,
    figsize: tuple = (10, 8),
    random_state: int = 42,
) -> Figure:
    """
    Visualize embedding space bằng t-SNE hoặc UMAP.

    Reduce embeddings từ D-dim xuống 2D rồi scatter plot với màu
    theo identity. Dùng để check xem embeddings có tách biệt theo
    identity không.

    Args:
        embeddings: (N, D) - không cần L2 normalize, t-SNE/UMAP tự handle.
        labels: (N,) integer identity labels.
        method: 'tsne' hoặc 'umap'.
        save_path: Output file.
        title: Plot title (default tự generate).
        label_names: Dict {label_idx: name} để hiện tên thay vì index.
        max_classes: Giới hạn số class hiển thị (chọn ngẫu nhiên).
            Quá nhiều class → plot rối.
        figsize: Size.
        random_state: Seed cho reproducibility.

    Returns:
        Figure object.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings phải 2D, nhận {embeddings.shape}")
    if len(embeddings) != len(labels):
        raise ValueError(
            f"Số embeddings ({len(embeddings)}) khác số labels ({len(labels)})"
        )

    method = method.lower()
    if method not in ("tsne", "umap"):
        raise ValueError(f"method phải là 'tsne' hoặc 'umap', nhận {method}")

    # Filter để giới hạn số class
    unique_labels = np.unique(labels)
    if len(unique_labels) > max_classes:
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(unique_labels, max_classes, replace=False)
        mask = np.isin(labels, chosen)
        embeddings = embeddings[mask]
        labels = labels[mask]
        logger.info(
            f"Quá nhiều classes ({len(unique_labels)}), chỉ plot "
            f"{max_classes} class ngẫu nhiên ({len(embeddings)} samples)"
        )

    # Reduce 2D
    logger.info(f"Reducing {embeddings.shape} -> 2D bằng {method}...")
    if method == "tsne":
        from sklearn.manifold import TSNE
        # perplexity nên < n_samples
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
    else:
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn không được cài. Cài bằng: pip install umap-learn"
            )
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
        )

    coords_2d = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Dùng tab20 cho ≤20 classes, viridis cho nhiều hơn
    cmap_name = "tab20" if n_classes <= 20 else "viridis"
    cmap = plt.get_cmap(cmap_name)

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = cmap(i / max(n_classes - 1, 1))
        name = label_names.get(int(lbl), str(lbl)) if label_names else str(lbl)
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            label=name,
            color=color,
            alpha=0.7,
            s=30,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel(f"{method.upper()} dimension 1")
    ax.set_ylabel(f"{method.upper()} dimension 2")
    ax.set_title(title or f"Embedding space ({method.upper()}, {n_classes} identities)")

    # Legend ngoài plot nếu nhiều class
    if n_classes <= 10:
        ax.legend(loc="best", fontsize=8)
    elif n_classes <= 20:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    # >20: bỏ legend (quá rối)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ====================================================================== #
# Similarity distribution (rất hữu ích cho chọn threshold)
# ====================================================================== #


def plot_similarity_distribution(
    similarities: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Similarity Distribution",
    threshold: Optional[float] = None,
    bins: int = 50,
    figsize: tuple = _DEFAULT_FIGSIZE,
) -> Figure:
    """
    Histogram của similarities cho positive vs negative pairs.

    Plot này cực hữu ích để:
    - Chọn threshold thủ công bằng cách nhìn vùng overlap
    - Đánh giá độ tách biệt của model (overlap ít = model tốt)
    - Hiểu vì sao TAR@FAR thấp (quá nhiều overlap)

    Args:
        similarities: (N,) cosine similarities.
        labels: (N,) 1=positive, 0=negative.
        save_path: Output file.
        title: Title.
        threshold: Nếu set, vẽ vertical line tại threshold đó.
        bins: Số bins cho histogram.
        figsize: Size.

    Returns:
        Figure object.
    """
    similarities = np.asarray(similarities).ravel()
    labels = np.asarray(labels).ravel()

    if similarities.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: {similarities.shape} vs {labels.shape}"
        )

    pos_sims = similarities[labels == 1]
    neg_sims = similarities[labels == 0]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot 2 histogram chồng lên nhau với alpha
    bins_array = np.linspace(
        min(similarities.min(), -1.0),
        max(similarities.max(), 1.0),
        bins,
    )

    ax.hist(
        neg_sims,
        bins=bins_array,
        alpha=0.6,
        label=f"Different person (n={len(neg_sims)})",
        color="tab:red",
        edgecolor="darkred",
    )
    ax.hist(
        pos_sims,
        bins=bins_array,
        alpha=0.6,
        label=f"Same person (n={len(pos_sims)})",
        color="tab:green",
        edgecolor="darkgreen",
    )

    # Threshold line nếu có
    if threshold is not None:
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Threshold={threshold:.3f}",
        )

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ====================================================================== #
# Training curves
# ====================================================================== #


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training Curves",
    figsize: tuple = (12, 5),
) -> Figure:
    """
    Plot training/validation curves theo epoch.

    Tự động chia subplot: loss bên trái, các metrics khác bên phải.

    Args:
        history: Dict {metric_name: [values per epoch]}.
            Convention: keys bắt đầu với 'train/' hoặc 'val/'.
            Loss-related keys (chứa 'loss') sẽ vào subplot riêng.
        save_path: Output file.
        title: Suptitle.
        figsize: Size.

    Returns:
        Figure object.

    Example:
        history = {
            'train/loss': [2.5, 1.8, 1.2, ...],
            'val/accuracy': [0.5, 0.7, 0.85, ...],
            'val/eer': [0.3, 0.2, 0.1, ...],
        }
    """
    if not history:
        raise ValueError("history rỗng")

    # Phân loại keys: loss vs metrics
    loss_keys = [k for k in history if "loss" in k.lower()]
    metric_keys = [k for k in history if k not in loss_keys]

    # Quyết định layout
    if loss_keys and metric_keys:
        fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax_loss, ax_metric]
        groups = [loss_keys, metric_keys]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        groups = [loss_keys or metric_keys]

    for ax, keys in zip(axes, groups):
        for key in keys:
            values = history[key]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker="o", markersize=3, label=key)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    if loss_keys and axes[0]:
        axes[0].set_title("Loss")
        axes[0].set_yscale("log")  # Loss thường xem log scale dễ hơn
    if metric_keys and (axes[1] if len(axes) > 1 else axes[0]):
        target_ax = axes[1] if len(axes) > 1 else axes[0]
        target_ax.set_title("Metrics")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ====================================================================== #
# Metrics comparison (cho objective của project)
# ====================================================================== #


def plot_metrics_comparison(
    metrics_per_scenario: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Metrics Comparison Across Scenarios",
    figsize: tuple = (12, 6),
) -> Figure:
    """
    Bar chart so sánh metrics giữa các scenario.

    Trực tiếp phục vụ objective của project: so sánh accuracy
    giữa "full vs full", "full vs masked", "masked vs masked".

    Args:
        metrics_per_scenario: Dict {scenario_name: {metric: value}}.
        metric_keys: Subset metrics muốn plot. None = lấy tất cả từ scenario đầu.
        save_path: Output file.
        title: Title.
        figsize: Size.

    Returns:
        Figure object.

    Example:
        results = {
            'Full vs Full':       {'accuracy': 0.99, 'eer': 0.01, 'auc': 0.999},
            'Full vs Masked':     {'accuracy': 0.92, 'eer': 0.08, 'auc': 0.97},
            'Masked vs Masked':   {'accuracy': 0.87, 'eer': 0.13, 'auc': 0.94},
        }
        plot_metrics_comparison(results)
    """
    if not metrics_per_scenario:
        raise ValueError("metrics_per_scenario rỗng")

    scenarios = list(metrics_per_scenario.keys())

    # Auto detect metric_keys nếu None
    if metric_keys is None:
        first_scenario = scenarios[0]
        metric_keys = list(metrics_per_scenario[first_scenario].keys())

    # Build data matrix: rows=metrics, cols=scenarios
    n_metrics = len(metric_keys)
    n_scenarios = len(scenarios)

    fig, ax = plt.subplots(figsize=figsize)

    # Bar positions
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_scenarios

    cmap = plt.get_cmap("Set2")
    for i, scenario in enumerate(scenarios):
        scenario_metrics = metrics_per_scenario[scenario]
        values = [scenario_metrics.get(k, 0) for k in metric_keys]
        offset = (i - n_scenarios / 2) * bar_width + bar_width / 2
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=scenario,
            color=cmap(i),
            edgecolor="black",
            linewidth=0.5,
        )
        # Annotate giá trị trên đầu bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig