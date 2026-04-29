"""
Config Utilities.

Cơ chế chính: cho phép 1 config YAML kế thừa từ nhiều config khác qua key _base_.

Ví dụ configs/train/exp_001_baseline.yaml:
    _base_:
      - configs/model/mobilenet_v2_arcface.yaml
      - configs/data/lfw.yaml

    training:
      num_epochs: 30        # override hoặc thêm mới

Cơ chế load:
    1. Load tất cả file trong _base_ theo thứ tự (resolve recursive nếu base
       lại có _base_).
    2. Deep merge từ trái sang phải (file sau override file trước).
    3. Merge với current file (override mạnh nhất).
    4. Xóa key _base_ khỏi kết quả.

Path resolution:
    Tất cả paths trong _base_ đều resolve RELATIVE TO PROJECT ROOT, không phải
    relative to file đang load. Điều này nhất quán và dễ debug.

Merge strategy:
    - Dict: deep merge (recursive)
    - List: replace (override list thay luôn base list)
    - Scalar: replace
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

logger = logging.getLogger(__name__)

# Project root được tính từ vị trí file này.
# src/utils/config.py -> .. -> .. -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Key đặc biệt cho inheritance
BASE_KEY = "_base_"


# ====================================================================== #
# Helper functions
# ====================================================================== #


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load 1 file YAML, return dict.

    Args:
        path: Đường dẫn tới file YAML.

    Returns:
        Dict được parse từ YAML. Trả về {} nếu file rỗng.

    Raises:
        FileNotFoundError: file không tồn tại.
        yaml.YAMLError: nếu YAML bị malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file không tồn tại: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # File trống thì yaml.safe_load trả None - chuẩn hóa thành dict rỗng
    if data is None:
        logger.warning(f"Config file rỗng: {path}")
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Config phải là YAML mapping (dict), nhận {type(data).__name__}: {path}"
        )

    return data


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge 2 configs, override ghi đè base.

    Quy tắc:
        - Nếu cả 2 đều là dict → recursive merge
        - Nếu khác type hoặc không phải dict → override thay luôn

    Note: Không modify input. Trả về dict mới (deep copy).

    Args:
        base: Config gốc.
        override: Config override.

    Returns:
        Dict đã merge.

    Examples:
        >>> base = {'a': 1, 'b': {'x': 1, 'y': 2}}
        >>> override = {'b': {'x': 10}, 'c': 3}
        >>> merge_configs(base, override)
        {'a': 1, 'b': {'x': 10, 'y': 2}, 'c': 3}
    """
    # Deep copy base để không modify input
    result = copy.deepcopy(base)

    for key, override_value in override.items():
        base_value = result.get(key)

        # Nếu cả 2 đều là dict → recursive merge
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = merge_configs(base_value, override_value)
        else:
            # Còn lại: override thay thế hoàn toàn (kể cả list)
            # deepcopy để tránh shared reference với input
            result[key] = copy.deepcopy(override_value)

    return result


# ====================================================================== #
# Main: load config với _base_ resolution
# ====================================================================== #


def _resolve_base_path(base_path: str) -> Path:
    """
    Resolve path trong _base_ về absolute path.

    Convention: paths luôn relative to PROJECT_ROOT.
    """
    p = Path(base_path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _load_with_inheritance(
    config_path: Path,
    visited: Set[Path],
) -> Dict[str, Any]:
    """
    Load config + resolve _base_ recursively.

    Args:
        config_path: File hiện tại đang load.
        visited: Set các path đã visit (để detect circular import).

    Returns:
        Config đã merge với tất cả base.
    """
    config_path = Path(config_path).resolve()

    # Detect circular inheritance
    if config_path in visited:
        chain = " -> ".join(str(p.name) for p in visited) + f" -> {config_path.name}"
        raise ValueError(f"Circular _base_ inheritance: {chain}")
    visited = visited | {config_path}

    # Load file hiện tại
    current = load_yaml(config_path)

    # Nếu không có _base_ → trả về luôn
    if BASE_KEY not in current:
        return current

    # Parse _base_: có thể là string đơn lẻ hoặc list
    base_value = current.pop(BASE_KEY)
    if isinstance(base_value, str):
        base_paths: List[str] = [base_value]
    elif isinstance(base_value, list):
        base_paths = base_value
        if not all(isinstance(p, str) for p in base_paths):
            raise ValueError(
                f"_base_ phải là string hoặc list of strings, "
                f"nhận: {base_value} ({config_path})"
            )
    else:
        raise ValueError(
            f"_base_ phải là string hoặc list, "
            f"nhận {type(base_value).__name__}: {config_path}"
        )

    # Load + merge tất cả base configs theo thứ tự
    # File sau trong list override file trước
    merged: Dict[str, Any] = {}
    for base_path_str in base_paths:
        base_abs = _resolve_base_path(base_path_str)
        if not base_abs.exists():
            raise FileNotFoundError(
                f"_base_ path không tồn tại: {base_abs} "
                f"(từ {config_path})"
            )
        # Recursive: base có thể có _base_ riêng
        base_config = _load_with_inheritance(base_abs, visited)
        merged = merge_configs(merged, base_config)

    # Cuối cùng: merge current (current override mọi thứ trong base)
    merged = merge_configs(merged, current)

    return merged


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load full config cho 1 experiment, resolve _base_ inheritance.

    Args:
        config_path: Path tới config file.

    Returns:
        Dict config đã được resolve hoàn toàn (không còn _base_ key).

    Examples:
        >>> config = load_config('configs/train/exp_001_baseline.yaml')
        >>> config['model']['backbone']  # từ configs/model/mobilenet_v2_arcface.yaml
        'mobilenet_v2'
        >>> config['training']['num_epochs']  # từ exp_001 trực tiếp
        30
    """
    config_path = Path(config_path)
    logger.info(f"Loading config: {config_path}")

    config = _load_with_inheritance(config_path, visited=set())

    logger.info(
        f"Config loaded: {len(config)} top-level keys: {list(config.keys())}"
    )
    return config


# ====================================================================== #
# Save config (cho reproducibility)
# ====================================================================== #


def save_config(config: Dict[str, Any], path: Path) -> None:
    """
    Save config dict ra file YAML.

    Dùng cho việc lưu lại config đã merge vào experiment folder để
    reproduce sau này (không phụ thuộc vào _base_ path nữa).

    Args:
        config: Dict cần save.
        path: Output path. Tự tạo parent dir.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,  # block style cho dễ đọc
            sort_keys=False,           # giữ thứ tự keys gốc
            allow_unicode=True,        # hỗ trợ tiếng Việt nếu có comment
            indent=2,
        )

    logger.info(f"Config saved: {path}")