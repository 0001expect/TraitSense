"""Central configuration for the MBTI personality prediction project."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class TfidfConfig:
    """Configurable TF-IDF feature engineering settings."""

    max_features: int = 15000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.9
    sublinear_tf: bool = True


@dataclass
class DataQualityConfig:
    """Dataset cleaning thresholds applied before train/test split."""

    min_word_count: int = 50
    min_char_count: int = 200
    remove_duplicates: bool = True


@dataclass
class TrainConfig:
    """Training / tuning switches. Quick mode keeps CV light for laptops."""

    random_state: int = 42
    test_size: float = 0.2
    use_class_weight: bool = True
    use_random_oversampler: bool = True
    tune_hyperparams: bool = True
    n_iter_search: int = 8
    cv_folds: int = 3
    scoring: str = "f1_weighted"
    quick_mode: bool = False
    skip_slow_models: bool = False  # skip Random Forest + XGBoost
    skip_ensemble: bool = False


@dataclass
class ProjectConfig:
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    axes: Dict[str, Tuple[str, str]] = field(
        default_factory=lambda: {
            "IE": ("I", "E"),
            "NS": ("N", "S"),
            "TF": ("T", "F"),
            "JP": ("J", "P"),
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def apply_quick_mode(self) -> None:
        """Faster local runs (~10–20 min): no RF/XGB tuning, smaller TF-IDF."""
        if not self.train.quick_mode:
            return
        self.tfidf.max_features = 5000
        self.train.n_iter_search = 3
        self.train.cv_folds = 2
        self.train.use_random_oversampler = False
        self.train.tune_hyperparams = False
        self.train.skip_slow_models = True


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y"}


CONFIG = ProjectConfig()
if _env_flag("MBTI_QUICK"):
    CONFIG.train.quick_mode = True
CONFIG.apply_quick_mode()
