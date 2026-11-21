"""TF-IDF feature engineering with configurable parameters."""

from __future__ import annotations

import sys
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CONFIG, TfidfConfig
from src.exception import CustomException
from src.logger import logging


def build_tfidf_vectorizer(config: TfidfConfig | None = None) -> TfidfVectorizer:
    """
    Build an improved TF-IDF vectorizer.

    Why: bigrams + sublinear TF + document frequency filters capture
    personality phrases better than unigrams alone, while controlling sparsity.
    """
    try:
        cfg = config or CONFIG.tfidf
        logging.info(
            "Building TF-IDF | max_features=%s ngram=%s min_df=%s max_df=%s",
            cfg.max_features,
            cfg.ngram_range,
            cfg.min_df,
            cfg.max_df,
        )
        return TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            sublinear_tf=cfg.sublinear_tf,
        )
    except Exception as e:
        raise CustomException(e, sys)


def vectorizer_params(vectorizer: TfidfVectorizer) -> dict[str, Any]:
    return {
        "max_features": vectorizer.max_features,
        "ngram_range": vectorizer.ngram_range,
        "min_df": vectorizer.min_df,
        "max_df": vectorizer.max_df,
        "sublinear_tf": vectorizer.sublinear_tf,
        "vocabulary_size": len(getattr(vectorizer, "vocabulary_", {}) or {}),
    }
