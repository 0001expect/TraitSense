"""Linear-model explainability: top influential TF-IDF terms."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.exception import CustomException


def _feature_names(vectorizer) -> np.ndarray:
    return np.array(vectorizer.get_feature_names_out())


def _coef_matrix(model) -> Optional[np.ndarray]:
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_)
    return None


def top_words_for_binary(
    model,
    vectorizer,
    positive_label: str,
    negative_label: str,
    top_k: int = 10,
) -> Dict[str, List[str]]:
    """
    For a binary linear model, return words pushing toward each class.

    Why: users can see *why* IE/NS/TF/JP leaned one way.
    """
    try:
        coef = _coef_matrix(model)
        if coef is None:
            return {positive_label: [], negative_label: []}
        weights = coef[0]
        names = _feature_names(vectorizer)
        top_pos_idx = np.argsort(weights)[-top_k:][::-1]
        top_neg_idx = np.argsort(weights)[:top_k]
        return {
            positive_label: names[top_pos_idx].tolist(),
            negative_label: names[top_neg_idx].tolist(),
        }
    except Exception as e:
        raise CustomException(e, sys)


def explain_instance_linear(
    model,
    vectorizer,
    cleaned_text: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Highlight tokens in *this* document with largest |tfidf * coef| contribution.

    For multiclass, uses the predicted class coefficient row when available.
    """
    try:
        coef = _coef_matrix(model)
        if coef is None:
            return []

        x = vectorizer.transform([cleaned_text])
        if hasattr(model, "predict"):
            pred = int(model.predict(x)[0])
        else:
            pred = 0

        # Binary: one row. Multiclass: one row per class.
        row = coef[0] if coef.shape[0] == 1 else coef[pred]
        x_dense = np.asarray(x.toarray()).ravel()
        contrib = x_dense * row
        names = _feature_names(vectorizer)

        nz = np.where(x_dense != 0)[0]
        if len(nz) == 0:
            return []

        order = nz[np.argsort(np.abs(contrib[nz]))[::-1][:top_k]]
        return [
            {
                "word": str(names[i]),
                "contribution": float(contrib[i]),
                "tfidf": float(x_dense[i]),
            }
            for i in order
        ]
    except Exception as e:
        raise CustomException(e, sys)


def collect_axis_explanations(
    axis_models: Dict[str, Any],
    vectorizer,
    cleaned_text: str,
    axes_letters: Dict[str, Sequence[str]],
    predicted_type: str,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Build a user-facing explanation for the four-dimension prediction."""
    try:
        influential: List[str] = []
        per_axis: Dict[str, Any] = {}

        for i, (axis_name, letters) in enumerate(axes_letters.items()):
            model = axis_models[axis_name]
            chosen = predicted_type[i]
            words = explain_instance_linear(model, vectorizer, cleaned_text, top_k=top_k)
            # Keep words whose contribution pushes toward the chosen class
            # Binary models: positive coef -> letters[1], negative -> letters[0]
            supporting = []
            for item in words:
                if chosen == letters[1] and item["contribution"] > 0:
                    supporting.append(item["word"])
                elif chosen == letters[0] and item["contribution"] < 0:
                    supporting.append(item["word"])
                elif abs(item["contribution"]) > 0:
                    supporting.append(item["word"])
            per_axis[axis_name] = {
                "chosen": chosen,
                "top_words": supporting[:top_k],
                "raw": words,
            }
            influential.extend(supporting[:3])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w in influential:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        return {
            "top_influential_words": unique[:12],
            "per_axis": per_axis,
        }
    except Exception as e:
        raise CustomException(e, sys)


LINEAR_MODELS = (LogisticRegression, LinearSVC, MultinomialNB)
