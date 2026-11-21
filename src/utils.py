"""Shared helpers for artifact IO and model evaluation."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import dill
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: Any) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path: str, data: Dict[str, Any]) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        raise CustomException(e, sys)


def load_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: Dict[str, Any]) -> Dict[str, Any]:
    """Fit each model and return accuracy / F1 plus the fitted estimator."""
    try:
        report: Dict[str, Any] = {}
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average="weighted")

            report[model_name] = {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "test_f1": float(test_f1),
                "model": model,
            }
            logging.info(
                f"{model_name} -> train_acc={train_accuracy:.4f}, "
                f"test_acc={test_accuracy:.4f}, test_f1={test_f1:.4f}"
            )
        return report
    except Exception as e:
        raise CustomException(e, sys)


def decision_confidence(model, features, predicted_index: int) -> float | None:
    """
    Return confidence in [0, 100].
    Prefer predict_proba; fall back to decision_function softmax/sigmoid style scaling.
    """
    try:
        import numpy as np

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            return float(probs[predicted_index]) * 100.0

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(features))
            if scores.ndim == 1:
                # Binary decision_function
                score = float(scores[0])
                prob_pos = 1.0 / (1.0 + np.exp(-score))
                conf = prob_pos if predicted_index == 1 else 1.0 - prob_pos
                return float(conf) * 100.0
            # Multiclass: softmax over margins
            row = scores[0]
            exp = np.exp(row - np.max(row))
            probs = exp / exp.sum()
            return float(probs[predicted_index]) * 100.0
        return None
    except Exception:
        return None
