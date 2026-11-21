"""Evaluation helpers: full classification metrics + confusion matrix plots."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_json


def compute_classification_metrics(
    y_true,
    y_pred,
    labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a rich metric dict beyond plain accuracy."""
    try:
        if labels is None and target_names is not None:
            labels = list(range(len(target_names)))
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "classification_report": report,
        }
        return metrics
    except Exception as e:
        raise CustomException(e, sys)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> str:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        plt.figure(figsize=(max(8, len(class_names) * 0.55), max(6, len(class_names) * 0.45)))
        sns.heatmap(
            cm,
            annot=len(class_names) <= 16,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(title)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(save_path, dpi=140)
        plt.close()
        logging.info(f"Saved confusion matrix -> {save_path}")
        return save_path
    except Exception as e:
        raise CustomException(e, sys)


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    save_json(path, metrics)


def print_metrics_summary(name: str, metrics: Dict[str, Any]) -> None:
    line = (
        f"[{name}] acc={metrics['accuracy']:.4f} | "
        f"f1_w={metrics['f1_weighted']:.4f} | f1_macro={metrics['f1_macro']:.4f} | "
        f"prec_w={metrics['precision_weighted']:.4f} | rec_w={metrics['recall_weighted']:.4f}"
    )
    print(line)
    logging.info(line)
