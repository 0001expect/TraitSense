"""
Train multiclass models, four binary axis models, ensemble voting,
hyperparameter tuning, and imbalance handling.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from src.components.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    print_metrics_summary,
    save_metrics,
)
from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging
from src.utils import save_json, save_object

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:  # pragma: no cover
    RandomOverSampler = None

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join("artifacts", "models")
    metrics_dir: str = os.path.join("artifacts", "metrics")
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    ensemble_model_path: str = os.path.join("artifacts", "models", "ensemble_voter.pkl")
    axis_models_path: str = os.path.join("artifacts", "models", "axis_models.pkl")
    all_models_path: str = os.path.join("artifacts", "models", "all_multiclass_models.pkl")
    config_path: str = os.path.join("artifacts", "config.json")
    training_metrics_path: str = os.path.join("artifacts", "metrics", "training_metrics.json")


class ModelTrainer:
    def __init__(self) -> None:
        self.cfg = ModelTrainerConfig()
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        os.makedirs(self.cfg.metrics_dir, exist_ok=True)

    def _base_multiclass_estimators(self, n_classes: int) -> Dict[str, Any]:
        cw = "balanced" if CONFIG.train.use_class_weight else None
        models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                max_iter=4000,
                solver="saga",
                tol=1e-3,
                class_weight=cw,
                random_state=CONFIG.train.random_state,
            ),
            "Linear SVC": LinearSVC(
                max_iter=4000,
                dual="auto",
                class_weight=cw,
                random_state=CONFIG.train.random_state,
            ),
            "Multinomial NB": MultinomialNB(),
        }
        if not CONFIG.train.skip_slow_models:
            models["Random Forest"] = RandomForestClassifier(
                n_estimators=120,
                max_depth=28,
                n_jobs=-1,
                class_weight=cw,
                random_state=CONFIG.train.random_state,
            )
            if HAS_XGBOOST:
                models["XGBoost"] = XGBClassifier(
                    n_estimators=120,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    n_jobs=-1,
                    random_state=CONFIG.train.random_state,
                    tree_method="hist",
                )
        return models

    def _param_distributions(self) -> Dict[str, dict]:
        # Note: do not pass deprecated LogisticRegression(penalty=...); sklearn>=1.8
        return {
            "Logistic Regression": {
                "C": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            },
            "Linear SVC": {
                "C": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            },
            "Multinomial NB": {
                "alpha": [0.05, 0.1, 0.5, 1.0, 1.5, 2.0],
            },
            "Random Forest": {
                "n_estimators": [80, 120, 160],
                "max_depth": [20, 28, 36, None],
                "min_samples_leaf": [1, 2, 4],
            },
            "XGBoost": {
                "n_estimators": [80, 120, 160],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
            },
        }

    @staticmethod
    def _search_space_size(params: dict) -> int:
        size = 1
        for values in params.values():
            size *= len(values)
        return size

    def _maybe_oversample(self, X, y, tag: str) -> Tuple[Any, Any]:
        if not CONFIG.train.use_random_oversampler or RandomOverSampler is None:
            return X, y
        logging.info(f"Applying RandomOverSampler for {tag}")
        ros = RandomOverSampler(random_state=CONFIG.train.random_state)
        return ros.fit_resample(X, y)

    def _fit_with_optional_tuning(self, name: str, estimator, X, y):
        params = self._param_distributions().get(name)
        if not CONFIG.train.tune_hyperparams or not params:
            estimator.fit(X, y)
            return estimator, {}, None

        # XGBoost imbalance: sample weights when class_weight isn't native
        fit_kwargs = {}
        if name == "XGBoost" and CONFIG.train.use_class_weight:
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y)

        n_iter = min(CONFIG.train.n_iter_search, self._search_space_size(params))
        search = RandomizedSearchCV(
            estimator,
            param_distributions=params,
            n_iter=n_iter,
            cv=CONFIG.train.cv_folds,
            scoring=CONFIG.train.scoring,
            n_jobs=-1,
            random_state=CONFIG.train.random_state,
            verbose=0,
        )
        search.fit(X, y, **fit_kwargs)
        logging.info(f"{name} best params: {search.best_params_}")
        print(f"{name} best params: {search.best_params_}")
        return search.best_estimator_, search.best_params_, float(search.best_score_)

    def _evaluate_and_plot(
        self,
        name: str,
        model,
        X_test,
        y_test,
        class_names: List[str],
        metrics_bucket: Dict[str, Any],
    ) -> np.ndarray:
        preds = model.predict(X_test)
        metrics = compute_classification_metrics(
            y_test, preds, target_names=class_names
        )
        print_metrics_summary(name, metrics)
        cm_path = os.path.join(self.cfg.metrics_dir, f"cm_{name.replace(' ', '_').lower()}.png")
        try:
            plot_confusion_matrix(
                y_test,
                preds,
                class_names=class_names,
                save_path=cm_path,
                title=f"Confusion Matrix — {name}",
            )
            metrics["confusion_matrix_path"] = cm_path
        except Exception as exc:  # plotting should not fail training
            logging.info(f"Confusion matrix skipped for {name}: {exc}")
        metrics_bucket[name] = {k: v for k, v in metrics.items() if k != "classification_report"}
        metrics_bucket[name]["classification_report"] = metrics["classification_report"]
        return preds

    def _train_multiclass(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        X_train, y_train = bundle["X_train"], bundle["y_train"]
        X_test, y_test = bundle["X_test"], bundle["y_test"]
        class_names = bundle["type_encoder"].classes_.tolist()
        n_classes = len(class_names)

        # Baseline without balancing (for comparison)
        compare: Dict[str, Any] = {}
        baseline = LogisticRegression(max_iter=1500, solver="saga", random_state=42)
        baseline.fit(X_train, y_train)
        base_pred = baseline.predict(X_test)
        compare["logistic_unbalanced"] = compute_classification_metrics(
            y_test, base_pred, target_names=class_names
        )
        print_metrics_summary("LogReg (no class_weight)", compare["logistic_unbalanced"])

        X_bal, y_bal = self._maybe_oversample(X_train, y_train, "16-class")
        if CONFIG.train.use_random_oversampler and RandomOverSampler is not None:
            bal_model = LogisticRegression(
                max_iter=1500,
                solver="saga",
                class_weight="balanced" if CONFIG.train.use_class_weight else None,
                random_state=42,
            )
            bal_model.fit(X_bal, y_bal)
            bal_pred = bal_model.predict(X_test)
            compare["logistic_oversampled"] = compute_classification_metrics(
                y_test, bal_pred, target_names=class_names
            )
            print_metrics_summary("LogReg (RandomOverSampler)", compare["logistic_oversampled"])

        fitted: Dict[str, Any] = {}
        best_params: Dict[str, Any] = {}
        metrics_out: Dict[str, Any] = {"imbalance_comparison": compare, "multiclass": {}}

        for name, est in self._base_multiclass_estimators(n_classes).items():
            logging.info(f"Training multiclass model: {name}")
            model, params, cv_score = self._fit_with_optional_tuning(name, est, X_train, y_train)
            fitted[name] = model
            best_params[name] = {"params": params, "cv_score": cv_score}
            save_object(
                os.path.join(self.cfg.model_dir, f"{name.replace(' ', '_').lower()}.pkl"),
                model,
            )
            self._evaluate_and_plot(
                name, model, X_test, y_test, class_names, metrics_out["multiclass"]
            )

        # Soft voting ensemble (LR + calibrated LinearSVC + NB)
        if CONFIG.train.skip_ensemble:
            logging.info("Skipping ensemble (skip_ensemble=True)")
        else:
            logging.info("Building VotingClassifier ensemble (LR + LinearSVC + NB)")
            cv_ensemble = 2 if CONFIG.train.quick_mode else 3
            soft_ensemble = VotingClassifier(
                estimators=[
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=2000,
                            solver="saga",
                            class_weight="balanced" if CONFIG.train.use_class_weight else None,
                            C=getattr(fitted["Logistic Regression"], "C", 1.0),
                            random_state=CONFIG.train.random_state,
                        ),
                    ),
                    (
                        "svc",
                        CalibratedClassifierCV(
                            LinearSVC(
                                max_iter=4000,
                                dual="auto",
                                class_weight="balanced" if CONFIG.train.use_class_weight else None,
                                C=getattr(fitted["Linear SVC"], "C", 1.0),
                                random_state=CONFIG.train.random_state,
                            ),
                            cv=cv_ensemble,
                        ),
                    ),
                    (
                        "nb",
                        MultinomialNB(alpha=getattr(fitted["Multinomial NB"], "alpha", 1.0)),
                    ),
                ],
                voting="soft",
                n_jobs=-1,
            )
            soft_ensemble.fit(X_train, y_train)
            fitted["Ensemble Soft Vote"] = soft_ensemble
            save_object(self.cfg.ensemble_model_path, soft_ensemble)
            self._evaluate_and_plot(
                "Ensemble Soft Vote",
                soft_ensemble,
                X_test,
                y_test,
                class_names,
                metrics_out["multiclass"],
            )

        # Pick best multiclass by weighted F1
        best_name = max(
            metrics_out["multiclass"],
            key=lambda n: metrics_out["multiclass"][n]["f1_weighted"],
        )
        best_model = fitted[best_name]
        save_object(self.cfg.trained_model_file_path, best_model)
        save_object(self.cfg.all_models_path, fitted)

        metrics_out["best_multiclass_model"] = best_name
        metrics_out["best_params"] = best_params
        return {
            "fitted": fitted,
            "metrics": metrics_out,
            "best_name": best_name,
            "best_model": best_model,
            "best_f1": metrics_out["multiclass"][best_name]["f1_weighted"],
        }

    def _train_axis_models(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Four independent binary classifiers (I/E, N/S, T/F, J/P).

        Why: 16-way classification is harder and more imbalanced; dimension-wise
        models usually generalize better and compose into a final type.
        """
        X_train, X_test = bundle["X_train"], bundle["X_test"]
        axis_models: Dict[str, Any] = {}
        axis_metrics: Dict[str, Any] = {}

        for axis_name, letters in CONFIG.axes.items():
            y_tr = bundle["axis_y_train"][axis_name]
            y_te = bundle["axis_y_test"][axis_name]
            enc = bundle["axis_encoders"][axis_name]

            # Compare balanced vs unbalanced logistic for this axis
            unbalanced = LogisticRegression(max_iter=1500, solver="liblinear", random_state=42)
            unbalanced.fit(X_train, y_tr)
            before = compute_classification_metrics(
                y_te, unbalanced.predict(X_test), target_names=list(letters)
            )

            est = LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced" if CONFIG.train.use_class_weight else None,
                random_state=CONFIG.train.random_state,
            )
            X_use, y_use = self._maybe_oversample(X_train, y_tr, axis_name)
            model, params, cv_score = self._fit_with_optional_tuning(
                "Logistic Regression", est, X_use, y_use
            )

            # Also try Linear SVC and pick better F1 (skip in quick mode)
            cand = {"Logistic Regression": model}
            svc_params = {}
            if not CONFIG.train.quick_mode:
                svc = LinearSVC(
                    max_iter=4000,
                    dual="auto",
                    class_weight="balanced" if CONFIG.train.use_class_weight else None,
                    random_state=CONFIG.train.random_state,
                )
                svc, svc_params, _ = self._fit_with_optional_tuning(
                    "Linear SVC", svc, X_use, y_use
                )
                cand["Linear SVC"] = svc

            best_axis_name = None
            best_axis_model = None
            best_f1 = -1.0
            for cname, cmodel in cand.items():
                m = compute_classification_metrics(
                    y_te, cmodel.predict(X_test), target_names=list(letters)
                )
                if m["f1_weighted"] > best_f1:
                    best_f1 = m["f1_weighted"]
                    best_axis_name = cname
                    best_axis_model = cmodel

            after = compute_classification_metrics(
                y_te, best_axis_model.predict(X_test), target_names=list(letters)
            )
            print_metrics_summary(f"Axis {axis_name} BEFORE balance", before)
            print_metrics_summary(f"Axis {axis_name} AFTER ({best_axis_name})", after)

            axis_models[axis_name] = best_axis_model
            save_object(
                os.path.join(self.cfg.model_dir, f"axis_{axis_name}.pkl"),
                best_axis_model,
            )
            axis_metrics[axis_name] = {
                "best_model": best_axis_name,
                "before_balance": {
                    k: before[k]
                    for k in ("accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted")
                },
                "after_balance": {
                    k: after[k]
                    for k in ("accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted")
                },
                "best_params": params if best_axis_name == "Logistic Regression" else svc_params,
                "cv_score": cv_score,
                "letters": list(letters),
                "encoder_classes": enc.classes_.tolist(),
            }

        # Compose 4 binary predictions into MBTI codes and score vs true type
        letter_preds = []
        for axis_name in CONFIG.axes:
            enc = bundle["axis_encoders"][axis_name]
            pred_idx = axis_models[axis_name].predict(X_test)
            letter_preds.append(enc.inverse_transform(pred_idx))

        composed = ["".join(parts) for parts in zip(*letter_preds)]
        # Map composed strings to type encoder indices where possible
        type_encoder = bundle["type_encoder"]
        true_labels = bundle["y_test"]
        pred_labels = []
        unknown = 0
        for code in composed:
            if code in type_encoder.classes_:
                pred_labels.append(type_encoder.transform([code])[0])
            else:
                unknown += 1
                pred_labels.append(-1)

        valid_mask = np.array(pred_labels) >= 0
        compose_metrics = compute_classification_metrics(
            true_labels[valid_mask],
            np.array(pred_labels)[valid_mask],
            target_names=type_encoder.classes_.tolist(),
        )
        print_metrics_summary("Composed 4-axis -> 16-type", compose_metrics)
        plot_confusion_matrix(
            true_labels[valid_mask],
            np.array(pred_labels)[valid_mask],
            class_names=type_encoder.classes_.tolist(),
            save_path=os.path.join(self.cfg.metrics_dir, "cm_four_axis_composed.png"),
            title="Confusion Matrix — 4-axis composition",
        )

        save_object(self.cfg.axis_models_path, axis_models)
        return {
            "axis_models": axis_models,
            "axis_metrics": axis_metrics,
            "compose_metrics": compose_metrics,
            "compose_unknown": unknown,
            "composed_sample": composed[:5],
        }

    def initiate_model_trainer(self, bundle: Dict[str, Any]) -> Tuple[float, str]:
        try:
            logging.info("Starting upgraded model training")
            save_json(self.cfg.config_path, CONFIG.to_dict())

            multi = self._train_multiclass(bundle)
            axis = self._train_axis_models(bundle)

            training_metrics = {
                **multi["metrics"],
                "four_axis": axis["axis_metrics"],
                "four_axis_composed": {
                    k: axis["compose_metrics"][k]
                    for k in (
                        "accuracy",
                        "f1_weighted",
                        "f1_macro",
                        "precision_weighted",
                        "recall_weighted",
                    )
                },
                "xgboost_available": HAS_XGBOOST,
                "random_oversampler_available": RandomOverSampler is not None,
            }
            save_metrics(training_metrics, self.cfg.training_metrics_path)

            # Prefer 4-axis composed F1 for reporting if stronger; still keep best multiclass as model.pkl
            report_name = f"{multi['best_name']} (+ 4-axis models)"
            report_score = multi["best_f1"]
            print(f"\nBest multiclass model: {multi['best_name']} | weighted F1={report_score:.4f}")
            print(
                f"4-axis composed accuracy: {axis['compose_metrics']['accuracy']:.4f} | "
                f"weighted F1={axis['compose_metrics']['f1_weighted']:.4f}"
            )
            return report_score, report_name
        except Exception as e:
            raise CustomException(e, sys)
