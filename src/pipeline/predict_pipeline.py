"""Inference: 4-axis models + optional ensemble, confidence, explanations."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from src.components.data_preprocessing import TextPreprocessor
from src.components.explainability import collect_axis_explanations
from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging
from src.mbti_profiles import DISCLAIMER, MBTI_DESCRIPTIONS, MBTI_PROFILES
from src.utils import decision_confidence, load_object

# Re-export for Flask / older imports
__all__ = [
    "PredictPipeline",
    "MBTI_DESCRIPTIONS",
    "MBTI_PROFILES",
    "DISCLAIMER",
]


class PredictPipeline:
    """
    Primary path: four binary axis models → composed MBTI type.
    Secondary: multiclass / ensemble vote for comparison in the UI.
    """

    def __init__(self) -> None:
        self.text_preprocessor = TextPreprocessor(preserve_negations=True)
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")
        self.axis_encoders_path = os.path.join("artifacts", "axis_encoders.pkl")
        self.axis_models_path = os.path.join("artifacts", "models", "axis_models.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.ensemble_path = os.path.join("artifacts", "models", "ensemble_voter.pkl")

        self._vectorizer = None
        self._type_encoder = None
        self._axis_encoders = None
        self._axis_models = None
        self._multiclass_model = None
        self._ensemble_model = None

    def clean_text(self, text: str) -> str:
        return self.text_preprocessor.clean_text(text)

    def _load_artifacts(self) -> None:
        if self._vectorizer is None:
            self._vectorizer = load_object(self.preprocessor_path)
            self._type_encoder = load_object(self.label_encoder_path)
            self._axis_encoders = load_object(self.axis_encoders_path)
            self._axis_models = load_object(self.axis_models_path)
            if os.path.exists(self.model_path):
                self._multiclass_model = load_object(self.model_path)
            if os.path.exists(self.ensemble_path):
                self._ensemble_model = load_object(self.ensemble_path)

    @staticmethod
    def combine_answers(answers: Dict[str, str]) -> str:
        """Merge questionnaire fields into one document for the model."""
        parts: List[str] = []
        for key, value in answers.items():
            value = (value or "").strip()
            if value:
                parts.append(value)
        return "\n".join(parts)

    def _predict_axes(self, features) -> Tuple[str, Dict[str, Any]]:
        letters: List[str] = []
        axis_details: Dict[str, Any] = {}

        for i, (axis_name, pair) in enumerate(CONFIG.axes.items()):
            model = self._axis_models[axis_name]
            enc = self._axis_encoders[axis_name]
            pred_idx = int(model.predict(features)[0])
            letter = enc.inverse_transform([pred_idx])[0]
            conf = decision_confidence(model, features, pred_idx)
            letters.append(letter)
            axis_details[axis_name] = {
                "letter": letter,
                "pair": list(pair),
                "confidence": conf,
            }
        return "".join(letters), axis_details

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Backward-compatible entry: returns a rich dict.
        Older callers expecting (type, confidence) should use predict_legacy.
        """
        try:
            cleaned = self.clean_text(text)
            if not cleaned or len(cleaned.split()) < 8:
                raise ValueError(
                    "Please provide more detail (combine answers totaling "
                    "roughly 80+ words) so the model has enough signal."
                )

            self._load_artifacts()
            features = self._vectorizer.transform([cleaned])

            personality_type, axis_details = self._predict_axes(features)
            axis_confidences = [
                d["confidence"] for d in axis_details.values() if d["confidence"] is not None
            ]
            overall_confidence = (
                float(sum(axis_confidences) / len(axis_confidences))
                if axis_confidences
                else None
            )

            explanation = collect_axis_explanations(
                self._axis_models,
                self._vectorizer,
                cleaned,
                CONFIG.axes,
                personality_type,
                top_k=6,
            )

            ensemble_type = None
            ensemble_confidence = None
            if self._ensemble_model is not None:
                e_idx = int(self._ensemble_model.predict(features)[0])
                ensemble_type = self._type_encoder.inverse_transform([e_idx])[0]
                ensemble_confidence = decision_confidence(self._ensemble_model, features, e_idx)

            multiclass_type = None
            multiclass_confidence = None
            if self._multiclass_model is not None:
                m_idx = int(self._multiclass_model.predict(features)[0])
                multiclass_type = self._type_encoder.inverse_transform([m_idx])[0]
                multiclass_confidence = decision_confidence(
                    self._multiclass_model, features, m_idx
                )

            profile = MBTI_PROFILES.get(personality_type, {})
            logging.info(f"Predicted {personality_type} (4-axis) | ensemble={ensemble_type}")

            return {
                "personality_type": personality_type,
                "confidence": overall_confidence,
                "axis_details": axis_details,
                "influential_words": explanation.get("top_influential_words", []),
                "explanation": explanation,
                "profile": profile,
                "description": MBTI_DESCRIPTIONS.get(personality_type, ""),
                "ensemble_type": ensemble_type,
                "ensemble_confidence": ensemble_confidence,
                "multiclass_type": multiclass_type,
                "multiclass_confidence": multiclass_confidence,
                "disclaimer": DISCLAIMER,
            }
        except Exception as e:
            raise CustomException(e, sys)

    def predict_legacy(self, text: str) -> Tuple[str, Optional[float]]:
        result = self.predict(text)
        return result["personality_type"], result["confidence"]
