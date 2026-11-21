"""Rule-based MBTI scoring for the Likert questionnaire (not ML)."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Mapping

from src.exception import CustomException
from src.mbti_profiles import DISCLAIMER, MBTI_PROFILES
from src.questionnaire.questions import QUESTIONNAIRE_ITEMS

# Axis letter pairs: index 0 = first letter, index 1 = second
AXIS_PAIRS = {
    "IE": ("I", "E"),
    "NS": ("N", "S"),
    "TF": ("T", "F"),
    "JP": ("J", "P"),
}


def score_questionnaire(answers: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Score answers keyed by question id (str or int) with Likert values 1–5.

    Positive centered score (+2..-2) pushes toward `towards` letter.
    """
    try:
        if len(answers) < len(QUESTIONNAIRE_ITEMS):
            missing = [
                q["id"] for q in QUESTIONNAIRE_ITEMS if str(q["id"]) not in {str(k) for k in answers}
            ]
            raise ValueError(f"Please answer all questions. Missing: {missing[:5]}...")

        # Raw push toward second letter of each pair (E/S/F/P): positive favors that letter
        deltas = {axis: 0.0 for axis in AXIS_PAIRS}
        max_abs = {axis: 0.0 for axis in AXIS_PAIRS}

        by_id = {q["id"]: q for q in QUESTIONNAIRE_ITEMS}

        for qid, raw in answers.items():
            q = by_id.get(int(qid))
            if q is None:
                continue
            value = int(raw)
            if value < 1 or value > 5:
                raise ValueError(f"Invalid Likert value for question {qid}: {value}")

            centered = float(value - 3)  # -2 .. +2
            axis = q["axis"]
            left, right = AXIS_PAIRS[axis]
            towards = q["towards"]

            if towards == right:
                deltas[axis] += centered
            elif towards == left:
                deltas[axis] -= centered
            else:
                raise ValueError(f"Invalid towards letter {towards} for axis {axis}")

            max_abs[axis] += 2.0

        letters: List[str] = []
        dimensions: Dict[str, Any] = {}
        confidences: List[float] = []

        for axis, (left, right) in AXIS_PAIRS.items():
            delta = deltas[axis]
            chosen = right if delta >= 0 else left
            # Strength: how far from neutral relative to theoretical max
            strength = abs(delta) / max_abs[axis] if max_abs[axis] else 0.0
            # Convert to % leaning toward chosen letter (50% = tie, 100% = max)
            pct_chosen = 50.0 + (strength * 50.0)
            pct_left = 50.0 - (delta / max_abs[axis] * 50.0) if max_abs[axis] else 50.0
            pct_right = 100.0 - pct_left

            letters.append(chosen)
            confidences.append(pct_chosen)
            dimensions[axis] = {
                "left": left,
                "right": right,
                "chosen": chosen,
                "delta": round(delta, 2),
                "percent_left": round(pct_left, 1),
                "percent_right": round(pct_right, 1),
                "confidence": round(pct_chosen, 1),
            }

        mbti_type = "".join(letters)
        overall_confidence = round(sum(confidences) / len(confidences), 1)
        profile = MBTI_PROFILES.get(mbti_type, {})

        return {
            "source": "questionnaire",
            "personality_type": mbti_type,
            "confidence": overall_confidence,
            "dimensions": dimensions,
            "profile": profile,
            "disclaimer": DISCLAIMER,
        }
    except Exception as e:
        raise CustomException(e, sys)
