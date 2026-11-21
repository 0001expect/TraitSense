"""Flask application for PersonaType — ML text analysis + Likert questionnaire."""

from __future__ import annotations

import os
import sys

from flask import Flask, jsonify, render_template, request

from src.exception import CustomException
from src.logger import logging
from src.mbti_profiles import DISCLAIMER
from src.pipeline.predict_pipeline import PredictPipeline
from src.questionnaire.questions import questions_payload
from src.questionnaire.scoring import score_questionnaire

app = Flask(__name__)

QUESTION_FIELDS = [
    ("about_you", "Describe yourself in detail (aim for 80–150+ words)"),
    ("weekends", "How do you usually spend weekends?"),
    ("decisions", "How do you make important decisions?"),
    ("friends", "How do your friends describe you?"),
    ("stress", "How do you react under stress?"),
    ("motivation", "What motivates you most?"),
]


@app.route("/")
def index():
    """Landing: choose assessment method."""
    return render_template("index.html", disclaimer=DISCLAIMER)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    """Existing free-text ML prediction flow (unchanged pipeline)."""
    form_values = {name: "" for name, _ in QUESTION_FIELDS}

    if request.method == "GET":
        return render_template(
            "home.html",
            questions=QUESTION_FIELDS,
            form_values=form_values,
            result=None,
            error=None,
            disclaimer=DISCLAIMER,
        )

    try:
        for name, _ in QUESTION_FIELDS:
            form_values[name] = (request.form.get(name) or "").strip()

        combined = PredictPipeline.combine_answers(form_values)
        word_count = len(combined.split())
        if word_count < 40:
            return render_template(
                "home.html",
                questions=QUESTION_FIELDS,
                form_values=form_values,
                result=None,
                error=(
                    f"Please write more detail across the questions "
                    f"(currently ~{word_count} words; aim for 80+)."
                ),
                disclaimer=DISCLAIMER,
            )

        logging.info("Running upgraded personality prediction")
        result = PredictPipeline().predict(combined)

        return render_template(
            "home.html",
            questions=QUESTION_FIELDS,
            form_values=form_values,
            result=result,
            error=None,
            disclaimer=DISCLAIMER,
        )
    except CustomException as e:
        logging.info(str(e))
        return render_template(
            "home.html",
            questions=QUESTION_FIELDS,
            form_values=form_values,
            result=None,
            error=str(e),
            disclaimer=DISCLAIMER,
        )
    except Exception as e:
        logging.info(str(e))
        raise CustomException(e, sys)


@app.route("/questionnaire")
def questionnaire_page():
    """Interactive 40-question Likert assessment (separate from ML)."""
    return render_template(
        "questionnaire.html",
        payload=questions_payload(),
        disclaimer=DISCLAIMER,
    )


@app.route("/questionnaire/score", methods=["POST"])
def questionnaire_score():
    """Score submitted Likert answers and return JSON for the results view."""
    try:
        data = request.get_json(silent=True) or {}
        answers = data.get("answers") or {}
        if not answers:
            return jsonify({"ok": False, "error": "No answers provided."}), 400

        result = score_questionnaire(answers)
        logging.info(f"Questionnaire scored as {result['personality_type']}")
        return jsonify({"ok": True, "result": result})
    except CustomException as e:
        logging.info(str(e))
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logging.info(str(e))
        return jsonify({"ok": False, "error": "Scoring failed. Please try again."}), 500


@app.route("/questionnaire/results")
def questionnaire_results_page():
    """
    Results page shell; details are loaded from sessionStorage by JS
    after a successful /questionnaire/score call.
    """
    return render_template("questionnaire_results.html", disclaimer=DISCLAIMER)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
