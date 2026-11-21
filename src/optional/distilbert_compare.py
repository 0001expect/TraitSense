"""Optional DistilBERT baseline for comparison with classical TF-IDF models.

This keeps the classical ML pipeline intact. Install extras only if you run this:

    pip install transformers torch datasets evaluate accelerate

Then:

    python -m src.optional.distilbert_compare

Expected outcome for a portfolio:
- DistilBERT often improves text classification when fine-tuned
- TF-IDF + Linear models remain faster, cheaper, and easier to explain
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DistilBERT vs classical metrics file")
    parser.add_argument(
        "--metrics",
        default=os.path.join("artifacts", "metrics", "training_metrics.json"),
        help="Path to classical training metrics JSON",
    )
    args = parser.parse_args()

    print(
        "Optional DistilBERT comparison\n"
        "--------------------------------\n"
        "1) Train the classical pipeline first so metrics exist.\n"
        "2) Install: pip install transformers torch datasets evaluate accelerate\n"
        "3) Fine-tune DistilBERT on notebook/data/mbti_1.csv (type + posts).\n"
        "4) Record accuracy / macro-F1 / weighted-F1 beside the classical scores.\n"
    )

    if os.path.exists(args.metrics):
        with open(args.metrics, encoding="utf-8") as f:
            metrics = json.load(f)
        print("Classical pipeline snapshot:")
        multi = metrics.get("multiclass", {})
        for name, vals in multi.items():
            if isinstance(vals, dict) and "accuracy" in vals:
                print(
                    f"  {name}: acc={vals['accuracy']:.4f} "
                    f"f1_w={vals['f1_weighted']:.4f} f1_macro={vals['f1_macro']:.4f}"
                )
        composed = metrics.get("four_axis_composed", {})
        if composed:
            print(
                "  4-axis composed: "
                f"acc={composed.get('accuracy', 0):.4f} "
                f"f1_w={composed.get('f1_weighted', 0):.4f}"
            )
    else:
        print(f"No classical metrics found at {args.metrics}. Train first.")

    print(
        "\nWhy keep both?\n"
        "- Classical TF-IDF models: interpretable coefficients, fast CPU training.\n"
        "- DistilBERT: stronger contextual understanding, heavier GPU/CPU cost.\n"
        "For a final-year project, showing this tradeoff is more valuable than\n"
        "replacing the classical stack entirely.\n"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
