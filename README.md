# TraitSense – MBTI Personality Prediction

A machine learning project that predicts Myers–Briggs (MBTI) personality types using written text.

The project uses text preprocessing, TF-IDF features, multiple classification models, and ensemble voting to explore how language patterns can be used to estimate personality types. It also includes four binary dimension models, model evaluation, explainability, and a Flask-based questionnaire interface.

> **Disclaimer:** Predictions are based on language patterns learned from the training data. They are only estimates and should not be considered a definitive psychological assessment.

## Architecture

```text
User answers (6 prompts)
        ↓
Text cleaning (shared preprocessor)
        ↓
TF-IDF (1–2 grams, 15k features)
        ↓
┌───────────────────────┬────────────────────────────┐
│ 4 binary models       │ Multiclass + soft ensemble │
│ I/E  N/S  T/F  J/P    │ LR / SVM / NB / RF / XGB   │
└──────────┬────────────┴─────────────┬──────────────┘
           ↓                          ↓
     Compose e.g. ENTP         Comparison in UI
           ↓
 Confidence + top influential words + profile
```

### Why four binary models?

Instead of predicting all 16 personality types directly, the project also predicts each of the four MBTI dimensions separately.

This allows the model to learn the letter pairs individually (I/E, N/S, T/F, J/P), which are then combined to form a personality type such as ENTP.

## Folder structure

```text
TraitSense/
├── application.py                 # Flask entry
├── requirements.txt
├── setup.py
├── artifacts/                     # Trained model files
│   ├── models/                    # Individual models and ensemble
│   ├── metrics/                   # Evaluation metrics and confusion matrices
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   ├── axis_encoders.pkl
│   ├── model.pkl                  # Best multiclass model
│   └── config.json
├── notebook/
│   ├── 1.EDA_MBTI.ipynb
│   └── data/mbti_1.csv            # Dataset (Kaggle)
├── src/
│   ├── config.py
│   ├── mbti_profiles.py
│   ├── exception.py / logger.py / utils.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── data_transformation.py
│   │   ├── feature_engineering.py
│   │   ├── model_trainer.py
│   │   ├── evaluation.py
│   │   ├── explainability.py
│   │   └── prediction.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   └── optional/distilbert_compare.py
├── static/css/style.css
└── templates/
```

## Dataset

1. Download the [MBTI Type dataset from Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type).
2. Place the downloaded `mbti_1.csv` file inside `notebook/data/`.

## Windows + VS Code setup

```powershell
cd E:\TraitSense
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the models

To run the complete training pipeline, including hyperparameter tuning, oversampling, and model comparison:

```powershell
python -m src.pipeline.train_pipeline
```

For quicker experimentation on a laptop, use:

```powershell
$env:MBTI_QUICK="1"
python -m src.pipeline.train_pipeline
```

The trained files are saved inside `artifacts/`, including:

- Best multiclass model and other model pickles
- Four binary dimension models
- Ensemble voter
- `training_metrics.json` and confusion-matrix PNGs
- `config.json`

## Run the Flask application

```powershell
python application.py
```

Once the server starts, open:

http://127.0.0.1:5000

## What was improved?

| Area | Upgrade |
|------|---------|
| Preprocessing | URLs, HTML, emojis, negations kept, lemmatization |
| Data quality | Duplicates and short/noisy posts removed, with printed stats |
| Imbalance | `class_weight=balanced` and RandomOverSampler comparison |
| Features | Configurable TF-IDF (15k, bigrams, sublinear_tf) |
| Models | LR, Linear SVM, NB, RF, XGBoost, and soft voting ensemble |
| Tuning | RandomizedSearchCV with cross-validation |
| Target design | Four binary MBTI dimensions combined into 16 types |
| Metrics | Accuracy, precision, recall, macro/weighted F1, confusion matrices |
| Explainability | Top influential TF-IDF terms for each prediction |
| UI | Multi-question form with personality profiles, careers, styles, and disclaimer |

## Optional DistilBERT comparison

The main classical ML pipeline works independently. If you want to experiment with a deep-learning approach, you can run:

```powershell
python -m src.optional.distilbert_compare
```

Install `transformers` and `torch` separately if you want to fine-tune DistilBERT.

## Limitations

- The dataset contains forum-style writing, which may differ from everyday conversations or personal writing.
- MBTI has its limitations as a personality assessment framework, so predictions should be treated as estimates.
- Class imbalance can still affect predictions for less-represented personality types.
- Feature importance shows correlations in the data, not cause-and-effect relationships.

## Future improvements

- Calibrated probabilities for each personality dimension
- Testing the model on different types of written text
- Further DistilBERT experiments with a frozen encoder
- API rate limiting and input-length restrictions for deployment