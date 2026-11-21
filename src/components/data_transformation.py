"""Clean text, build TF-IDF features, and derive 16-class + 4-axis labels."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.components.data_preprocessing import TextPreprocessor
from src.components.feature_engineering import build_tfidf_vectorizer, vectorizer_params
from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging
from src.utils import save_json, save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_file_path: str = os.path.join("artifacts", "label_encoder.pkl")
    cleaned_train_path: str = os.path.join("artifacts", "train_cleaned.csv")
    cleaned_test_path: str = os.path.join("artifacts", "test_cleaned.csv")
    transform_meta_path: str = os.path.join("artifacts", "metrics", "transform_meta.json")


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.preprocessor = TextPreprocessor(preserve_negations=True)

    def clean_text(self, text: str) -> str:
        """Public wrapper so predict pipeline / notebooks can reuse cleaning."""
        return self.preprocessor.clean_text(text)

    def get_data_transformer_object(self):
        return build_tfidf_vectorizer()

    @staticmethod
    def add_axis_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Split 16-class type into four binary personality dimensions."""
        out = df.copy()
        out["IE"] = out["type"].str[0]
        out["NS"] = out["type"].str[1]
        out["TF"] = out["type"].str[2]
        out["JP"] = out["type"].str[3]
        return out

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Dict[str, Any]:
        """
        Returns a dict with matrices and label encoders for 16-class + axes.

        Why return a dict: the trainer now needs both multiclass and four
        binary targets without re-reading CSVs.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data for transformation")

            logging.info("Cleaning text posts (lemmatization + stopwords + noise removal)")
            train_df["clean_posts"] = train_df["posts"].astype(str).map(self.preprocessor.clean_text)
            test_df["clean_posts"] = test_df["posts"].astype(str).map(self.preprocessor.clean_text)

            train_df = train_df[train_df["clean_posts"].str.split().str.len() >= 10]
            test_df = test_df[test_df["clean_posts"].str.split().str.len() >= 10]

            train_df = self.add_axis_labels(train_df)
            test_df = self.add_axis_labels(test_df)

            train_df.to_csv(self.data_transformation_config.cleaned_train_path, index=False)
            test_df.to_csv(self.data_transformation_config.cleaned_test_path, index=False)

            type_encoder = LabelEncoder()
            y_train = type_encoder.fit_transform(train_df["type"])
            y_test = type_encoder.transform(test_df["type"])

            axis_encoders: Dict[str, LabelEncoder] = {}
            axis_y_train: Dict[str, Any] = {}
            axis_y_test: Dict[str, Any] = {}
            for axis_name, letters in CONFIG.axes.items():
                enc = LabelEncoder()
                enc.fit(list(letters))
                axis_encoders[axis_name] = enc
                axis_y_train[axis_name] = enc.transform(train_df[axis_name])
                axis_y_test[axis_name] = enc.transform(test_df[axis_name])

            vectorizer = self.get_data_transformer_object()
            logging.info("Fitting TF-IDF on training posts")
            X_train = vectorizer.fit_transform(train_df["clean_posts"])
            X_test = vectorizer.transform(test_df["clean_posts"])

            save_object(self.data_transformation_config.preprocessor_obj_file_path, vectorizer)
            save_object(self.data_transformation_config.label_encoder_file_path, type_encoder)
            save_object(os.path.join("artifacts", "axis_encoders.pkl"), axis_encoders)

            meta = {
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "n_features": int(X_train.shape[1]),
                "tfidf": vectorizer_params(vectorizer),
                "classes": type_encoder.classes_.tolist(),
                "axes": {k: list(v) for k, v in CONFIG.axes.items()},
            }
            save_json(self.data_transformation_config.transform_meta_path, meta)
            logging.info(f"Transformation meta: {meta}")

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "axis_y_train": axis_y_train,
                "axis_y_test": axis_y_test,
                "type_encoder": type_encoder,
                "axis_encoders": axis_encoders,
                "vectorizer": vectorizer,
                "train_df": train_df,
                "test_df": test_df,
                "preprocessor_path": self.data_transformation_config.preprocessor_obj_file_path,
            }
        except Exception as e:
            raise CustomException(e, sys)
