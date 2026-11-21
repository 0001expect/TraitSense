"""Load, clean, summarize, and split the Kaggle MBTI dataset."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging
from src.utils import save_json


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    stats_path: str = os.path.join("artifacts", "metrics", "dataset_stats.json")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
        self.quality = CONFIG.data_quality

    def _print_and_save_stats(self, df: pd.DataFrame, stage: str) -> dict:
        counts = df["type"].value_counts().to_dict()
        stats = {
            "stage": stage,
            "rows": int(len(df)),
            "unique_types": int(df["type"].nunique()),
            "missing_type": int(df["type"].isna().sum()),
            "missing_posts": int(df["posts"].isna().sum()),
            "avg_word_count": float(df["word_count"].mean()) if "word_count" in df else None,
            "median_word_count": float(df["word_count"].median()) if "word_count" in df else None,
            "class_distribution": {str(k): int(v) for k, v in counts.items()},
        }

        print("\n========== Dataset Summary ==========")
        print(f"Stage              : {stage}")
        print(f"Rows               : {stats['rows']}")
        print(f"Unique MBTI types  : {stats['unique_types']}")
        if stats["avg_word_count"] is not None:
            print(f"Avg word count     : {stats['avg_word_count']:.1f}")
            print(f"Median word count  : {stats['median_word_count']:.1f}")
        print("Class distribution :")
        for mbti_type, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / max(len(df), 1)
            print(f"  {mbti_type}: {count} ({pct:.1f}%)")
        print("=====================================\n")

        logging.info(str(stats))
        return stats

    def _quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove empty / tiny / duplicate / noisy posts.

        Why: very short forum scrapes and duplicates inflate accuracy and
        hide imbalance; filtering yields a more realistic training set.
        """
        before = len(df)
        df = df.dropna(subset=["type", "posts"]).copy()
        df["posts"] = df["posts"].astype(str).str.strip()
        df = df[df["posts"].str.len() > 0]
        df["word_count"] = df["posts"].str.split().str.len()
        df["char_count"] = df["posts"].str.len()

        df = df[df["word_count"] >= self.quality.min_word_count]
        df = df[df["char_count"] >= self.quality.min_char_count]

        # Drop posts that are almost entirely URLs / separators
        noise_ratio = df["posts"].str.count(r"https?://|\|\|\|") / df["word_count"].clip(lower=1)
        df = df[noise_ratio < 0.35]

        if self.quality.remove_duplicates:
            df = df.drop_duplicates(subset=["posts"])
            df = df.drop_duplicates(subset=["type", "posts"])

        # Keep only valid 4-letter MBTI codes
        valid_types = {
            a + b + c + d
            for a in CONFIG.axes["IE"]
            for b in CONFIG.axes["NS"]
            for c in CONFIG.axes["TF"]
            for d in CONFIG.axes["JP"]
        }
        df = df[df["type"].isin(valid_types)]

        logging.info(f"Quality filter: {before} -> {len(df)} rows")
        return df.reset_index(drop=True)

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        logging.info("Entered the data ingestion method")
        try:
            raw_path = os.path.join("notebook", "data", "mbti_1.csv")
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"Dataset not found at {raw_path}. "
                    "Download mbti_1.csv from Kaggle and place it in notebook/data/"
                )

            df = pd.read_csv(raw_path)
            logging.info("Read the MBTI dataset as dataframe")

            if "type" not in df.columns or "posts" not in df.columns:
                raise ValueError("Dataset must contain 'type' and 'posts' columns")

            df = df[["type", "posts"]]
            raw_stats = self._print_and_save_stats(
                df.assign(word_count=df["posts"].astype(str).str.split().str.len()),
                stage="raw",
            )

            df = self._quality_filter(df)
            clean_stats = self._print_and_save_stats(df, stage="cleaned")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            save_json(
                self.ingestion_config.stats_path,
                {"raw": raw_stats, "cleaned": clean_stats},
            )

            logging.info("Train test split initiated (stratified by type)")
            train_set, test_set = train_test_split(
                df,
                test_size=CONFIG.train.test_size,
                random_state=CONFIG.train.random_state,
                stratify=df["type"],
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
