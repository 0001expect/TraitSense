"""Training pipeline: ingest → transform → train/evaluate → persist artifacts."""

from __future__ import annotations

import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("======= Upgraded training pipeline started =======")
            if CONFIG.train.quick_mode:
                print("Quick mode ON (MBTI_QUICK=1) — skipping RF/XGB + hyperparameter search.")
                logging.info("Quick mode enabled")

            train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()
            bundle = DataTransformation().initiate_data_transformation(
                train_data_path, test_data_path
            )
            best_score, best_model_name = ModelTrainer().initiate_model_trainer(bundle)

            logging.info(
                f"======= Training finished | {best_model_name} | "
                f"weighted F1={best_score:.4f} ======="
            )
            print(f"Best model: {best_model_name}")
            print(f"Best model weighted F1: {best_score:.4f}")
            print("Artifacts saved under artifacts/ (models, metrics, config).")
            return best_score, best_model_name
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().run_pipeline()
