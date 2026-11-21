"""Reusable text preprocessing utilities for MBTI posts and user answers."""

from __future__ import annotations

import html
import re
import sys
from typing import Iterable, List, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.exception import CustomException
from src.logger import logging

# Keep polarity cues that matter for personality / opinion text
NEGATION_KEEP: Set[str] = {"not", "no", "never", "nor", "neither", "n't"}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def ensure_nltk_data(packages: Optional[Iterable[str]] = None) -> None:
    """Download required NLTK assets once if missing."""
    packages = list(packages or ["stopwords", "wordnet", "omw-1.4"])
    for package in packages:
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            logging.info(f"Downloading NLTK package: {package}")
            nltk.download(package, quiet=True)


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_html_tags(text: str) -> str:
    text = HTML_TAG_PATTERN.sub(" ", text)
    return html.unescape(text)


def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub(" ", text)


def remove_forum_separators(text: str) -> str:
    """Kaggle MBTI posts join multiple posts with |||."""
    return text.replace("|||", " ")


def remove_punctuation_and_numbers(text: str) -> str:
    return NON_ALPHA_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


def build_stopwords(preserve_negations: bool = True) -> Set[str]:
    ensure_nltk_data()
    stops = set(stopwords.words("english"))
    if preserve_negations:
        stops -= NEGATION_KEEP
    return stops


class TextPreprocessor:
    """
    Modular text cleaner shared by training and inference.

    Why this matters: forum posts contain URLs, HTML, emojis, and noise that
    dilute TF-IDF signal. Shared cleaning avoids train/serve skew.
    """

    def __init__(self, preserve_negations: bool = True, min_token_len: int = 2) -> None:
        ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = build_stopwords(preserve_negations=preserve_negations)
        self.min_token_len = min_token_len

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for token in text.split():
            if len(token) < self.min_token_len:
                continue
            if token in self.stop_words:
                continue
            tokens.append(self.lemmatizer.lemmatize(token))
        return tokens

    def clean_text(self, text: Optional[str]) -> str:
        try:
            if text is None or not isinstance(text, str):
                return ""
            cleaned = text
            cleaned = to_lowercase(cleaned)
            cleaned = remove_urls(cleaned)
            cleaned = remove_html_tags(cleaned)
            cleaned = remove_emojis(cleaned)
            cleaned = remove_forum_separators(cleaned)
            cleaned = remove_punctuation_and_numbers(cleaned)
            cleaned = normalize_whitespace(cleaned)
            tokens = self.tokenize_and_lemmatize(cleaned)
            return " ".join(tokens)
        except Exception as e:
            raise CustomException(e, sys)

    def clean_many(self, texts: Iterable[str]) -> List[str]:
        return [self.clean_text(t) for t in texts]
