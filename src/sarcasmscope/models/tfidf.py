"""Offline baseline: character + word TF-IDF with logistic regression.

Character n-grams make this robust to spelling variation and to Roman-script transliteration,
which is why it is a strong baseline for code-mixed text. It needs no GPU and no downloads.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from ..text.scripts import tokenize_words


class TfidfLR:
    kind = "tfidf_lr"

    def __init__(self, C: float = 4.0, char_range=(2, 5), word_range=(1, 2), min_df: int = 1,
                 max_features: int = 300000, class_weight="balanced", seed: int = 42):
        self.C, self.char_range, self.word_range = C, tuple(char_range), tuple(word_range)
        self.min_df, self.max_features, self.class_weight, self.seed = min_df, max_features, class_weight, seed
        self.pipe = None

    def _build(self):
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_range, sublinear_tf=True,
                               min_df=self.min_df, max_features=self.max_features, lowercase=False)
        word = TfidfVectorizer(tokenizer=tokenize_words, token_pattern=None, ngram_range=self.word_range,
                               sublinear_tf=True, min_df=self.min_df, max_features=self.max_features, lowercase=False)
        clf = LogisticRegression(C=self.C, max_iter=3000, class_weight=self.class_weight, random_state=self.seed)
        return make_pipeline(FeatureUnion([("char", char), ("word", word)]), clf)

    def fit(self, texts, y, **_):
        self.pipe = self._build()
        self.pipe.fit(list(texts), np.asarray(y))
        return self

    def predict_proba(self, texts, contexts=None) -> np.ndarray:
        """(n, 2) array; column 1 = P(sarcastic)."""
        return self.pipe.predict_proba(list(texts))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TfidfLR":
        return joblib.load(path)
