"""Reusable preprocessing pipeline (numeric + categorical + text)."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_DESCRIPTION, TEXT_NAME,
)


def build_sparse_preprocessor() -> ColumnTransformer:
    """Sparse TF-IDF preprocessor (used by the final LightGBM model)."""
    numeric_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # min_frequency groups rare brands / sub-categories into one column;
    # handle_unknown lets the app accept brands never seen in training
    categorical_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=30,
        )),
    ])

    name_tfidf = TfidfVectorizer(
        stop_words="english", max_features=3000, ngram_range=(1, 2), min_df=3
    )
    description_tfidf = TfidfVectorizer(
        stop_words="english", max_features=5000, ngram_range=(1, 2), min_df=5
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_preprocessor, NUMERIC_FEATURES),
            ("cat", categorical_preprocessor, CATEGORICAL_FEATURES),
            ("name_text", name_tfidf, TEXT_NAME),
            ("desc_text", description_tfidf, TEXT_DESCRIPTION),
        ],
        remainder="drop",
    )
