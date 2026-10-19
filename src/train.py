"""Train the final model and save it with its metadata.

Run from the project root:
    python -m src.train --data data/train.tsv --version v1
Add --mlflow to also log the run to MLflow (sqlite:///mlflow.db).
"""
import argparse
import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import ALL_FEATURES, create_features
from src.preprocessing import build_sparse_preprocessor

RANDOM_STATE = 42

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=63,
    colsample_bytree=0.3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


def load_data(path: str, sample_size: int) -> pd.DataFrame:
    """Load the Mercari train.tsv, take a seeded sample and remove price = 0."""
    full = pd.read_csv(path, sep="\t")
    if sample_size and sample_size < len(full):
        full = full.sample(n=sample_size, random_state=RANDOM_STATE)
    data = full.reset_index(drop=True)
    data = data[data["price"] > 0].reset_index(drop=True)
    return data


def build_pipeline() -> Pipeline:
    from lightgbm import LGBMRegressor

    return Pipeline([
        ("preprocessing", build_sparse_preprocessor()),
        ("model", LGBMRegressor(**LGBM_PARAMS)),
    ])


def evaluate(pipeline, X_test, y_test_log, y_test_price) -> dict:
    pred_log = np.clip(pipeline.predict(X_test), 0, None)
    pred_price = np.expm1(pred_log)
    return {
        "MAE": mean_absolute_error(y_test_price, pred_price),
        "RMSE": mean_squared_error(y_test_price, pred_price) ** 0.5,
        "R2": r2_score(y_test_price, pred_price),
        "RMSLE": mean_squared_error(y_test_log, pred_log) ** 0.5,
        "R2_log": r2_score(y_test_log, pred_log),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the Mercari price model")
    parser.add_argument("--data", default="data/train.tsv")
    parser.add_argument("--sample", type=int, default=100_000,
                        help="rows to sample from the full file (0 = use all)")
    parser.add_argument("--output", default="models")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    data = load_data(args.data, args.sample)
    data["log_price"] = np.log1p(data["price"])
    data = create_features(data)

    X = data[ALL_FEATURES]
    X_train, X_test, y_train, y_test, y_train_price, y_test_price = train_test_split(
        X, data["log_price"], data["price"], test_size=0.20, random_state=RANDOM_STATE
    )
    print("Train:", X_train.shape, "| Test:", X_test.shape)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)          # trained on log1p(price)
    metrics = evaluate(pipeline, X_test, y_test, y_test_price)
    print({k: round(float(v), 4) for k, v in metrics.items()})

    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, f"price_model_{args.version}.joblib")
    joblib.dump(pipeline, model_path)

    import lightgbm
    metadata = {
        "model_version": args.version,
        "model": "LightGBM on sparse TF-IDF (name + description) + brand + category + numeric",
        "target": "log1p(price); predictions converted back to USD with expm1",
        "trained_on": str(datetime.date.today()),
        "train_rows": int(len(X_train)),
        "random_state": RANDOM_STATE,
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    meta_path = os.path.join(args.output, f"price_model_{args.version}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved:", model_path, "and", meta_path)

    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("mercari-price-prediction")
        with mlflow.start_run(run_name=f"train.py - LightGBM {args.version}"):
            mlflow.set_tag("stage", "script")
            mlflow.log_params({f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()
                               if k in ["n_estimators", "learning_rate",
                                        "num_leaves", "colsample_bytree"]})
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("train_rows", len(X_train))
            for name, value in metrics.items():
                mlflow.log_metric(name, float(value))
            mlflow.log_artifact(meta_path)


if __name__ == "__main__":
    main()
