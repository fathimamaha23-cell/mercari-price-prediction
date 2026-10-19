"""Single and batch inference.

Batch inference from the project root:
    python -m src.predict --input data/sample_new_listings.csv --output data/predictions.csv
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.features import RAW_COLUMNS, create_features

DEFAULT_MODEL_PATH = os.path.join("models", "price_model_v1.joblib")


def load_model(path: str = DEFAULT_MODEL_PATH):
    return joblib.load(path)


def load_metadata(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Read the version / metrics file that sits next to the model file."""
    meta_path = model_path.replace(".joblib", "_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def predict_price(listings: pd.DataFrame, model) -> np.ndarray:
    """Return the estimated selling price in USD for each listing."""
    missing = [c for c in RAW_COLUMNS if c not in listings.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = listings[RAW_COLUMNS].copy()
    # blank brand / description are treated as missing, like in training
    df["brand_name"] = df["brand_name"].replace(r"^\s*$", np.nan, regex=True)
    df["item_description"] = df["item_description"].replace(r"^\s*$", np.nan, regex=True)

    features = create_features(df)
    pred_log = np.clip(model.predict(features), 0, None)
    return np.round(np.expm1(pred_log), 2)


def main():
    parser = argparse.ArgumentParser(description="Batch price prediction")
    parser.add_argument("--input", required=True, help="CSV with raw listing columns")
    parser.add_argument("--output", default="data/predictions.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    model = load_model(args.model)
    listings = pd.read_csv(args.input)
    listings["predicted_price_usd"] = predict_price(listings, model)
    listings.to_csv(args.output, index=False)

    version = load_metadata(args.model).get("model_version", "unknown")
    print(f"Predicted {len(listings)} listings with model {version} -> {args.output}")


if __name__ == "__main__":
    main()
