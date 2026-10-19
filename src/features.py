"""Feature engineering for the Mercari price prediction project.

The same function is used for training and for inference, so the model
always receives features built in exactly the same way.
"""
import pandas as pd

# Raw columns a new listing must provide
RAW_COLUMNS = [
    "name",
    "item_condition_id",
    "category_name",
    "brand_name",
    "shipping",
    "item_description",
]

NUMERIC_FEATURES = [
    "item_condition_id", "shipping", "name_length", "name_word_count",
    "desc_length", "desc_word_count", "no_description", "rm_count", "has_brand",
]
CATEGORICAL_FEATURES = ["brand_name", "cat_main", "cat_sub1", "cat_sub2"]
TEXT_NAME = "name"
TEXT_DESCRIPTION = "item_description"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_NAME, TEXT_DESCRIPTION]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features from raw listing columns."""
    df = df.copy()

    # fill missing values
    df["name"] = df["name"].fillna("").astype(str)
    df["item_description"] = df["item_description"].fillna("").astype(str)
    df["brand_name"] = df["brand_name"].fillna("missing")
    df["category_name"] = df["category_name"].fillna("missing/missing/missing")

    # text length features
    df["name_length"] = df["name"].str.len()
    df["name_word_count"] = df["name"].str.split().str.len()
    df["desc_length"] = df["item_description"].str.len()
    df["desc_word_count"] = df["item_description"].str.split().str.len()

    # description quality flags
    df["no_description"] = (
        (df["item_description"] == "") |
        (df["item_description"] == "No description yet")
    ).astype(int)
    df["rm_count"] = df["item_description"].str.count(r"\[rm\]")  # price removed by Mercari

    # brand flag
    df["has_brand"] = (df["brand_name"] != "missing").astype(int)

    # split category path into 3 levels
    cats = df["category_name"].str.split("/", n=2, expand=True)
    for position in range(3):
        if position not in cats.columns:       # category without all 3 levels
            cats[position] = None
    df["cat_main"] = cats[0].fillna("missing")
    df["cat_sub1"] = cats[1].fillna("missing")
    df["cat_sub2"] = cats[2].fillna("missing")

    return df
