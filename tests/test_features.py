import pandas as pd

from src.features import ALL_FEATURES, RAW_COLUMNS, create_features


def sample_listing(**overrides):
    row = {
        "name": "Nike Air Max shoes",
        "item_condition_id": 2,
        "category_name": "Women/Shoes/Athletic",
        "brand_name": "Nike",
        "shipping": 0,
        "item_description": "Worn twice [rm] with box",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_all_model_features_are_created():
    features = create_features(sample_listing())
    assert all(col in features.columns for col in ALL_FEATURES)


def test_category_is_split_into_three_levels():
    row = create_features(sample_listing()).iloc[0]
    assert (row["cat_main"], row["cat_sub1"], row["cat_sub2"]) == ("Women", "Shoes", "Athletic")


def test_missing_brand_and_category_get_default_values():
    row = create_features(sample_listing(brand_name=None, category_name=None)).iloc[0]
    assert row["brand_name"] == "missing"
    assert row["has_brand"] == 0
    assert row["cat_main"] == "missing"


def test_no_description_flag_and_rm_count():
    row = create_features(sample_listing(item_description="No description yet")).iloc[0]
    assert row["no_description"] == 1
    row = create_features(sample_listing()).iloc[0]
    assert row["rm_count"] == 1


def test_short_category_path_does_not_crash():
    row = create_features(sample_listing(category_name="Women")).iloc[0]
    assert row["cat_sub1"] == "missing" and row["cat_sub2"] == "missing"


def test_raw_columns_constant_matches_listing():
    assert set(RAW_COLUMNS) <= set(sample_listing().columns)
