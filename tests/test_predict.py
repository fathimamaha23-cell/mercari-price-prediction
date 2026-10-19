import os

import pandas as pd
import pytest

from src.predict import DEFAULT_MODEL_PATH, load_model, predict_price

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_MODEL_PATH),
    reason="trained model file not found - run python -m src.train first",
)


def test_prediction_is_positive_and_one_per_listing():
    listings = pd.DataFrame([
        {"name": "Nike shoes", "item_condition_id": 2, "category_name": "Women/Shoes/Athletic",
         "brand_name": "Nike", "shipping": 0, "item_description": "Worn twice."},
        {"name": "Baby onesie", "item_condition_id": 1, "category_name": "Kids/Girls 0-24 Mos/One-Pieces",
         "brand_name": "", "shipping": 1, "item_description": ""},
    ])
    prices = predict_price(listings, load_model())
    assert len(prices) == 2
    assert (prices >= 0).all()


def test_missing_column_raises_error():
    with pytest.raises(ValueError):
        predict_price(pd.DataFrame([{"name": "x"}]), model=None)
