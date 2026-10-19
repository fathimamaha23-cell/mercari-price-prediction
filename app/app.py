"""Streamlit app: estimate the selling price of a marketplace listing.

Run from the project root:
    python -m streamlit run app/app.py
"""
import sys
from pathlib import Path

# make `src` importable when Streamlit runs this file from the app/ folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.features import RAW_COLUMNS
from src.predict import load_metadata, load_model, predict_price

MODEL_PATH = ROOT / "models" / "price_model_v1.joblib"
CATEGORY_FILE = ROOT / "models" / "category_options.csv"
BRAND_FILE = ROOT / "models" / "brand_options.csv"

CONDITION_LABELS = {
    1: "1 - Best condition",
    2: "2 - Very good",
    3: "3 - Good",
    4: "4 - Fair",
    5: "5 - Worst condition",
}
NO_BRAND = "(no brand)"

st.set_page_config(page_title="Marketplace Price Predictor", page_icon="🏷️")


@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))


@st.cache_data
def get_options():
    categories = pd.read_csv(CATEGORY_FILE)
    brands = pd.read_csv(BRAND_FILE)["brand_name"].tolist()
    return categories, brands


# ---- check that the needed files exist ----
for needed in (MODEL_PATH, CATEGORY_FILE, BRAND_FILE):
    if not needed.exists():
        st.error(f"Missing file: {needed.relative_to(ROOT)}. See the README for setup steps.")
        st.stop()

model = get_model()
metadata = load_metadata(str(MODEL_PATH))
categories, brands = get_options()

# ---- sidebar: model information ----
with st.sidebar:
    st.header("Model information")
    st.write(f"**Version:** {metadata.get('model_version', 'unknown')}")
    st.write("**Model:** LightGBM on TF-IDF text features")
    st.write(f"**Trained on:** {metadata.get('trained_on', 'unknown')}")
    test_metrics = metadata.get("test_metrics", {})
    if test_metrics:
        st.write(f"**Test MAE:** \\${test_metrics.get('MAE', 0):.2f}")
        st.write(f"**Test RMSLE:** {test_metrics.get('RMSLE', 0):.4f}")
    st.caption("Trained on Mercari Price Suggestion Challenge data (Kaggle).")

# ---- main page ----
st.title("🏷️ Marketplace Price Predictor")
st.write("Enter the details of a second-hand product listing to get an estimated selling price.")

single_tab, batch_tab = st.tabs(["Single listing", "Batch (CSV file)"])

# ===== single listing =====
with single_tab:
    name = st.text_input("Product name", placeholder="e.g. Nike Air Max running shoes size 9")

    col1, col2, col3 = st.columns(3)
    with col1:
        main_options = sorted(categories["cat_main"].unique())
        cat_main = st.selectbox("Main category", main_options)
    with col2:
        sub1_options = sorted(categories.loc[categories["cat_main"] == cat_main, "cat_sub1"].unique())
        cat_sub1 = st.selectbox("Sub-category", sub1_options)
    with col3:
        sub2_options = sorted(categories.loc[
            (categories["cat_main"] == cat_main) & (categories["cat_sub1"] == cat_sub1), "cat_sub2"
        ].unique())
        cat_sub2 = st.selectbox("Product type", sub2_options)

    col4, col5 = st.columns(2)
    with col4:
        brand = st.selectbox("Brand (type to search)", [NO_BRAND] + brands)
        condition = st.selectbox(
            "Item condition", list(CONDITION_LABELS.keys()),
            format_func=lambda value: CONDITION_LABELS[value], index=1
        )
    with col5:
        shipping_label = st.radio("Shipping", ["Buyer pays shipping", "Seller pays shipping"])

    description = st.text_area("Product description", placeholder="Describe the item (optional)")

    if st.button("Predict price", type="primary"):
        if not name.strip():
            st.warning("Please enter a product name.")
        else:
            listing = pd.DataFrame([{
                "name": name,
                "item_condition_id": condition,
                "category_name": f"{cat_main}/{cat_sub1}/{cat_sub2}",
                "brand_name": None if brand == NO_BRAND else brand,
                "shipping": 1 if shipping_label == "Seller pays shipping" else 0,
                "item_description": description,
            }])
            price = predict_price(listing, model)[0]
            st.metric("Estimated selling price", f"${price:,.2f}")
            mae = test_metrics.get("MAE")
            if mae:
                # "\\$" stops Streamlit from reading two $ signs as a math formula
                st.caption(
                    f"On held-out test listings the model's average error was about \\${mae:.2f}. "
                    "Expensive items (above about \\$100) are often under-estimated."
                )

# ===== batch =====
with batch_tab:
    st.write("Upload a CSV file with these columns: " + ", ".join(f"`{c}`" for c in RAW_COLUMNS))
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            batch["predicted_price_usd"] = predict_price(batch, model)
            st.dataframe(batch)
            st.download_button(
                "Download predictions as CSV",
                batch.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except ValueError as error:
            st.error(str(error))
