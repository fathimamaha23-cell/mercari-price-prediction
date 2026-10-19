# Mercari Product Price Prediction

## Project Overview

This project predicts an estimated selling price for a second-hand
product listing using information such as the product name, category,
brand, item condition, shipping option, and description.

The project uses the Mercari Price Suggestion Challenge dataset from
Kaggle for model development. The trained model is used by a Streamlit
web app to estimate prices for new listings.

## Features

-   Explore the dataset and examine data quality.
-   Preprocess product listing data and prepare text features.
-   Train and evaluate individual machine-learning models.
-   Tune model hyperparameters.
-   Explain model predictions using SHAP and LIME, where supported by
    the notebooks.
-   Predict a price for one product through the Streamlit interface.
-   Upload a CSV file to predict prices for multiple products and
    download the results.

## Project Structure

``` text
ProductPricePrediction/
├── app/
│   └── app.py
├── data/
│   ├── split_and_preprocessors.joblib
│   └── sample_new_listings.csv
├── models/
│   ├── brand_options.csv
│   ├── category_options.csv
│   ├── price_model_v1.joblib
│   └── price_model_v1_metadata.json
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing_Feature_Engineering.ipynb
│   └── 03_Modeling_Tuning.ipynb
├── src/
│   ├── features.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
├── requirements.txt
└── requirements-notebooks.txt
```

The exact files in your local project may differ slightly. Keep any
additional files required by your code and saved model.

## Requirements

-   Python compatible with the package versions in `requirements.txt`
-   pip

## Setup on Windows

Open PowerShell or Command Prompt in the project root folder (the folder
containing `requirements.txt`).

### 1. Create a virtual environment

``` powershell
python -m venv .venv
```

### 2. Install the app requirements

You can run the virtual environment's Python directly, without
activating it:

``` powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run the Streamlit App

**1. Install the required dependencies**

```bash
python -m pip install -r requirements.txt
```

**2. Start the app**

```bash
python -m streamlit run app/app.py
```

**3. Open the app**

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

Keep the terminal open while using the app.

## Using the App

### Single listing

1.  Enter a product name.
2.  Select the main category, sub-category, and product type.
3.  Choose a brand and item condition.
4.  Select the shipping option.
5.  Optionally enter a product description.
6.  Click **Predict price** to see the estimated selling price.

### Batch prediction

1.  Open the **Batch (CSV file)** tab.
2.  Upload a CSV file containing the input columns listed by the app.
3.  Review the predicted prices.
4.  Download the resulting CSV.

`data/sample_new_listings.csv` can be used as a sample input if its
columns match those requested by the app.

## Model and Supporting Files

The app expects the following files under `models/`:

-   `price_model_v1.joblib` --- trained prediction model.
-   `category_options.csv` --- category options for the interface.
-   `brand_options.csv` --- brand options for the interface.
-   `price_model_v1_metadata.json` --- model metadata, if provided with
    the trained model.

Do not remove these files unless you have confirmed the app no longer
depends on them.

## Notebooks

The `notebooks/` folder contains the project's analysis and modelling
workflow:

1.  `01_EDA.ipynb` --- exploratory data analysis.
2.  `02_Preprocessing_Feature_Engineering.ipynb` --- preprocessing and
    feature engineering.
3.  `03_Modeling_Tuning.ipynb` --- model training, evaluation, and
    hyperparameter tuning.

To run the notebooks, install the additional notebook dependencies:

``` powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
```

Then start JupyterLab:

``` powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

## Dataset and Large Files

The original training dataset is from the Kaggle Mercari Price
Suggestion Challenge. It is not included in the final project ZIP or
GitHub repository. Obtain it from the official Kaggle competition if you
need to rerun training.

The Streamlit app can run without the original training dataset when the
required trained model and supporting files are present.

Do not commit virtual environments, Python cache folders, Jupyter
checkpoints, test caches, generated tracking folders, or large raw
datasets to GitHub.

## Notes

-   The displayed price is a model estimate, not a guaranteed selling
    price.
-   Model performance can vary across product categories and price
    ranges.
-   Run the app from the project root so that its relative file paths
    resolve correctly.
