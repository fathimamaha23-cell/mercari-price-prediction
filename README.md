# Mercari Product Price Prediction

## Project Overview

This project predicts an estimated selling price for a second-hand product listing using information such as the product name, category, brand, item condition, shipping option, and description.

The project uses the Mercari Price Suggestion Challenge dataset from Kaggle for model development. The trained model is used by a Streamlit web application to estimate prices for new product listings.

## Features

- Explore the dataset and examine data quality through exploratory data analysis (EDA).
- Preprocess product listing data and prepare text features.
- Train and evaluate individual machine-learning models.
- Tune model hyperparameters.
- Predict the estimated selling price of a single product through the Streamlit interface.
- Upload a CSV file to predict prices for multiple products and download the results.
- Support model interpretation using SHAP and LIME where implemented in the notebooks.

## Project Structure

```text
ProductPricePrediction/
├── app/
│   └── app.py
├── data/
│   ├── sample_new_listings.csv
│   └── processed/
│       └── split_and_preprocessors.joblib
├── models/
│   ├── brand_options.csv
│   ├── category_options.csv
│   ├── model_results.csv
│   ├── price_model_v1.joblib
│   └── price_model_v1_metadata.json
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing_Feature_Engineering.ipynb
│   └── 03_Modeling_Tuning.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── test_features.py
│   └── test_predict.py
├── requirements.txt
├── requirements-notebooks.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

The exact files in the project may differ depending on the version. Keep any additional files required by the application, trained model, or training workflow.

## Requirements

- Python compatible with the package versions listed in `requirements.txt`
- pip

## Setup on Windows

Open PowerShell or Command Prompt in the project root directory, which contains `requirements.txt`.

### 1. Install the required dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Run the Streamlit application

```powershell
python -m streamlit run app/app.py
```

### 3. Open the application

Open the local URL displayed in the terminal, usually:

http://localhost:8501

Keep the terminal open while using the application.

## Using the App

### Single Listing Prediction

1. Enter the product name.
2. Select the main category, sub-category, and product type.
3. Choose the brand and item condition.
4. Select the shipping option.
5. Optionally enter a product description.
6. Click **Predict price** to view the estimated selling price.

### Batch Prediction

1. Open the **Batch (CSV file)** tab.
2. Upload a CSV file containing the input columns expected by the application.
3. Review the predicted prices.
4. Download the resulting CSV file.

The file `data/sample_new_listings.csv` can be used as a sample input if its columns match the format expected by the application.

## Model and Supporting Files

The application uses the trained model and supporting files stored in the `models/` directory.

- `price_model_v1.joblib` — trained model used for price prediction.
- `category_options.csv` — category options used by the interface.
- `brand_options.csv` — brand options used by the interface.
- `price_model_v1_metadata.json` — metadata describing the trained model and its evaluation metrics.
- `model_results.csv` — comparison of model evaluation results, if included in the project.

Do not remove model or supporting files unless you have confirmed that the application and training workflow no longer depend on them.

## Running the Jupyter Notebooks

The `notebooks/` directory contains the project's analysis and modelling workflow:

1. `01_EDA.ipynb` — exploratory data analysis.
2. `02_Preprocessing_Feature_Engineering.ipynb` — preprocessing and feature engineering.
3. `03_Modeling_Tuning.ipynb` — model training, evaluation, and hyperparameter tuning.

### 1. Install the additional notebook dependencies

```powershell
python -m pip install -r requirements-notebooks.txt
```

### 2. Start JupyterLab

```powershell
python -m jupyterlab
```

### 3. Open a notebook

Open the required notebook from the `notebooks/` directory in JupyterLab.

Make sure the necessary dependencies are installed before running the notebooks.

## Dataset and Large Files

The project uses the Mercari Price Suggestion Challenge dataset from Kaggle for model development.

The original training dataset is not included in the final project ZIP or GitHub repository. Obtain the dataset from the Kaggle competition if you need to rerun the training workflow.

The Streamlit application can make predictions without the original training dataset, provided that the trained model and all required supporting files are available.

Avoid committing virtual environments, Python cache folders, Jupyter checkpoints, test caches, generated tracking files, and large raw datasets to GitHub.

## Testing

The project includes automated tests for feature processing and prediction functionality.

Run the tests from the project root directory:

```powershell
python -m pytest -v
```

## Live demo
Try the deployed application here:
https://mercari-price-predicto.streamlit.app/
## Notes

- Predicted prices are estimates and are not guaranteed selling prices.
- Model performance may vary across product categories and price ranges.
- Run the application from the project root directory so that relative file paths resolve correctly.
- The application requires the trained model and supporting files to be present at the expected paths.

