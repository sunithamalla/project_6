# Rossmann Sales Forecasting — End-to-End ML Project

This project forecasts Rossmann store sales for future dates and serves predictions through a Flask web app.

## Business objective

Forecast daily sales for multiple stores several weeks ahead using:
- promotions
- competition
- school/state holidays
- seasonality
- store type and assortment
- historical sales patterns

## Architecture

```text
train.csv + store.csv
        |
        v
Data cleaning / feature engineering
        |
        v
LightGBM regression on log1p(Sales)
        |
        +--> time-based validation (RMSPE)
        |
        v
Saved model + feature metadata
        |
        v
Recursive future forecasting
        |
        v
Flask API / Web UI
```

## Project structure

```text
rossmann_sales_forecasting_solution/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── store.csv
├── src/
│   ├── config.py
│   ├── features.py
│   ├── metrics.py
│   ├── train.py
│   └── predict.py
├── artifacts/
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py
├── requirements.txt
└── README.md
```

## Windows setup

Open PowerShell in the project root:

```powershell
cd C:\Users\LENOVO\Downloads\rossmann_sales_forecasting_solution
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Make sure the CSV files are inside the project's `data` folder.

## Train

```powershell
python src\train.py
```

This creates:

```text
artifacts/rossmann_lgbm.joblib
artifacts/model_meta.json
artifacts/validation_predictions.csv
artifacts/validation_metrics.json
```

## Generate test predictions

```powershell
python src\predict.py
```

Output:

```text
artifacts/submission.csv
```

The output contains:

```text
Id,Sales
```

with closed stores predicted as zero.

## Run the web app

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Important implementation note

The test set does not contain `Customers`, so `Customers` is intentionally excluded from the prediction features. This prevents train/test feature mismatch and reflects the real forecasting situation.

The forecasting logic uses lagged historical sales and recursively uses previous predictions for future lag values. This is appropriate for a multi-step forecast where actual future sales are unavailable.

## Why RMSPE?

Rossmann-style evaluation commonly uses:

```text
RMSPE = sqrt(mean(((actual - predicted) / actual)^2))
```

Rows with actual sales equal to zero are excluded from the RMSPE denominator.

## Production improvements

For a stronger production version:
1. Add MLflow experiment tracking.
2. Add GitHub Actions CI/CD.
3. Add Docker.
4. Add model/data validation.
5. Add monitoring for prediction drift.
6. Retrain on a schedule.
7. Add a database instead of CSV files.
8. Add authentication to the Flask service.
