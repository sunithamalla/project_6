import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Allows "python src/train.py" from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    TRAIN_PATH, TEST_PATH, STORE_PATH, MODEL_PATH, META_PATH,
    VALIDATION_PRED_PATH, METRICS_PATH
)
from src.features import make_features, FEATURES
from src.metrics import rmspe


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    store = pd.read_csv(STORE_PATH)

    # Keep only dates available in the historical training data for model fitting.
    train["Date"] = pd.to_datetime(train["Date"])
    cutoff = train["Date"].max() - pd.Timedelta(days=42)
    print(f"Historical maximum date: {train['Date'].max().date()}")
    print(f"Validation starts: {cutoff.date()}")

    # Combine train + test so store metadata/category mappings are identical.
    from src.features import prepare_merged
    combined = prepare_merged(train, test, store)

    # Generate features. Lag features for training are based only on historical Sales.
    featured, mappings = make_features(combined)

    train_feat = featured[featured["is_train"].eq(1)].copy()
    fit = train_feat[train_feat["Date"] < cutoff].copy()
    valid = train_feat[train_feat["Date"] >= cutoff].copy()

    # Closed days have zero sales. For the ML model we focus on open stores.
    fit = fit[fit["Open"].fillna(0).eq(1)].copy()
    valid_open = valid[valid["Open"].fillna(0).eq(1)].copy()

    X_fit = fit[FEATURES]
    y_fit = np.log1p(fit["Sales"].clip(lower=0))

    X_valid = valid_open[FEATURES]
    y_valid = valid_open["Sales"].values

    print(f"Training rows: {len(fit):,}")
    print(f"Validation rows: {len(valid_open):,}")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, np.log1p(y_valid))],
        callbacks=[
            lgb.early_stopping(80, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    valid_pred = np.maximum(0, np.expm1(model.predict(X_valid)))
    score = rmspe(y_valid, valid_pred)
    print(f"Validation RMSPE: {score:.5f}")

    # Train a final model on all open historical rows using the selected iteration count.
    final_rounds = model.best_iteration_ or 800
    full = train_feat[train_feat["Open"].fillna(0).eq(1)].copy()
    X_full = full[FEATURES]
    y_full = np.log1p(full["Sales"].clip(lower=0))

    final_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=final_rounds,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_full, y_full)

    joblib.dump(final_model, MODEL_PATH)

    meta = {
        "features": FEATURES,
        "category_mappings": mappings,
        "validation_cutoff": str(cutoff.date()),
        "validation_rmspe": score,
        "best_iteration": int(final_rounds),
        "target_transform": "log1p",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    out = valid[["Store", "Date", "Sales", "Open"]].copy()
    out["PredictedSales"] = 0.0
    out.loc[valid_open.index, "PredictedSales"] = valid_pred
    out.to_csv(VALIDATION_PRED_PATH, index=False)
    METRICS_PATH.write_text(json.dumps({"RMSPE": score}, indent=2))

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved validation predictions: {VALIDATION_PRED_PATH}")


if __name__ == "__main__":
    main()
