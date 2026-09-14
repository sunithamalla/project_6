import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TEST_PATH, TRAIN_PATH, STORE_PATH, MODEL_PATH, META_PATH, SUBMISSION_PATH
from src.features import prepare_merged, make_features


def main():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run from the project root:\n"
            "python src\\train.py"
        )

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    store = pd.read_csv(STORE_PATH)

    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    features = meta["features"]
    mappings = meta["category_mappings"]

    # Build the full historical + future timeline.
    combined = prepare_merged(train, test, store)
    combined["Date"] = pd.to_datetime(combined["Date"])

    # Forecast one date at a time. After each date, predictions are written into
    # Sales so future lag/rolling features can use earlier predictions.
    future_dates = sorted(
        combined.loc[combined["is_train"].eq(0), "Date"].unique()
    )

    print(f"Forecasting {len(future_dates)} future dates...")

    for i, date in enumerate(future_dates, 1):
        current = combined["Date"].eq(date) & combined["is_train"].eq(0)

        # Recalculate lags/rolling features because prior future sales may
        # already have been filled with predictions.
        featured, _ = make_features(combined, mappings)
        row = featured.loc[current].copy()

        # Closed stores are deterministically zero.
        open_mask = row["Open"].fillna(0).eq(1)
        if open_mask.any():
            pred = np.maximum(
                0,
                np.expm1(model.predict(row.loc[open_mask, features]))
            )
            combined.loc[row.index[open_mask], "Sales"] = pred

        combined.loc[row.index[~open_mask], "Sales"] = 0.0

        if i % 7 == 0 or i == len(future_dates):
            print(f"Processed {i}/{len(future_dates)} dates")

    result = combined[combined["is_train"].eq(0)][["Id", "Sales"]].copy()
    result["Sales"] = result["Sales"].clip(lower=0).round(2)
    result.to_csv(SUBMISSION_PATH, index=False)

    print(f"Saved submission: {SUBMISSION_PATH}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
