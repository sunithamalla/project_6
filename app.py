from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.config import MODEL_PATH, META_PATH, TRAIN_PATH, STORE_PATH
from src.features import prepare_merged, make_features

app = Flask(__name__)

MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else None


def forecast_one(store_id, date, promo=0, open_=1, school_holiday=0, state_holiday="0"):
    if MODEL is None:
        raise RuntimeError("Model not found. Run python src\\train.py first.")

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    store = pd.read_csv(STORE_PATH)

    date = pd.to_datetime(date)
    hist = train[train["Store"].eq(int(store_id))].copy()
    future = pd.DataFrame([{
        "Id": -1,
        "Store": int(store_id),
        "DayOfWeek": int(date.dayofweek + 1),
        "Date": date,
        "Open": int(open_),
        "Promo": int(promo),
        "StateHoliday": str(state_holiday),
        "SchoolHoliday": int(school_holiday),
    }])

    # For the web demo, the user's selected date is appended to the store's
    # history. Lags therefore use the latest observed historical sales.
    combined = prepare_merged(hist, future, store)
    featured, _ = make_features(combined, META["category_mappings"])
    row = featured[featured["is_train"].eq(0)].tail(1)

    if int(open_) == 0:
        return 0.0

    pred = float(np.maximum(0, np.expm1(MODEL.predict(row[META["features"]]))[0]))
    return round(pred, 2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    required = ["store", "date"]
    missing = [x for x in required if x not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        prediction = forecast_one(
            store_id=data["store"],
            date=data["date"],
            promo=data.get("promo", 0),
            open_=data.get("open", 1),
            school_holiday=data.get("school_holiday", 0),
            state_holiday=data.get("state_holiday", "0"),
        )
        return jsonify({
            "store": int(data["store"]),
            "date": data["date"],
            "predicted_sales": prediction,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
