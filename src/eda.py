import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import TRAIN_PATH, STORE_PATH, ARTIFACT_DIR


def main():
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    store = pd.read_csv(STORE_PATH)
    train["Date"] = pd.to_datetime(train["Date"])

    print(train.info())
    print("\nMissing values:\n", train.isna().sum())
    print("\nSales summary:\n", train["Sales"].describe())

    merged = train.merge(store, on="Store", how="left")
    daily = train.groupby("Date", as_index=False)["Sales"].sum()

    plt.figure(figsize=(12, 5))
    plt.plot(daily["Date"], daily["Sales"])
    plt.title("Total Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "daily_sales.png", dpi=150)
    plt.close()

    by_type = merged.groupby("StoreType", as_index=False)["Sales"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(by_type["StoreType"], by_type["Sales"])
    plt.title("Average Sales by Store Type")
    plt.xlabel("Store Type")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "sales_by_store_type.png", dpi=150)
    plt.close()

    print("EDA charts saved in artifacts/.")


if __name__ == "__main__":
    main()
