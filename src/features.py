import numpy as np
import pandas as pd


# These are deliberately explicit so training and prediction use the same schema.
BASE_NUMERIC = [
    "Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
    "Promo2SinceYear",
    "Year", "Month", "Day", "WeekOfYear", "DayOfYear",
    "IsMonthStart", "IsMonthEnd", "IsWeekend",
    "CompetitionOpenMonths", "Promo2AgeWeeks",
    "Promo2Active", "Promo2StartMonthFlag",
]

CATEGORICAL = ["StateHoliday", "StoreType", "Assortment"]

LAG_COLS = ["SalesLag1", "SalesLag7", "SalesLag14", "SalesLag28"]
ROLL_COLS = ["SalesRoll7", "SalesRoll14", "SalesRoll28"]

FEATURES = BASE_NUMERIC + CATEGORICAL + LAG_COLS + ROLL_COLS


def prepare_merged(train: pd.DataFrame, test: pd.DataFrame, store: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    store = store.copy()

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    # Normalize holiday values so mixed "0"/0/a/b/c do not create duplicate categories.
    for df in (train, test):
        df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0.0": "0"})

    store_cols = [
        "Store", "StoreType", "Assortment", "CompetitionDistance",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
        "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    ]
    train = train.merge(store[store_cols], on="Store", how="left")
    test = test.merge(store[store_cols], on="Store", how="left")

    train["is_train"] = 1
    test["is_train"] = 0
    test["Sales"] = np.nan

    combined = pd.concat([train, test], ignore_index=True, sort=False)
    combined = combined.sort_values(["Store", "Date"]).reset_index(drop=True)
    return combined


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["Date"]

    df["Year"] = d.dt.year
    df["Month"] = d.dt.month
    df["Day"] = d.dt.day
    df["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    df["DayOfYear"] = d.dt.dayofyear
    df["IsMonthStart"] = d.dt.is_month_start.astype(int)
    df["IsMonthEnd"] = d.dt.is_month_end.astype(int)
    df["IsWeekend"] = (d.dt.dayofweek >= 5).astype(int)

    # Competition age.
    comp_date = pd.to_datetime(
        dict(
            year=df["CompetitionOpenSinceYear"].fillna(d.dt.year).astype(int),
            month=df["CompetitionOpenSinceMonth"].fillna(1).astype(int),
            day=1,
        ),
        errors="coerce",
    )
    df["CompetitionOpenMonths"] = (
        (d.dt.year - comp_date.dt.year) * 12
        + d.dt.month - comp_date.dt.month
    ).clip(lower=0).fillna(0)

    # Promo2 age in weeks.
    promo2_start = pd.to_datetime(
        df["Promo2SinceYear"].fillna(d.dt.year).astype(int).astype(str)
        + "-01-01"
    ) + pd.to_timedelta(
        (df["Promo2SinceWeek"].fillna(1).astype(int) - 1) * 7, unit="D"
    )
    df["Promo2AgeWeeks"] = (
        (d - promo2_start).dt.days.div(7).clip(lower=0).fillna(0)
    )

    df["Promo2Active"] = 0
    df["Promo2StartMonthFlag"] = 0

    mask = df["Promo2"].fillna(0).astype(int).eq(1)
    df.loc[mask, "Promo2Active"] = 1

    intervals = df["PromoInterval"].fillna("").astype(str)
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    starts = []
    for interval, month in zip(intervals, d.dt.month):
        starts.append(int(month_names[int(month)] in interval.split(",")) if interval else 0)
    df["Promo2StartMonthFlag"] = np.array(starts, dtype=int)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag/rolling features from the current Sales column.

    During test prediction, Sales for future dates is filled recursively with
    model predictions, so the same function can be reused.
    """
    df = df.copy().sort_values(["Store", "Date"]).reset_index(drop=True)
    g = df.groupby("Store", sort=False)["Sales"]

    for lag in (1, 7, 14, 28):
        df[f"SalesLag{lag}"] = g.shift(lag)

    # Shift first to avoid including today's target.
    shifted = g.shift(1)
    for window in (7, 14, 28):
        df[f"SalesRoll{window}"] = (
            shifted.groupby(df["Store"], sort=False)
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df


def encode_categories(df: pd.DataFrame, mappings=None):
    df = df.copy()
    if mappings is None:
        mappings = {}

    for col in CATEGORICAL:
        values = df[col].fillna("Unknown").astype(str)
        if col not in mappings:
            uniques = sorted(values.unique().tolist())
            mappings[col] = {v: i for i, v in enumerate(uniques)}
        df[col] = values.map(mappings[col]).fillna(-1).astype(int)

    return df, mappings


def make_features(df: pd.DataFrame, mappings=None):
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df, mappings = encode_categories(df, mappings)

    # Fill numerical missing values with robust zero/default values.
    for col in BASE_NUMERIC + LAG_COLS + ROLL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df, mappings
