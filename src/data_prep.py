"""Build a weekly modelling table for Brent crude price forecasting.

Raw sources (data/raw/) come at three frequencies:
  * weekly  - Brent, WTI, US crude stocks (ex-SPR), US crude field production
  * daily   - Nominal Broad U.S. Dollar Index (resampled to weekly mean)
  * monthly - CPI, Fed funds rate, gold, Henry Hub gas, US industrial production

Monthly series are published with a delay, so each one is shifted by one month
before being forward-filled onto the weekly calendar. This keeps the model from
"seeing" macro data that would not have been available on the forecast date.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "brent_weekly_features.csv"


def _read(name, date_col, value_col, new_name, fmt=None):
    df = pd.read_csv(RAW / name)
    df = df[[date_col, value_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col], format=fmt)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna().rename(columns={date_col: "date", value_col: new_name}).set_index("date").sort_index()


def load_weekly():
    brent = _read("brent_weekly.csv", "observation_date",
                  "Brent crude oil price in USD per barrel", "brent", "%m/%d/%Y")
    wti = _read("wti_weekly.csv", "Observation_Date", "WCOILWTICO", "wti", "%m/%d/%Y")
    stocks = _read("us_crude_stocks_weekly.csv", "Date",
                   "Weekly U.S. Ending Stocks excluding SPR of Crude Oil  (Thousand Barrels)",
                   "us_stocks", "%b %d, %Y")
    prod = _read("us_crude_production_weekly.csv", "Date",
                 "Weekly U.S. Field Production of Crude Oil  (Thousand Barrels per Day)",
                 "us_production", "%b %d, %Y")
    usd = _read("usd_index_daily.csv", "observation_date", "DTWEXBGS", "usd_index", "%Y-%m-%d")

    # Brent dates are Fridays; align everything to week ending Friday.
    to_week = lambda s: s.resample("W-FRI").last()
    weekly = pd.concat(
        [to_week(brent), to_week(wti), to_week(stocks), to_week(prod),
         usd.resample("W-FRI").mean()],
        axis=1, sort=True,
    )
    return weekly


def load_monthly():
    series = [
        _read("cpi_monthly.csv", "observation_date", "CPIAUCSL", "cpi", "%Y-%m-%d"),
        _read("fedfunds_monthly.csv", "observation_date", "FEDFUNDS", "fed_funds", "%m/%d/%Y"),
        _read("gold_monthly.csv", "date", "gold_price_usd", "gold", "%m/%d/%Y"),
        _read("henryhub_gas_monthly.csv", "observation_date", "MHHNGSP", "henry_hub", "%Y-%m-%d"),
        _read("industrial_production_monthly.csv", "observation_date",
              "Industrial_Production", "industrial_prod", "%m/%d/%Y"),
    ]
    monthly = pd.concat(series, axis=1, sort=True)
    # One-month publication lag: the January value becomes usable in March.
    monthly.index = monthly.index + pd.offsets.MonthBegin(2)
    return monthly


def build_table():
    weekly = load_weekly()
    monthly = load_monthly()
    df = weekly.join(monthly.reindex(weekly.index, method="ffill"))
    df = df.loc["2000-01-01":]
    df = df[df["brent"].notna()]

    # Fill short gaps in exogenous series (holidays, missing weeks) forward only.
    exog = [c for c in df.columns if c != "brent"]
    df[exog] = df[exog].ffill()

    # Target: next week's log return of Brent.
    df["log_brent"] = np.log(df["brent"])
    df["target_ret_1w"] = df["log_brent"].shift(-1) - df["log_brent"]

    # Price-based features (all known at week t).
    for lag in (1, 2, 4, 8, 13, 26):
        df[f"ret_{lag}w"] = df["log_brent"] - df["log_brent"].shift(lag)
    df["vol_4w"] = df["log_brent"].diff().rolling(4).std()
    df["vol_13w"] = df["log_brent"].diff().rolling(13).std()
    df["dist_ma_13w"] = df["log_brent"] - df["log_brent"].rolling(13).mean()
    df["dist_ma_52w"] = df["log_brent"] - df["log_brent"].rolling(52).mean()
    df["brent_wti_spread"] = df["brent"] - df["wti"]

    # Fundamentals / macro, expressed as changes so they are roughly stationary.
    df["stocks_chg_4w"] = df["us_stocks"].pct_change(4)
    df["stocks_vs_52w"] = df["us_stocks"] / df["us_stocks"].rolling(52).mean() - 1
    df["prod_chg_13w"] = df["us_production"].pct_change(13)
    df["usd_chg_4w"] = df["usd_index"].pct_change(4)
    df["gold_chg_13w"] = df["gold"].pct_change(13)
    df["gas_chg_13w"] = df["henry_hub"].pct_change(13)
    df["cpi_yoy"] = df["cpi"].pct_change(52)
    df["ip_yoy"] = df["industrial_prod"].pct_change(52)
    df["fed_funds_chg_26w"] = df["fed_funds"].diff(26)

    return df


if __name__ == "__main__":
    table = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index_label="week_ending")
    print(f"Saved {len(table):,} weekly rows x {table.shape[1]} columns -> {OUT.relative_to(ROOT)}")
    print(f"Range: {table.index.min().date()} to {table.index.max().date()}")
