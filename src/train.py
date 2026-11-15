"""Walk-forward evaluation of one-week-ahead Brent crude forecasts.

Each model predicts next week's log return; the forecast price is
current price * exp(predicted return). Models are refit on an expanding
window every REFIT_WEEKS weeks across the test period, so every forecast
uses only data available at that time.

Outputs:
  results/metrics.csv          - MAE / RMSE / MAPE / direction accuracy per model
  results/test_predictions.csv - weekly forecasts for every model
  figures/*.png                - charts used in the README
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import build_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TRAIN_START = "2007-01-01"   # first year with all features (USD index starts 2006)
TEST_START = "2022-01-01"
REFIT_WEEKS = 13
SEED = 42

FEATURES = [
    "ret_1w", "ret_2w", "ret_4w", "ret_8w", "ret_13w", "ret_26w",
    "vol_4w", "vol_13w", "dist_ma_13w", "dist_ma_52w", "brent_wti_spread",
    "stocks_chg_4w", "stocks_vs_52w", "prod_chg_13w", "usd_chg_4w",
    "gold_chg_13w", "gas_chg_13w", "cpi_yoy", "ip_yoy", "fed_funds_chg_26w",
]
TARGET = "target_ret_1w"


def make_models():
    return {
        "Ridge regression": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))),
        "Random forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=SEED),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=30,
            l2_regularization=1.0, random_state=SEED),
    }


def walk_forward(df):
    test_idx = df.index[df.index >= TEST_START]
    preds = {name: pd.Series(index=test_idx, dtype=float) for name in make_models()}

    for start in range(0, len(test_idx), REFIT_WEEKS):
        block = test_idx[start:start + REFIT_WEEKS]
        # Training rows must have a known target, i.e. end one week before the block.
        train = df[df.index < block[0]].iloc[:-1]
        for name, model in make_models().items():
            model.fit(train[FEATURES], train[TARGET])
            preds[name].loc[block] = model.predict(df.loc[block, FEATURES])
    return pd.DataFrame(preds)


def evaluate(df, ret_preds):
    test = df.loc[ret_preds.index]
    actual_next = test["brent"] * np.exp(test[TARGET])
    out = pd.DataFrame({"week_ending": test.index, "brent_now": test["brent"],
                        "actual_next_week": actual_next,
                        "Naive (no change)": test["brent"]})
    for name in ret_preds:
        out[name] = test["brent"] * np.exp(ret_preds[name])
    out = out.set_index("week_ending")

    rows = []
    for name in ["Naive (no change)", *ret_preds.columns]:
        err = out[name] - out["actual_next_week"]
        row = {
            "model": name,
            "MAE ($/bbl)": err.abs().mean(),
            "RMSE ($/bbl)": np.sqrt((err ** 2).mean()),
            "MAPE (%)": (err.abs() / out["actual_next_week"]).mean() * 100,
        }
        if name != "Naive (no change)":
            row["Direction accuracy (%)"] = (
                np.sign(ret_preds[name]) == np.sign(test[TARGET])).mean() * 100
        rows.append(row)
    return out, pd.DataFrame(rows).set_index("model")


def plot_history(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["brent"], color="#1f4e79", lw=1)
    ax.axvspan(pd.Timestamp(TEST_START), df.index.max(), color="#f2b134", alpha=0.15,
               label="Test period")
    ax.set_title("Brent crude oil, weekly price (USD per barrel)")
    ax.set_ylabel("USD / barrel")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "brent_price_history.png", dpi=150)
    plt.close(fig)


def plot_forecasts(out, best):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(out.index, out["actual_next_week"], color="#222", lw=1.2, label="Actual")
    ax.plot(out.index, out[best], color="#d1495b", lw=1, alpha=0.9, label=f"{best} forecast")
    ax.set_title("One-week-ahead forecasts on the test period")
    ax.set_ylabel("USD / barrel")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_forecasts.png", dpi=150)
    plt.close(fig)


def plot_importance(df):
    train = df[df.index < TEST_START].iloc[:-1]
    valid = df[(df.index >= TEST_START)]
    valid = valid[valid[TARGET].notna()]
    model = make_models()["Gradient boosting"].fit(train[FEATURES], train[TARGET])
    imp = permutation_importance(model, valid[FEATURES], valid[TARGET],
                                 n_repeats=20, random_state=SEED, scoring="neg_mean_absolute_error")
    s = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    s = s.tail(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s.values, color="#1f4e79")
    ax.set_title("Permutation importance (gradient boosting, test period)")
    ax.set_xlabel("Increase in MAE of weekly return when shuffled")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)
    return s


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    df = build_table()
    df = df.loc[TRAIN_START:]
    df = df.dropna(subset=FEATURES)
    df = df[df[TARGET].notna()]

    ret_preds = walk_forward(df)
    out, metrics = evaluate(df, ret_preds)

    out.round(3).to_csv(RESULTS / "test_predictions.csv")
    metrics.round(3).to_csv(RESULTS / "metrics.csv")

    best = metrics.drop("Naive (no change)")["MAE ($/bbl)"].idxmin()
    plot_history(build_table())
    plot_forecasts(out, best)
    imp = plot_importance(df)

    print(f"Train: {df.index.min().date()} .. {pd.Timestamp(TEST_START).date()}  |  "
          f"Test: {out.index.min().date()} .. {out.index.max().date()} ({len(out)} weeks)\n")
    print(metrics.round(3).to_string())
    print("\nTop features:\n", imp.sort_values(ascending=False).round(5).to_string())


if __name__ == "__main__":
    main()
