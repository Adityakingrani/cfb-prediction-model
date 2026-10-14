"""
Walk-forward backtest. For every week from 2017 to 2025, train ONLY on games played
before that week, predict that week, and score the predictions.

Compares: simple baseline, Ridge regression, LightGBM, and the betting market (benchmark).

Run from the repo root:  python src\\backtest.py
"""
import warnings
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
import lightgbm as lgb

warnings.filterwarnings("ignore")
DB_PATH = "data/cfb.duckdb"
TEST_SEASONS = range(2017, 2026)
EXCLUDE_2020_FROM_TRAINING = True
MARGIN_CAP = 45   # cap extreme blowouts in the training target

RIDGE_FEATURES = ["in_margin_w", "in_ppa_edge_w", "in_sr_edge_w",
                  "pr_margin_w", "pr_ppa_edge_w", "pr_sp_w", "pr_ret_ppa_w", "pr_talent_w", "pr_rec_w",
                  "home_field", "rest_diff"]
LGB_FEATURES = ["in_margin", "in_ppa_edge", "in_sr_edge",
                "pr_margin", "pr_ppa_edge", "pr_sp", "pr_ret_ppa", "pr_talent", "pr_rec",
                "w_inseason", "home_field", "rest_diff", "postseason"]
HOME_FIELD_PTS = 2.5


def win_prob(pred_margin, sigma):
    return norm.cdf(pred_margin / sigma)


def fit_predict(train, test):
    y = train["margin"].clip(-MARGIN_CAP, MARGIN_CAP)
    out = {}

    # baseline: in-season rating blended with last year's rating, plus home field
    def base(d):
        return d["in_margin_w"] + d["pr_margin_w"] + HOME_FIELD_PTS * d["home_field"]
    sig = np.std(train["margin"] - base(train))
    out["baseline"] = (base(test), sig)

    ridge = Ridge(alpha=1.0).fit(train[RIDGE_FEATURES], y)
    sig = np.std(train["margin"] - ridge.predict(train[RIDGE_FEATURES]))
    out["ridge"] = (ridge.predict(test[RIDGE_FEATURES]), sig)

    gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                            min_child_samples=40, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, verbose=-1, random_state=7)
    gbm.fit(train[LGB_FEATURES], y)
    # in-sample residuals understate error for trees, so borrow ridge's sigma
    out["lightgbm"] = (gbm.predict(test[LGB_FEATURES]), out["ridge"][1])

    m = train.dropna(subset=["market_spread"])
    sig = np.std(m["margin"] + m["market_spread"])
    out["market"] = (-test["market_spread"], sig)
    return out


def score(df, pred_col, prob_col):
    d = df.dropna(subset=[pred_col, prob_col])
    d = d[d["margin"] != 0]
    p = d[prob_col].clip(1e-4, 1 - 1e-4)
    y = (d["margin"] > 0).astype(int)
    decided = d[d[pred_col] != 0]
    return {
        "games": len(d),
        "accuracy": ((decided[pred_col] > 0) == (decided["margin"] > 0)).mean(),
        "brier": ((p - y) ** 2).mean(),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
        "margin_mae": (d[pred_col] - d["margin"]).abs().mean(),
    }


def main():
    con = duckdb.connect(DB_PATH)
    f = con.execute("SELECT * FROM features WHERE completed").df()
    f = f.sort_values(["season", "period"]).reset_index(drop=True)
    # seasons with no prior-year data (e.g. 2014 has no 2013 ratings) -> treat prior edge as neutral
    feat_cols = sorted(set(RIDGE_FEATURES) | set(LGB_FEATURES))
    print("Missing values filled with 0:", {c: int(f[c].isna().sum()) for c in feat_cols if f[c].isna().any()} or "none")
    f[feat_cols] = f[feat_cols].fillna(0)

    rows = []
    for season in TEST_SEASONS:
        if season == 2020:
            continue
        for period in sorted(f.loc[f["season"] == season, "period"].unique()):
            train = f[(f["season"] < season) | ((f["season"] == season) & (f["period"] < period))]
            if EXCLUDE_2020_FROM_TRAINING:
                train = train[train["season"] != 2020]
            test = f[(f["season"] == season) & (f["period"] == period)].copy()
            if len(train) < 500 or test.empty:
                continue
            for name, (pred, sig) in fit_predict(train, test).items():
                test[f"{name}_pred"] = np.asarray(pred, dtype=float)
                test[f"{name}_prob"] = win_prob(test[f"{name}_pred"], sig)
            rows.append(test)
        print(f"  backtested {season}")

    bt = pd.concat(rows, ignore_index=True)
    models = ["baseline", "ridge", "lightgbm", "market"]

    # compare all models on the SAME games (ones that have a market line)
    common = bt.dropna(subset=["market_pred"])

    print("\n================ OVERALL (2017-2025, excl. 2020, same games for all) ================")
    print(pd.DataFrame({m: score(common, f"{m}_pred", f"{m}_prob") for m in models}).T
          .to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nACCURACY BY SEASON")
    by = {m: common.groupby("season").apply(lambda d: score(d, f"{m}_pred", f"{m}_prob")["accuracy"])
          for m in models}
    print(pd.DataFrame(by).to_string(float_format=lambda x: f"{x:.1%}"))

    print("\nEARLY vs LATE SEASON (accuracy)")
    common["phase"] = np.where(common["week"] <= 4, "weeks 1-4", "week 5+")
    common.loc[common["postseason"] == 1, "phase"] = "postseason"
    ph = {m: common.groupby("phase").apply(lambda d: score(d, f"{m}_pred", f"{m}_prob")["accuracy"])
          for m in models}
    print(pd.DataFrame(ph).to_string(float_format=lambda x: f"{x:.1%}"))

    best = min(["baseline", "ridge", "lightgbm"],
               key=lambda m: score(common, f"{m}_pred", f"{m}_prob")["brier"])
    print(f"\nCALIBRATION — {best} (when the model says X%, how often does that team win?)")
    fav_prob = np.maximum(common[f"{best}_prob"], 1 - common[f"{best}_prob"])
    fav_won = np.where(common[f"{best}_prob"] >= 0.5, common["margin"] > 0, common["margin"] < 0)
    cal = pd.DataFrame({"prob": fav_prob, "won": fav_won})
    cal["bucket"] = pd.cut(cal["prob"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    print(cal.groupby("bucket", observed=True).agg(games=("won", "size"), predicted=("prob", "mean"),
                                                   actual=("won", "mean"))
          .to_string(float_format=lambda x: f"{x:.1%}"))
    print(f"\nBest model by Brier score: {best}")

    keep = ["game_id", "season", "week", "home", "away", "margin"] + \
           [f"{m}_{k}" for m in models for k in ("pred", "prob")]
    con.register("bt_df", bt[keep])
    con.execute("CREATE OR REPLACE TABLE backtest_preds AS SELECT * FROM bt_df")
    con.close()
    print(f"Saved `backtest_preds`: {len(bt):,} predictions")


if __name__ == "__main__":
    main()