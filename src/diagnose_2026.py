"""
Is the model slipping in 2026, or is it just a chaotic season?
Replays every 2026 week (train only on earlier games), then compares model vs market by
confidence tier, and against the same weeks of 2017-2025.

Run from the repo root:  python src\\diagnose_2026.py
"""
import warnings
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
DB_PATH = "data/cfb.duckdb"
FEATURES = ["in_margin_w", "in_ppa_edge_w", "in_sr_edge_w",
            "pr_margin_w", "pr_ppa_edge_w", "pr_sp_w", "pr_ret_ppa_w", "pr_talent_w", "pr_rec_w",
            "home_field", "rest_diff"]


def replay(f, season, max_period):
    hist = f[f["completed"] & (f["season"] != 2020)]
    rows = []
    for period in sorted(hist.loc[(hist["season"] == season) & (hist["period"] <= max_period), "period"].unique()):
        train = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["period"] < period))]
        test = hist[(hist["season"] == season) & (hist["period"] == period) & (hist["margin"] != 0)].copy()
        if len(train) < 500 or test.empty:
            continue
        m = Ridge(alpha=1.0).fit(train[FEATURES], train["margin"].clip(-45, 45))
        sig = np.std(train["margin"] - m.predict(train[FEATURES]))
        test["pred"] = m.predict(test[FEATURES])
        test["prob"] = norm.cdf(test["pred"] / sig)
        rows.append(test)
    return pd.concat(rows) if rows else pd.DataFrame()


def summarize(d, label):
    d = d.copy()
    d["fav_prob"] = np.maximum(d["prob"], 1 - d["prob"])
    d["model_right"] = (d["pred"] > 0) == (d["margin"] > 0)
    d["tier"] = pd.cut(d["fav_prob"], [0.5, 0.6, 0.75, 1.0], labels=["LOW", "MED", "HIGH"], include_lowest=True)
    m = d.dropna(subset=["market_spread"])
    m = m[m["market_spread"] != 0]
    d["market_right"] = np.nan
    d.loc[m.index, "market_right"] = ((m["market_spread"] < 0) == (m["margin"] > 0)).astype(float)
    print(f"\n{label}: {len(d)} games | model {d['model_right'].mean():.1%} | "
          f"market favorite {d['market_right'].mean():.1%}")
    t = d.groupby("tier", observed=True).agg(games=("model_right", "size"), predicted=("fav_prob", "mean"),
                                             model_won=("model_right", "mean"), market_won=("market_right", "mean"))
    print(t.to_string(float_format=lambda x: f"{x:.1%}"))
    return d


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    f = con.execute("SELECT * FROM features").df()
    con.close()
    f[FEATURES] = f[FEATURES].fillna(0)
    last = int(f.loc[(f["season"] == 2026) & f["completed"], "period"].max())

    cur = summarize(replay(f, 2026, last), f"2026 weeks 1-{last}")
    past = pd.concat([replay(f, s, last) for s in range(2017, 2026) if s != 2020])
    summarize(past, f"2017-2025, same weeks 1-{last}")

    late = cur[cur["week"] >= max(last - 1, 1)]
    summarize(late, f"2026 last two weeks ({max(last - 1, 1)}-{last})")

    print("\nTEAMS THE MODEL MISSED MOST IN 2026 (avg points the model was too high on them)")
    long = pd.concat([
        pd.DataFrame({"team": cur["home"], "err": cur["pred"] - cur["margin"]}),
        pd.DataFrame({"team": cur["away"], "err": -(cur["pred"] - cur["margin"])}),
    ])
    agg = long.groupby("team")["err"].agg(["mean", "size"])
    print(agg[agg["size"] >= 4].sort_values("mean", ascending=False).head(10).round(1).to_string())


if __name__ == "__main__":
    main()
