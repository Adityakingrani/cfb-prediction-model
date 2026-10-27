"""
Do the two upgrades actually make the model more accurate?

Walk-forward backtest (2017-2025, excluding 2020) of four versions on the same games:
  current      the model you run today
  low_prior    + teams new to FBS start near the bottom of FBS instead of average
  qb           + quarterback value (assumes last game's starter, like the live model does by default)
  low_prior+qb both upgrades
plus an upper bound you can't fully get live:
  qb_perfect   QB features using who ACTUALLY started (= if every late QB change were entered as an override)

The winner (lowest Brier score) is saved to model_config.json, and friday.py uses it automatically.

Run from the repo root:  python src\\compare_variants.py
"""
import json
import warnings

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
DB_PATH = "data/cfb.duckdb"
TEST_SEASONS = [s for s in range(2017, 2026) if s != 2020]
PRIORS = ["pr_margin_w", "pr_ppa_edge_w", "pr_sp_w", "pr_ret_ppa_w", "pr_talent_w", "pr_rec_w"]
INSEASON = ["in_margin_w", "in_ppa_edge_w", "in_sr_edge_w"]
CONTEXT = ["home_field", "rest_diff"]
VARIANTS = {
    "current": INSEASON + PRIORS + CONTEXT,
    "low_prior": INSEASON + [f"{p}_low" for p in PRIORS] + CONTEXT,
    "qb": INSEASON + PRIORS + CONTEXT + ["qb_edge", "qb_change"],
    "low_prior+qb": INSEASON + [f"{p}_low" for p in PRIORS] + CONTEXT + ["qb_edge", "qb_change"],
    "qb_perfect": INSEASON + [f"{p}_low" for p in PRIORS] + CONTEXT + ["qb_edge_actual", "qb_change_actual"],
}
SELECTABLE = ["current", "low_prior", "qb", "low_prior+qb"]


def score(d, m):
    d = d[d["margin"] != 0]
    p = d[f"{m}_prob"].clip(1e-4, 1 - 1e-4)
    y = (d["margin"] > 0).astype(int)
    return {"games": len(d), "accuracy": ((d[f"{m}_pred"] > 0) == (d["margin"] > 0)).mean(),
            "brier": ((p - y) ** 2).mean(), "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
            "margin_mae": (d[f"{m}_pred"] - d["margin"]).abs().mean()}


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    f = con.execute("SELECT * FROM features WHERE completed").df()
    con.close()
    cols = sorted({c for v in VARIANTS.values() for c in v})
    f[cols] = f[cols].fillna(0)
    f = f.sort_values(["season", "period"])
    if (f["qb_edge"] == 0).all():
        print("!! QB features are all zero - run python src\\pull_qb.py and python src\\build_features.py first")

    rows = []
    for season in TEST_SEASONS:
        for period in sorted(f.loc[f["season"] == season, "period"].unique()):
            train = f[((f["season"] < season) | ((f["season"] == season) & (f["period"] < period))) & (f["season"] != 2020)]
            test = f[(f["season"] == season) & (f["period"] == period)].copy()
            if len(train) < 500 or test.empty:
                continue
            for name, feats in VARIANTS.items():
                m = Ridge(alpha=1.0).fit(train[feats], train["margin"].clip(-45, 45))
                sig = np.std(train["margin"] - m.predict(train[feats]))
                test[f"{name}_pred"] = m.predict(test[feats])
                test[f"{name}_prob"] = norm.cdf(test[f"{name}_pred"] / sig)
            ms = train.dropna(subset=["market_spread"])
            test["market_pred"] = -test["market_spread"]
            test["market_prob"] = norm.cdf(test["market_pred"] / np.std(ms["margin"] + ms["market_spread"]))
            rows.append(test)
        print(f"  backtested {season}")
    bt = pd.concat(rows, ignore_index=True)
    common = bt.dropna(subset=["market_pred"])
    names = list(VARIANTS) + ["market"]

    print("\n================ OVERALL (2017-2025 excl. 2020, same games) ================")
    print(pd.DataFrame({n: score(common, n) for n in names}).T.to_string(float_format=lambda x: f"{x:.4f}"))

    common["fav"] = np.maximum(common["current_prob"], 1 - common["current_prob"])
    close = common[common["fav"] < 0.6]
    print(f"\nCLOSE GAMES ONLY (current model under 60%, {len(close)} games) - accuracy")
    print(pd.Series({n: score(close, n)["accuracy"] for n in names}).to_string(float_format=lambda x: f"{x:.1%}"))

    newb = common[common["new_fbs_game"] == 1]
    if len(newb):
        print(f"\nGAMES WITH A TEAM NEW TO FBS ({len(newb)} games)")
        print(pd.DataFrame({n: score(newb, n) for n in ["current", "low_prior", "market"]}).T[["accuracy", "margin_mae"]]
              .to_string(float_format=lambda x: f"{x:.3f}"))

    qbc = common[common["qb_change_actual"].abs() > 0.5]
    if len(qbc):
        print(f"\nGAMES WHERE A TEAM'S STARTER CHANGED ({len(qbc)} games) - accuracy")
        print(pd.Series({n: score(qbc, n)["accuracy"] for n in names}).to_string(float_format=lambda x: f"{x:.1%}"))

    briers = {n: score(common, n)["brier"] for n in SELECTABLE}
    best = min(briers, key=briers.get)
    gain = briers["current"] - briers[best]
    if best != "current" and gain < 0.0003:
        print(f"\n{best} is better by only {gain:.4f} Brier - too small to matter, keeping current.")
        best = "current"
    json.dump({"chosen": best, "features": VARIANTS[best], "brier": round(briers[best], 5),
               "brier_current": round(briers["current"], 5)}, open("model_config.json", "w"), indent=2)
    print(f"\nWINNER: {best}  (Brier {briers[best]:.4f} vs current {briers['current']:.4f})")
    print("Saved model_config.json - friday.py will use this version from now on.")


if __name__ == "__main__":
    main()
