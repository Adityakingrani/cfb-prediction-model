"""
Weekly run: refresh this season's data, rebuild features, train the model,
and predict every FBS vs FBS game in the upcoming week.

Run from the repo root:
    python src\\friday.py              # normal weekly run
    python src\\friday.py --no-pull    # skip the API refresh (use data already in the database)
"""
import os
import sys
import json
import subprocess
import time
import argparse
import warnings
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import norm
from sklearn.linear_model import Ridge

import build_features

warnings.filterwarnings("ignore")
DB_PATH = "data/cfb.duckdb"
OUT_DIR = "predictions"
LOCAL_TZ = "America/New_York"
MARGIN_CAP = 45
FEATURES = ["in_margin_w", "in_ppa_edge_w", "in_sr_edge_w",
            "pr_margin_w", "pr_ppa_edge_w", "pr_sp_w", "pr_ret_ppa_w", "pr_talent_w", "pr_rec_w",
            "home_field", "rest_diff"]
BASE = "https://api.collegefootballdata.com"


# ---------------------------------------------------------------- 1. refresh current season
def refresh(con, season):
    load_dotenv()
    key = os.getenv("CFBD_API_KEY")
    if not key:
        raise SystemExit("No API key found. Check your .env file.")
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    def get(endpoint, **params):
        r = requests.get(f"{BASE}{endpoint}", headers=headers, params=params, timeout=60)
        if r.status_code == 429:
            raise SystemExit("429 Rate limited: monthly API limit hit.")
        r.raise_for_status()
        time.sleep(0.3)
        return r.json()

    def replace_season(table, frames, record_path=None, meta=None):
        frames = [f for f in frames if f[1]]
        if not frames:
            return 0
        dfs = []
        for st, data in frames:
            df = (pd.json_normalize(data, record_path=record_path, meta=meta, sep="_", errors="ignore")
                  if record_path else pd.json_normalize(data, sep="_"))
            df["pull_year"], df["pull_season_type"] = season, st
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
        df = df[[c for c in df.columns if c in cols]]
        con.execute(f"DELETE FROM {table} WHERE pull_year = {season}")
        con.register("new_df", df)
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM new_df")
        con.unregister("new_df")
        return len(df)

    calls = 0
    out = {}
    for table, endpoint, rp, meta in [
        ("games", "/games", None, None),
        ("adv_game_stats", "/stats/game/advanced", None, None),
        ("lines", "/lines", "lines", ["id", "season", "seasonType", "week",
                                      "homeTeam", "awayTeam", "homeScore", "awayScore"]),
    ]:
        frames = []
        for st in ("regular", "postseason"):
            frames.append((st, get(endpoint, year=season, seasonType=st)))
            calls += 1
        out[table] = replace_season(table, frames, rp, meta)
    print(f"Refreshed {season}: " + ", ".join(f"{k} {v:,} rows" for k, v in out.items())
          + f"  ({calls} API calls)")


# ---------------------------------------------------------------- 2. model helpers
def train_model(train):
    model = Ridge(alpha=1.0).fit(train[FEATURES], train["margin"].clip(-MARGIN_CAP, MARGIN_CAP))
    sigma = np.std(train["margin"] - model.predict(train[FEATURES]))
    return model, sigma


def confidence(p):
    p = max(p, 1 - p)
    return "HIGH" if p >= 0.75 else "MED" if p >= 0.60 else "LOW"


def season_record(f, season):
    """Walk-forward record for this season so far: each week predicted using only earlier games."""
    hist = f[f["completed"] & (f["season"] != 2020)]
    done = hist[hist["season"] == season]
    model_right = market_right = model_n = market_n = 0
    for period in sorted(done["period"].unique()):
        train = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["period"] < period))]
        test = done[(done["period"] == period) & (done["margin"] != 0)]
        model, _ = train_model(train)
        pred = model.predict(test[FEATURES])
        model_right += int(((pred > 0) == (test["margin"] > 0)).sum())
        model_n += len(test)
        m = test.dropna(subset=["market_spread"])
        m = m[m["market_spread"] != 0]
        market_right += int(((m["market_spread"] < 0) == (m["margin"] > 0)).sum())
        market_n += len(m)
    return model_right, model_n, market_right, market_n


# ---------------------------------------------------------------- 3. predict
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true", help="skip refreshing data from the API")
    ap.add_argument("--week", type=int, help="predict a specific week instead of the next one")
    ap.add_argument("--no-app", action="store_true", help="don't open the Streamlit app at the end")
    args = ap.parse_args()

    con = duckdb.connect(DB_PATH)
    season = con.execute("SELECT MAX(season) FROM games").fetchone()[0]
    if not args.no_pull:
        refresh(con, season)
    con.close()

    print("Rebuilding features...")
    build_features.main()

    con = duckdb.connect(DB_PATH, read_only=True)
    f = con.execute("SELECT * FROM features").df()
    con.close()
    f[FEATURES] = f[FEATURES].fillna(0)

    cur = f[f["season"] == season]
    upcoming = cur[~cur["completed"]]
    if upcoming.empty:
        raise SystemExit(f"No upcoming games left in {season}.")
    if args.week:
        week_games = upcoming[(upcoming["week"] == args.week) & (upcoming["postseason"] == 0)]
    else:
        # next week = earliest unplayed game that hasn't already kicked off
        # (so a postponed/cancelled game from an old week can't get the script stuck)
        kick = pd.to_datetime(upcoming["start_date"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6)
        future = upcoming[kick.isna() | (kick >= cutoff)]
        if future.empty:
            raise SystemExit(f"No upcoming games left in {season}.")
        next_period = future["period"].min()
        week_games = upcoming[upcoming["period"] == next_period]
    if week_games.empty:
        raise SystemExit("No unplayed games found for that week.")
    week = int(week_games["week"].iloc[0])
    label = f"BOWLS / PLAYOFF" if week_games["postseason"].iloc[0] else f"WEEK {week}"

    train = f[f["completed"] & (f["season"] != 2020)]
    model, sigma = train_model(train)

    g = week_games.copy()
    g["pred_margin"] = model.predict(g[FEATURES])
    g["home_prob"] = norm.cdf(g["pred_margin"] / sigma)
    g["pick"] = np.where(g["pred_margin"] >= 0, g["home"], g["away"])
    g["win_pct"] = np.maximum(g["home_prob"], 1 - g["home_prob"])
    g["pick_margin"] = g["pred_margin"].abs()
    g["conf"] = g["home_prob"].apply(confidence)
    g["flag"] = (g["market_spread"].notna() & (g["market_spread"] != 0)
                 & ((g["market_spread"] < 0) != (g["pred_margin"] > 0)))
    g["kickoff"] = pd.to_datetime(g["start_date"], utc=True).dt.tz_convert(LOCAL_TZ)
    g["matchup"] = np.where(g["home_field"] == 1, g["away"] + " @ " + g["home"], g["away"] + " vs " + g["home"])
    g = g.sort_values(["kickoff", "matchup"])

    # ---- print
    mr, mn, kr, kn = season_record(f, season)
    now = datetime.now().strftime("%a %b %d, %I:%M %p")
    W = 104
    print("\n" + "=" * W)
    print(f" CFB PREDICTIONS - {season} {label}          generated {now}")
    if mn:
        print(f" Model record this season: {mr}-{mn - mr} ({mr / mn:.1%}) | "
              f"Market favorite: {kr}-{kn - kr} ({kr / kn:.1%})")
    print("=" * W)
    print(f" {'KICKOFF':<15}{'MATCHUP':<40}{'PICK':<22}{'WIN %':>6}{'MARGIN':>8}{'MARKET':>8}  CONF")
    for _, r in g.iterrows():
        kick = r["kickoff"].strftime("%a %I:%M %p") if pd.notna(r["kickoff"]) else "TBD"
        mkt = f"{r['market_spread']:+.1f}" if pd.notna(r["market_spread"]) else "--"
        print(f" {kick:<15}{r['matchup'][:39]:<40}{r['pick'][:21]:<22}{r['win_pct']:>6.0%}"
              f"{r['pick_margin']:>+8.1f}{mkt:>8}  {r['conf']:<4}{'  !' if r['flag'] else ''}")
    print("-" * W)
    counts = g["conf"].value_counts()
    print(f" {len(g)} games | HIGH: {counts.get('HIGH', 0)} | MEDIUM: {counts.get('MED', 0)} | LOW: {counts.get('LOW', 0)}")
    print(f" MARKET = betting spread from the HOME team's view (negative = home favored), shown for comparison only")
    print(f" ! = model picks a different winner than the market ({int(g['flag'].sum())} games) - check injuries/news")
    print("=" * W)

    # ---- save
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "bowls" if g["postseason"].iloc[0] else f"week{week:02d}"
    path = os.path.join(OUT_DIR, f"{season}_{tag}.csv")
    g["neutral"] = g["home_field"] == 0
    out = g[["kickoff", "away", "home", "pick", "win_pct", "pick_margin", "market_spread", "conf", "flag",
             "neutral", "game_id"]].copy()
    out["kickoff"] = out["kickoff"].dt.strftime("%Y-%m-%d %I:%M %p")
    out["win_pct"] = (out["win_pct"] * 100).round(1)
    out["pick_margin"] = out["pick_margin"].round(1)
    out["market_spread"] = out["market_spread"].round(1)
    out.rename(columns={"flag": "disagrees_with_market"}).to_csv(path, index=False)
    summary = {"season": int(season), "label": label.title(), "file": os.path.basename(path),
               "generated": now, "model_right": mr, "model_games": mn,
               "market_right": kr, "market_games": kn}
    with open(path.replace(".csv", ".json"), "w") as fh:
        json.dump(summary, fh)
    print(f"Saved {path}")

    if not args.no_app:
        print("\nOpening the predictions app in your browser... (press Ctrl+C here to stop it)")
        app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "main.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app])


if __name__ == "__main__":
    main()