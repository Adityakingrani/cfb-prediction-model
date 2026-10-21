"""
Grade a week of saved predictions against the actual final scores.

Run from the repo root after the weekend's games are done:
    python src\\grade_week.py --week 7
    python src\\grade_week.py --week 7 --no-pull    # skip refreshing scores from the API
"""
import os
import argparse
import requests
import duckdb
import pandas as pd
from dotenv import load_dotenv

DB_PATH = "data/cfb.duckdb"
BASE = "https://api.collegefootballdata.com"


def refresh_scores(season):
    load_dotenv()
    key = os.getenv("CFBD_API_KEY")
    r = requests.get(f"{BASE}/games", params={"year": season, "seasonType": "regular"},
                     headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    g = pd.json_normalize(r.json())
    return g[["id", "homeTeam", "awayTeam", "homePoints", "awayPoints"]].rename(columns={"id": "game_id"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args()

    path = os.path.join("predictions", f"{args.season}_week{args.week:02d}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Can't find {path}")
    pred = pd.read_csv(path)

    if args.no_pull:
        con = duckdb.connect(DB_PATH, read_only=True)
        res = con.execute("SELECT id AS game_id, homePoints, awayPoints FROM games WHERE season = ?",
                          [args.season]).df()
        con.close()
    else:
        res = refresh_scores(args.season)[["game_id", "homePoints", "awayPoints"]]

    d = pred.merge(res, on="game_id", how="left")
    d = d[d["homePoints"].notna() & d["awayPoints"].notna()].copy()
    d = d[d["homePoints"] != d["awayPoints"]]
    if d.empty:
        raise SystemExit("No finished games found for that week yet.")
    d["winner"] = d.apply(lambda r: r["home"] if r["homePoints"] > r["awayPoints"] else r["away"], axis=1)
    d["correct"] = d["pick"] == d["winner"]
    d["score"] = d["away"] + " " + d["awayPoints"].astype(int).astype(str) + ", " + \
                 d["home"] + " " + d["homePoints"].astype(int).astype(str)

    n, c = len(d), int(d["correct"].sum())
    print("\n" + "=" * 70)
    print(f" WEEK {args.week} RESULTS: {c}-{n - c}  ({c / n:.1%} correct, {n} of {len(pred)} games graded)")
    print("=" * 70)

    print("\nBY CONFIDENCE")
    for tier in ["HIGH", "MED", "LOW"]:
        t = d[d["conf"] == tier]
        if len(t):
            print(f"  {tier:<5} {int(t['correct'].sum())}-{int((~t['correct']).sum())}  "
                  f"({t['correct'].mean():.0%}, avg predicted {t['win_pct'].mean():.0f}%)")

    print("\nBIGGEST MISSES (most confident wrong picks)")
    for _, r in d[~d["correct"]].sort_values("win_pct", ascending=False).head(5).iterrows():
        print(f"  Picked {r['pick']} at {r['win_pct']:.0f}%  ->  {r['score']}")

    print("\nBEST CALLS (closest margins on correct picks)")
    d["actual_margin"] = (d["homePoints"] - d["awayPoints"]).abs()
    good = d[d["correct"]].assign(err=lambda x: (x["actual_margin"] - x["pick_margin"]).abs())
    for _, r in good.sort_values("err").head(5).iterrows():
        print(f"  {r['pick']} by {r['pick_margin']:.1f} predicted  ->  {r['score']}")

    print("\nSAMPLE GAMES ON THE COVER SLIDE")
    for h in ["Texas", "Tennessee", "Michigan"]:
        row = d[d["home"] == h]
        for _, r in row.iterrows():
            print(f"  {'✓' if r['correct'] else '✗'} {r['away']} @ {r['home']}: picked {r['pick']} "
                  f"({r['win_pct']:.0f}%)  ->  {r['score']}")
    print()


if __name__ == "__main__":
    main()