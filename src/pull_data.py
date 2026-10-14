"""
Pull historical college football data from CollegeFootballData.com
and save it to data/cfb.duckdb.

Run from the repo root:  python src\pull_data.py
"""
import os
import time
import requests
import pandas as pd
import duckdb
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("CFBD_API_KEY")
if not KEY:
    raise SystemExit("No API key found. Check your .env file.")

BASE = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
YEARS = range(2014, 2027)            # 2014 through 2026
SEASON_TYPES = ["regular", "postseason"]
DB_PATH = "data/cfb.duckdb"

calls = 0
failures = []


def get(endpoint, **params):
    """One API call. Logs failures instead of crashing."""
    global calls
    calls += 1
    try:
        r = requests.get(f"{BASE}{endpoint}", headers=HEADERS, params=params, timeout=60)
    except requests.RequestException as e:
        failures.append(f"{endpoint} {params} -> {e}")
        return []
    if r.status_code == 401:
        raise SystemExit("401 Unauthorized: your API key is wrong or not active yet.")
    if r.status_code == 429:
        raise SystemExit("429 Rate limited: you've hit the API call limit. Stop and tell Claude.")
    if r.status_code != 200:
        failures.append(f"{endpoint} {params} -> HTTP {r.status_code}")
        return []
    time.sleep(0.3)  # be polite to the API
    return r.json()


def pull(endpoint, by_season_type=False, record_path=None, meta=None):
    """Pull an endpoint for every year (and season type if needed) into one DataFrame."""
    frames = []
    for year in YEARS:
        types = SEASON_TYPES if by_season_type else [None]
        for st in types:
            params = {"year": year}
            if st:
                params["seasonType"] = st
            data = get(endpoint, **params)
            if not data:
                continue
            if record_path:
                df = pd.json_normalize(data, record_path=record_path, meta=meta,
                                       sep="_", errors="ignore")
            else:
                df = pd.json_normalize(data, sep="_")
            df["pull_year"] = year
            if st:
                df["pull_season_type"] = st
            frames.append(df)
        print(f"  {endpoint}  {year} done")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save(con, name, df):
    if df.empty:
        print(f"  !! {name}: no data, skipped")
        return
    try:
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
        print(f"  saved {name}: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        failures.append(f"save {name} -> {e}")
        print(f"  !! {name}: save failed ({e})")


def main():
    os.makedirs("data", exist_ok=True)
    con = duckdb.connect(DB_PATH)

    print("\n[1/8] Games")
    save(con, "games", pull("/games", by_season_type=True))

    print("\n[2/8] Betting lines (benchmark only)")
    save(con, "lines", pull("/lines", by_season_type=True, record_path="lines",
                            meta=["id", "season", "seasonType", "week",
                                  "homeTeam", "awayTeam", "homeScore", "awayScore"]))

    print("\n[3/8] Advanced game stats (EPA, success rate, etc.)")
    save(con, "adv_game_stats", pull("/stats/game/advanced", by_season_type=True))

    print("\n[4/8] Pregame win probabilities (benchmark only)")
    save(con, "pregame_wp", pull("/metrics/wp/pregame", by_season_type=True))

    print("\n[5/8] SP+ ratings (end of season)")
    save(con, "sp_ratings", pull("/ratings/sp"))

    print("\n[6/8] Returning production")
    save(con, "returning_production", pull("/player/returning"))

    print("\n[7/8] Recruiting team rankings")
    save(con, "recruiting", pull("/recruiting/teams"))

    print("\n[8/8] Team talent composite")
    save(con, "talent", pull("/talent"))

    print("\n================ SUMMARY ================")
    print(f"API calls used: {calls}")
    for (t,) in con.execute("SHOW TABLES").fetchall():
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<22} {n:>8,} rows")
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures:
            print("  -", f)
    else:
        print("\nNo failures.")
    con.close()


if __name__ == "__main__":
    main()