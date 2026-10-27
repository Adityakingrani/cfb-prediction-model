"""
Pull quarterback passing stats for every FBS game from CollegeFootballData.com box scores
and save them to the `qb_games` table in data/cfb.duckdb.

Run from the repo root:
    python src\\pull_qb.py                  # full history 2014-now (one time, ~220 API calls, ~3 minutes)
    python src\\pull_qb.py --season 2026    # only weeks of 2026 not already saved (used by friday.py)
"""
import argparse
import os
import time

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

DB_PATH = "data/cfb.duckdb"
BASE = "https://api.collegefootballdata.com"
FIRST_SEASON = 2014


def headers():
    load_dotenv()
    key = os.getenv("CFBD_API_KEY")
    if not key:
        raise SystemExit("No API key found. Check your .env file.")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def to_num(x):
    try:
        return float(str(x).replace("--", "").strip() or "nan")
    except ValueError:
        return float("nan")


def parse(data, season, week, season_type):
    """Box score JSON -> one row per passer per game."""
    rows = []
    for game in data:
        gid = game.get("id")
        for t in game.get("teams", []):
            team = t.get("team") or t.get("school")
            for cat in t.get("categories", []):
                if cat.get("name") != "passing":
                    continue
                stats = {}
                for typ in cat.get("types", []):
                    for a in typ.get("athletes", []):
                        key = (str(a.get("id")), a.get("name"))
                        stats.setdefault(key, {})[typ.get("name")] = a.get("stat")
                for (pid, name), s in stats.items():
                    if name in (None, "", " TEAM", "TEAM") or str(pid) in ("-9999", "None"):
                        continue
                    cmp_att = str(s.get("C/ATT", "0/0")).split("/")
                    att = to_num(cmp_att[1]) if len(cmp_att) == 2 else float("nan")
                    rows.append({"game_id": gid, "season": season, "week": week, "season_type": season_type,
                                 "team": team, "qb_id": pid, "qb_name": name,
                                 "cmp": to_num(cmp_att[0]) if len(cmp_att) == 2 else float("nan"), "att": att,
                                 "yds": to_num(s.get("YDS")), "td": to_num(s.get("TD")), "ints": to_num(s.get("INT"))})
    return rows


def fetch(session, params, tries=6):
    """GET with retries: waits 5s, 10s, 20s, 40s, 60s between attempts if the connection drops."""
    wait = 5
    for attempt in range(1, tries + 1):
        try:
            r = session.get(f"{BASE}/games/players", params=params, timeout=60)
            if r.status_code in (429, 502, 503, 504):
                raise requests.ConnectionError(f"HTTP {r.status_code}")
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == tries:
                raise SystemExit(f"\nConnection keeps dropping ({e}). Everything pulled so far is saved - "
                                 f"wait 15-30 minutes (or switch networks) and run the same command again; "
                                 f"it will pick up where it stopped.")
            print(f"    connection dropped, retrying in {wait}s (attempt {attempt}/{tries - 1})...")
            time.sleep(wait)
            wait = min(wait * 2, 60)


def save_rows(con, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    con.register("q_df", df)
    if "qb_games" not in tables:
        con.execute("CREATE TABLE qb_games AS SELECT * FROM q_df")
    else:
        con.execute("DELETE FROM qb_games WHERE game_id IN (SELECT DISTINCT game_id FROM q_df)")
        con.execute("INSERT INTO qb_games BY NAME SELECT * FROM q_df")
    con.unregister("q_df")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="only pull missing weeks of this season")
    args = ap.parse_args()
    con = duckdb.connect(DB_PATH)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

    weeks = con.execute("""
        SELECT DISTINCT season, week, seasonType FROM games
        WHERE homePoints IS NOT NULL AND season >= ? AND (homeClassification = 'fbs' OR awayClassification = 'fbs')
        ORDER BY 1, 3 DESC, 2
    """, [args.season or FIRST_SEASON]).fetchall()
    if not args.season and "qb_games" in tables:
        have = set(con.execute("SELECT DISTINCT season, week, season_type FROM qb_games").fetchall())
        skipped = len([w for w in weeks if w in have])
        weeks = [w for w in weeks if w not in have]
        if skipped:
            print(f"Resuming: {skipped} weeks already saved, {len(weeks)} to go.")
    if args.season:
        weeks = [w for w in weeks if w[0] == args.season]
        if "qb_games" in tables:
            have = set(con.execute("SELECT DISTINCT season, week, season_type FROM qb_games WHERE season = ?",
                                   [args.season]).fetchall())
            latest = max([w for w in have], default=None, key=lambda w: (w[2] == "regular", w[1]))
            # re-pull the most recent saved week too, in case it was incomplete
            weeks = [w for w in weeks if (w[0], w[1], w[2]) not in have or (w[0], w[1], w[2]) == latest]

    session = requests.Session()
    session.headers.update(headers())
    rows, calls, total = [], 0, 0
    current = None
    for season, week, st in weeks:
        if current is not None and season != current:      # save after every season
            save_rows(con, rows)
            total += len(rows)
            print(f"  saved {current}")
            rows = []
        current = season
        try:
            r = fetch(session, {"year": season, "week": week, "seasonType": st, "category": "passing"})
        except SystemExit:
            save_rows(con, rows)
            con.close()
            raise
        calls += 1
        if r.status_code != 200:
            print(f"  !! {season} {st} week {week}: HTTP {r.status_code}")
            continue
        got = parse(r.json(), season, week, st)
        if not got and r.json():
            print(f"  !! {season} week {week}: unexpected format. First item keys: {list(r.json()[0].keys())}")
        rows += got
        time.sleep(1.0)   # slower pace to avoid being throttled
        if week == 1 or st == "postseason":
            print(f"  {season} {st}: pulling...")

    save_rows(con, rows)
    total += len(rows)
    if not total and not calls:
        print("Nothing new to pull.")
        con.close()
        return
    n = con.execute("SELECT COUNT(*), COUNT(DISTINCT game_id) FROM qb_games").fetchone()
    print(f"Saved {total:,} passer rows ({calls} API calls). qb_games now has {n[0]:,} rows across {n[1]:,} games.")
    con.close()


if __name__ == "__main__":
    main()
