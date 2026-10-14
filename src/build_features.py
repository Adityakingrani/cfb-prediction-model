"""
Build one row per FBS-vs-FBS game with features that use ONLY information
available before kickoff. Saves to the `features` table in data/cfb.duckdb.

Run from the repo root:  python src\\build_features.py
"""
import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

DB_PATH = "data/cfb.duckdb"
PRIOR_GAMES = 4      # how many games of evidence it takes for in-season data to weigh as much as the prior
RIDGE_ALPHA = 2.0    # shrinkage for opponent-adjusted ratings (higher = more pulled toward average)
POSTSEASON_OFFSET = 20


# ---------------------------------------------------------------- load
def load(con):
    games = con.execute("""
        SELECT id AS game_id, season, week, seasonType AS season_type,
               startDate AS start_date, neutralSite AS neutral, conferenceGame AS conf_game,
               homeTeam AS home, awayTeam AS away,
               homePoints AS home_pts, awayPoints AS away_pts
        FROM games
        WHERE homeClassification = 'fbs' AND awayClassification = 'fbs'
    """).df().drop_duplicates("game_id")

    all_games = con.execute("""
        SELECT id AS game_id, season, startDate AS start_date, homeTeam AS home, awayTeam AS away
        FROM games
    """).df().drop_duplicates("game_id")

    adv = con.execute("""
        SELECT gameId AS game_id, team,
               offense_ppa, offense_successRate AS offense_sr
        FROM adv_game_stats
    """).df().drop_duplicates(["game_id", "team"])

    lines = con.execute("""
        SELECT id AS game_id, MEDIAN(TRY_CAST(spread AS DOUBLE)) AS market_spread
        FROM lines
        WHERE TRY_CAST(spread AS DOUBLE) IS NOT NULL
        GROUP BY id
    """).df()

    sp = con.execute("SELECT year, team, rating AS sp_rating FROM sp_ratings").df()
    ret = con.execute("SELECT season, team, percentPPA AS ret_ppa FROM returning_production").df()
    talent = con.execute("SELECT * FROM talent").df()
    recruit = con.execute("SELECT * FROM recruiting").df()
    return games, all_games, adv, lines, sp, ret, talent, recruit


def pick(df, options):
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(f"None of {options} found in columns {list(df.columns)}")


# ---------------------------------------------------------------- prep
def prep_games(games):
    g = games.copy()
    g["start_date"] = pd.to_datetime(g["start_date"], utc=True, errors="coerce")
    g["neutral"] = g["neutral"].fillna(False).astype(bool)
    g["conf_game"] = g["conf_game"].fillna(False).astype(bool)
    g["postseason"] = (g["season_type"] == "postseason").astype(int)
    g["period"] = g["week"] + g["postseason"] * POSTSEASON_OFFSET
    g["completed"] = g["home_pts"].notna() & g["away_pts"].notna()
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["home_win"] = np.where(g["completed"], (g["margin"] > 0).astype(float), np.nan)
    return g.sort_values(["season", "period", "start_date"]).reset_index(drop=True)


def team_game_rows(g, adv):
    """Two rows per completed game (one per team), with location and stats."""
    done = g[g["completed"]]
    loc = np.where(done["neutral"], 0, 1)
    home = pd.DataFrame({
        "game_id": done["game_id"], "season": done["season"], "period": done["period"],
        "team": done["home"], "opponent": done["away"], "loc": loc,
        "pts_margin": done["margin"],
    })
    away = pd.DataFrame({
        "game_id": done["game_id"], "season": done["season"], "period": done["period"],
        "team": done["away"], "opponent": done["home"], "loc": -loc,
        "pts_margin": -done["margin"],
    })
    tg = pd.concat([home, away], ignore_index=True)
    return tg.merge(adv, on=["game_id", "team"], how="left")


# ---------------------------------------------------------------- opponent-adjusted ratings
def fit_off_def(tg, metric, teams):
    """value = avg + off[team] + def[opponent] + hfa*loc. Higher def = worse defense."""
    d = tg.dropna(subset=[metric])
    if len(d) < 20:
        return None
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(d), len(teams)
    r = np.arange(n)
    X = csr_matrix((np.ones(2 * n),
                    (np.r_[r, r], np.r_[d["team"].map(idx).values, T + d["opponent"].map(idx).values])),
                   shape=(n, 2 * T))
    X = hstack([X, csr_matrix(d[["loc"]].values.astype(float))]).tocsr()
    m = Ridge(alpha=RIDGE_ALPHA).fit(X, d[metric].values)
    return pd.DataFrame({"team": teams, f"off_{metric}": m.coef_[:T], f"def_{metric}": m.coef_[T:2 * T]})


def fit_margin(tg, teams):
    """pts_margin = rating[team] - rating[opponent] + hfa*loc  (an SRS-style power rating)."""
    if len(tg) < 20:
        return None
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(tg), len(teams)
    r = np.arange(n)
    X = csr_matrix((np.r_[np.ones(n), -np.ones(n)],
                    (np.r_[r, r], np.r_[tg["team"].map(idx).values, tg["opponent"].map(idx).values])),
                   shape=(n, T))
    X = hstack([X, csr_matrix(tg[["loc"]].values.astype(float))]).tocsr()
    y = tg["pts_margin"].clip(-38, 38).values   # cap blowouts so garbage time doesn't dominate
    m = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(X, y)
    return pd.DataFrame({"team": teams, "margin_rtg": m.coef_[:T]})


def ratings_from(tg_subset, teams):
    out = pd.DataFrame({"team": teams})
    for part in (fit_margin(tg_subset, teams),
                 fit_off_def(tg_subset, "offense_ppa", teams),
                 fit_off_def(tg_subset, "offense_sr", teams)):
        if part is not None:
            out = out.merge(part, on="team", how="left")
    gp = tg_subset.groupby("team").size().rename("games_played")
    return out.merge(gp, on="team", how="left").fillna({"games_played": 0})


RATING_COLS = ["margin_rtg", "off_offense_ppa", "def_offense_ppa", "off_offense_sr", "def_offense_sr"]


def build_ratings(g, tg):
    """Ratings for each (season, period) using only games from EARLIER periods. Plus final ratings per season."""
    weekly, final = [], []
    for season, sg in g.groupby("season"):
        teams = sorted(set(sg["home"]) | set(sg["away"]))
        stg = tg[tg["season"] == season]
        for period in sorted(sg["period"].unique()):
            r = ratings_from(stg[stg["period"] < period], teams)
            r["season"], r["period"] = season, period
            weekly.append(r)
        f = ratings_from(stg, teams)
        f["season"] = season
        final.append(f)
        print(f"  ratings {season} done")
    weekly = pd.concat(weekly, ignore_index=True)
    final = pd.concat(final, ignore_index=True)
    for c in RATING_COLS:
        if c not in weekly:
            weekly[c] = np.nan
        if c not in final:
            final[c] = np.nan
    return weekly, final


# ---------------------------------------------------------------- priors & context
def build_priors(final, sp, ret, talent, recruit):
    prev = final[["season", "team"] + RATING_COLS].copy()
    prev["season"] += 1
    prev = prev.rename(columns={c: f"prev_{c}" for c in RATING_COLS})

    sp_prev = sp.copy()
    sp_prev["season"] = sp_prev["year"] + 1
    sp_prev = sp_prev[["season", "team", "sp_rating"]].rename(columns={"sp_rating": "prev_sp"})
    sp_prev = sp_prev[sp_prev["team"].notna()]

    ret = ret[["season", "team", "ret_ppa"]]

    t_team, t_year = pick(talent, ["team", "school"]), pick(talent, ["year", "season"])
    t_val = pick(talent, ["talent"])
    tal = talent.rename(columns={t_team: "team", t_year: "season", t_val: "talent"})[["season", "team", "talent"]]
    tal["talent"] = pd.to_numeric(tal["talent"], errors="coerce")

    r_team, r_year = pick(recruit, ["team", "school"]), pick(recruit, ["year", "season"])
    r_val = pick(recruit, ["points"])
    rec = recruit.rename(columns={r_team: "team", r_year: "season", r_val: "rec_points"})[["season", "team", "rec_points"]]
    rec["rec_points"] = pd.to_numeric(rec["rec_points"], errors="coerce")
    rec = rec.sort_values(["team", "season"])
    rec["rec_4yr"] = rec.groupby("team")["rec_points"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    rec = rec[["season", "team", "rec_4yr"]]

    pri = prev
    for df in (sp_prev, ret, tal, rec):
        pri = pri.merge(df, on=["season", "team"], how="outer")
    return pri.drop_duplicates(["season", "team"])


def rest_days(all_games):
    a = all_games.copy()
    a["start_date"] = pd.to_datetime(a["start_date"], utc=True, errors="coerce")
    long = pd.concat([
        a[["game_id", "season", "start_date", "home"]].rename(columns={"home": "team"}),
        a[["game_id", "season", "start_date", "away"]].rename(columns={"away": "team"}),
    ]).sort_values(["team", "start_date"])
    long["rest"] = long.groupby(["team", "season"])["start_date"].diff().dt.days
    long["rest"] = long["rest"].clip(upper=21).fillna(14)
    return long[["game_id", "team", "rest"]].drop_duplicates(["game_id", "team"])


# ---------------------------------------------------------------- assemble
def side(df, prefix, team_col):
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c not in ("season", "period", "team", "game_id")}) \
             .rename(columns={"team": team_col})


def assemble(g, weekly, priors, rest, lines):
    f = g.copy()
    f = f.merge(side(weekly, "h", "home"), on=["season", "period", "home"], how="left")
    f = f.merge(side(weekly, "a", "away"), on=["season", "period", "away"], how="left")
    f = f.merge(side(priors, "h", "home"), on=["season", "home"], how="left")
    f = f.merge(side(priors, "a", "away"), on=["season", "away"], how="left")
    f = f.merge(side(rest, "h", "home"), on=["game_id", "home"], how="left")
    f = f.merge(side(rest, "a", "away"), on=["game_id", "away"], how="left")
    f = f.merge(lines, on="game_id", how="left")

    # fill missing priors with that season's average (new FBS teams etc.)
    prior_cols = [c for c in f.columns if c[2:] in ("prev_margin_rtg", "prev_off_offense_ppa", "prev_def_offense_ppa",
                                                    "prev_off_offense_sr", "prev_def_offense_sr",
                                                    "prev_sp", "ret_ppa", "talent", "rec_4yr")]
    for c in prior_cols:
        f[f"{c}_missing"] = f[c].isna().astype(int)
        f[c] = f[c].fillna(f.groupby("season")[c].transform("mean"))
    for c in [c for c in f.columns if c[2:] in RATING_COLS]:
        f[c] = f[c].fillna(0.0)
    f[["h_games_played", "a_games_played"]] = f[["h_games_played", "a_games_played"]].fillna(0)
    f[["h_rest", "a_rest"]] = f[["h_rest", "a_rest"]].fillna(14)

    # weight on in-season evidence vs prior (0 = all prior, ->1 = all in-season)
    gp = (f["h_games_played"] + f["a_games_played"]) / 2
    f["w_inseason"] = gp / (gp + PRIOR_GAMES)

    # matchup features, home minus away
    f["in_margin"] = f["h_margin_rtg"] - f["a_margin_rtg"]
    f["in_ppa_edge"] = (f["h_off_offense_ppa"] + f["a_def_offense_ppa"]) - (f["a_off_offense_ppa"] + f["h_def_offense_ppa"])
    f["in_sr_edge"] = (f["h_off_offense_sr"] + f["a_def_offense_sr"]) - (f["a_off_offense_sr"] + f["h_def_offense_sr"])

    f["pr_margin"] = f["h_prev_margin_rtg"] - f["a_prev_margin_rtg"]
    f["pr_ppa_edge"] = (f["h_prev_off_offense_ppa"] + f["a_prev_def_offense_ppa"]) - (f["a_prev_off_offense_ppa"] + f["h_prev_def_offense_ppa"])
    f["pr_sp"] = f["h_prev_sp"] - f["a_prev_sp"]
    f["pr_ret_ppa"] = f["h_ret_ppa"] - f["a_ret_ppa"]
    f["pr_talent"] = f["h_talent"] - f["a_talent"]
    f["pr_rec"] = f["h_rec_4yr"] - f["a_rec_4yr"]

    w = f["w_inseason"]
    for c in ("in_margin", "in_ppa_edge", "in_sr_edge"):
        f[f"{c}_w"] = f[c] * w
    for c in ("pr_margin", "pr_ppa_edge", "pr_sp", "pr_ret_ppa", "pr_talent", "pr_rec"):
        f[f"{c}_w"] = f[c] * (1 - w)

    f["home_field"] = (~f["neutral"]).astype(int)
    f["conf_game"] = f["conf_game"].astype(int)
    f["rest_diff"] = f["h_rest"] - f["a_rest"]
    f["covid_2020"] = (f["season"] == 2020).astype(int)
    return f


KEEP = ["game_id", "season", "week", "period", "season_type", "start_date", "home", "away",
        "completed", "home_pts", "away_pts", "margin", "home_win",
        "w_inseason", "h_games_played", "a_games_played",
        "in_margin", "in_ppa_edge", "in_sr_edge",
        "pr_margin", "pr_ppa_edge", "pr_sp", "pr_ret_ppa", "pr_talent", "pr_rec",
        "in_margin_w", "in_ppa_edge_w", "in_sr_edge_w",
        "pr_margin_w", "pr_ppa_edge_w", "pr_sp_w", "pr_ret_ppa_w", "pr_talent_w", "pr_rec_w",
        "home_field", "conf_game", "postseason", "rest_diff", "covid_2020",
        "market_spread"]


def report(f):
    print("\n================ FEATURE TABLE ================")
    s = f.groupby("season").agg(games=("game_id", "size"), completed=("completed", "sum"),
                                 has_market=("market_spread", lambda x: x.notna().sum()))
    print(s.to_string())

    done = f[f["completed"] & (f["season"] < 2026)]
    print("\nCorrelation with final margin (completed games, 2014-2025):")
    for c in ("in_margin", "pr_margin", "pr_sp", "pr_talent", "pr_ret_ppa", "in_ppa_edge"):
        sub = done[done["w_inseason"] > 0.5] if c.startswith("in_") else done
        print(f"  {c:<14} {sub[c].corr(sub['margin']):+.3f}")
    m = done.dropna(subset=["market_spread"])
    print(f"  {'-market_spread':<14} {(-m['market_spread']).corr(m['margin']):+.3f}   (should be clearly positive)")

    late = done[done["w_inseason"] > 0.5]
    pick_home = (late["in_margin"] + 2.5 * late["home_field"]) > 0
    print(f"\nQuick sanity, weeks with real in-season data: rating+home-field picks winner "
          f"{(pick_home == (late['margin'] > 0)).mean():.1%} of {len(late):,} games")
    lm = late.dropna(subset=["market_spread"])
    lm = lm[lm["market_spread"] != 0]
    print(f"Market favorite wins {((lm['market_spread'] < 0) == (lm['margin'] > 0)).mean():.1%} of {len(lm):,} games (benchmark)")


def main():
    con = duckdb.connect(DB_PATH)
    print("Loading...")
    games, all_games, adv, lines, sp, ret, talent, recruit = load(con)
    g = prep_games(games)
    tg = team_game_rows(g, adv)
    print("Building week-by-week ratings (only past games each week)...")
    weekly, final = build_ratings(g, tg)
    priors = build_priors(final, sp, ret, talent, recruit)
    f = assemble(g, weekly, priors, rest_days(all_games), lines)[KEEP]

    con.register("f_df", f)
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM f_df")
    con.unregister("f_df")
    con.close()
    report(f)
    print(f"\nSaved `features`: {len(f):,} rows, {len(f.columns)} columns")


if __name__ == "__main__":
    main()