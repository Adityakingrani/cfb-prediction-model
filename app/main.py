"""
CFB Predictions app. Reads the files friday.py saves in the predictions folder.

Opens automatically at the end of friday.py, or run it yourself from the repo root:
    streamlit run app\\main.py
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

PRED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "predictions")
CONF_ORDER = {"HIGH": 0, "MED": 1, "LOW": 2}
CONF_STYLE = {
    "HIGH": ("#1f7a4d", "#e3f4ea", "High"),
    "MED": ("#9a6700", "#fff4d6", "Medium"),
    "LOW": ("#57606a", "#eef0f2", "Low"),
}

st.set_page_config(page_title="CFB Predictions", page_icon="🏈", layout="wide")
st.markdown("""
<style>
  .kick {color:#6e7781; font-size:0.85rem; margin-bottom:2px;}
  .matchup {font-size:1.05rem; font-weight:600; margin-bottom:6px;}
  .pickline {font-size:1.0rem; margin-bottom:4px;}
  .badge {display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.75rem;
          font-weight:600; margin-left:6px; vertical-align:middle;}
  .detail {color:#57606a; font-size:0.88rem; line-height:1.5;}
  .flag {color:#bc4c00; font-size:0.85rem; font-weight:600; margin-top:6px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
def list_weeks():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*.csv")), reverse=True)
    weeks = []
    for f in files:
        meta = {}
        j = f.replace(".csv", ".json")
        if os.path.exists(j):
            with open(j) as fh:
                meta = json.load(fh)
        name = os.path.basename(f).replace(".csv", "")
        season, tag = name.split("_", 1)
        label = meta.get("label") or (tag.replace("week", "Week ").replace("Week 0", "Week ").title())
        weeks.append({"file": f, "title": f"{season} {label}", "meta": meta})
    return weeks


@st.cache_data
def load(path, mtime):
    d = pd.read_csv(path)
    d["kickoff_dt"] = pd.to_datetime(d["kickoff"], format="%Y-%m-%d %I:%M %p", errors="coerce")
    if "neutral" not in d:
        d["neutral"] = False
    return d


def market_text(r):
    s = r["market_spread"]
    if pd.isna(s):
        return "No line yet"
    if s == 0:
        return "Pick'em"
    fav = r["home"] if s < 0 else r["away"]
    return f"{fav} -{abs(s):g}"


def card(r):
    fg, bg, word = CONF_STYLE.get(r["conf"], CONF_STYLE["LOW"])
    sep = "vs" if r["neutral"] else "@"
    kick = r["kickoff_dt"].strftime("%a %I:%M %p").replace(" 0", " ") if pd.notna(r["kickoff_dt"]) else "Time TBD"
    with st.container(border=True):
        st.markdown(f'<div class="kick">{kick}{" · neutral site" if r["neutral"] else ""}</div>'
                    f'<div class="matchup">{r["away"]} {sep} {r["home"]}</div>'
                    f'<div class="pickline">Pick: <b>{r["pick"]}</b>'
                    f'<span class="badge" style="color:{fg};background:{bg}">{word}</span></div>',
                    unsafe_allow_html=True)
        st.progress(min(max(r["win_pct"] / 100, 0.0), 1.0), text=f"{r['win_pct']:.0f}% to win")
        st.markdown(f'<div class="detail">Model: {r["pick"]} by {r["pick_margin"]:.1f}<br>'
                    f'Market: {market_text(r)}</div>'
                    + ('<div class="flag">⚑ Model disagrees with the market. Check injury news.</div>'
                       if r["disagrees_with_market"] else ""),
                    unsafe_allow_html=True)


# ---------------------------------------------------------------- page
weeks = list_weeks()
if not weeks:
    st.title("🏈 CFB Predictions")
    st.info("No predictions yet. Run `python src\\friday.py` first.")
    st.stop()

with st.sidebar:
    st.header("Week")
    choice = st.selectbox("Week", [w["title"] for w in weeks], label_visibility="collapsed")
    week = next(w for w in weeks if w["title"] == choice)
    st.header("Filters")
    team = st.text_input("Team", placeholder="e.g. Michigan State")
    confs = st.multiselect("Confidence", ["HIGH", "MED", "LOW"], default=["HIGH", "MED", "LOW"],
                           format_func=lambda c: CONF_STYLE[c][2])
    only_flags = st.toggle("Only games where model disagrees with market")
    sort_by = st.radio("Sort by", ["Kickoff time", "Confidence"], horizontal=True)

df = load(week["file"], os.path.getmtime(week["file"]))
meta = week["meta"]

st.title(f"🏈 {week['title']} Predictions")
if meta.get("generated"):
    st.caption(f"Generated {meta['generated']} · Win % and margins are model estimates; "
               "the market line is shown for comparison only and is never used by the model.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Games", len(df))
c2.metric("High confidence", int((df["conf"] == "HIGH").sum()))
c3.metric("Disagree with market", int(df["disagrees_with_market"].sum()))
if meta.get("model_games"):
    mr, mn = meta["model_right"], meta["model_games"]
    kr, kn = meta["market_right"], meta["market_games"]
    c4.metric("Model record (season)", f"{mr}-{mn - mr}", f"{mr / mn:.1%}", delta_color="off")
    c5.metric("Market favorite (season)", f"{kr}-{kn - kr}", f"{kr / kn:.1%}", delta_color="off")

view = df[df["conf"].isin(confs)]
if team:
    t = team.strip().lower()
    view = view[view["home"].str.lower().str.contains(t, regex=False)
                | view["away"].str.lower().str.contains(t, regex=False)]
if only_flags:
    view = view[view["disagrees_with_market"]]
if sort_by == "Confidence":
    view = view.assign(_c=view["conf"].map(CONF_ORDER)).sort_values(["_c", "win_pct"], ascending=[True, False])
else:
    view = view.sort_values(["kickoff_dt", "away"])

tab_cards, tab_table = st.tabs(["Cards", "Table"])

with tab_cards:
    if view.empty:
        st.info("No games match these filters.")
    elif sort_by == "Kickoff time":
        for day, games in view.groupby(view["kickoff_dt"].dt.strftime("%A, %b %d").fillna("Time TBD"), sort=False):
            st.subheader(day)
            cols = st.columns(3)
            for i, (_, r) in enumerate(games.iterrows()):
                with cols[i % 3]:
                    card(r)
    else:
        cols = st.columns(3)
        for i, (_, r) in enumerate(view.iterrows()):
            with cols[i % 3]:
                card(r)

with tab_table:
    table = view.assign(
        Kickoff=view["kickoff_dt"].dt.strftime("%a %I:%M %p"),
        Matchup=view["away"] + " @ " + view["home"],
        Market=view.apply(market_text, axis=1),
    )[["Kickoff", "Matchup", "pick", "win_pct", "pick_margin", "Market", "conf", "disagrees_with_market"]]
    st.dataframe(
        table.rename(columns={"pick": "Pick", "win_pct": "Win %", "pick_margin": "Margin",
                              "conf": "Confidence", "disagrees_with_market": "Disagrees w/ market"}),
        hide_index=True, width="stretch",
        column_config={"Win %": st.column_config.ProgressColumn("Win %", min_value=50, max_value=100, format="%.0f%%"),
                       "Margin": st.column_config.NumberColumn(format="%.1f")},
    )