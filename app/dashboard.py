"""Interactive dashboard.  Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze import APP_COLORS, APPS, C, churn_by_topic, load, nice, rating_drag  # noqa: E402

st.set_page_config(page_title="Q-Commerce Voice of Customer", layout="wide")
COLOR = alt.Scale(domain=APPS, range=[APP_COLORS[a] for a in APPS])


@st.cache_data
def data() -> pd.DataFrame:
    df = load()
    df["app"] = df["app"].astype(str)
    return df


df_all = data()
st.title("What India's quick-commerce customers complain about")
st.caption(f"{len(df_all):,} Google Play reviews, {df_all['date'].min():%d %b} – {df_all['date'].max():%d %b %Y}. "
           "Topics labelled by Claude on a sample and by a distilled classifier on the rest.")

# One filter row scopes everything below it
f1, f2 = st.columns([2, 3])
apps = f1.multiselect("Apps", APPS, default=APPS)
d0, d1 = df_all["date"].min().date(), df_all["date"].max().date()
start, end = f2.slider("Dates", min_value=d0, max_value=d1, value=(d0, d1))
df = df_all[df_all["app"].isin(apps) & df_all["date"].dt.date.between(start, end)]
if df.empty:
    st.warning("No reviews match these filters.")
    st.stop()

# Scorecard tiles
cols = st.columns(len(apps))
for col, app in zip(cols, apps):
    a = df[df["app"] == app]
    col.metric(app, f"{a['stars'].mean():.2f} ★", f"{100 * (a['stars'] == 1).mean():.0f}% one-star",
               delta_color="off")
    col.caption(f"{len(a):,} reviews · {100 * a['churn_intent'].mean():.1f}% threaten to leave")

tab1, tab2, tab3, tab4 = st.tabs(["Pain points", "Priority matrix", "Trends", "Read the reviews"])

with tab1:
    rates = (100 * df.groupby("app")[C].mean()).reset_index().melt("app", var_name="topic", value_name="per100")
    rates["topic"] = rates["topic"].map(nice)
    order = rates.groupby("topic")["per100"].mean().sort_values(ascending=False).index.tolist()
    heat = alt.Chart(rates).mark_rect(stroke="white", strokeWidth=2).encode(
        x=alt.X("app:N", sort=APPS, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("topic:N", sort=order, title=None),
        color=alt.Color("per100:Q", scale=alt.Scale(scheme="blues"), title="per 100 reviews"),
        tooltip=["app", "topic", alt.Tooltip("per100:Q", format=".1f", title="per 100 reviews")],
    )
    text = heat.mark_text(fontSize=11).encode(text=alt.Text("per100:Q", format=".1f"),
                                              color=alt.value("#0b0b0b"))
    st.altair_chart((heat + text).properties(height=420), use_container_width=True)
    st.caption("Complaints per 100 reviews. The denominator is every review, so apps compare fairly.")

with tab2:
    drag = rating_drag(df)
    churn = churn_by_topic(df)
    share = (100 * df[C].mean()).rename(lambda c: nice(c))
    pm = pd.DataFrame({"topic": share.index, "per100": share.values, "star_drop": -drag.reindex(share.index).values})
    pm = pm.merge(churn.reset_index(), on="topic", how="left")
    chart = alt.Chart(pm).mark_circle(opacity=0.8, stroke="white", strokeWidth=2, color=APP_COLORS["Blinkit"]).encode(
        x=alt.X("per100:Q", title="How common (complaints per 100 reviews)"),
        y=alt.Y("star_drop:Q", title="How damaging (star drop when raised)"),
        size=alt.Size("churn_rate:Q", title="% threatening to leave", scale=alt.Scale(range=[60, 900])),
        tooltip=["topic", alt.Tooltip("per100:Q", format=".1f"), alt.Tooltip("star_drop:Q", format=".2f"),
                 alt.Tooltip("churn_rate:Q", format=".0f", title="% threaten to leave")],
    )
    labels = chart.mark_text(align="left", dx=10, fontSize=11, color="#52514e").encode(text="topic")
    st.altair_chart((chart + labels).properties(height=480), use_container_width=True)
    st.caption("Top-right = fix first. Star drop comes from a regression of stars on complaint flags with app "
               "fixed effects. It is correlational, not causal.")

with tab3:
    topic = st.selectbox("Complaint", C, format_func=nice)
    w = df.groupby(["week", "app"])[topic].mean().mul(100).reset_index(name="per100")
    wk = df.groupby("week").size()
    w = w[w["week"].map(wk) >= 200]
    line = alt.Chart(w).mark_line(strokeWidth=2).encode(
        x=alt.X("week:T", title=None), y=alt.Y("per100:Q", title="per 100 reviews"),
        color=alt.Color("app:N", scale=COLOR, title="App"),
        tooltip=["app", alt.Tooltip("week:T", format="%d %b"), alt.Tooltip("per100:Q", format=".1f")],
    )
    st.altair_chart(line.properties(height=380), use_container_width=True)

with tab4:
    topic = st.selectbox("Show reviews complaining about", C, format_func=nice, key="read")
    hits = df[(df[topic] == 1)].sort_values("helpful_votes", ascending=False)
    st.write(f"{len(hits):,} reviews. Most-upvoted first.")
    st.dataframe(hits[["app", "stars", "date", "helpful_votes", "text"]].head(200),
                 use_container_width=True, hide_index=True)
