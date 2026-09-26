"""Turn labelled reviews into business answers.

  1. Scorecard      - how does each app rate, and how often do customers complain?
  2. Pain points    - what do customers complain about, app by app?
  3. Rating impact  - which complaints drag the star rating down the most?
  4. Churn triggers - which complaints make customers threaten to leave, and to whom?
  5. Trends         - which complaints are rising week on week?
  6. Service        - does the company reply to unhappy reviewers?

Usage:  python src/analyze.py      (writes outputs/insights.json and outputs/charts/*.png)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import CHARTS, LABELLED_REVIEWS, OUTPUTS
from taxonomy import TOPIC_LABELS, TOPICS

APPS = ["Blinkit", "Zepto", "Instamart", "bigbasket"]  # fixed order = fixed colour
APP_COLORS = dict(zip(APPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
C = [f"c_{t}" for t in TOPICS]
P = [f"p_{t}" for t in TOPICS]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load() -> pd.DataFrame:
    df = pd.read_parquet(LABELLED_REVIEWS)
    df["app"] = pd.Categorical(df["app"], APPS, ordered=True)
    df["any_complaint"] = df[C].max(axis=1)
    return df


def nice(col: str) -> str:
    return TOPIC_LABELS[col[2:]]


# --- 1. Scorecard -----------------------------------------------------------------

def scorecard(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("app", observed=True)
    neg = df[df["stars"] <= 2].groupby("app", observed=True)
    return pd.DataFrame({
        "Reviews": g.size(),
        "Avg stars": g["stars"].mean(),
        "% 1-star": 100 * g["stars"].apply(lambda s: (s == 1).mean()),
        "% negative": 100 * g["sentiment"].apply(lambda s: (s == "negative").mean()),
        "Complaints per 100 reviews": 100 * g[C].sum().sum(axis=1) / g.size(),
        "% threatening to leave": 100 * g["churn_intent"].mean(),
        "Reply rate to 1-2 star (%)": 100 * neg["company_replied"].mean(),
    })


# --- 2. Pain points ---------------------------------------------------------------

def complaint_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Complaints per 100 reviews, topic x app. Denominator = all reviews, so apps compare fairly."""
    rates = 100 * df.groupby("app", observed=True)[C].mean().T
    rates.index = [nice(c) for c in rates.index]
    return rates.loc[rates.mean(axis=1).sort_values(ascending=False).index]


def praise_rates(df: pd.DataFrame) -> pd.DataFrame:
    rates = 100 * df.groupby("app", observed=True)[P].mean().T
    rates.index = [nice(c) for c in rates.index]
    return rates.loc[rates.mean(axis=1).sort_values(ascending=False).index]


# --- 3. Rating impact -------------------------------------------------------------

def rating_drag(df: pd.DataFrame) -> pd.Series:
    """OLS of stars on complaint flags + app fixed effects (substantive reviews only).
    Coefficient = average star drop when a review raises that complaint, all else equal."""
    sub = df[df["substantive"]]
    X = pd.concat([sub[C].astype(float), pd.get_dummies(sub["app"], drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0)
    coef, *_ = np.linalg.lstsq(X.to_numpy(), sub["stars"].to_numpy(dtype=float), rcond=None)
    return pd.Series(coef[1:1 + len(C)], index=[nice(c) for c in C])


def stars_at_stake(df: pd.DataFrame, drag: pd.Series) -> pd.DataFrame:
    """Rough upper bound on the average-rating gain if a complaint were fixed:
    share of all reviews raising it x its star drag. Correlational, not causal."""
    share = df.groupby("app", observed=True)[C].mean().T
    share.index = [nice(c) for c in share.index]
    return share.mul(-drag, axis=0).clip(lower=0)


# --- 4. Churn ---------------------------------------------------------------------

def churn_by_topic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in C:
        hit = df[df[c] == 1]
        if len(hit) >= 30:
            rows.append({"topic": nice(c), "reviews": len(hit), "churn_rate": 100 * hit["churn_intent"].mean()})
    return pd.DataFrame(rows).sort_values("churn_rate", ascending=False).set_index("topic")


def switching_flows(df: pd.DataFrame) -> pd.DataFrame:
    """From the Claude-labelled sample: where do customers say they are going?"""
    s = df[(df["label_source"] == "claude") & df["switch_to"].notna() & (df["switch_to"] != "None")]
    s = s[s["switch_to"] != s["app"].astype(str)]
    return pd.crosstab(s["app"], s["switch_to"])


# --- 5. Trends --------------------------------------------------------------------

def weekly_trend(df: pd.DataFrame, topic_col: str) -> pd.DataFrame:
    w = df.groupby(["week", "app"], observed=True)[topic_col].mean().unstack() * 100
    counts = df.groupby("week").size()
    return w[counts.reindex(w.index) >= 200]  # drop thin partial weeks at the edges


def rising_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Last 4 weeks vs the 4 weeks before, per app and topic (pp change per 100 reviews)."""
    weeks = sorted(df["week"].unique())
    if len(weeks) < 9:
        return pd.DataFrame()
    recent, prior = weeks[-5:-1], weeks[-9:-5]  # skip the final partial week
    rows = []
    for app in APPS:
        a = df[df["app"] == app]
        for c in C:
            r, p = 100 * a[a["week"].isin(recent)][c].mean(), 100 * a[a["week"].isin(prior)][c].mean()
            rows.append({"app": app, "topic": nice(c), "prior": p, "recent": r, "change_pp": r - p})
    return pd.DataFrame(rows).sort_values("change_pp", ascending=False)


# --- 6. Service -------------------------------------------------------------------

def reply_stats(df: pd.DataFrame) -> pd.DataFrame:
    neg = df[df["stars"] <= 2]
    g = neg.groupby("app", observed=True)
    return pd.DataFrame({"Reply rate (%)": 100 * g["company_replied"].mean(),
                         "Median reply time (hrs)": g["reply_hours"].median()})


# --- Charts -----------------------------------------------------------------------

def _style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(CHARTS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def chart_heatmap(rates: pd.DataFrame, title: str, name: str):
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rates) + 1.2))
    vmax = rates.to_numpy().max()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#f4f8fe"] + BLUES)
    ax.imshow(rates.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(rates.shape[1]), rates.columns)
    ax.set_yticks(range(len(rates)), rates.index)
    ax.xaxis.tick_top()
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            v = rates.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 * vmax else INK)
    ax.tick_params(length=0, labelcolor=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=28)
    _save(fig, name)


def chart_priority_matrix(df: pd.DataFrame, drag: pd.Series, churn: pd.DataFrame):
    share = 100 * df[C].mean()
    share.index = [nice(c) for c in share.index]
    topics = [t for t in share.index if t in churn.index]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _style(ax, grid_axis="both")
    x, y = share[topics], -drag[topics]
    size = 30 + 12 * churn.loc[topics, "churn_rate"]
    ax.scatter(x, y, s=size, color=APP_COLORS["Blinkit"], alpha=0.75, edgecolor=SURFACE, linewidth=2)
    for t in topics:
        ax.annotate(t, (x[t], y[t]), xytext=(7, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.axvline(x.median(), color=AXIS, linewidth=0.8)
    ax.axhline(y.median(), color=AXIS, linewidth=0.8)
    ax.text(ax.get_xlim()[1], ax.get_ylim()[1], "Fix first: common and damaging", ha="right", va="top",
            fontsize=8.5, color=MUTED)
    ax.set_xlabel("How common: complaints per 100 reviews (all apps)", color=INK2)
    ax.set_ylabel("How damaging: star drop when raised", color=INK2)
    ax.set_title("Pain-point priority matrix (bubble size = % threatening to leave)", loc="left", color=INK,
                 fontsize=11)
    _save(fig, "priority_matrix.png")


def chart_churn(churn: pd.DataFrame):
    c = churn.sort_values("churn_rate")
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(c) + 1))
    _style(ax)
    ax.barh(c.index, c["churn_rate"], color=APP_COLORS["Blinkit"], height=0.6)
    for i, v in enumerate(c["churn_rate"]):
        ax.text(v + 0.4, i, f"{v:.0f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("% of reviews raising this complaint that threaten to leave", color=INK2)
    ax.set_title("Which complaints push customers out the door", loc="left", color=INK, fontsize=11)
    _save(fig, "churn_triggers.png")


def chart_trend(df: pd.DataFrame, topic_col: str, name: str):
    w = weekly_trend(df, topic_col)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _style(ax, grid_axis="y")
    for app in w.columns:
        ax.plot(w.index, w[app], color=APP_COLORS[app], linewidth=2)
        ax.text(w.index[-1], w[app].iloc[-1], f"  {app}", color=INK2, fontsize=8.5, va="center")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("per 100 reviews", color=INK2)
    ax.set_title(f"{nice(topic_col)} complaints, weekly", loc="left", color=INK, fontsize=11)
    fig.autofmt_xdate()
    _save(fig, name)


def chart_stars(sc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    _style(ax)
    apps = sc.index[::-1]
    ax.barh(apps, sc.loc[apps, "% 1-star"], color=[APP_COLORS[a] for a in apps], height=0.6)
    for i, a in enumerate(apps):
        ax.text(sc.at[a, "% 1-star"] + 0.4, i, f"{sc.at[a, '% 1-star']:.0f}%  (avg {sc.at[a, 'Avg stars']:.2f}★)",
                va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("% of reviews that are 1-star", color=INK2)
    ax.set_title("Share of 1-star reviews", loc="left", color=INK, fontsize=11)
    _save(fig, "one_star_share.png")


def main() -> None:
    df = load()
    sc = scorecard(df)
    rates = complaint_rates(df)
    praise = praise_rates(df)
    drag = rating_drag(df)
    stake = stars_at_stake(df, drag)
    churn = churn_by_topic(df)
    flows = switching_flows(df)
    rising = rising_complaints(df)
    replies = reply_stats(df)

    chart_stars(sc)
    chart_heatmap(rates, "Complaints per 100 reviews", "complaint_heatmap.png")
    chart_heatmap(praise, "Praise per 100 reviews", "praise_heatmap.png")
    chart_priority_matrix(df, drag, churn)
    chart_churn(churn)
    top_topic = C[int(np.argmax(df[C].mean().to_numpy()))]
    chart_trend(df, top_topic, "trend_top_complaint.png")

    top_stake = {app: stake[app].sort_values(ascending=False).head(3).round(3).to_dict() for app in stake.columns}
    insights = {
        "period": [str(df["date"].min().date()), str(df["date"].max().date())],
        "n_reviews": int(len(df)),
        "n_substantive": int(df["substantive"].sum()),
        "n_claude_labelled": int((df["label_source"] == "claude").sum()),
        "scorecard": sc.round(2).to_dict(orient="index"),
        "complaint_rates": rates.round(2).to_dict(),
        "praise_rates": praise.round(2).to_dict(),
        "rating_drag": drag.round(3).to_dict(),
        "stars_at_stake_top3": top_stake,
        "churn_by_topic": churn.round(1).to_dict(orient="index"),
        "switching_flows": flows.to_dict(orient="index"),
        "rising_complaints": rising.head(8).round(2).to_dict(orient="records"),
        "reply_stats": replies.round(1).to_dict(orient="index"),
        "top_complaint_overall": nice(top_topic),
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "insights.json").write_text(json.dumps(insights, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 160)
    print(sc.round(2), "\n")
    print(rates.round(1), "\n")
    print("Star drag:\n", drag.round(2).sort_values(), "\n")
    print("Churn by topic:\n", churn.round(1), "\n")
    print("Switching flows:\n", flows, "\n")
    print("Rising:\n", rising.head(8).round(2), "\n")
    print(f"Charts -> {CHARTS}")


if __name__ == "__main__":
    main()
