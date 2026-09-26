"""Build a self-contained executive report (outputs/report.html) from insights.json.

Usage:  python src/report.py
"""
import base64
import html
import json

import pandas as pd

from paths import CHARTS, OUTPUTS
from taxonomy import TOPIC_LABELS, TOPICS

OWNER = {TOPIC_LABELS[k]: owner for k, (_, owner) in TOPICS.items()}


def img(name: str) -> str:
    path = CHARTS / name
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(name)}">'


def table(df: pd.DataFrame, fmt: str = "{:.1f}") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = ""
    for idx, r in df.iterrows():
        cells = "".join(f"<td>{fmt.format(v) if isinstance(v, (int, float)) and pd.notna(v) else html.escape(str(v))}</td>"
                        for v in r)
        rows += f"<tr><th>{html.escape(str(idx))}</th>{cells}</tr>"
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"


def main() -> None:
    ins = json.loads((OUTPUTS / "insights.json").read_text(encoding="utf-8"))
    sc = pd.DataFrame(ins["scorecard"]).T
    sc.index.name = "App"
    drag = pd.Series(ins["rating_drag"]).sort_values()
    churn = pd.DataFrame(ins["churn_by_topic"]).T.sort_values("churn_rate", ascending=False)
    rates = pd.DataFrame(ins["complaint_rates"])
    worst_app = sc["Avg stars"].idxmin()
    best_app = sc["Avg stars"].idxmax()
    top = ins["top_complaint_overall"]
    most_damaging = drag.index[0]
    top_churn = churn.index[0]

    # Per-app priorities: biggest "stars at stake"
    recs = []
    for app, topics in ins["stars_at_stake_top3"].items():
        items = list(topics.items())
        lead, gain = items[0]
        rest = ", ".join(t for t, _ in items[1:])
        recs.append((app, lead, gain, OWNER.get(lead, ""), rest))
    rec_rows = "".join(
        f"<tr><th>{a}</th><td><b>{html.escape(t)}</b><br><span class='m'>Owner: {html.escape(o)}</span></td>"
        f"<td>up to +{g:.2f} ★</td><td>{html.escape(r)}</td></tr>" for a, t, g, o, r in recs)

    rising = pd.DataFrame(ins["rising_complaints"])
    rising_html = ""
    if not rising.empty:
        rising = rising[rising["change_pp"] > 0].head(5).set_index("app")
        rising.index.name = "App"
        rising = rising.rename(columns={"topic": "Complaint", "prior": "Prior 4 wks", "recent": "Last 4 wks",
                                        "change_pp": "Change (pts)"})
        rising_html = table(rising)

    val_path = OUTPUTS / "human_vs_claude.csv"
    clf_path = OUTPUTS / "classifier_report.csv"
    validation = []
    if val_path.exists():
        v = pd.read_csv(val_path)
        t = v[v["label"] != "sentiment"]
        validation.append(f"Claude agreed with a human annotator on {v.iloc[0]['agreement']:.0%} of sentiment "
                          f"labels and reached a macro F1 of {t['f1'].mean():.2f} on complaint topics "
                          "(blind hand-labelled sample).")
    if clf_path.exists():
        c = pd.read_csv(clf_path)
        ct = c[c["label"].str.startswith("c_")]
        validation.append(f"The distilled classifier reproduces Claude's complaint labels with a macro F1 of "
                          f"{ct['f1'].mean():.2f} (5-fold out-of-fold), at near-zero cost per review.")
    validation_html = "".join(f"<li>{html.escape(x)}</li>" for x in validation)

    p0, p1 = ins["period"]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q-Commerce Voice of Customer</title>
<style>
:root {{ --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --accent:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d0d0d; --card:#1a1a19; --ink:#fff;
  --ink2:#c3c2b7; --line:#2c2c2a; --accent:#3987e5; }} }}
:root[data-theme="dark"] {{ --bg:#0d0d0d; --card:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --line:#2c2c2a; --accent:#3987e5; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:900px; margin:0 auto; padding:40px 16px 80px; }}
h1 {{ font-size:28px; line-height:1.2; margin:0 0 6px; }}
h2 {{ font-size:19px; margin:44px 0 6px; }}
.m {{ color:var(--muted); font-size:13px; }}
.lead {{ color:var(--ink2); margin:0 0 24px; }}
.box {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent); padding:16px 20px; border-radius:6px; }}
.box li {{ margin:6px 0; }}
img {{ max-width:100%; height:auto; background:#fcfcfb; border-radius:6px; border:1px solid var(--line); margin-top:10px; }}
.tw {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; margin-top:10px; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
thead th {{ color:var(--ink2); font-weight:600; }}
</style></head><body><main>
<p class="m">Voice-of-customer study · Google Play, India · {p0} to {p1}</p>
<h1>What India's quick-commerce customers complain about, and what each app should fix first</h1>
<p class="lead">{ins['n_reviews']:,} public reviews of Blinkit, Zepto, Swiggy Instamart and bigbasket.
{ins['n_substantive']:,} had enough text to classify; Claude labelled {ins['n_claude_labelled']:,} of them and a
distilled classifier labelled the rest.</p>

<div class="box"><b>Key findings</b><ul>
<li><b>{html.escape(top)}</b> is the most common complaint across the category.</li>
<li><b>{html.escape(most_damaging)}</b> does the most damage to ratings: a review that raises it scores
{abs(drag.iloc[0]):.2f} stars lower, all else equal.</li>
<li><b>{html.escape(top_churn)}</b> is the strongest churn trigger: {churn.iloc[0]['churn_rate']:.0f}% of customers
raising it say they are leaving or switching.</li>
<li>{html.escape(best_app)} rates highest ({sc.at[best_app, 'Avg stars']:.2f}★); {html.escape(worst_app)} rates lowest
({sc.at[worst_app, 'Avg stars']:.2f}★).</li>
</ul></div>

<h2>Recommendation: one priority fix per app</h2>
<p class="m">Ranked by "stars at stake": how often the complaint is raised × how much it lowers the rating.
It is an upper bound from correlations, useful for ranking rather than forecasting.</p>
<div class="tw"><table><thead><tr><th>App</th><th>Fix first</th><th>Rating upside</th><th>Next</th></tr></thead>
<tbody>{rec_rows}</tbody></table></div>

<h2>Scorecard</h2>
{table(sc)}
{img('one_star_share.png')}

<h2>Where each app is weakest</h2>
<p class="m">Complaints per 100 reviews. Every review is in the denominator, so apps with more one-word reviews are not flattered.</p>
{img('complaint_heatmap.png')}

<h2>Not all complaints are equal: prioritise by frequency and damage</h2>
{img('priority_matrix.png')}

<h2>What drives customers to leave</h2>
{img('churn_triggers.png')}

<h2>What is getting worse</h2>
{rising_html or '<p class="m">Not enough weeks of data to compare.</p>'}
{img('trend_top_complaint.png')}

<h2>What customers value</h2>
{img('praise_heatmap.png')}

<h2>Method and limitations</h2>
<ul>
<li>Reviews collected from Google Play (India) with reviewer names and photos removed at collection.</li>
<li>Claude (Anthropic) labelled a stratified sample against a 12-topic taxonomy built around the customer
journey; a TF-IDF + logistic-regression classifier trained on those labels scaled them to all reviews.</li>
{validation_html}
<li>App-store reviewers skew towards extreme experiences, so rates describe reviewers, not all customers.</li>
<li>Rating impact comes from a regression with app fixed effects. It shows association, not causation.</li>
</ul>
</main></body></html>"""
    out = OUTPUTS / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
