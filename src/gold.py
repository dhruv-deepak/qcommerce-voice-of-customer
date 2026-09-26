"""Human validation: how often does Claude agree with a person?

  python src/gold.py export     -> data/labels/gold_to_label.csv (150 reviews, blank label columns)
  (fill in the CSV yourself: sentiment + one column per topic, 1 if the review complains about it)
  python src/gold.py evaluate   -> agreement between your labels and Claude's

Label blind: do not look at Claude's labels while filling in the sheet.
"""
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from paths import CLEAN_REVIEWS, GOLD_LABELS, GOLD_TEMPLATE, LLM_LABELS, OUTPUTS
from taxonomy import TOPICS

N = 150


def export() -> None:
    labelled = pd.read_json(LLM_LABELS, lines=True)["review_id"]
    reviews = pd.read_parquet(CLEAN_REVIEWS, columns=["review_id", "app", "stars", "text"])
    pick = reviews[reviews["review_id"].isin(labelled)].sample(N, random_state=11)
    for col in ["sentiment", *TOPICS, "churn_intent"]:
        pick[col] = ""
    pick.to_csv(GOLD_TEMPLATE, index=False, encoding="utf-8-sig")
    print(f"Wrote {N} reviews to {GOLD_TEMPLATE}.\nFill it in, save it as {GOLD_LABELS.name} "
          "in the same folder, then run: python src/gold.py evaluate")


def evaluate() -> None:
    human = pd.read_csv(GOLD_LABELS, encoding="utf-8-sig")
    llm = pd.read_json(LLM_LABELS, lines=True).drop_duplicates("review_id", keep="last")
    m = human.merge(llm, on="review_id", suffixes=("_human", ""))
    rows = [{"label": "sentiment", "support": len(m),
             "agreement": (m["sentiment_human"].str.strip().str.lower() == m["sentiment"]).mean(),
             "kappa": cohen_kappa_score(m["sentiment_human"].str.strip().str.lower(), m["sentiment"]),
             "f1": f1_score(m["sentiment_human"].str.strip().str.lower(), m["sentiment"], average="macro")}]
    for t in TOPICS:
        h = m[t].fillna(0).astype(int)
        c = m["complaints"].map(lambda xs, t=t: int(t in xs))
        if h.sum() + c.sum() == 0:
            continue
        rows.append({"label": t, "support": int(h.sum()), "agreement": (h == c).mean(),
                     "kappa": cohen_kappa_score(h, c), "f1": f1_score(h, c, zero_division=0)})
    out = pd.DataFrame(rows)
    OUTPUTS.mkdir(exist_ok=True)
    out.to_csv(OUTPUTS / "human_vs_claude.csv", index=False)
    print(out.round(2).to_string(index=False))
    topics = out[out["label"] != "sentiment"]
    print(f"\nClaude vs human on {len(m)} reviews: sentiment agreement {out.iloc[0]['agreement']:.0%}, "
          f"complaint-topic macro F1 {topics['f1'].mean():.2f}")


if __name__ == "__main__":
    {"export": export, "evaluate": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
