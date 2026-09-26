"""Distil Claude's labels into a fast local classifier, then label every review.

TF-IDF on word n-grams plus character n-grams (robust to Hinglish spelling
variants like "bahut / bhut / bohot") feeding one logistic regression per
label. Per-label decision thresholds are tuned on 5-fold out-of-fold
predictions, which also give an honest estimate of agreement with Claude.

Usage:  python src/train_classifier.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from paths import CLEAN_REVIEWS, LABELLED_REVIEWS, LLM_LABELS, MODEL_DIR, OUTPUTS
from taxonomy import SENTIMENTS, TOPICS

MIN_POSITIVES = 15  # labels rarer than this in the sample are not modelled


class Featuriser:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                    max_features=150_000)

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


def load_training() -> pd.DataFrame:
    labels = pd.read_json(LLM_LABELS, lines=True).drop_duplicates("review_id", keep="last")
    reviews = pd.read_parquet(CLEAN_REVIEWS, columns=["review_id", "text", "stars"])
    df = labels.merge(reviews, on="review_id")
    for t in TOPICS:
        df[f"c_{t}"] = df["complaints"].map(lambda xs, t=t: t in xs).astype(int)
        df[f"p_{t}"] = df["praises"].map(lambda xs, t=t: t in xs).astype(int)
    df["churn_intent"] = df["churn_intent"].astype(int)
    return df


def model_input(df: pd.DataFrame) -> pd.Series:
    # The star rating is a strong, free signal; expose it as a token
    return df["stars"].astype(str) + "star " + df["text"]


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.arange(0.15, 0.86, 0.05)
    return float(grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])])


def main() -> None:
    df = load_training()
    print(f"Training on {len(df):,} Claude-labelled reviews")
    feats = Featuriser()
    X = feats.fit_transform(model_input(df))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    binary_labels = [f"c_{t}" for t in TOPICS] + [f"p_{t}" for t in TOPICS] + ["churn_intent"]
    models, thresholds, report = {}, {}, []
    for label in binary_labels:
        y = df[label].to_numpy()
        if y.sum() < MIN_POSITIVES:
            continue
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
        oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        thr = best_threshold(y, oof)
        pred = oof >= thr
        report.append({"label": label, "positives_in_sample": int(y.sum()), "threshold": thr,
                       "precision": precision_score(y, pred, zero_division=0),
                       "recall": recall_score(y, pred, zero_division=0),
                       "f1": f1_score(y, pred, zero_division=0)})
        models[label], thresholds[label] = clf.fit(X, y), thr

    sent_clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
    oof = cross_val_predict(sent_clf, X, df["sentiment"], cv=cv)
    report.append({"label": "sentiment (3-class)", "positives_in_sample": len(df), "threshold": np.nan,
                   "precision": np.nan, "recall": np.nan, "f1": f1_score(df["sentiment"], oof, average="macro")})
    models["sentiment"] = sent_clf.fit(X, df["sentiment"])

    rep = pd.DataFrame(report)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rep.to_csv(OUTPUTS / "classifier_report.csv", index=False)
    topic_rows = rep[rep["label"].str.startswith("c_")]
    print(rep.round(2).to_string(index=False))
    print(f"\nComplaint topics: macro F1 {topic_rows['f1'].mean():.2f} vs Claude labels (5-fold out-of-fold)")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"featuriser": feats, "models": models, "thresholds": thresholds}, MODEL_DIR / "classifier.joblib")

    # --- Label every review ----------------------------------------------------
    allr = pd.read_parquet(CLEAN_REVIEWS)
    sub = allr[allr["substantive"]].copy()
    Xa = feats.transform(model_input(sub))
    for label, clf in models.items():
        if label == "sentiment":
            sub["sentiment"] = clf.predict(Xa)
        else:
            sub[label] = (clf.predict_proba(Xa)[:, 1] >= thresholds[label]).astype(int)
    for label in binary_labels:
        if label not in sub:
            sub[label] = 0
    # Where Claude labelled the review itself, prefer Claude's label
    llm = df.set_index("review_id")[binary_labels + ["sentiment", "switch_to"]]
    sub = sub.set_index("review_id")
    overlap = llm.index.intersection(sub.index)
    sub.loc[overlap, binary_labels + ["sentiment"]] = llm.loc[overlap, binary_labels + ["sentiment"]]
    sub["switch_to"] = llm["switch_to"].reindex(sub.index)
    sub["label_source"] = np.where(sub.index.isin(overlap), "claude", "classifier")
    sub = sub.reset_index()

    short = allr[~allr["substantive"]].copy()
    short["sentiment"] = pd.cut(short["stars"], [0, 2, 3, 5], labels=["negative", "neutral", "positive"]).astype(str)
    short["label_source"] = "stars_only"
    out = pd.concat([sub, short], ignore_index=True)
    out[binary_labels] = out[binary_labels].fillna(0).astype(int)
    out.to_parquet(LABELLED_REVIEWS, index=False)
    print(f"Labelled {len(out):,} reviews -> {LABELLED_REVIEWS}")
    print(json.dumps(out["label_source"].value_counts().to_dict()))


if __name__ == "__main__":
    main()
