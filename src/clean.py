"""Clean raw reviews and flag the ones substantive enough to classify.

One-word reviews ("nice", "good") carry a star rating but no reason, so they
count towards rating statistics but are not sent for topic classification.
"""
import re

import pandas as pd

from paths import CLEAN_REVIEWS, RAW_REVIEWS

MIN_WORDS = 6
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def main() -> None:
    df = pd.read_parquet(RAW_REVIEWS)
    df["text"] = df["content"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"] != ""].copy()

    df["word_count"] = df["text"].str.split().str.len()
    df["substantive"] = df["word_count"] >= MIN_WORDS
    df["script"] = df["text"].map(lambda t: "Devanagari" if DEVANAGARI.search(t) else "Latin")
    df["date"] = pd.to_datetime(df["at"]).dt.normalize()
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    df["company_replied"] = df["replyContent"].notna()
    df["reply_hours"] = (pd.to_datetime(df["repliedAt"]) - pd.to_datetime(df["at"])).dt.total_seconds() / 3600

    df = df.rename(columns={"reviewId": "review_id", "score": "stars", "thumbsUpCount": "helpful_votes",
                            "appVersion": "app_version"})
    cols = ["review_id", "app", "stars", "text", "word_count", "substantive", "script", "date", "week",
            "helpful_votes", "app_version", "company_replied", "reply_hours"]
    df = df[cols].reset_index(drop=True)

    CLEAN_REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_REVIEWS, index=False)
    summary = df.groupby("app").agg(reviews=("review_id", "size"), substantive=("substantive", "mean"),
                                    avg_stars=("stars", "mean"), one_star=("stars", lambda s: (s == 1).mean()))
    print(f"Saved {len(df):,} clean reviews ({df['substantive'].sum():,} substantive) to {CLEAN_REVIEWS}")
    print(summary.round(3))


if __name__ == "__main__":
    main()
