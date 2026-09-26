"""Collect recent public Google Play reviews for India's quick-commerce apps.

Reviewer names and profile images are dropped at collection time: the analysis
only needs the text, rating, date, app version and whether the company replied.

Each app is saved as soon as it finishes, so a re-run skips apps already done
(pass --refresh to fetch everything again).

Usage:  python src/scrape.py --since 2026-07-27
"""
import argparse
import time
from datetime import datetime

import pandas as pd
from google_play_scraper import Sort, reviews

from paths import RAW_REVIEWS

APPS = {
    "Blinkit": "com.grofers.customerapp",
    "Zepto": "com.zeptoconsumerapp",
    "Instamart": "in.swiggy.android.instamart",
    "bigbasket": "com.bigbasket.mobileapp",
}
KEEP = ["reviewId", "score", "at", "content", "thumbsUpCount", "appVersion", "replyContent", "repliedAt"]
PER_APP_DIR = RAW_REVIEWS.parent / "by_app"
RETRY_WAITS = [5, 20, 60]  # seconds; the endpoint sometimes returns an empty page when throttling


def fetch_page(app_id: str, token):
    for wait in [0, *RETRY_WAITS]:
        time.sleep(wait)
        try:
            batch, next_token = reviews(app_id, lang="en", country="in", sort=Sort.NEWEST,
                                        count=200, continuation_token=token)
        except Exception as e:  # the library surfaces network errors as bare exceptions
            print(f"\n  request error ({e}); retrying")
            continue
        if batch:
            return batch, next_token
    return [], None


def fetch_app(app_name: str, app_id: str, since: datetime, max_reviews: int) -> pd.DataFrame:
    rows, token = [], None
    while len(rows) < max_reviews:
        batch, token = fetch_page(app_id, token)
        if not batch:
            break
        rows.extend({k: r.get(k) for k in KEEP} for r in batch)
        oldest = batch[-1]["at"]
        print(f"  {app_name}: {len(rows):>6,} reviews, back to {oldest:%d %b %Y}", end="\r", flush=True)
        if oldest < since or token is None:
            break
        time.sleep(0.5)  # be polite to the endpoint
    print()
    df = pd.DataFrame(rows, columns=KEEP)
    df["app"] = app_name
    df = df[pd.to_datetime(df["at"]) >= since]
    if not df.empty and df["at"].min() > since + pd.Timedelta(days=2):
        print(f"  WARNING: {app_name} only reaches back to {df['at'].min():%d %b}; re-run with --refresh {app_name}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2026-07-27")
    p.add_argument("--max-per-app", type=int, default=90_000)
    p.add_argument("--refresh", nargs="*", default=None, help="app names to re-fetch (none listed = all)")
    args = p.parse_args()
    since = datetime.fromisoformat(args.since)
    refresh = set(APPS) if args.refresh == [] else set(args.refresh or [])

    PER_APP_DIR.mkdir(parents=True, exist_ok=True)
    for name, app_id in APPS.items():
        path = PER_APP_DIR / f"{name}.parquet"
        if path.exists() and name not in refresh:
            print(f"{name}: already saved, skipping")
            continue
        print(f"Fetching {name} ({app_id})", flush=True)
        df = fetch_app(name, app_id, since, args.max_per_app)
        if df.empty:
            print(f"  {name}: no reviews returned, not saved. Re-run later to retry.")
            continue
        df.to_parquet(path, index=False)

    frames = [pd.read_parquet(f) for f in sorted(PER_APP_DIR.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("reviewId")
    df.to_parquet(RAW_REVIEWS, index=False)
    print(f"\nSaved {len(df):,} reviews to {RAW_REVIEWS}")
    print(df.groupby("app").agg(reviews=("reviewId", "size"), first=("at", "min"), last=("at", "max")))


if __name__ == "__main__":
    main()
