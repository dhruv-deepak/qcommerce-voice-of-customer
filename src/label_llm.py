"""Use Claude as an expert annotator on a stratified sample of reviews.

The labels become training data for a cheap local classifier (train_classifier.py),
so we pay for the LLM on ~3,000 reviews instead of ~100,000.

Needs ANTHROPIC_API_KEY in the environment or in a .env file at the project root.

Usage:  python src/label_llm.py --per-app 750
        python src/label_llm.py --per-app 750 --model claude-haiku-4-5   # cheaper
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd
from dotenv import load_dotenv

from paths import CLEAN_REVIEWS, LLM_LABELS, ROOT
from taxonomy import COMPETITORS, SENTIMENTS, TOPICS, taxonomy_prompt

BATCH_SIZE = 20
SEED = 7
PRICES = {  # USD per million tokens (input, output)
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

SYSTEM = f"""You are an analyst labelling Google Play reviews of Indian quick-commerce apps
(Blinkit, Zepto, Swiggy Instamart, bigbasket) for a customer-experience study.

Reviews are often in Hinglish, Hindi, or informal English with typos. Read them as a
native Indian user would. For each review return:

- sentiment: the customer's overall feeling (positive, neutral, negative).
- complaints: every topic the customer complains about. Empty if none.
- praises: every topic the customer praises. Empty if none.
- churn_intent: true only if the customer says they have stopped, will stop, uninstalled,
  or are switching to another service.
- switch_to: the service they say they use or will use instead; "None" if not named.

Topics (use only these keys):
{taxonomy_prompt()}

Label what is written, not what might be implied. A review can have several topics.
Generic praise such as "good app" has no topic."""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "sentiment": {"type": "string", "enum": SENTIMENTS},
                    "complaints": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
                    "praises": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
                    "churn_intent": {"type": "boolean"},
                    "switch_to": {"type": "string", "enum": COMPETITORS},
                },
                "required": ["id", "sentiment", "complaints", "praises", "churn_intent", "switch_to"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def sample_reviews(per_app: int) -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_REVIEWS)
    df = df[df["substantive"]]
    return pd.concat(g.sample(min(per_app, len(g)), random_state=SEED)
                     for _, g in df.groupby("app")).reset_index(drop=True)


def already_labelled() -> set[str]:
    if not LLM_LABELS.exists():
        return set()
    with LLM_LABELS.open(encoding="utf-8") as f:
        return {json.loads(line)["review_id"] for line in f}


def request_kwargs(model: str) -> dict:
    """Opus/Sonnet get low-effort adaptive thinking and server-side refusal fallbacks;
    Haiku 4.5 supports neither."""
    if model.startswith("claude-haiku"):
        return {}
    return {"output_config_extra": {"effort": "low"}, "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default"}


def label_batch(client: anthropic.Anthropic, model: str, batch: pd.DataFrame) -> tuple[list[dict], object]:
    numbered = "\n".join(f"[{i}] ({r.stars} stars) {r.text}" for i, r in enumerate(batch.itertuples()))
    extra = request_kwargs(model)
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}, **extra.pop("output_config_extra", {})}
    response = client.beta.messages.create(
        model=model,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Label these {len(batch)} reviews:\n\n{numbered}"}],
        output_config=output_config,
        **extra,
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    rows = []
    for lab in json.loads(text)["labels"]:
        if 0 <= lab["id"] < len(batch):
            r = batch.iloc[lab["id"]]
            rows.append({"review_id": r["review_id"], "app": r["app"], **{k: v for k, v in lab.items() if k != "id"}})
    return rows, response.usage


def setup_problem(e: Exception) -> str | None:
    """Explain errors caused by missing or unusable credentials, which no retry will fix."""
    if isinstance(e, TypeError) and "authentication" in str(e).lower():
        return "no Anthropic API key found. Copy .env.example to .env, paste your key after ANTHROPIC_API_KEY=, and re-run."
    if isinstance(e, anthropic.AuthenticationError):
        return "the API key was rejected. Check the key in .env (no quotes or spaces) and re-run."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "this API key is not allowed to use this model. Check the key's permissions in the Anthropic Console."
    if isinstance(e, anthropic.BadRequestError) and "credit" in str(e).lower():
        return "the account has no credit left. Add credit in the Anthropic Console (Billing) and re-run."
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-app", type=int, default=750)
    p.add_argument("--model", default="claude-opus-5", choices=list(PRICES))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic(max_retries=5)

    sample = sample_reviews(args.per_app)
    done = already_labelled()
    todo = sample[~sample["review_id"].isin(done)]
    batches = [todo.iloc[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    print(f"{len(sample):,} sampled, {len(done):,} already labelled, {len(batches)} batches to send to {args.model}")

    LLM_LABELS.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    tokens_in = tokens_out = failed = 0
    with ThreadPoolExecutor(args.workers) as pool, LLM_LABELS.open("a", encoding="utf-8") as out:
        futures = [pool.submit(label_batch, client, args.model, b) for b in batches]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                rows, usage = fut.result()
            except Exception as e:
                problem = setup_problem(e)
                if problem:  # every remaining batch would fail the same way, so stop now
                    pool.shutdown(cancel_futures=True)
                    raise SystemExit(f"\nStopped: {problem}") from None
                if not isinstance(e, (anthropic.APIError, RuntimeError, json.JSONDecodeError, StopIteration)):
                    raise
                failed += 1
                print(f"\n  batch failed: {e}")
                continue
            with lock:
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            tokens_in += usage.input_tokens
            tokens_out += usage.output_tokens
            price_in, price_out = PRICES[args.model]
            cost = (tokens_in * price_in + tokens_out * price_out) / 1e6
            print(f"  {n}/{len(batches)} batches, ~${cost:.2f} so far", end="\r")

    print(f"\nDone. {len(already_labelled()):,} reviews labelled in {LLM_LABELS} ({failed} batches failed; "
          "re-run to retry them).")


if __name__ == "__main__":
    main()
