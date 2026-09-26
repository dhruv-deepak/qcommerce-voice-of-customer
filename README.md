# What India's quick-commerce customers complain about

A voice-of-customer study of **128,000+ Google Play reviews** of Blinkit, Zepto, Swiggy Instamart and bigbasket
(27 Jul – 25 Sep 2026). It answers three business questions:

1. **What** do customers complain about, app by app?
2. **Which** complaints hurt ratings and push customers to switch apps the most?
3. **What** should each app fix first?

> Work in progress. Findings, charts and the executive report will be added here once the analysis run is complete.

## Approach

| Step | What happens | Code |
|---|---|---|
| Collect | Public Play Store reviews for 60 days; reviewer names and photos dropped at collection | `src/scrape.py` |
| Clean | Normalise text; flag reviews with enough words to classify | `src/clean.py` |
| Label | An LLM (Claude) labels a stratified sample against a 12-topic taxonomy built around the customer journey, handling Hinglish | `src/label_llm.py`, `src/taxonomy.py` |
| Validate | Hand-label 150 reviews blind and measure agreement with the LLM | `src/gold.py` |
| Scale | Train a TF-IDF + logistic-regression classifier on the LLM labels and label every review | `src/train_classifier.py` |
| Analyse | Complaint rates, rating impact (regression), churn triggers, weekly trends, reply rates | `src/analyze.py` |
| Present | Executive report + interactive Streamlit dashboard | `src/report.py`, `app/dashboard.py` |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add your Anthropic API key
python run_pipeline.py
streamlit run app/dashboard.py
```

Raw review text is not committed to this repository; only code and aggregated results are published.
