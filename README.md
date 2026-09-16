# SentientSpend AI — v5

Every one of the 17 advertised features is now backed by real logic. Nothing is simulated, mocked, or hardcoded.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create an account, tick "Load 12 months of realistic demo data", and everything below has data to work with immediately.

## What each feature actually does now

| Feature | Implementation |
|---|---|
| Chat with your data | Parses the question for period, category and intent, then queries the real DataFrame. Handles totals, categories, biggest expense, top merchants, savings, forecast, runway, subscriptions, anomalies. Falls back to a suggestion list, never a canned answer. |
| Receipt OCR | pytesseract on a grayscale, upscaled image. Finds the grand total (prefers lines containing "total"), merchant, and date, auto-tags the category, then shows an editable form before saving. |
| Anomaly detection | Isolation Forest on amount, day-of-week, month and category-relative z-score. Known recurring bills are excluded first, then flags are filtered for materiality, so rent doesn't get flagged every month. |
| Smart auto-tagging | ~120 merchant keywords with word-boundary matching, plus fuzzy matching via thefuzz. Returns a confidence score, and suggests a category live as you type a description. |
| Roast mode | Built from your actual numbers: savings rate, worst category, largest single transaction as a % of budget, weekend share, annual subscription bleed. |
| Financial runway | Net savings ÷ daily burn rate over the last 90 days. |
| Health score | Transparent 0–100: savings rate (40) + budget adherence (25) + spending consistency (15) + runway (10) + anomaly-free (10), with a per-component breakdown bar chart in the Budget tab. |
| Spending heatmap | GitHub-style calendar of real daily expense totals for the last 26 weeks, plus a category × month heatmap. |
| Subscription detector | Requires ≥3 charges across ≥3 months, stable amounts (CV ≤ 0.25) and a consistent billing gap or billing day. Reports cadence, next expected date and annual cost. |
| Family accounts (RBAC) | Real `family_accounts` table. Admins add members by username and see per-member spend vs budget; members can only see themselves. Enforced by `can_view_member()`. |
| Multi-bank CSV/Excel ETL | Detects the delimiter, skips statement preamble to find the real header row, scores every column to map date/description/debit/credit/amount/type, detects day-first vs month-first dates, repairs numbers split by unquoted thousands separators, auto-tags categories, and skips duplicates. Tested against HDFC, Chase and SBI layouts. |
| Multi-currency + crypto | Live FX from open.er-api.com cached for an hour, with a static fallback; all figures re-render in the chosen currency. Crypto via yfinance, then CoinGecko. |
| PDF reports | reportlab document with KPI band, monthly table, category breakdown, subscriptions and your ML persona. |
| Gamified challenges | Three types (save / spend-less / no-spend). Progress is computed from transactions inside the challenge window, auto-completes when the window ends, and unlocks 8 badges. |
| Safe-to-spend | (budget − this month's spend) ÷ days remaining in the month. |
| Behavioural nudges | Intercepts an expense ≥25% of budget or one that busts the month, explains why, and waits for confirm/skip. Nothing is written until you confirm. |
| Voice entry | Web Speech API mic widget plus a parser that handles "spent 450 on uber yesterday", "paid 1.2k for netflix", "received 50000 salary" — amounts with k/lakh, relative dates, income vs expense, and category. Preview before saving. |

## Other fixes

- Passwords are PBKDF2-SHA256 with a per-user salt; existing SHA-256 hashes are upgraded on next login.
- Fixed the crash when loading challenges (7 columns unpacked into 6).
- Fixed the `re` module being shadowed by the signup email variable.
- Added schema migrations, so an existing `sentientspend.db` keeps working.
- Transactions can now be edited, not just deleted, and are paginated.
- KMeans clusters are named from their centroids instead of a fixed 0/1/2 map, and the fake "94.2% confidence" was replaced with real cluster share.

## Optional extras

| Env var | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | Adds an LLM toggle to the chat (only aggregates are sent, never raw rows) |
| `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `SMTP_PORT` | Enables the budget alert email |
| `SENTIENTSPEND_DB` | Custom database path |

Without Tesseract, the receipt scanner still accepts the image and lets you type the amount — it tells you what's missing instead of faking a result.