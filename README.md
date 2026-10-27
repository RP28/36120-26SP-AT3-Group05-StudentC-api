# Student C — Trend Category API (AT3 Group 05)

FastAPI app with **four application Python files**, following the assignment layout. Supports BTC, ETH, XRP, SOL, DOGE, XMR, XLM, and TRX. Do not interpret placeholder output as a live cryptocurrency forecast.

## Setup

Requires Python 3.12.13 (per brief). In the project directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Use `http://127.0.0.1:8000/docs` to test. The bundled `.env` sets `APP_PROFILE=local`; it has **no credentials**, is excluded from Git, and is meant to be edited locally. When moving to Git, create your own `.env` (or retain this local copy); do not commit it.

Both `POST /features` and `POST /predict` accept:

```json
{"token":"BTC","forecast_horizon":7}
```

`/features` returns the requested token, the latest completed UTC market-data date (`YYYY-MM-DD`), and Student C's indicators nested under `features`. Example response (illustrative values):

```json
{
  "token": "BTC",
  "date": "2026-10-26",
  "features": {
    "Volatility20": 0.002,
    "MaxPrice20": 75000.0,
    "MaxDrawdown": 0.015,
    "GR": 0.004
  }
}
```

The five mandatory routes are `/`, `/health`, `/features`, `/predict`, and `/model-metadata`.

## Spring-style profiles (only APP_PROFILE is needed)

| APP_PROFILE | Market OHLC and cap/volume | Other students /features | Model |
|---|---|---|---|
| `local` | both mocked | mocked | placeholder |
| `dev` | real Kraken→CoinDesk and CMC→CoinGecko | mocked | placeholder |
| `prod` | real with fallbacks | real parallel student calls | trained artifact |

No `MARKET_BACKEND`, `FEATURE_BACKEND`, or `MODEL_BACKEND` switches are needed: `config.py` derives them from `APP_PROFILE`. No secrets are shipped.

## Render environment variables

In Render Web Service → Environment, add `APP_PROFILE=dev` initially. For production set `APP_PROFILE=prod` and `STUDENT_A_URL`, `STUDENT_B_URL`, `STUDENT_D_URL`, and deploy the real `models/model.pkl`. Set `CMC_API_KEY`, `COINGECKO_API_KEY`, and `COINDESK_API_KEY` based on your provider access (CoinGecko may work without a key subject to its rules). URLs use the exact names listed in `config.py` and can be overridden with `KRAKEN_URL`, `COINDESK_URL`, `CMC_URL`, `COINGECKO_BASE_URL`. Also set `GITHUB_URL` and optionally `HISTORY_DAYS` and `REQUEST_TIMEOUT_SECONDS`.

**Important:** CMC historical quotes can require paid access; if missing or unavailable the code tries CoinGecko. A successful fallback still must provide `HISTORY_DAYS` consecutive UTC dates. If neither combination can, the API returns HTTP 503, **not synthetic data**. CoinGecko's granularity and quota depend on plan and date range; production testing is necessary. `dev` should only be used when real market APIs can provide enough aligned dates.

Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. If using Render's Python runtime, choose Python 3.12 where available. Install command: `pip install -r requirements.txt`.

## Model artifact contract

`local` and `dev` use an explicit placeholder. `prod` loads a **trusted** pickle at `MODEL_PATH` (default `models/model.pkl`) and expects a dict with `model`, `feature_columns`, `class_labels`, and `forecast_horizon=7`, optionally `preprocessor`. The estimator needs `predict` and `predict_proba`. Build and save this artifact after experimentation with compatible library versions. Each worker loads its own artifact on startup.

## Reliability and data semantics

OHLC and 24-hour USD market-volume/capitalisation metrics are requested concurrently; only failures or insufficient alignment trigger their corresponding secondary providers. Sources are normalized by UTC date and the service requires exactly `HISTORY_DAYS` consecutive completed UTC days ending yesterday. CoinGecko rolling 24-hour volume is not exactly UTC-day traded volume. Cross-provider market caps and OHLC closes may not share an exact snapshot time.

A remote feature outage results in missing model values (`NaN`) but only a model trained to tolerate such missingness can safely predict. HTTP timeouts are controlled by `REQUEST_TIMEOUT_SECONDS`. Do not commit API keys.

## Integration checks

1. With `APP_PROFILE=local`, test `/health`, `/features` and `/predict` from `/docs`.
2. Switch to `dev`, set available API keys, restart and test all 8 tokens. A market-provider failure should attempt fallback; a 503 means historical coverage is unavailable.
3. When trained model and teammates' URLs are available, set `prod`, restart, test real `/features`, `/predict`, and degraded provider conditions.
4. Check `http://127.0.0.1:8000/openapi.json` for the published request schema.
