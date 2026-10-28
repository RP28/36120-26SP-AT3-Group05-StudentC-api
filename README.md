# Student C — Trend Category API (AT3 Group 05)

FastAPI app that supports BTC, ETH, XRP, SOL, DOGE, XMR, XLM, and TRX. Uses the experiment notebook's serialized preprocessing and seven-day trend prediction function, plus its saved model metadata.

## Setup

Requires Python 3.12.13. In the project directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps mlweave==0.2.3
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Only MLWeave is installed from TestPyPI. The first command skips dependency installation. The second resolves MLWeave's dependencies and all other packages against PyPI, as specified in `requirements.txt`. Alternatively, run `uv sync`: `pyproject.toml` assigns only MLWeave to an explicit TestPyPI index.

Use `http://127.0.0.1:8000/docs` to test. Create or edit your local `.env` with `APP_PROFILE=local`.

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

`/predict` passes the complete market history and separate Student A, B and D dictionaries to the saved function. It returns the notebook fields `coin`, `date`, `forecast_horizon_days`, `predicted_trend` and all five class `probabilities`. The existing API fields `prediction`, `confidence`, `forecast_horizon` and `prediction_date` are retained. `date` is the observation date. `prediction_date` is seven days later. `confidence` is the probability of the predicted class.

`/model-metadata` serves the saved JSON's `model` schema (23 predictors), gain-based `feature_importance`, `permutation_importance` and `created_at`, together with the forecast horizon, class labels and development flag.

## App profiles

| APP_PROFILE | Market OHLC and cap/volume | Other students /features | Model |
|---|---|---|---|
| `local` | both mocked | mocked | trained artifact |
| `dev` | real Kraken→CoinDesk and CMC→CoinGecko | mocked | trained artifact |
| `prod` | real with fallbacks | real parallel student calls | trained artifact |

No `MARKET_BACKEND`, `FEATURE_BACKEND`, or `MODEL_BACKEND` switches are needed: `config.py` derives them from `APP_PROFILE`.

Local and dev predictions include `development_only=true` because they use synthetic inputs. All profiles load the actual saved model.

## Render environment variables

In Render Web Service → Environment, add `APP_PROFILE=dev` initially. For production set `APP_PROFILE=prod` and `STUDENT_A_URL`, `STUDENT_B_URL`, `STUDENT_D_URL`, and deploy both files in `models/`. Set `CMC_API_KEY`, `COINGECKO_API_KEY`, and `COINDESK_API_KEY` based on your provider access. URLs use the exact names listed in `config.py` and can be overridden with `KRAKEN_URL`, `COINDESK_URL`, `CMC_URL`, `COINGECKO_BASE_URL`. Also set `GITHUB_URL` and optionally `HISTORY_DAYS` (35–365 with default 90) and `REQUEST_TIMEOUT_SECONDS`.


Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Use Python 3.12.13. Build command: `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps mlweave==0.2.3 && python -m pip install -r requirements.txt`.

## Model artifact contract

Every profile loads the **trusted** callable at `MODEL_PATH` (default `models/predict_seven_day_trend.pkl`) with cloudpickle, and JSON at `MODEL_METADATA_PATH` (default `models/model_metadata.json`). Both are loaded once per worker on startup.

The function accepts `market_df` with exactly `date`, `open`, `high`, `low`, `close`, `volume`, `marketCap`, plus `student_a`, `student_b`, `student_d` and `coin`. It requires at least 35 consecutive daily observations. Its fitted pipeline computes Student C's indicators and the additional MACD/Bollinger indicators, reconstructs TrendScore, preserves the categorical coin predictor and selects the trained model's feature order.

External feature requirements match notebook section L:

| Student | Indicators |
|---|---|
| A | `DailyReturn`, `Return5`, `Return20` |
| B | `MA5`, `MA20`, `TrendStrength` |
| D | `SMA20Volume`, `RelativeVolume` |

Student services receive `POST /features` with `{token, forecast_horizon}`. Responses may contain a flat feature dictionary or indicators nested under `features`. If supplied, the response token and date must match the requested coin and latest completed UTC day. Extra indicators are ignored. Missing values, nulls or service failures become NaN in the saved pipeline. A/B failures also leave TrendScore missing, matching the notebook's native XGBoost missing-value evaluation.

## Reliability and data semantics

OHLC and 24-hour USD market-volume/capitalisation metrics are requested concurrently. Only failures or insufficient alignment trigger their corresponding secondary providers. Sources are normalized by UTC date and the service requires exactly `HISTORY_DAYS` consecutive completed UTC days ending yesterday.

A remote feature outage results in missing model values (`NaN`), handled by this saved XGBoost model. HTTP timeouts are controlled by `REQUEST_TIMEOUT_SECONDS`.

`/predict` first checks an in-memory prediction cache. Its key includes the app profile, coin, forecast horizon, history length, last completed UTC date and loaded model version. A hit returns the complete saved response immediately, skipping market-data requests, student feature calls and inference. Entries expire 30 minutes after calculation. Hits do not extend their expiry. Per-key locks prevent concurrent identical requests from duplicating a successful calculation while allowing different coins to run concurrently.

Expired and older-date prediction entries are pruned when predictions are requested. Failed calculations and predictions with missing student indicators are not cached, allowing subsequent requests to retry those services. All the caches are local to each worker. Restarting clears them, and a new model load starts with an empty prediction cache.

## Integration checks

1. With `APP_PROFILE=local`, test `/health`, `/features` and `/predict` from `/docs`.
2. Switch to `dev`, set available API keys, restart and test all 8 tokens. A market-provider failure should attempt fallback. A 503 means historical coverage is unavailable.
3. For teammates' URLs are available, set `prod`, restart, test real `/features`, `/predict`, and degraded provider conditions.
4. Check `http://127.0.0.1:8000/openapi.json` for the published request schema.
