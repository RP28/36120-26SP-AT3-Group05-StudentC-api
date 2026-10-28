# Student C — Trend Category API (AT3 Group 05)

Group 05 · Student C: Ratnadeep Patra · Student ID: 26294153.

Repository: https://github.com/RP28/36120-26SP-AT3-Group05-StudentC-api

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

The API pins the brief's versions for the runtime libraries it uses, including Python 3.12.13, pandas 2.2.2, scikit-learn 1.6.1, FastAPI 0.139.0, Uvicorn 0.50.0, XGBoost 3.3.0, joblib 1.5.3 and matplotlib 3.11.0. Notebook and frontend dependencies are maintained in their respective repositories.

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

`/predict` passes the complete market history and separate Student A, B and D dictionaries to the saved function. The public response contains exactly these four fields (illustrative values):

```json
{
  "prediction": "Sideways",
  "confidence": 0.84,
  "forecast_horizon": 7,
  "prediction_date": "2026-11-03"
}
```

`prediction_date` is seven days after the latest completed UTC observation used by the model. `confidence` is that predicted class's probability, between 0 and 1. Student C's labels are Strong Downtrend, Downtrend, Sideways, Uptrend and Strong Uptrend. Pydantic response models enforce and document the prediction and feature response structures in `/docs` and `/openapi.json`. Successful predictions return HTTP 200.

`/model-metadata` exposes the required top-level `target`, `algorithm`, `forecast_horizon`, `features`, `feature_importance` and `classification_labels`, plus a target description. Importance values are normalized gain proportions (0–1) keyed by the exact feature names. The metadata comes from the saved JSON. Its original gain details, permutation importance, creation time and custom estimator class are also available as `feature_importance_details`, `permutation_importance`, `created_at` and `model_class`.

Invalid requests return HTTP 422 with a meaningful `error`. Unsupported coins also return `supported_tokens`, matching the brief's example. Other request errors include a `details` list identifying invalid fields. Tokens must match one of the eight uppercase tickers; `forecast_horizon` must be the integer 7. Unavailable market history or failed inference returns HTTP 503. A student feature-service outage alone does not prevent prediction: the model uses missing values for unavailable indicators.

Every response includes an `X-Correlation-ID` header, including cached responses, validation errors and server errors. The API accepts an incoming UUID in that header and normalizes it. An absent or invalid value generates a new UUID. IDs are kept outside the JSON body and prediction cache. A cached prediction therefore receives a fresh ID unless the caller supplies one. Application logs include the ID, and outgoing market-data and student-service requests carry the same header. Concurrent requests keep separate request contexts. Unexpected request failures are logged with their ID and return HTTP 500 with `{"detail":"Internal server error"}`.

Streamlit can read the ID independently of the assignment's prediction response:

```python
correlation_id = response.headers.get("X-Correlation-ID")
prediction = response.json()
```

## App profiles

| APP_PROFILE | Market OHLC and cap/volume | Other students /features | Model |
|---|---|---|---|
| `local` | both mocked | mocked | trained artifact |
| `dev` | real Kraken→CoinDesk and CMC→CoinGecko | mocked | trained artifact |
| `prod` | real with fallbacks | real parallel student calls | trained artifact |

No `MARKET_BACKEND`, `FEATURE_BACKEND`, or `MODEL_BACKEND` switches are needed: `config.py` derives them from `APP_PROFILE`.

All profiles load the actual saved model. `/health` identifies the active profile and each backend. Local and dev use synthetic inputs and are development profiles. The assessed Render deployment must use `APP_PROFILE=prod`, with real market data and HTTP calls to the other students' feature services.

## Render environment variables

In Render Web Service → Environment, set `APP_PROFILE=prod`, `STUDENT_A_URL`, `STUDENT_B_URL`, `STUDENT_D_URL`, and deploy both files in `models/`. Set `CMC_API_KEY`, `COINGECKO_API_KEY`, and `COINDESK_API_KEY` based on your provider access. URLs use the exact names listed in `config.py` and can be overridden with `KRAKEN_URL`, `COINDESK_URL`, `CMC_URL`, `COINGECKO_BASE_URL`. `GITHUB_URL` defaults to this repository's URL and can be overridden. Optionally set `HISTORY_DAYS` (35–365 with default 90) and `REQUEST_TIMEOUT_SECONDS`.


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
3. With teammates' URLs configured, set `prod`, restart, and verify `/health` reports both market and feature backends as `http`. Test real `/features`, `/predict`, and unavailable student feature services.
4. Check `http://127.0.0.1:8000/openapi.json` for the published request schema.
