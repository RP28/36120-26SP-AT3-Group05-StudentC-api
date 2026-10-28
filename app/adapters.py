"""Market-data adapters and normalizers for Student C.

Market data has two independent branches, fetched concurrently:
  OHLC: Kraken -> CoinDesk
  market-wide USD volume / market cap: CoinGecko -> CoinMarketCap

Fallbacks are lazy: a secondary provider is contacted only when its branch is
unusable or the two selected providers cannot supply the required aligned days.
All providers normalize to UTC-dated dictionaries. Mock mode needs no API keys.
"""

import asyncio
import logging
import math
from datetime import date, datetime, timedelta, timezone
import httpx
import copy
import time
from .config import Settings, STUDENT_FEATURES

logger = logging.getLogger(__name__)

COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "XRP": "ripple", "SOL": "solana", "DOGE": "dogecoin", "XMR": "monero", "XLM": "stellar", "TRX": "tron"}
KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "XRP": "XRPUSD", "SOL": "SOLUSD", "DOGE": "DOGEUSD", "XMR": "XMRUSD", "XLM": "XLMUSD", "TRX": "TRXUSD"}
CMC_IDS = {"BTC": 1, "ETH": 1027, "XRP": 52, "SOL": 5426, "DOGE": 74, "XMR": 328, "XLM": 512, "TRX": 1958}

MARKET_CACHE_TTL_SECONDS = 1800  # 30 minutes
_market_cache = {}
_market_cache_locks = {}

# Small fetch buffer for APIs that include today's unclosed day or have gaps.
# It does not guarantee that every free API plan permits this lookback.
HISTORY_BUFFER_DAYS = 3
PROVIDER_ERRORS = (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError)

class MarketDataError(ValueError):
    """All usable provider combinations failed to supply aligned market data."""

def mock_ohlc(token: str, days: int) -> list[dict]:
    """Deterministic, synthetic closed candles (NOT actual market prices)."""
    base = {"BTC": 65000, "ETH": 3000, "XRP": 0.6, "SOL": 150, "DOGE": 0.15, "XMR": 160, "XLM": 0.12, "TRX": 0.13}[token]
    today = datetime.now(timezone.utc).date()
    rows = []
    for i in range(days):
        d = today - timedelta(days=days - i)
        close = base * (1 + 0.0015 * i + 0.012 * math.sin(i / 5))
        opening = close * (1 - 0.006 * math.sin(i / 3))
        rows.append({
            "date": d.isoformat(), "open": opening,
            "high": max(opening, close) * 1.01,
            "low": min(opening, close) * 0.99, "close": close
        })
    return rows

def mock_metrics(token: str, days: int) -> list[dict]:
    """Synthetic daily USD total volume and market cap."""
    today = datetime.now(timezone.utc).date()
    mult = list(COINGECKO_IDS).index(token) + 1
    return [
        {
            "date": (today - timedelta(days=days - i)).isoformat(),
            "volume": 1_000_000_000 / mult * (1 + 0.015 * i),
            "marketCap": 30_000_000_000 / mult * (1 + 0.002 * i)
        }
        for i in range(days)
    ]

def _date(timestamp: int | float) -> str:
    return datetime.fromtimestamp(float(timestamp), timezone.utc).date().isoformat()

def kraken_to_ohlc(payload: dict) -> list[dict]:
    if payload.get("error"):
        raise ValueError(f"Kraken API errors: {payload['error']}")
    result = payload.get("result", {})
    pair = next((value for key, value in result.items() if key != "last" and isinstance(value, list)), None)
    if pair is None:
        raise ValueError("Kraken OHLC array absent")
    # Kraken documents the final candle as uncommitted.
    return [
        {
            "date": _date(r[0]), "open": float(r[1]),
            "high": float(r[2]), "low": float(r[3]), "close": float(r[4])
        }
        for r in pair[:-1]
    ]

def coindesk_to_ohlc(payload: dict) -> list[dict]:
    if payload.get("Err") or not isinstance(payload.get("Data"), list):
        raise ValueError("Unexpected CoinDesk OHLC response")
    return [
        {
            "date": _date(r["TIMESTAMP"]), "open": float(r["OPEN"]),
            "high": float(r["HIGH"]), "low": float(r["LOW"]),
            "close": float(r["CLOSE"])
        }
        for r in payload["Data"] if r.get("OPEN") is not None
    ]

def coingecko_to_metrics(payload: dict) -> list[dict]:
    """Choose latest sample per UTC day, separately for cap and 24h volume.

    CoinGecko `total_volumes` is market-wide rolling volume, not exchange volume.
    It is not necessarily the volume traded during a UTC calendar day.
    """
    caps = {}
    vols = {}
    for timestamp, value in payload.get("market_caps", []):
        if value is not None:
            day = _date(timestamp / 1000)
            if day not in caps or timestamp > caps[day][0]:
                caps[day] = (timestamp, float(value))
    for timestamp, value in payload.get("total_volumes", []):
        if value is not None:
            day = _date(timestamp / 1000)
            if day not in vols or timestamp > vols[day][0]:
                vols[day] = (timestamp, float(value))
    return [{"date": day, "volume": vols[day][1], "marketCap": caps[day][1]} for day in sorted(caps.keys() & vols.keys())]

def cmc_to_metrics(payload: dict, token: str) -> list[dict]:
    """Normalize CMC historical quotes; verify actual account response shape."""
    data = payload.get("data", {})
    entry = data.get(str(CMC_IDS[token]), {}) if isinstance(data, dict) else {}
    if not entry and isinstance(data, dict):
        entry = data
    quotes = entry.get("quotes", []) if isinstance(entry, dict) else []
    if not isinstance(quotes, list):
        raise ValueError("Unexpected CoinMarketCap quote structure")
    rows = []
    for quote in quotes:
        usd = quote.get("quote", {}).get("USD", {})
        if not usd:
            continue
        day = str(quote.get("timestamp") or usd.get("timestamp") or "")[:10]
        if not day:
            continue
        rows.append({
            "date": day,
            "volume": float(usd["volume_24h"]),
            "marketCap": float(usd["market_cap"])
        })
    if not rows:
        raise ValueError("No usable CoinMarketCap historical metrics returned")
    return rows

def _validate_rows(rows: list[dict], fields: tuple[str, ...], provider: str) -> list[dict]:
    """Validate per-provider observations before accepting a response."""
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{provider} returned no records")
    validated = []
    for row in rows:
        day = date.fromisoformat(row["date"]).isoformat()
        record = {"date": day}
        for field in fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{provider} returned non-finite {field} for {day}")
            if field in ("open", "high", "low", "close", "marketCap") and value <= 0:
                raise ValueError(f"{provider} returned invalid {field} for {day}")
            if field == "volume" and value < 0:
                raise ValueError(f"{provider} returned negative volume for {day}")
            record[field] = value
        if "high" in fields and (record["high"] < max(record["open"], record["close"]) or record["low"] > min(record["open"], record["close"])):
            raise ValueError(f"{provider} returned inconsistent OHLC for {day}")
        validated.append(record)
    if len({item["date"] for item in validated}) != len(validated):
        # Some providers legitimately emit several snapshots in one UTC day.
        # Keep the last observation explicitly rather than duplicating dates.
        by_date = {item["date"]: item for item in validated}
        return [by_date[day] for day in sorted(by_date)]
    return validated

def _merge_for_history(ohlc_rows: list[dict], metric_rows: list[dict], days: int) -> list[dict]:
    """Require all of the last `days` *completed* UTC dates, without gaps."""
    today = datetime.now(timezone.utc).date()
    required = [(today - timedelta(days=offset)).isoformat() for offset in range(days, 0, -1)]
    ohlc_by_date = {row["date"]: row for row in ohlc_rows}
    metrics_by_date = {row["date"]: row for row in metric_rows}
    available = ohlc_by_date.keys() & metrics_by_date.keys()
    missing = [day for day in required if day not in available]
    if missing:
        raise MarketDataError(f"Only {days - len(missing)}/{days} required completed UTC days align (missing {len(missing)}, "
                              f"first missing: {missing[0]}); provider coverage may be delayed or incomplete")
    return [{**ohlc_by_date[day], **metrics_by_date[day]} for day in required]


async def logged_get(client: httpx.AsyncClient, provider: str, url: str, **kwargs) -> httpx.Response:
    start = time.perf_counter()
    logger.info("Downstream request started: provider=%s", provider)
    try:
        response = await client.get(url, **kwargs)
        response.raise_for_status()
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("Downstream request succeeded: provider=%s, status=%d, duration_ms=%.2f", provider, response.status_code, duration_ms)
        return response
    except httpx.HTTPError as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.warning("Downstream request failed: provider=%s, duration_ms=%.2f, error=%s", provider, duration_ms, type(exc).__name__)
        raise

async def _fetch_market_data_uncached(token: str, cfg: Settings, client: httpx.AsyncClient) -> list[dict]:
    """Concurrent OHLC/metrics branches with lazy primary -> secondary fallback.

    `history_days` means exactly that many consecutive, completed UTC daily
    observations, ending yesterday. No partial day, sparse history or fake values.
    """
    token = token.upper()
    if token not in KRAKEN_PAIRS:
        raise ValueError(f"Unsupported cryptocurrency: {token}")
    days = cfg.history_days
    if days < 1:
        raise ValueError("history_days must be positive")
    if cfg.market_backend == "mock":
        return _merge_for_history(mock_ohlc(token, days), mock_metrics(token, days), days)
    # Provider functions retain the same normalized shape across APIs.
    async def fetch_kraken():
        response = await logged_get(client, "kraken", cfg.kraken_url, params={"pair": KRAKEN_PAIRS[token], "interval": 1440})
        rows = kraken_to_ohlc(response.json())
        logger.info("OHLC normalized: provider=kraken, token=%s, rows=%d", token, len(rows))
        return rows
    async def fetch_coindesk():
        if not cfg.coindesk_api_key:
            raise ValueError("CoinDesk API key is not configured")
        response = await logged_get(client, "coindesk", cfg.coindesk_url,
            headers={"authorization": f"Apikey {cfg.coindesk_api_key}"} if cfg.coindesk_api_key else {},
            params={"market": "kraken", "instrument": f"{token}-USD", "limit": days + HISTORY_BUFFER_DAYS, "aggregate": 1, "response_format": "JSON"})
        rows = coindesk_to_ohlc(response.json())
        logger.info("OHLC normalized: provider=coindesk, token=%s, rows=%d", token, len(rows))
        return rows
    async def fetch_cmc():
        if not cfg.cmc_api_key:
            raise ValueError("CoinMarketCap API key is not configured")
        response = await logged_get(client, "cmc", cfg.cmc_url,
            headers={"X-CMC_PRO_API_KEY": cfg.cmc_api_key},
            params={"id": CMC_IDS[token], "count": days + HISTORY_BUFFER_DAYS, "interval": "daily", "convert": "USD"})
        rows = cmc_to_metrics(response.json(), token)
        logger.info("Volume/Market cap normalized: provider=cmc, token=%s, rows=%d", token, len(rows))
        return rows
    async def fetch_coingecko():
        if not cfg.coingecko_api_key:
            raise ValueError("CoinGecko API key is not configured")
        response = await logged_get(client, "coingecko", f"{cfg.coingecko_base_url.rstrip('/')}/coins/{COINGECKO_IDS[token]}/market_chart",
            headers={"x-cg-demo-api-key": cfg.coingecko_api_key} if cfg.coingecko_api_key else {},
            params={"vs_currency": "usd", "days": days + HISTORY_BUFFER_DAYS})
        rows = coingecko_to_metrics(response.json())
        logger.info("Volume/Market cap normalized: provider=coingecko, token=%s, rows=%d", token, len(rows))
        return rows
    # Fixed provider order in every environment: primary then lazy fallback.
    ohlc_providers = {"kraken": fetch_kraken, "coindesk": fetch_coindesk}
    metrics_providers = {"coingecko": fetch_coingecko, "coinmarketcap": fetch_cmc}
    ohlc_order = ["kraken", "coindesk"]
    metrics_order = ["coingecko", "coinmarketcap"]
    ohlc_results: dict[str, list[dict]] = {}
    metrics_results: dict[str, list[dict]] = {}
    failures: list[str] = []
    async def load(name: str, providers: dict, fields: tuple[str, ...]):
        try:
            records = _validate_rows(await providers[name](), fields, name)
            return records
        except PROVIDER_ERRORS as exc:
            logger.warning("Market provider %s failed: %s", name, exc)
            failures.append(f"{name}: {exc}")
            return None
    async def branch(order, providers, fields, result_dict):
        for name in order:
            logger.info("Trying market provider: provider=%s, attempt=%d", name, order.index(name) + 1)
            rows = await load(name, providers, fields)
            if rows is not None:
                result_dict[name] = rows
                # A provider must contain enough records to satisfy history_days.
                # A full inner-join/consecutive check follows below.
                if len({r["date"] for r in rows if r["date"] < datetime.now(timezone.utc).date().isoformat()}) >= days:
                    return
                logger.warning("Market provider %s supplied fewer than %d completed days", name, days)
                failures.append(f"{name}: fewer than {days} completed dates")
        return
    # Fetch primary branches in parallel; fallback sequentially inside each branch.
    await asyncio.gather(
        branch(ohlc_order, ohlc_providers, ("open", "high", "low", "close"), ohlc_results),
        branch(metrics_order, metrics_providers, ("volume", "marketCap"), metrics_results)
    )
    # Try any already fetched provider combinations, retaining primary preference.
    def try_merge():
        for o_name in ohlc_order:
            if o_name not in ohlc_results:
                continue
            for m_name in metrics_order:
                if m_name not in metrics_results:
                    continue
                try:
                    result = _merge_for_history(ohlc_results[o_name], metrics_results[m_name], days)
                    logger.info(
                        "Market data aligned: token=%s, ohlc_provider=%s, metrics_provider=%s, aligned_days=%d, first_date=%s, last_date=%s",
                        token, o_name, m_name, len(result), result[0]["date"], result[-1]["date"])
                    return result
                except MarketDataError as exc:
                    failures.append(f"{o_name}+{m_name}: {exc}")
        return None
    aligned = try_merge()
    if aligned is not None:
        return aligned
    # Branches may each have `days` records but on different UTC dates.
    # Only now call alternatives that weren't needed/queried earlier.
    extra_calls = []
    extra_names = []
    for name in ohlc_order:
        if name not in ohlc_results:
            extra_calls.append(load(name, ohlc_providers, ("open", "high", "low", "close")))
            extra_names.append(("ohlc", name))
    for name in metrics_order:
        if name not in metrics_results:
            extra_calls.append(load(name, metrics_providers, ("volume", "marketCap")))
            extra_names.append(("metrics", name))
    if extra_calls:
        extras = await asyncio.gather(*extra_calls)
        for (kind, name), rows in zip(extra_names, extras):
            if rows is not None:
                (ohlc_results if kind == "ohlc" else metrics_results)[name] = rows
        aligned = try_merge()
        if aligned is not None:
            return aligned
    raise MarketDataError(
        f"Could not obtain {days} consecutive aligned completed UTC days for {token}. "
        + " | ".join(failures[-8:])
    )


async def fetch_market_data(token: str, cfg: Settings, client: httpx.AsyncClient) -> list[dict]:
    last_completed_day = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    cache_key = (cfg.app_profile, token, cfg.history_days, last_completed_day)
    lock = _market_cache_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        cached = _market_cache.get(cache_key)
        if cached and now < cached["expires_at"]:
            logger.info("Market cache HIT: token=%s, days=%d", token, cfg.history_days)
            return copy.deepcopy(cached["data"])
        logger.info("Market cache MISS: token=%s, days=%d", token, cfg.history_days)
        # Existing primary/fallback retrieval logic
        rows = await _fetch_market_data_uncached(token, cfg, client)
        _market_cache[cache_key] = {
            "data": copy.deepcopy(rows),
            "expires_at": time.monotonic() + MARKET_CACHE_TTL_SECONDS
        }
        # Discard outdated date entries
        for key in list(_market_cache):
            if key[3] != last_completed_day:
                del _market_cache[key]
                _market_cache_locks.pop(key, None)
        logger.info("Market cache UPDATED: token=%s, rows=%d", token, len(rows))
        return copy.deepcopy(rows)

async def fetch_remote_features(token: str, horizon: int, cfg: Settings, client: httpx.AsyncClient) -> dict:
    """Fetch the notebook's selected indicators, grouped by student.

    Accept a flat feature object or the shared {token, date, features} envelope.
    Missing indicators stay absent so the saved callable supplies NaN.
    """
    if cfg.feature_backend == "mock":
        prices = [row["close"] for row in mock_ohlc(token, cfg.history_days)]
        volumes = [row["volume"] for row in mock_metrics(token, cfg.history_days)]
        ma5, ma20 = sum(prices[-5:])/5, sum(prices[-20:])/20
        volume_ma20 = sum(volumes[-20:])/20
        return {
            "A": {"DailyReturn": prices[-1]/prices[-2]-1,
                  "Return5": prices[-1]/prices[-6]-1, "Return20": prices[-1]/prices[-21]-1},
            "B": {"MA5": ma5, "MA20": ma20, "TrendStrength": ma5/ma20-1},
            "D": {"SMA20Volume": volume_ma20, "RelativeVolume": volumes[-1]/volume_ma20},
        }
    endpoints = {"A": cfg.student_a_url, "B": cfg.student_b_url, "D": cfg.student_d_url}
    expected = STUDENT_FEATURES
    async def call(name, url):
        try:
            if not url:
                raise ValueError(f"Student {name} URL is not configured")
            logger.info("Fetching remote features: student=%s, token=%s", name, token)
            response = await client.post(f"{url.rstrip('/')}/features", json={"token": token, "forecast_horizon": horizon})
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Expected root-level JSON object")
            if body.get("token", token) != token:
                raise ValueError("Feature response token does not match request")
            expected_date = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
            if body.get("date", expected_date) != expected_date:
                raise ValueError("Feature response does not describe the latest completed UTC day")
            source = body.get("features", body)
            if not isinstance(source, dict):
                raise ValueError("Expected a feature dictionary")
            values = {}
            for feature in expected[name]:
                value = source.get(feature)
                if value is None:
                    continue
                if isinstance(value, bool):
                    raise ValueError(f"{feature} must be numeric, not boolean")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{feature} must be finite")
                values[feature] = number
            logger.info("Remote features retrieved: student=%s, features=%s", name, list(values.keys()))
            return values
        except PROVIDER_ERRORS as exc:
            logger.warning("Remote feature service failed: student=%s, error_type=%s; missing features will use NaN", name, type(exc).__name__)
            return {}  # model input uses NaN for missing features
    results = await asyncio.gather(*(call(name, url) for name, url in endpoints.items()))
    grouped = dict(zip(endpoints, results))
    logger.info("Remote feature retrieval completed: token=%s, retrieved_features=%d",
                token, sum(len(values) for values in grouped.values()))
    return grouped
