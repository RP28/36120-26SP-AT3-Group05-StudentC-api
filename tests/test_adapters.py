import asyncio
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app import adapters
from app.config import COINS, STUDENT_FEATURES

pytestmark = pytest.mark.anyio


def timestamp(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def market_transport(rows, modes=None):
    modes = modes or {}
    calls = []

    def respond(request):
        name = {"api.kraken.com": "kraken", "data-api.coindesk.com": "coindesk",
                "api.coingecko.com": "coingecko", "pro-api.coinmarketcap.com": "cmc"}[request.url.host]
        calls.append(name)
        mode = modes.get(name)
        if mode == "error":
            return httpx.Response(503)
        selected = copy.deepcopy(rows[:-1] if mode == "short" else rows)
        if mode == "gap":
            selected[-1]["date"] = (datetime.fromisoformat(rows[0]["date"])-timedelta(days=1)).date().isoformat()
        if name == "kraken":
            values = [[timestamp(r["date"]), r["open"], r["high"], r["low"], r["close"], 0, 1, 1] for r in selected]
            values.append([timestamp(rows[-1]["date"])+86400, 1, 1, 1, 1, 0, 1, 1])
            payload = {"error": [], "result": {"PAIR": values, "last": 0}}
        elif name == "coindesk":
            payload = {"Data": [{"TIMESTAMP": timestamp(r["date"]), **{key.upper(): r[key] for key in ("open", "high", "low", "close")}} for r in selected]}
        elif name == "coingecko":
            payload = {"market_caps": [[timestamp(r["date"])*1000, r["marketCap"]] for r in selected],
                       "total_volumes": [[timestamp(r["date"])*1000, r["volume"]] for r in selected]}
        else:
            payload = {"data": {"quotes": [{"timestamp": r["date"]+"T00:00:00Z", "quote": {"USD": {
                "market_cap": r["marketCap"]*2, "volume_24h": r["volume"]*2,
            }}} for r in selected]}}
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("coin", COINS)
async def test_mock_history(coin, settings):
    cfg = settings()
    async with httpx.AsyncClient() as client:
        rows = await adapters._fetch_market_data_uncached(coin, cfg, client)
    assert len(rows) == 35 and rows[-1]["date"] == "2026-10-27"
    assert all(row["low"] <= row["open"] <= row["high"] and row["volume"] > 0 for row in rows)


async def test_ohlc_normalizers():
    assert adapters.kraken_to_ohlc({"result": {"last": 5, "pair": [[0, "2", "3", "1", "2", 0], [86400, 3, 4, 2, 3]]}}) == [
        {"date": "1970-01-01", "open": 2.0, "high": 3.0, "low": 1.0, "close": 2.0}]
    assert adapters.coindesk_to_ohlc({"Data": [{"OPEN": None}, {"TIMESTAMP": 0, "OPEN": 2, "HIGH": 3, "LOW": 1, "CLOSE": 2}]})[0]["date"] == "1970-01-01"


@pytest.mark.parametrize("function,payload", [
    (adapters.kraken_to_ohlc, {"error": ["bad pair"]}),
    (adapters.kraken_to_ohlc, {"result": {"last": 2}}),
    (adapters.coindesk_to_ohlc, {"Err": "denied"}),
    (adapters.coindesk_to_ohlc, {"Data": {}}),
])
async def test_bad_ohlc(function, payload):
    with pytest.raises(ValueError):
        function(payload)


async def test_coingecko_daily_alignment_and_latest_sample():
    payload = {"market_caps": [[20, 30], [10, 15], [30, None], [86400000, 99]],
               "total_volumes": [[20, 10], [10, 5], [30, None], [172800000, 4]]}
    assert adapters.coingecko_to_metrics(payload) == [{"date": "1970-01-01", "volume": 10.0, "marketCap": 30.0}]
    assert adapters.coingecko_to_metrics({}) == []


@pytest.mark.parametrize("wrapped", [True, False])
async def test_cmc_shapes_and_skipped_quotes(wrapped):
    quotes = [{}, {"quote": {"USD": {"volume_24h": 3, "market_cap": 4}}},
              {"quote": {"USD": {"timestamp": "2026-10-27T00:00:00Z", "volume_24h": 3, "market_cap": 4}}}]
    data = {"quotes": quotes}
    assert adapters.cmc_to_metrics({"data": {"1": data} if wrapped else data}, "BTC") == [
        {"date": "2026-10-27", "volume": 3.0, "marketCap": 4.0}]


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": {"1": []}}, {"data": {"quotes": "bad"}}])
async def test_cmc_invalid(payload):
    with pytest.raises(ValueError):
        adapters.cmc_to_metrics(payload, "BTC")


@pytest.mark.parametrize("rows,fields", [
    ([], ("close",)), ({}, ("close",)),
    ([{"date": "2026-10-27", "close": float("nan")}], ("close",)),
    ([{"date": "2026-10-27", "close": 0}], ("close",)),
    ([{"date": "2026-10-27", "volume": -1}], ("volume",)),
    ([{"date": "2026-10-27", "open": 2, "close": 2, "high": 1, "low": 1}], ("open", "high", "low", "close")),
    ([{"date": "2026-10-27", "open": 2, "close": 2, "high": 4, "low": 3}], ("open", "high", "low", "close")),
])
async def test_invalid_provider_rows(rows, fields):
    with pytest.raises(ValueError):
        adapters._validate_rows(rows, fields, "test")


async def test_duplicate_provider_dates_keep_last():
    rows = [{"date": "2026-10-27", "volume": 0}, {"date": "2026-10-26", "volume": 1},
            {"date": "2026-10-27", "volume": 2}]
    assert adapters._validate_rows(rows, ("volume",), "test") == [rows[1], rows[2]]


async def test_merge_requires_completed_aligned_dates(market):
    assert adapters._merge_for_history(market, market, 35) == market
    with pytest.raises(adapters.MarketDataError, match="34/35"):
        adapters._merge_for_history(market, market[:-1], 35)


@pytest.mark.parametrize("modes,expected", [
    ({}, ["kraken", "coingecko"]),
    ({"coingecko": "error"}, ["kraken", "coingecko", "cmc"]),
    ({"kraken": "error"}, ["kraken", "coindesk", "coingecko"]),
    ({"coingecko": "short"}, ["kraken", "coingecko", "cmc"]),
    ({"kraken": "short"}, ["kraken", "coindesk", "coingecko"]),
    ({"coingecko": "gap"}, ["kraken", "coingecko", "coindesk", "cmc"]),
    ({"kraken": "gap", "coingecko": "gap"}, ["kraken", "coingecko", "coindesk", "cmc"]),
])
async def test_market_provider_fallbacks(production, market, modes, expected):
    transport, calls = market_transport(market, modes)
    async with httpx.AsyncClient(transport=transport) as client:
        rows = await adapters._fetch_market_data_uncached("BTC", production, client)
    assert len(rows) == 35 and rows[-1]["date"] == "2026-10-27"
    assert sorted(calls) == sorted(expected)
    if "cmc" in calls:
        assert calls.index("coingecko") < calls.index("cmc")


@pytest.mark.parametrize("modes", [
    dict.fromkeys(["kraken", "coindesk", "coingecko", "cmc"], "short"),
    dict.fromkeys(["kraken", "coindesk", "coingecko", "cmc"], "error"),
    {"coingecko": "gap", "cmc": "error", "coindesk": "error"},
])
async def test_market_exhausts_fallbacks(production, market, modes):
    transport, _ = market_transport(market, modes)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(adapters.MarketDataError, match="Could not obtain 35"):
            await adapters._fetch_market_data_uncached("BTC", production, client)


async def test_missing_keys_and_invalid_market_inputs(production, market):
    cfg = production.model_copy(update={"cmc_api_key": "", "coindesk_api_key": "", "coingecko_api_key": ""})
    transport, _ = market_transport(market, {"kraken": "error"})
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(adapters.MarketDataError):
            await adapters._fetch_market_data_uncached("BTC", cfg, client)
        with pytest.raises(ValueError, match="Unsupported"):
            await adapters._fetch_market_data_uncached("BAD", cfg, client)
        with pytest.raises(ValueError, match="positive"):
            await adapters._fetch_market_data_uncached("BTC", cfg.model_copy(update={"history_days": 0}), client)


async def test_market_cache_hit_expiry_pruning_and_copies(monkeypatch, settings, market):
    clock = [10.0]
    monkeypatch.setattr(adapters, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    fetch = AsyncMock(return_value=market)
    monkeypatch.setattr(adapters, "_fetch_market_data_uncached", fetch)
    old_key = ("local", "BTC", 35, "2026-10-26")
    adapters._market_cache[old_key] = {"data": [], "expires_at": 100}
    adapters._market_cache_locks[old_key] = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        first = await adapters.fetch_market_data("BTC", settings(), client)
        first[-1]["close"] = -1
        cached = await adapters.fetch_market_data("BTC", settings(), client)
        assert cached[-1]["close"] == market[-1]["close"]
        assert fetch.await_count == 1 and old_key not in adapters._market_cache_locks
        clock[0] += 1800
        await adapters.fetch_market_data("BTC", settings(), client)
        assert fetch.await_count == 2


async def test_market_cache_coalesces_concurrent_requests(monkeypatch, settings, market):
    async def fetch(*args):
        await asyncio.sleep(0)
        return market
    fetcher = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(adapters, "_fetch_market_data_uncached", fetcher)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(adapters.fetch_market_data("BTC", settings(), client) for _ in range(5)))
    assert fetcher.await_count == 1 and all(result == market for result in results)


async def test_mock_external_indicators(settings):
    async with httpx.AsyncClient() as client:
        features = await adapters.fetch_remote_features("ETH", 7, settings(), client)
    assert {student: set(values) for student, values in features.items()} == {
        student: set(values) for student, values in STUDENT_FEATURES.items()}
    assert features["B"]["TrendStrength"] == pytest.approx(features["B"]["MA5"]/features["B"]["MA20"]-1)


async def test_remote_features_flat_nested_and_extra_columns(production, external):
    def respond(request):
        assert request.method == "POST" and request.url.path == "/features"
        student = request.url.host[0].upper()
        features = {**external[student], "ignored": "not numeric"}
        return httpx.Response(200, json=features if student == "A" else {
            "token": "BTC", "date": "2026-10-27", "features": features})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert await adapters.fetch_remote_features("BTC", 7, production, client) == external


@pytest.mark.parametrize("body", [
    [], {"token": "ETH"}, {"date": "2026-10-26"}, {"features": []},
    {"DailyReturn": True}, {"DailyReturn": "bad"}, {"DailyReturn": "inf"},
])
async def test_remote_invalid_features_become_missing(production, body):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        assert await adapters.fetch_remote_features("BTC", 7, production, client) == {"A": {}, "B": {}, "D": {}}


async def test_remote_null_outages_and_missing_url(production):
    cfg = production.model_copy(update={"student_a_url": ""})
    def respond(request):
        if request.url.host == "b.test":
            return httpx.Response(502)
        return httpx.Response(200, json={"SMA20Volume": None, "RelativeVolume": 2})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert await adapters.fetch_remote_features("BTC", 7, cfg, client) == {
            "A": {}, "B": {}, "D": {"RelativeVolume": 2.0}}
