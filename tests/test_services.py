import asyncio
import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import cloudpickle
import httpx
import pytest

from app import adapters, services
from app.config import COINS

pytestmark = pytest.mark.anyio


async def test_student_c_formulas_and_order(market):
    rows = market[-21:]
    previous = [row["close"] for row in rows[:-1]]
    returns = [(row["close"]-prior)/prior for row, prior in zip(rows[1:], previous)]
    gaps = [abs(row["open"]-prior)/prior for row, prior in zip(rows[1:], previous)]
    high = max(row["high"] for row in rows[-20:])
    values = services.student_c_features(list(reversed(rows)))
    assert values == pytest.approx({
        "Volatility20": statistics.stdev(returns), "MaxPrice20": high,
        "MaxDrawdown": (high-rows[-1]["close"])/high, "GR": statistics.mean(gaps),
    })


@pytest.mark.parametrize("mode", ["short", "gap"])
async def test_student_features_reject_insufficient_history(market, mode):
    rows = market[-20:] if mode == "short" else market
    if mode == "gap":
        rows[-1]["date"] = "2026-10-29"
    with pytest.raises(ValueError, match="21 consecutive"):
        services.student_c_features(rows)


async def test_nonfinite_features_are_json_null(market):
    for row in market:
        row["close"] = row["high"] = row["open"] = 0
    result = services.student_c_features(market)
    assert result == {"Volatility20": None, "MaxPrice20": 0.0, "MaxDrawdown": None, "GR": None}
    json.dumps(result, allow_nan=False)


def write_artifacts(tmp_path, settings, metadata, function=None):
    path = tmp_path / "model.pkl"
    with path.open("wb") as file:
        cloudpickle.dump(function if function is not None else (lambda **kwargs: {}), file)
    info = tmp_path / "metadata.json"
    info.write_text(json.dumps(metadata))
    return settings(model_path=str(path), model_metadata_path=str(info))


async def test_load_callable_and_metadata(tmp_path, settings, metadata):
    cfg = write_artifacts(tmp_path, settings, metadata)
    artifact = services.load_artifact(cfg)
    assert callable(artifact["predict_function"]) and artifact["metadata"] == metadata
    assert artifact["class_labels"] == services.CLASS_LABELS
    assert artifact["version"] == metadata["created_at"]
    metadata.pop("created_at")
    assert services.load_artifact(write_artifacts(tmp_path, settings, metadata))["version"] == "unknown"


async def test_missing_or_noncallable_model(tmp_path, settings, metadata):
    with pytest.raises(FileNotFoundError, match="Trained model"):
        services.load_artifact(settings(model_path=str(tmp_path/"absent.pkl")))
    cfg = write_artifacts(tmp_path, settings, metadata, function={"old": "artifact"})
    with pytest.raises(ValueError, match="callable"):
        services.load_artifact(cfg)


@pytest.mark.parametrize("columns,count", [({}, 0), ([], 0), ([1], 1), (["coin", "coin"], 2), (["coin"], 2)])
async def test_invalid_metadata_schema(tmp_path, settings, metadata, columns, count):
    metadata["model"].update(features=columns, feature_count=count)
    with pytest.raises(ValueError, match="feature schema"):
        services.load_artifact(write_artifacts(tmp_path, settings, metadata))


async def test_wrong_forecast_target(tmp_path, settings, metadata):
    metadata["model"]["target"] = "today"
    with pytest.raises(ValueError, match="TrendCategory7d"):
        services.load_artifact(write_artifacts(tmp_path, settings, metadata))


def setup_inference(monkeypatch, market, external, artifact):
    monkeypatch.setattr(services, "fetch_market_data", AsyncMock(return_value=market))
    monkeypatch.setattr(services, "fetch_remote_features", AsyncMock(return_value=external))
    spy = Mock(wraps=artifact["predict_function"])
    artifact["predict_function"] = spy
    return spy


@pytest.mark.parametrize("profile", ["local", "prod"])
async def test_prediction_contract_and_inputs(monkeypatch, market, external, artifact, settings, production, profile):
    spy = setup_inference(monkeypatch, market, external, artifact)
    async with httpx.AsyncClient() as client:
        result, complete = await services._predict_uncached("BTC", 7, settings() if profile == "local" else production, client, artifact)
    assert result == {"prediction": "Sideways", "confidence": .7, "forecast_horizon": 7, "prediction_date": "2026-11-03"}
    assert complete is True
    inputs = spy.call_args.kwargs
    assert inputs["market_df"].to_dict("records") == market
    assert inputs["student_a"] == external["A"] and inputs["coin"] == "BTC"


@pytest.mark.parametrize("change,error", [
    ({"predicted_trend": "INVALID"}, "unknown trend"),
    ({"probabilities": {}}, "probabilities"),
    ({"probabilities": dict(zip(services.CLASS_LABELS, [math.nan, .1, .7, .1, .05]))}, "probabilities"),
    ({"probabilities": dict(zip(services.CLASS_LABELS, [-.05, .1, .8, .1, .05]))}, "probabilities"),
    ({"probabilities": dict(zip(services.CLASS_LABELS, [0, 0, .5, 0, 0]))}, "probabilities"),
    ({"coin": "ETH"}, "identifiers"), ({"date": "2026-10-26"}, "identifiers"),
    ({"forecast_horizon_days": 1}, "identifiers"),
])
async def test_invalid_model_outputs(monkeypatch, market, external, artifact, served, settings, change, error):
    setup_inference(monkeypatch, market, external, artifact)
    artifact["predict_function"] = lambda **kwargs: {**served, **change}
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match=error):
            await services._predict_uncached("BTC", 7, settings(), client, artifact)


async def test_missing_features_predict_but_are_not_cached(monkeypatch, market, external, artifact, settings):
    external["A"] = {}
    spy = setup_inference(monkeypatch, market, external, artifact)
    async with httpx.AsyncClient() as client:
        first = await services.predict("BTC", 7, settings(), client, artifact)
        second = await services.predict("BTC", 7, settings(), client, artifact)
    assert first == second and spy.call_count == 2 and not artifact["prediction_cache"]


async def test_prediction_cache_expiry_and_copies(monkeypatch, market, external, artifact, settings):
    spy = setup_inference(monkeypatch, market, external, artifact)
    seconds = [10.0]
    monkeypatch.setattr(services, "time", SimpleNamespace(monotonic=lambda: seconds[0], perf_counter=lambda: seconds[0]))
    async with httpx.AsyncClient() as client:
        first = await services.predict("BTC", 7, settings(), client, artifact)
        first["confidence"] = 0
        second = await services.predict("BTC", 7, settings(), client, artifact)
        assert second["confidence"] == .7 and spy.call_count == 1
        seconds[0] = 1809
        await services.predict("BTC", 7, settings(), client, artifact)
        assert spy.call_count == 1
        seconds[0] = 1810
        await services.predict("BTC", 7, settings(), client, artifact)
        assert spy.call_count == 2


async def test_prediction_date_rollover_and_lock_cleanup(monkeypatch, settings, artifact):
    clock = [datetime(2026, 10, 28, tzinfo=timezone.utc)]
    monkeypatch.setattr(services, "datetime", SimpleNamespace(now=lambda tz: clock[0]))
    async def calculate(*args):
        return {"prediction_date": (clock[0].date()+timedelta(days=6)).isoformat()}, True
    work = AsyncMock(side_effect=calculate)
    monkeypatch.setattr(services, "_predict_uncached", work)
    cfg = settings()
    locked_key = ("local", "ETH", 7, 35, "2026-10-26", artifact["version"])
    held = asyncio.Lock()
    await held.acquire()
    artifact["prediction_cache_locks"] = {locked_key: held}
    async with httpx.AsyncClient() as client:
        await services.predict("BTC", 7, cfg, client, artifact)
        assert locked_key in artifact["prediction_cache_locks"]
        held.release()
        clock[0] += timedelta(days=1)
        await services.predict("BTC", 7, cfg, client, artifact)
    assert work.await_count == 2 and len(artifact["prediction_cache"]) == 1
    assert locked_key not in artifact["prediction_cache_locks"]


async def test_stale_results_not_cached(monkeypatch, settings, artifact):
    work = AsyncMock(return_value=({"prediction_date": "2026-11-02"}, True))
    monkeypatch.setattr(services, "_predict_uncached", work)
    await services.predict("BTC", 7, settings(), None, artifact)
    assert not artifact["prediction_cache"]


async def test_prediction_cache_concurrency_and_key_separation(monkeypatch, settings, artifact):
    active, peak = 0, 0
    async def calculate(*args):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(0)
        active -= 1
        return {"prediction_date": "2026-11-03"}, True
    work = AsyncMock(side_effect=calculate)
    monkeypatch.setattr(services, "_predict_uncached", work)
    cfg = settings()
    await asyncio.gather(*(services.predict("BTC", 7, cfg, None, artifact) for _ in range(8)))
    assert work.await_count == 1
    await asyncio.gather(services.predict("ETH", 7, cfg, None, artifact), services.predict("SOL", 7, cfg, None, artifact))
    assert peak == 2
    await services.predict("BTC", 7, cfg.model_copy(update={"history_days": 90}), None, artifact)
    await services.predict("BTC", 7, cfg.model_copy(update={"app_profile": "dev"}), None, artifact)
    artifact["version"] = "new"
    await services.predict("BTC", 7, cfg, None, artifact)
    assert work.await_count == 6


async def test_failed_prediction_is_not_cached(monkeypatch, settings, artifact):
    monkeypatch.setattr(services, "_predict_uncached", AsyncMock(side_effect=ValueError("unavailable")))
    with pytest.raises(ValueError):
        await services.predict("BTC", 7, settings(), None, artifact)
    assert not artifact["prediction_cache"]


async def test_bundled_model_smoke(settings):
    """Exercise the real serialized pipeline without AT3 or network access."""
    root = Path(__file__).resolve().parents[1]
    cfg = settings(model_path=str(root/"models/predict_seven_day_trend.pkl"),
                   model_metadata_path=str(root/"models/model_metadata.json"))
    artifact = services.load_artifact(cfg)
    async with httpx.AsyncClient() as client:
        for coin in COINS:
            result = await services.predict(coin, 7, cfg, client, artifact)
            assert result["prediction"] in services.CLASS_LABELS
            assert 0 <= result["confidence"] <= 1 and result["forecast_horizon"] == 7
