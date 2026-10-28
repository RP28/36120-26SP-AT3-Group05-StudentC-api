"""Deterministic inputs, isolated settings, and a ban on live HTTP requests."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import httpx
import pytest

from app import adapters, config, services


NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    for field in config.Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    config.get_settings.cache_clear()
    adapters._market_cache.clear()
    adapters._market_cache_locks.clear()
    monkeypatch.setattr(adapters, "datetime", FrozenDatetime)
    monkeypatch.setattr(services, "datetime", FrozenDatetime)
    token = config.CORRELATION_ID.set("-")

    async def no_network(*args, **kwargs):
        raise AssertionError("Unit tests must not contact live APIs")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    yield
    config.CORRELATION_ID.reset(token)
    config.get_settings.cache_clear()
    adapters._market_cache.clear()
    adapters._market_cache_locks.clear()


@pytest.fixture
def settings():
    def make(**overrides):
        return config.Settings(_env_file=None, **{"history_days": 35, **overrides})
    return make


@pytest.fixture
def production(settings):
    return settings(app_profile="prod", student_a_url="https://a.test",
                    student_b_url="https://b.test", student_d_url="https://d.test",
                    coingecko_api_key="gecko-test", cmc_api_key="cmc-test",
                    coindesk_api_key="desk-test")


@pytest.fixture
def market():
    return [{"date": (NOW.date() - timedelta(days=35-i)).isoformat(),
             "open": 100.0+i, "high": 103.0+i, "low": 99.0+i, "close": 102.0+i,
             "volume": 1000.0+i*10, "marketCap": 100000.0+i*100}
            for i in range(35)]


@pytest.fixture
def external():
    return {student: {feature: 1.0 for feature in features}
            for student, features in config.STUDENT_FEATURES.items()}


@pytest.fixture
def metadata():
    path = Path(__file__).resolve().parents[1] / "models/model_metadata.json"
    return json.loads(path.read_text())


@pytest.fixture
def served():
    return {"coin": "BTC", "date": "2026-10-27", "forecast_horizon_days": 7,
            "predicted_trend": "Sideways",
            "probabilities": dict(zip(services.CLASS_LABELS, [.05, .10, .70, .10, .05]))}


@pytest.fixture
def artifact(metadata, served):
    def inference(**kwargs):
        result = copy.deepcopy(served)
        result["coin"] = kwargs["coin"]
        result["date"] = kwargs["market_df"]["date"].iloc[-1]
        return result
    return {"predict_function": inference, "metadata": metadata,
            "class_labels": list(services.CLASS_LABELS), "version": metadata["created_at"]}
