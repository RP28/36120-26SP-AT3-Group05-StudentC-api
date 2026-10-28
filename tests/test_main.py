import asyncio
import logging
from unittest.mock import AsyncMock
from uuid import UUID

from fastapi.testclient import TestClient
import httpx
import pytest

from app import main
from app.config import COINS, CORRELATION_ID, CORRELATION_ID_HEADER, FEATURES_C


@pytest.fixture
def api(monkeypatch, settings, artifact):
    cfg = settings()
    monkeypatch.setattr(main, "get_settings", lambda: cfg)
    monkeypatch.setattr(main, "load_artifact", lambda cfg: artifact)
    # Ensure startup replaces any schema generated before the model was loaded.
    main.app.openapi()
    with TestClient(main.app) as client:
        yield client
    assert main.app.state.http.is_closed


def test_service_information_health_and_metadata(api, metadata):
    info = api.get("/")
    assert info.status_code == 200
    assert set(info.json()["endpoints"]) == {"/", "/health", "/features", "/predict", "/model-metadata"}
    assert info.json()["github"].startswith("https://github.com/")
    health = api.get("/health")
    assert health.json() == {"status": "ok", "profile": "local", "market_backend": "mock",
                             "feature_backend": "mock", "model_backend": "trained"}
    result = api.get("/model-metadata").json()
    assert result["features"] == metadata["model"]["features"]
    assert set(result["feature_importance"]) == set(result["features"])
    assert sum(result["feature_importance"].values()) == pytest.approx(1)
    assert result["permutation_importance"] == metadata["permutation_importance"]
    assert result["classification_labels"] == list(main.PredictionResponse.model_fields["prediction"].annotation.__args__)


@pytest.mark.parametrize("coin", COINS)
def test_prediction_and_features_contract(api, coin):
    body = {"token": coin, "forecast_horizon": 7}
    features = api.post("/features", json=body)
    assert features.status_code == 200 and features.json()["token"] == coin
    assert set(features.json()["features"]) == set(FEATURES_C)
    response = api.post("/predict", json=body)
    assert response.status_code == 200
    assert response.json() == {"prediction": "Sideways", "confidence": .7,
                               "forecast_horizon": 7, "prediction_date": "2026-11-03"}
    assert api.post("/predict", json=body).json() == response.json()


@pytest.mark.parametrize("endpoint", ["/features", "/predict"])
@pytest.mark.parametrize("body", [
    {}, {"token": "BTC"}, {"token": "BTC", "forecast_horizon": 1},
    {"token": "BTC", "forecast_horizon": "7"}, {"token": "BTC", "forecast_horizon": 7.0},
    {"token": "BTC", "forecast_horizon": True}, {"token": 1, "forecast_horizon": 7},
    {"token": "BTC", "forecast_horizon": 7, "extra": 1},
])
def test_invalid_requests(api, endpoint, body):
    response = api.post(endpoint, json=body)
    assert response.status_code == 422
    assert response.json()["error"] == "Invalid request" and response.json()["details"]
    UUID(response.headers[CORRELATION_ID_HEADER])


def test_unknown_token_and_malformed_json(api):
    unsupported = api.post("/predict", json={"token": "BAD", "forecast_horizon": 7})
    assert unsupported.status_code == 422
    assert unsupported.json() == {"error": "Unsupported cryptocurrency", "supported_tokens": list(COINS)}
    malformed = api.post("/predict", content="{bad", headers={"Content-Type": "application/json"})
    assert malformed.status_code == 422 and malformed.json()["error"] == "Invalid request"


@pytest.mark.parametrize("supplied", ["btc-test-123", "48F8D2B9-8EF1-4D5C-B739-D1E240A4F532", "a"*128])
def test_incoming_correlation_id_is_preserved(api, supplied):
    response = api.get("/health", headers={"x-correlation-id": supplied})
    assert response.headers[CORRELATION_ID_HEADER] == supplied


@pytest.mark.parametrize("supplied", [None, "", "invalid id", "a"*129, "<bad>"])
def test_missing_or_invalid_correlation_id_is_generated(api, supplied):
    response = api.get("/health", headers={} if supplied is None else {CORRELATION_ID_HEADER: supplied})
    UUID(response.headers[CORRELATION_ID_HEADER])
    assert CORRELATION_ID.get() == "-"


def test_cache_response_has_request_specific_id(api):
    body = {"token": "BTC", "forecast_horizon": 7}
    first = api.post("/predict", json=body, headers={CORRELATION_ID_HEADER: "first"})
    second = api.post("/predict", json=body, headers={CORRELATION_ID_HEADER: "second"})
    assert first.json() == second.json()
    assert first.headers[CORRELATION_ID_HEADER] == "first"
    assert second.headers[CORRELATION_ID_HEADER] == "second"
    assert CORRELATION_ID_HEADER not in next(iter(main.app.state.artifact["prediction_cache"].values()))["data"]


@pytest.mark.parametrize("endpoint,target", [("/features", "fetch_market_data"), ("/predict", "predict")])
def test_unavailable_data_returns_503(api, monkeypatch, endpoint, target):
    monkeypatch.setattr(main, target, AsyncMock(side_effect=ValueError("unavailable")))
    response = api.post(endpoint, json={"token": "BTC", "forecast_horizon": 7},
                        headers={CORRELATION_ID_HEADER: "failed-request"})
    assert response.status_code == 503 and response.json()["detail"]
    assert response.headers[CORRELATION_ID_HEADER] == "failed-request"


def test_unexpected_failure_returns_correlated_500(api, monkeypatch):
    monkeypatch.setattr(main, "predict", AsyncMock(side_effect=RuntimeError("internal details")))
    response = api.post("/predict", json={"token": "BTC", "forecast_horizon": 7},
                        headers={CORRELATION_ID_HEADER: "server-error"})
    assert response.status_code == 500 and response.json() == {"detail": "Internal server error"}
    assert response.headers[CORRELATION_ID_HEADER] == "server-error"


def test_features_expose_only_owned_indicators(api, monkeypatch):
    monkeypatch.setattr(main, "student_c_features", lambda rows: {**dict.fromkeys(FEATURES_C, 1.0), "extra": 99})
    result = api.post("/features", json={"token": "BTC", "forecast_horizon": 7})
    assert set(result.json()["features"]) == set(FEATURES_C)


def test_openapi_examples_sections_and_headers(api):
    schema = api.get("/openapi.json").json()
    expected = {"/": ("get", "Service Information"), "/health": ("get", "Health"),
                "/features": ("post", "Feature Engineering"), "/predict": ("post", "Predictions"),
                "/model-metadata": ("get", "Model Metadata")}
    for path, (method, tag) in expected.items():
        operation = schema["paths"][path][method]
        assert operation["tags"] == [tag]
        parameter = next(p for p in operation["parameters"] if p["name"] == CORRELATION_ID_HEADER)
        assert parameter["in"] == "header" and parameter["required"] is False
        assert CORRELATION_ID_HEADER in operation["responses"]["200"]["headers"]
    models = schema["components"]["schemas"]
    assert models["CoinRequest"]["examples"] == [{"token": "BTC", "forecast_horizon": 7}]
    assert set(models["PredictionResponse"]["properties"]) == {"prediction", "confidence", "forecast_horizon", "prediction_date"}
    assert schema["paths"]["/model-metadata"]["get"]["responses"]["200"]["content"]["application/json"]["example"] == api.get("/model-metadata").json()
    assert api.get("/docs").status_code == 200


def test_logging_filter_uses_current_request_context():
    record = logging.makeLogRecord({"msg": "test"})
    token = CORRELATION_ID.set("log-test")
    try:
        assert main.CorrelationIDFilter().filter(record) is True
        assert record.correlation_id == "log-test"
    finally:
        CORRELATION_ID.reset(token)


@pytest.mark.anyio
async def test_outgoing_correlation_headers_and_context_isolation():
    seen = []
    async def transport(request):
        await asyncio.sleep(0)
        seen.append(request.headers.get(CORRELATION_ID_HEADER))
        return httpx.Response(200)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport), event_hooks={"request": [main.propagate_correlation_id]}) as client:
        await client.get("https://test")
        async def request(value):
            token = CORRELATION_ID.set(value)
            try:
                await client.get("https://test")
            finally:
                CORRELATION_ID.reset(token)
        await asyncio.gather(request("one"), request("two"))
        assert CORRELATION_ID_HEADER not in client.headers
    assert seen[0] is None and set(seen[1:]) == {"one", "two"}
