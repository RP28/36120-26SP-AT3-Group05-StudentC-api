import asyncio
import logging
from unittest.mock import AsyncMock
from uuid import UUID

from fastapi.testclient import TestClient
import httpx
import pytest

from app import adapters, main, services
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
    assert set(info.json()["endpoints"]) == {"/", "/health", "/history", "/features", "/predict", "/model-metadata"}
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
                "/history": ("get", "Historical Data"),
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


@pytest.mark.parametrize("coin", COINS)
def test_history_returns_completed_ohlc_only(api, coin):
    response = api.get("/history", params={"token": coin},
                       headers={CORRELATION_ID_HEADER: "history-test"})
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"token", "data"} and result["token"] == coin
    assert len(result["data"]) == 35
    assert [row["date"] for row in result["data"]] == sorted(row["date"] for row in result["data"])
    assert result["data"][0]["date"] == "2026-09-23" and result["data"][-1]["date"] == "2026-10-27"
    assert all(set(row) == {"date", "open", "high", "low", "close"} for row in result["data"])
    assert all(row["low"] <= row["open"] <= row["high"] for row in result["data"])
    assert response.headers[CORRELATION_ID_HEADER] == "history-test"


@pytest.mark.parametrize("params", [{}, {"token": "btc"}, {"token": "INVALID"},
                                  {"token": "BTC", "days": 1}])
def test_history_rejects_invalid_queries(api, params):
    response = api.get("/history", params=params)
    assert response.status_code == 422
    if params.get("token") in ("btc", "INVALID"):
        assert response.json() == {"error": "Unsupported cryptocurrency", "supported_tokens": list(COINS)}
    else:
        assert response.json()["error"] == "Invalid request"


@pytest.mark.parametrize("error", [ValueError("unavailable"), httpx.ReadTimeout("timeout"),
                                 KeyError("missing date"), TypeError("invalid data")])
def test_history_provider_errors_are_correlated_503(api, monkeypatch, error):
    monkeypatch.setattr(main, "fetch_market_data", AsyncMock(side_effect=error))
    response = api.get("/history", params={"token": "BTC"},
                       headers={CORRELATION_ID_HEADER: "history-failure"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Unable to obtain required historical data"}
    assert response.headers[CORRELATION_ID_HEADER] == "history-failure"


@pytest.mark.parametrize("order", [("history", "features", "predict"),
                                 ("predict", "history", "features"),
                                 ("features", "history", "predict")])
def test_history_reuses_market_cache_across_endpoints(api, monkeypatch, order):
    fetch = AsyncMock(wraps=adapters._fetch_market_data_uncached)
    students = AsyncMock(wraps=services.fetch_remote_features)
    monkeypatch.setattr(adapters, "_fetch_market_data_uncached", fetch)
    monkeypatch.setattr(services, "fetch_remote_features", students)
    for endpoint in order:
        response = (api.get("/history", params={"token": "BTC"}) if endpoint == "history"
                    else api.post(f"/{endpoint}", json={"token": "BTC", "forecast_horizon": 7}))
        assert response.status_code == 200, response.text
    assert fetch.await_count == students.await_count == 1
    cached = next(iter(adapters._market_cache.values()))["data"]
    assert len(cached) == 35 and "volume" in cached[0] and "marketCap" in cached[0]


def test_history_openapi_contract_and_example(api):
    schema = api.get("/openapi.json").json()
    operation = schema["paths"]["/history"]["get"]
    assert "requestBody" not in operation
    token = next(parameter for parameter in operation["parameters"] if parameter["name"] == "token")
    assert token["in"] == "query" and token["required"] is True
    assert token["schema"]["examples"] == ["BTC"]
    assert all(parameter["name"] != "forecast_horizon" for parameter in operation["parameters"])
    model = schema["components"]["schemas"]["HistoryResponse"]
    main.HistoryResponse.model_validate(model["examples"][0])


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
