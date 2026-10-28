"""AT3 Group 05 Student C — Trend Category FastAPI application."""
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal
from uuid import uuid4
from starlette.concurrency import run_in_threadpool
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .config import COINS, CORRELATION_ID, CORRELATION_ID_HEADER, FEATURES_C, get_settings
from .adapters import fetch_market_data
from .services import load_artifact, predict, student_c_features
import logging
import re
import time

CORRELATION_ID_EXAMPLE = "48f8d2b9-8ef1-4d5c-b739-d1e240a4f532"
CORRELATION_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
CORRELATION_RESPONSE_HEADERS = {
    CORRELATION_ID_HEADER: {
        "description": "Request correlation ID, preserved in logs and downstream requests.",
        "schema": {"type": "string", "maxLength": 128},
        "example": CORRELATION_ID_EXAMPLE,
    }
}


class CorrelationIDFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = CORRELATION_ID.get()
        return True


async def propagate_correlation_id(request: httpx.Request):
    """Set the header on each outgoing request, without mutating shared headers."""
    correlation_id = CORRELATION_ID.get()
    if correlation_id != "-":
        request.headers[CORRELATION_ID_HEADER] = correlation_id

class CoinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"token": "BTC", "forecast_horizon": 7}],
    })
    token: str = Field(examples=["BTC"])
    forecast_horizon: int = Field(strict=True, ge=7, le=7, examples=[7], description="Forecast horizon in days; must equal 7")

    @field_validator("token")
    @classmethod
    def check_token(cls, value):
        if value not in COINS:
            raise ValueError(f"Unsupported cryptocurrency; supported tokens: {', '.join(COINS)}")
        return value

class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"prediction": "Sideways", "confidence": 0.84,
                      "forecast_horizon": 7, "prediction_date": "2026-11-03"}],
    })
    prediction: Literal["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]
    confidence: float = Field(ge=0, le=1, description="Probability associated with the predicted class")
    forecast_horizon: Literal[7]
    prediction_date: date

class FeaturesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"token": "BTC", "date": "2026-10-27", "features": {
            "Volatility20": 0.002, "MaxPrice20": 75000.0, "MaxDrawdown": 0.015, "GR": 0.004,
        }}],
    })
    token: str
    date: date
    features: dict[str, float | None]

class RequestErrorResponse(BaseModel):
    error: str
    supported_tokens: list[str] | None = None
    details: list[dict[str, str]] | None = None

REQUEST_ERROR_DOCUMENTATION = {
    "model": RequestErrorResponse,
    "description": "Invalid request; examples show unsupported coin and invalid forecast horizon.",
    "headers": CORRELATION_RESPONSE_HEADERS,
    "content": {"application/json": {"examples": {
        "unsupported_coin": {"summary": "Unsupported cryptocurrency", "value": {
            "error": "Unsupported cryptocurrency", "supported_tokens": list(COINS),
        }},
        "invalid_horizon": {"summary": "Forecast horizon must equal 7", "value": {
            "error": "Invalid request", "details": [
                {"field": "forecast_horizon", "message": "Input should be greater than or equal to 7"},
            ],
        }},
    }}},
}


def unavailable_response_documentation(detail: str) -> dict:
    return {"description": detail, "headers": CORRELATION_RESPONSE_HEADERS,
            "content": {"application/json": {"example": {"detail": detail}}}}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Student C Trend Service")
    cfg = get_settings()
    app.state.cfg = cfg
    logger.info("Configuration loaded: profile=%s, market=%s, features=%s, model=%s, history_days=%d",
        cfg.app_profile, cfg.market_backend, cfg.feature_backend, cfg.model_backend, cfg.history_days)
    logger.info("Loading model artifact")
    app.state.artifact = await run_in_threadpool(load_artifact, cfg)
    logger.info("Model artifact loaded: version=%s", app.state.artifact.get("version", "unknown"))
    # Use the loaded artifact for an accurate metadata example, including all predictors.
    for route in app.routes:
        if getattr(route, "path", None) == "/model-metadata":
            route.responses[200] = {
                "description": "Metadata from the currently loaded model artifact.",
                "headers": CORRELATION_RESPONSE_HEADERS,
                "content": {"application/json": {"example": model_metadata_response(app.state.artifact)}},
            }
    app.openapi_schema = None
    async with httpx.AsyncClient(timeout=cfg.request_timeout_seconds,
                                 event_hooks={"request": [propagate_correlation_id]}) as client:
        app.state.http = client
        logger.info("HTTP client initialized")
        logger.info("Student C Trend Service ready")
        try:
            yield
        finally:
            logger.info("Shutting down Student C Trend Service")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | correlation_id=%(correlation_id)s | %(message)s")
for handler in logging.getLogger().handlers:
    handler.addFilter(CorrelationIDFilter())
logger = logging.getLogger(__name__)

async def accept_correlation_id(correlation_id: str | None = Header(
    default=None, alias=CORRELATION_ID_HEADER,
    examples=[CORRELATION_ID_EXAMPLE],
    description="Optional correlation ID, preserved exactly. Use 1-128 characters, starting with a letter or digit, followed by letters, digits, dots, underscores, colons or hyphens. Missing or invalid values generate a UUID.",
)):
    """Declare the incoming header in OpenAPI and middleware selects the request ID."""
    return correlation_id


app=FastAPI(title="Student C — Trend Category",version="0.1.0",lifespan=lifespan,
            openapi_tags=[
                {"name": "Service Information", "description": "API overview and available endpoints."},
                {"name": "Health", "description": "Service status and active backend configuration."},
                {"name": "Feature Engineering", "description": "Student C's engineered cryptocurrency indicators."},
                {"name": "Predictions", "description": "Seven-day cryptocurrency trend predictions and confidence scores."},
                {"name": "Model Metadata", "description": "Model details, predictors, class labels and feature importance."},
            ],
            description="Seven-day cryptocurrency trend classification.",
            dependencies=[Depends(accept_correlation_id)],
            responses={200: {"headers": CORRELATION_RESPONSE_HEADERS},
                       500: unavailable_response_documentation("Internal server error")})


@app.middleware("http")
async def correlate_request(request: Request, call_next):
    incoming_id = request.headers.get(CORRELATION_ID_HEADER)
    correlation_id = (incoming_id if incoming_id and re.fullmatch(CORRELATION_ID_PATTERN, incoming_id)
                      else str(uuid4()))
    context_token = CORRELATION_ID.set(correlation_id)
    request.state.correlation_id = correlation_id
    started = time.perf_counter()
    try:
        logger.info("Request started: method=%s, path=%s", request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled request error: method=%s, path=%s", request.method, request.url.path)
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        logger.info("Request completed: method=%s, path=%s, status=%d, duration_ms=%.2f",
                    request.method, request.url.path, response.status_code,
                    (time.perf_counter() - started) * 1000)
        return response
    finally:
        CORRELATION_ID.reset(context_token)


@app.exception_handler(RequestValidationError)
async def invalid_request(request:Request, exc:RequestValidationError):
    errors = exc.errors()
    if any(error["loc"] == ("body", "token") and error["type"] == "value_error" for error in errors):
        return JSONResponse(status_code=422, content={
            "error": "Unsupported cryptocurrency", "supported_tokens": list(COINS),
        })
    return JSONResponse(status_code=422, content={
        "error": "Invalid request",
        "details": [{"field": ".".join(map(str, error["loc"][1:])), "message": error["msg"]}
                    for error in errors],
    })

@app.get("/", tags=["Service Information"], responses={200: {"headers": CORRELATION_RESPONSE_HEADERS,
    "content": {"application/json": {"example": {
        "project": "Student C Trend Category API",
        "description": "Seven-day-ahead cryptocurrency trend classification",
        "algorithm": "XGBoost Classifier", "version": "0.1.0",
        "github": "https://github.com/RP28/36120-26SP-AT3-Group05-StudentC-api",
        "endpoints": ["/", "/health", "/features", "/predict", "/model-metadata"],
    }}}}})
async def root():
    return {"project":"Student C Trend Category API","description":"Seven-day-ahead cryptocurrency trend classification",
            "algorithm":"XGBoost Classifier",
            "version":"0.1.0","github":get_settings().github_url,
            "endpoints":["/","/health","/features","/predict","/model-metadata"]}

@app.get("/health", tags=["Health"], responses={200: {"headers": CORRELATION_RESPONSE_HEADERS,
    "content": {"application/json": {"examples": {
        "local": {"summary": "Local development profile", "value": {
            "status": "ok", "profile": "local", "market_backend": "mock",
            "feature_backend": "mock", "model_backend": "trained",
        }},
        "production": {"summary": "Production profile", "value": {
            "status": "ok", "profile": "prod", "market_backend": "http",
            "feature_backend": "http", "model_backend": "trained",
        }},
    }}}}})
async def health(request:Request):
    cfg=request.app.state.cfg
    return {"status":"ok","profile":cfg.app_profile,"market_backend":cfg.market_backend,
            "feature_backend":cfg.feature_backend,"model_backend":cfg.model_backend}

def model_metadata_response(artifact:dict)->dict:
    saved=artifact["metadata"]
    importance={row["feature"]: float(row["importance_percentage"])/100
                for row in saved["feature_importance"]["results"]}
    return {"target":"Trend Category (TC)",
            "target_description": (
                "Five-class cryptocurrency trend seven days after the latest completed UTC observation. "
                "Labels use TrendScore = 0.7 * (MA5 / MA20 - 1) + 0.3 * Return20: "
                "Strong Downtrend below -0.10; Downtrend from -0.10 to below -0.02; "
                "Sideways from -0.02 through 0.02; Uptrend above 0.02 through 0.10; "
                "Strong Uptrend above 0.10."
            ),
            "algorithm":"XGBoost Classifier","forecast_horizon":7,
            "features":list(saved["model"]["features"]),"feature_importance":importance,
            "classification_labels":list(artifact["class_labels"]),
            "model_class":saved["model"]["name"],
            "feature_importance_details":saved["feature_importance"],
            "permutation_importance":saved["permutation_importance"],"created_at":saved["created_at"]}

@app.get("/model-metadata", tags=["Model Metadata"])
async def metadata(request:Request):
    return model_metadata_response(request.app.state.artifact)

@app.post("/features", tags=["Feature Engineering"], response_model=FeaturesResponse, responses={
    422: REQUEST_ERROR_DOCUMENTATION,
    503: unavailable_response_documentation("Unable to obtain required historical data"),
})
async def features(payload:CoinRequest, request:Request):
    try:
        rows = await fetch_market_data(payload.token, request.app.state.cfg, request.app.state.http)
        values = await run_in_threadpool(student_c_features, rows)
        return {
            "token": payload.token,
            "date": max(row["date"] for row in rows),
            "features": {feature: values[feature] for feature in FEATURES_C},
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        logger.exception("Feature request failed: token=%s", payload.token)
        raise HTTPException(status_code=503, detail="Unable to obtain required historical data")

@app.post("/predict", tags=["Predictions"], response_model=PredictionResponse, responses={
    422: REQUEST_ERROR_DOCUMENTATION,
    503: unavailable_response_documentation("Unable to obtain required data or generate prediction"),
})
async def prediction(payload:CoinRequest,request:Request):
    try:
        return await predict(payload.token,payload.forecast_horizon,request.app.state.cfg,request.app.state.http,request.app.state.artifact)
    except (httpx.HTTPError,ValueError,KeyError,TypeError):
        logger.exception("Predict request failed: token=%s", payload.token)
        raise HTTPException(status_code=503, detail="Unable to obtain required data or generate prediction")
