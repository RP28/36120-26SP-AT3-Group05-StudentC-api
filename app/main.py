"""AT3 Group 05 Student C — Trend Category FastAPI application."""
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal
from starlette.concurrency import run_in_threadpool
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .config import COINS, FEATURES_C, get_settings
from .adapters import fetch_market_data
from .services import load_artifact, predict, student_c_features
import logging

class CoinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str
    forecast_horizon: int = Field(strict=True, ge=7, le=7, description="Forecast horizon in days; must equal 7")

    @field_validator("token")
    @classmethod
    def check_token(cls, value):
        if value not in COINS:
            raise ValueError(f"Unsupported cryptocurrency; supported tokens: {', '.join(COINS)}")
        return value

class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: Literal["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]
    confidence: float = Field(ge=0, le=1, description="Probability associated with the predicted class")
    forecast_horizon: Literal[7]
    prediction_date: date

class FeaturesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str
    date: date
    features: dict[str, float | None]

class RequestErrorResponse(BaseModel):
    error: str
    supported_tokens: list[str] | None = None
    details: list[dict[str, str]] | None = None

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
    async with httpx.AsyncClient(timeout=cfg.request_timeout_seconds) as client:
        app.state.http = client
        logger.info("HTTP client initialized")
        logger.info("Student C Trend Service ready")
        try:
            yield
        finally:
            logger.info("Shutting down Student C Trend Service")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app=FastAPI(title="Student C — Trend Category",version="0.1.0",lifespan=lifespan)

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

@app.get("/")
async def root():
    return {"project":"Student C Trend Category API","description":"Seven-day-ahead cryptocurrency trend classification",
            "algorithm":"XGBoost Classifier",
            "version":"0.1.0","github":get_settings().github_url,
            "endpoints":["/","/health","/features","/predict","/model-metadata"]}

@app.get("/health")
async def health(request:Request):
    cfg=request.app.state.cfg
    return {"status":"ok","profile":cfg.app_profile,"market_backend":cfg.market_backend,
            "feature_backend":cfg.feature_backend,"model_backend":cfg.model_backend}

@app.get("/model-metadata")
async def metadata(request:Request):
    artifact=request.app.state.artifact
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

@app.post("/features", response_model=FeaturesResponse, responses={422: {"model": RequestErrorResponse}})
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

@app.post("/predict", response_model=PredictionResponse, responses={422: {"model": RequestErrorResponse}})
async def prediction(payload:CoinRequest,request:Request):
    try:
        return await predict(payload.token,payload.forecast_horizon,request.app.state.cfg,request.app.state.http,request.app.state.artifact)
    except (httpx.HTTPError,ValueError,KeyError,TypeError):
        logger.exception("Predict request failed: token=%s", payload.token)
        raise HTTPException(status_code=503, detail="Unable to obtain required data or generate prediction")
