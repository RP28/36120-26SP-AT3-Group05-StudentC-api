"""AT3 Group 05 Student C — Trend Category FastAPI application."""
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, field_validator
from .config import COINS, get_settings
from .adapters import fetch_market_data
from .services import load_artifact, predict, student_c_features
import logging
import time

class CoinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str
    forecast_horizon: int

    @field_validator("token")
    @classmethod
    def check_token(cls, value):
        if value not in COINS:
            raise ValueError(f"Unsupported cryptocurrency; supported tokens: {', '.join(COINS)}")
        return value

    @field_validator("forecast_horizon")
    @classmethod
    def check_horizon(cls, value):
        if value != 7:
            raise ValueError("forecast_horizon must equal 7")
        return value

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

@app.get("/")
async def root():
    return {"project":"Student C Trend Category API","description":"Seven-day-ahead cryptocurrency trend classification",
            "algorithm":"OrderedTrendXGBClassifier",
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
    return {**artifact["metadata"],"forecast_horizon":7,
            "classification_labels":list(artifact["class_labels"]),
            "development_only":request.app.state.cfg.app_profile!="prod"}

@app.post("/features")
async def features(payload:CoinRequest, request:Request):
    try:
        rows = await fetch_market_data(payload.token, request.app.state.cfg, request.app.state.http)
        values = await run_in_threadpool(student_c_features, rows)
        return {
            "token": payload.token,
            "date": max(row["date"] for row in rows),
            "features": values,
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        logger.exception("Feature request failed: token=%s", payload.token)
        raise HTTPException(status_code=503, detail="Unable to obtain required historical data")

@app.post("/predict")
async def prediction(payload:CoinRequest,request:Request):
    try:
        return await predict(payload.token,payload.forecast_horizon,request.app.state.cfg,request.app.state.http,request.app.state.artifact)
    except (httpx.HTTPError,ValueError,KeyError,TypeError):
        logger.exception("Predict request failed: token=%s", payload.token)
        raise HTTPException(status_code=503, detail="Unable to obtain required data or generate prediction")
