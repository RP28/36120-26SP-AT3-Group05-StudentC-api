"""Student C indicators and saved-model inference orchestration."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import copy
import math
import json
import cloudpickle
import pandas as pd
import numpy as np
import httpx
import logging
import time
import asyncio
from starlette.concurrency import run_in_threadpool
from .adapters import fetch_market_data, fetch_remote_features
from .config import Settings, STUDENT_FEATURES

CLASS_LABELS = ["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]
PREDICTION_CACHE_TTL_SECONDS = 1800  # 30 minutes

logger = logging.getLogger(__name__)

def student_c_features(rows:list[dict]) -> dict:
    """Matches AT3 notebook formulas: 20-return sample std (ddof=1), max high,
    (max-high minus close)/max-high, mean 20 absolute overnight gaps.
    Requires 21 consecutive daily observations for volatility and gap risk.
    """
    logger.debug("Student C feature calculation started: rows=%d", len(rows))
    df=pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"],utc=True)
    # Never roll across a missing calendar day.
    if len(df)<21 or not df["date"].diff().iloc[-21:].dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Student C features require 21 consecutive completed UTC days")
    previous=df["close"].shift(1).replace(0,np.nan)
    daily_returns=(df["close"]-previous)/previous
    gap=(df["open"]-previous).abs()/previous
    high=df["high"].rolling(20,min_periods=20).max().iloc[-1]
    result={
        "Volatility20":float(daily_returns.rolling(20,min_periods=20).std(ddof=1).iloc[-1]),
        "MaxPrice20":float(high),
        "MaxDrawdown":float((high-df["close"].iloc[-1])/high) if high else math.nan,
        "GR":float(gap.rolling(20,min_periods=20).mean().iloc[-1]),
    }
    logger.info("Student C feature calculation completed: features=%s", list(result.keys()))
    return {k:(v if math.isfinite(v) else None) for k,v in result.items()}


def load_artifact(cfg:Settings):
    path=Path(cfg.model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {path}")
    with path.open("rb") as f:
        prediction_function=cloudpickle.load(f)
    if not callable(prediction_function):
        raise ValueError("Model artifact must contain the notebook's prediction callable")
    with Path(cfg.model_metadata_path).open(encoding="utf-8") as f:
        metadata=json.load(f)
    model_info=metadata["model"]
    columns=model_info["features"]
    if (not isinstance(columns,list) or not columns
            or not all(isinstance(column,str) for column in columns)
            or len(set(columns)) != len(columns)
            or model_info["feature_count"] != len(columns)):
        raise ValueError("Model metadata must describe a unique, complete feature schema")
    if model_info["target"] != "TrendCategory7d":
        raise ValueError("Model metadata target must be TrendCategory7d")
    return {"predict_function":prediction_function,"metadata":metadata,
            "class_labels":CLASS_LABELS,"version":metadata.get("created_at","unknown")}


async def predict(token:str,horizon:int,cfg:Settings,client:httpx.AsyncClient,artifact:dict)->dict:
    """Reuse a completed prediction before making any downstream requests.

    Cache state belongs to the loaded artifact, so each worker/model load starts
    fresh. Per-key locks let different coins run concurrently while identical
    requests share one successful calculation.
    """
    last_completed_day = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    cache_key = (cfg.app_profile, token, horizon, cfg.history_days, last_completed_day, artifact["version"])
    cache = artifact.setdefault("prediction_cache", {})
    locks = artifact.setdefault("prediction_cache_locks", {})
    now = time.monotonic()
    for key, entry in list(cache.items()):
        if key[4] != last_completed_day or now >= entry["expires_at"]:
            del cache[key]
    for key, lock in list(locks.items()):
        if key[4] != last_completed_day and not lock.locked():
            del locks[key]
    lock = locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = cache.get(cache_key)
        if cached and time.monotonic() < cached["expires_at"]:
            logger.info("Prediction cache HIT: token=%s, date=%s", token, last_completed_day)
            return copy.deepcopy(cached["data"])
        logger.info("Prediction cache MISS: token=%s, date=%s", token, last_completed_day)
        result, complete_features = await _predict_uncached(token, horizon, cfg, client, artifact)
        current_day = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        if complete_features and result["date"] == last_completed_day == current_day:
            cache[cache_key] = {
                "data": copy.deepcopy(result),
                "expires_at": time.monotonic() + PREDICTION_CACHE_TTL_SECONDS
            }
            logger.info("Prediction cache UPDATED: token=%s, date=%s", token, last_completed_day)
        else:
            logger.info("Prediction cache SKIP: token=%s; missing features or UTC date changed", token)
        return result


async def _predict_uncached(token:str,horizon:int,cfg:Settings,client:httpx.AsyncClient,artifact:dict)->tuple[dict,bool]:
    start = time.perf_counter()
    logger.info("Prediction started: token=%s, horizon=%d", token, horizon)
    market_task=fetch_market_data(token,cfg,client)
    external_task=fetch_remote_features(token,horizon,cfg,client)
    market, external=await asyncio.gather(market_task,external_task)
    logger.info("Prediction dependencies resolved: market_rows=%d, remote_features=%d",
                len(market), sum(len(values) for values in external.values()))
    served=await run_in_threadpool(
        artifact["predict_function"], market_df=pd.DataFrame(market),
        student_a=external["A"], student_b=external["B"], student_d=external["D"], coin=token,
    )
    prediction=served["predicted_trend"]
    labels=artifact["class_labels"]
    if prediction not in labels:
        raise ValueError("Model returned an unknown trend class")
    probabilities=served["probabilities"]
    if (set(probabilities) != set(labels)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()),1.0,abs_tol=1e-5)):
        raise ValueError("Model returned invalid class probabilities")
    if (served["coin"] != token or served["date"] != market[-1]["date"]
            or served["forecast_horizon_days"] != horizon):
        raise ValueError("Model returned inconsistent prediction identifiers")
    confidence=float(probabilities[prediction])
    result={**served,"prediction":prediction,"confidence":confidence,"forecast_horizon":horizon,
            "prediction_date":(date.fromisoformat(served["date"])+timedelta(days=horizon)).isoformat()}
    if cfg.app_profile != "prod": result["development_only"]=True
    logger.info("Prediction completed: token=%s, prediction=%s, confidence=%.4f, duration_ms=%.2f", token, prediction, confidence, (time.perf_counter() - start) * 1000)
    complete_features = all(
        feature in external[student]
        for student, features in STUDENT_FEATURES.items()
        for feature in features
    )
    return result, complete_features
