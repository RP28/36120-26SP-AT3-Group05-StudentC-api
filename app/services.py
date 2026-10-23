"""Student C indicators and saved-model inference orchestration."""
from datetime import date, timedelta
from pathlib import Path
import math
import pickle
import pandas as pd
import numpy as np
import httpx
import logging
import time
import asyncio
from starlette.concurrency import run_in_threadpool
from .adapters import fetch_market_data, fetch_remote_features
from .config import Settings, FEATURES_C

CLASS_LABELS = ["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]

logger = logging.getLogger(__name__)

class PlaceholderModel:
    """Development-only, deterministic stub; never market advice."""
    def predict(self, X):
        return np.asarray(["Sideways"] * len(X))
    def predict_proba(self, X):
        return np.tile(np.asarray([[.05,.10,.70,.10,.05]]),(len(X),1))


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
    if cfg.model_backend == "placeholder":
        return {"model":PlaceholderModel(),"preprocessor":None,"feature_columns":["MA5","MA20","Return20", "Volatility20"],"class_labels":CLASS_LABELS,"forecast_horizon":7,"version":"mock"}
    path=Path(cfg.model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {path}")
    # Only load trusted, locally-produced pickle files.
    with path.open("rb") as f:
        artifact=pickle.load(f)
    required={"model","feature_columns","class_labels","forecast_horizon"}
    if not isinstance(artifact,dict) or not required.issubset(artifact):
        raise ValueError(f"Invalid model artifact: missing {required-set(artifact) if isinstance(artifact,dict) else required}")
    if artifact["forecast_horizon"]!=7:
        raise ValueError("Model artifact forecast horizon must equal 7")
    return artifact


async def predict(token:str,horizon:int,cfg:Settings,client:httpx.AsyncClient,artifact:dict)->dict:
    # Market data and third-party features start together. Each market request
    # itself fans out concurrently into OHLC and market metrics.
    start = time.perf_counter()
    logger.info("Prediction started: token=%s, horizon=%d", token, horizon)

    market_task=fetch_market_data(token,cfg,client)
    external_task=fetch_remote_features(token,horizon,cfg,client)
    market, external=await asyncio.gather(market_task,external_task)
    logger.info("Prediction dependencies resolved: market_rows=%d, remote_features=%d", len(market), len(external))

    local_features=await run_in_threadpool(student_c_features, market)
    features={**external,**local_features}
    cols=list(artifact["feature_columns"])
    X=pd.DataFrame([{k:features.get(k,np.nan) for k in cols}],columns=cols)
    X=X.apply(pd.to_numeric,errors="coerce")
    if artifact.get("preprocessor") is not None:
        X=await run_in_threadpool(artifact["preprocessor"].transform, X)
    logger.info("Model input prepared: shape=%s, missing_features=%s", X.shape, X.columns[X.isna().any()].tolist())
    
    model=artifact["model"]
    prediction=(await run_in_threadpool(model.predict, X))[0]
    labels=artifact["class_labels"]
    if isinstance(prediction,(int,np.integer)):
        prediction=labels[int(prediction)]
    prediction=str(prediction)
    if prediction not in labels:
        raise ValueError("Model returned a class not included in artifact class_labels")
    if not hasattr(model,"predict_proba"):
        raise ValueError("Classification model must expose predict_proba")
    probs=(await run_in_threadpool(model.predict_proba, X))[0]
    classes=getattr(model,"classes_",None)
    if classes is None:
        confidence=float(probs[labels.index(prediction)])
    else:
        index=next((i for i,c in enumerate(classes) if (str(c)==prediction or (isinstance(c,(int,np.integer)) and labels[int(c)]==prediction))),None)
        if index is None: raise ValueError("Predicted class not present in model.classes_")
        confidence=float(probs[index])
    result={"prediction":prediction,"confidence":confidence,"forecast_horizon":7,
            "prediction_date":(date.fromisoformat(market[-1]["date"])+timedelta(days=7)).isoformat()}
    if cfg.model_backend=="placeholder": result["development_only"]=True
    logger.info("Prediction completed: token=%s, prediction=%s, confidence=%.4f, duration_ms=%.2f", token, prediction, confidence, (time.perf_counter() - start) * 1000)
    return result
