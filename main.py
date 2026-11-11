from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from threading import Lock
import time

app = FastAPI(title="AI Forex Predictor")
PAIRS = {"AUD/USD": "AUDUSD=X", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"}
HORIZONS = {"15m": 3, "30m": 6, "1h": 12}

# Reuse market data instead of repeatedly calling Yahoo Finance.
DATA_CACHE = {}
CACHE_LOCK = Lock()
CACHE_SECONDS = 300
STALE_SECONDS = 21600

def get_market_data(symbol: str):
    now = time.time()
    with CACHE_LOCK:
        cached = DATA_CACHE.get(symbol)
        if cached and now - cached["time"] < CACHE_SECONDS:
            return cached["data"], False

    try:
        data = yf.download(
            symbol, period="5d", interval="5m", progress=False,
            auto_adjust=True, threads=False, timeout=20
        )
        if data is None or data.empty:
            raise ValueError("The market-data provider returned no prices.")
        with CACHE_LOCK:
            DATA_CACHE[symbol] = {"time": now, "data": data}
        return data, False
    except Exception as exc:
        with CACHE_LOCK:
            cached = DATA_CACHE.get(symbol)
        if cached and now - cached["time"] < STALE_SECONDS:
            return cached["data"], True
        message = str(exc)
        if "rate" in message.lower() or "too many requests" in message.lower():
            raise ValueError("Yahoo Finance is temporarily rate-limiting requests. Please wait 10–15 minutes and try again.")
        raise ValueError("Unable to retrieve market data right now. Please try again later.")

def forecast(symbol: str, steps: int):
    data, using_cached_data = get_market_data(symbol)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    if len(close) < 80:
        raise ValueError("Not enough recent market data is available.")

    features = pd.DataFrame(index=close.index)
    for lag in range(1, 13):
        features[f"ret_{lag}"] = close.pct_change(lag, fill_method=None)
    features["ma_5"] = close.rolling(5).mean() / close - 1
    features["ma_20"] = close.rolling(20).mean() / close - 1
    features["vol_12"] = close.pct_change(fill_method=None).rolling(12).std()

    target = close.shift(-steps) / close - 1
    training = features.join(target.rename("target")).replace([np.inf, -np.inf], np.nan).dropna()
    latest = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(training) < 50 or latest.empty:
        raise ValueError("Not enough clean market data to build a forecast.")

    # Small ridge-regression model avoids heavy compiled ML dependencies.
    x = training.drop(columns=["target"]).to_numpy(dtype=float)
    y = training["target"].to_numpy(dtype=float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    x = (x - x_mean) / x_std
    x = np.column_stack([np.ones(len(x)), x])
    alpha = 1.0
    coefficients = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)

    latest_x = latest.iloc[[-1]].to_numpy(dtype=float)
    latest_x = (latest_x - x_mean) / x_std
    latest_x = np.column_stack([np.ones(len(latest_x)), latest_x])
    predicted_return = float((latest_x @ coefficients)[0])
    last_price = float(close.iloc[-1])
    predicted_price = last_price * (1 + predicted_return)
    direction = "UP" if predicted_return > 0.00015 else "DOWN" if predicted_return < -0.00015 else "SIDEWAYS"

    return {
        "price": last_price,
        "predicted_price": predicted_price,
        "change_pct": predicted_return * 100,
        "direction": direction,
        "data_time": close.index[-1].isoformat(),
        "observations": int(len(training)),
        "cached_data": using_cached_data
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/predict")
def predict(pair: str = "AUD/USD", horizon: str = "15m"):
    if pair not in PAIRS or horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="Unsupported currency pair or forecast horizon.")
    try:
        result = forecast(PAIRS[pair], HORIZONS[horizon])
        return {"pair": pair, "horizon": horizon, **result}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="The forecast service encountered an unexpected error. Please try again shortly.")

@app.get("/")
def home():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")
