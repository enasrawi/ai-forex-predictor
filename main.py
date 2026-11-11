from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime, timezone
import os

app = FastAPI(title="AI Forex Predictor")
PAIRS = {"AUD/USD": "AUDUSD=X", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"}
HORIZONS = {"15m": 3, "30m": 6, "1h": 12}

def forecast(symbol: str, steps: int):
    data = yf.download(symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    if data is None or data.empty or len(data) < 80:
        raise ValueError("Not enough recent market data is available.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    frame = pd.DataFrame({"close": close})
    for lag in range(1, 13):
        frame[f"ret_{lag}"] = frame["close"].pct_change(lag)
    frame["ma_5"] = frame["close"].rolling(5).mean() / frame["close"] - 1
    frame["ma_20"] = frame["close"].rolling(20).mean() / frame["close"] - 1
    frame["vol_12"] = frame["close"].pct_change().rolling(12).std()
    frame["target"] = frame["close"].shift(-steps) / frame["close"] - 1
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna()
    features = [c for c in clean.columns if c not in ("close", "target")]
    if len(clean) < 50:
        raise ValueError("Not enough clean data to build a forecast.")
    split = max(30, int(len(clean) * 0.8))
    model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.06, max_leaf_nodes=15, random_state=7)
    model.fit(clean[features].iloc[:split], clean["target"].iloc[:split])
    predicted_return = float(model.predict(clean[features].iloc[[-1]])[0])
    last_price = float(close.iloc[-1])
    predicted_price = last_price * (1 + predicted_return)
    direction = "UP" if predicted_return > 0.00015 else "DOWN" if predicted_return < -0.00015 else "SIDEWAYS"
    return {
        "price": last_price,
        "predicted_price": predicted_price,
        "change_pct": predicted_return * 100,
        "direction": direction,
        "data_time": close.index[-1].isoformat(),
        "observations": int(len(clean))
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/predict")
def predict(pair: str = "AUD/USD", horizon: str = "15m"):
    if pair not in PAIRS or horizon not in HORIZONS:
        return {"error": "Unsupported currency pair or forecast horizon."}
    try:
        result = forecast(PAIRS[pair], HORIZONS[horizon])
        return {"pair": pair, "horizon": horizon, **result}
    except Exception as exc:
        return {"error": str(exc), "pair": pair, "horizon": horizon}

@app.get("/")
def home():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")
