#!/usr/bin/env python3
"""
BTC AI Analyst — fetches OHLCV data from Binance public API,
computes TA indicators, asks Claude for a structured buy/sell/hold signal,
and writes it to ledger.json.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import anthropic
import pandas as pd
import requests
import ta as ta_lib
from dotenv import load_dotenv

load_dotenv()

LEDGER_PATH = os.path.join(os.path.dirname(__file__), "ledger.json")
BINANCE_BASE = "https://api.binance.com/api/v3"


def fetch_klines(interval: str = "4h", limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance public API (no auth required)."""
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base", "taker_buy_quote", "unused",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_current_price() -> float:
    """Fetch current BTC/USDT spot price from Binance."""
    url = f"{BINANCE_BASE}/ticker/price"
    params = {"symbol": "BTCUSDT"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["price"])


def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute RSI, MACD, Bollinger Bands, EMAs using the `ta` library."""
    close = df["close"]

    rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
    macd_obj = ta_lib.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
    ema9 = ta_lib.trend.EMAIndicator(close, window=9).ema_indicator()
    ema21 = ta_lib.trend.EMAIndicator(close, window=21).ema_indicator()

    return {
        "rsi": round(float(rsi.iloc[-1]), 2),
        "macd_value": round(float(macd_obj.macd().iloc[-1]), 2),
        "macd_signal": round(float(macd_obj.macd_signal().iloc[-1]), 2),
        "macd_hist": round(float(macd_obj.macd_diff().iloc[-1]), 2),
        "bb_upper": round(float(bb.bollinger_hband().iloc[-1]), 2),
        "bb_mid": round(float(bb.bollinger_mavg().iloc[-1]), 2),
        "bb_lower": round(float(bb.bollinger_lband().iloc[-1]), 2),
        "ema9": round(float(ema9.iloc[-1]), 2),
        "ema21": round(float(ema21.iloc[-1]), 2),
    }


def load_ledger() -> dict:
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {
        "capital": 10000.0,
        "position": None,
        "trades": [],
        "last_signal": None,
    }


def save_ledger(ledger: dict):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def build_prompt(price: float, df_4h: pd.DataFrame, df_1d: pd.DataFrame,
                 indicators_4h: dict, indicators_1d: dict, vol_ratio: float, position) -> str:
    last_7_daily = df_1d.tail(7)[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    last_7_daily["timestamp"] = last_7_daily["timestamp"].dt.strftime("%Y-%m-%d")
    daily_str = last_7_daily.to_string(index=False)

    last_12_4h = df_4h.tail(12)[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    last_12_4h["timestamp"] = last_12_4h["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    h4_str = last_12_4h.to_string(index=False)

    pos_str = "None (flat)"
    if position:
        entry = position.get("entry_price", 0)
        unreal = round((price - entry) / entry * 100, 2)
        pos_str = f"LONG @ ${entry:,.2f} (unrealized: {unreal:+.2f}%)"

    return f"""You are a professional crypto technical analyst. Analyze the following BTC/USDT market data and provide a trading signal for the next 1-3 days.

CURRENT PRICE: ${price:,.2f}
TIMESTAMP: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

LAST 7 DAILY CANDLES:
{daily_str}

LAST 12 FOUR-HOUR CANDLES:
{h4_str}

DAILY TIMEFRAME INDICATORS:
- RSI(14): {indicators_1d['rsi']} {"[OVERSOLD]" if indicators_1d['rsi'] < 30 else "[OVERBOUGHT]" if indicators_1d['rsi'] > 70 else ""}
- MACD: {indicators_1d['macd_value']}, Signal: {indicators_1d['macd_signal']}, Histogram: {indicators_1d['macd_hist']}
- Bollinger Bands: Upper={indicators_1d['bb_upper']:,.2f}, Mid={indicators_1d['bb_mid']:,.2f}, Lower={indicators_1d['bb_lower']:,.2f}
- EMA9: {indicators_1d['ema9']:,.2f}, EMA21: {indicators_1d['ema21']:,.2f} {"[EMA BULLISH]" if indicators_1d['ema9'] > indicators_1d['ema21'] else "[EMA BEARISH]"}

4-HOUR TIMEFRAME INDICATORS:
- RSI(14): {indicators_4h['rsi']} {"[OVERSOLD]" if indicators_4h['rsi'] < 30 else "[OVERBOUGHT]" if indicators_4h['rsi'] > 70 else ""}
- MACD: {indicators_4h['macd_value']}, Signal: {indicators_4h['macd_signal']}, Histogram: {indicators_4h['macd_hist']} {"[BULLISH CROSS]" if indicators_4h['macd_hist'] > 0 else "[BEARISH CROSS]"}
- Bollinger Bands: Upper={indicators_4h['bb_upper']:,.2f}, Mid={indicators_4h['bb_mid']:,.2f}, Lower={indicators_4h['bb_lower']:,.2f}
- EMA9: {indicators_4h['ema9']:,.2f}, EMA21: {indicators_4h['ema21']:,.2f} {"[EMA BULLISH]" if indicators_4h['ema9'] > indicators_4h['ema21'] else "[EMA BEARISH]"}
- Volume vs 20-period average: {vol_ratio:.2f}x

CURRENT PAPER POSITION: {pos_str}

Based on the above data, provide your analysis. Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
{{
  "signal": "BUY" or "SELL" or "HOLD",
  "confidence": integer 0-100,
  "reasoning": "2-3 sentence technical explanation citing specific indicator values",
  "target_1d_pct": float (expected % change in 1 day, e.g. 2.5 or -1.8),
  "target_2d_pct": float (expected % change in 2 days),
  "target_3d_pct": float (expected % change in 3 days),
  "key_level_support": float (nearest support price),
  "key_level_resistance": float (nearest resistance price),
  "risk": "LOW" or "MEDIUM" or "HIGH"
}}"""


def call_claude(prompt: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def main():
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] BTC Analyst running...")

    print("Fetching 4h klines (100 candles = ~16 days)...")
    df_4h = fetch_klines(interval="4h", limit=100)

    print("Fetching daily klines (60 candles)...")
    df_1d = fetch_klines(interval="1d", limit=60)

    print("Fetching current price...")
    price = fetch_current_price()

    vol_sma20 = df_4h["volume"].rolling(20).mean().iloc[-1]
    vol_ratio = df_4h["volume"].iloc[-1] / vol_sma20 if vol_sma20 > 0 else 1.0

    print("Computing indicators...")
    indicators_4h = compute_indicators(df_4h)
    indicators_1d = compute_indicators(df_1d)

    ledger = load_ledger()
    position = ledger.get("position")

    print("Calling Claude for analysis...")
    prompt = build_prompt(price, df_4h, df_1d, indicators_4h, indicators_1d, vol_ratio, position)
    signal = call_claude(prompt)
    signal["price_at_signal"] = price
    signal["timestamp"] = datetime.now(timezone.utc).isoformat()

    ledger["last_signal"] = signal
    save_ledger(ledger)

    print("\n--- SIGNAL ---")
    print(json.dumps(signal, indent=2))
    print("--------------\n")

    return signal


if __name__ == "__main__":
    main()
