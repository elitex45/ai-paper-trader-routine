#!/usr/bin/env python3
"""
BTC AI Analyst — fetches OHLCV data, computes TA indicators,
asks Claude for a structured buy/sell/hold signal, and writes it to ledger.json.
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
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def fetch_ohlcv(days: int = 30) -> pd.DataFrame:
    """Fetch daily OHLCV from CoinGecko (no auth required)."""
    url = f"{COINGECKO_BASE}/coins/bitcoin/ohlc"
    params = {"vs_currency": "usd", "days": days}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_current_price() -> float:
    """Fetch current BTC spot price."""
    url = f"{COINGECKO_BASE}/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["bitcoin"]["usd"])


def fetch_volume_data(days: int = 30) -> pd.DataFrame:
    """Fetch daily volume via market_chart endpoint."""
    url = f"{COINGECKO_BASE}/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    volumes = data.get("total_volumes", [])
    df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


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


def build_prompt(price: float, df: pd.DataFrame, indicators: dict, vol_ratio: float, position) -> str:
    last_7 = df.tail(7)[["timestamp", "open", "high", "low", "close"]].copy()
    last_7["timestamp"] = last_7["timestamp"].dt.strftime("%Y-%m-%d")
    candles_str = last_7.to_string(index=False)

    pos_str = "None (flat)"
    if position:
        entry = position.get("entry_price", 0)
        unreal = round((price - entry) / entry * 100, 2)
        pos_str = f"LONG @ ${entry:,.2f} (unrealized: {unreal:+.2f}%)"

    return f"""You are a professional crypto technical analyst. Analyze the following BTC/USD market data and provide a trading signal for the next 1-3 days.

CURRENT PRICE: ${price:,.2f}
TIMESTAMP: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

LAST 7 DAILY CANDLES (OHLCV):
{candles_str}

CURRENT INDICATORS:
- RSI(14): {indicators['rsi']} {"[OVERSOLD]" if indicators['rsi'] < 30 else "[OVERBOUGHT]" if indicators['rsi'] > 70 else ""}
- MACD: {indicators['macd_value']}, Signal: {indicators['macd_signal']}, Histogram: {indicators['macd_hist']} {"[BULLISH CROSS]" if indicators['macd_hist'] > 0 else "[BEARISH CROSS]"}
- Bollinger Bands: Upper={indicators['bb_upper']:,.2f}, Mid={indicators['bb_mid']:,.2f}, Lower={indicators['bb_lower']:,.2f}
- EMA9: {indicators['ema9']:,.2f}, EMA21: {indicators['ema21']:,.2f} {"[EMA BULLISH]" if indicators['ema9'] > indicators['ema21'] else "[EMA BEARISH]"}
- Volume vs 20d average: {vol_ratio:.2f}x

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
    # Strip any accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def main():
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] BTC Analyst running...")

    # Fetch data with a small delay between calls to respect rate limits
    print("Fetching OHLCV data...")
    df = fetch_ohlcv(days=30)
    time.sleep(1)

    print("Fetching current price...")
    price = fetch_current_price()
    time.sleep(1)

    print("Fetching volume data...")
    vol_df = fetch_volume_data(days=21)

    # Merge volume into OHLCV df
    df = df.merge(vol_df, on="timestamp", how="left")
    df["volume"] = df["volume"].fillna(0)

    # Volume ratio: current vs 20-day avg
    vol_sma20 = df["volume"].rolling(20).mean().iloc[-1]
    vol_ratio = df["volume"].iloc[-1] / vol_sma20 if vol_sma20 > 0 else 1.0

    print("Computing indicators...")
    indicators = compute_indicators(df)

    ledger = load_ledger()
    position = ledger.get("position")

    print("Calling Claude for analysis...")
    prompt = build_prompt(price, df, indicators, vol_ratio, position)
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
