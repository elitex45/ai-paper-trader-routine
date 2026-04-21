#!/usr/bin/env python3
"""
Paper Trade Engine — reads the latest signal from ledger.json,
executes buy/sell/stop-loss/take-profit logic, and updates the ledger.
"""

import json
import os
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

LEDGER_PATH = os.path.join(os.path.dirname(__file__), "ledger.json")
BINANCE_BASE = "https://api.binance.com/api/v3"

POSITION_SIZE_PCT = 0.50   # Use 50% of capital per trade
BUY_CONFIDENCE_MIN = 65    # Minimum confidence to open a long
SELL_CONFIDENCE_MIN = 60   # Minimum confidence to close on SELL signal

RR_TIERS = [
    (85, 1.0),   # 85%+ confidence -> minimum 1:1 R:R
    (75, 1.5),   # 75-84% confidence -> minimum 1.5:1 R:R
    (0,  2.0),   # below 75% -> minimum 2:1 R:R
]


def min_rr_for_confidence(confidence: int) -> float:
    for threshold, min_rr in RR_TIERS:
        if confidence >= threshold:
            return min_rr
    return RR_TIERS[-1][1]


def fetch_current_price() -> float:
    url = f"{BINANCE_BASE}/ticker/price"
    params = {"symbol": "BTCUSDT"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["price"])


def load_ledger() -> dict:
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {"capital": 10000.0, "position": None, "trades": [], "last_signal": None}


def save_ledger(ledger: dict):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def compute_rr(price: float, signal: dict) -> tuple[float, float, float, float]:
    """Compute risk, reward, and R:R ratio from signal's support/resistance levels.
    Returns (risk_per_unit, reward_per_unit, rr_ratio, stop_loss, take_profit)."""
    support = signal.get("key_level_support", 0)
    resistance = signal.get("key_level_resistance", 0)

    risk = price - support
    reward = resistance - price

    if risk <= 0 or reward <= 0:
        return 0, 0, 0, 0, 0

    rr_ratio = reward / risk
    stop_loss = round(support, 2)
    take_profit = round(resistance, 2)
    return risk, reward, rr_ratio, stop_loss, take_profit


def open_position(ledger: dict, price: float, signal: dict, stop_loss: float, take_profit: float):
    capital = ledger["capital"]
    size_usd = round(capital * POSITION_SIZE_PCT, 2)
    btc_amount = round(size_usd / price, 8)
    trade_id = len(ledger["trades"]) + 1

    position = {
        "entry_price": price,
        "size_usd": size_usd,
        "btc_amount": btc_amount,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "stop_loss_price": stop_loss,
        "take_profit_price": take_profit,
        "trade_id": trade_id,
    }

    ledger["capital"] -= size_usd
    ledger["position"] = position

    buy_record = {
        "id": trade_id,
        "type": "BUY",
        "price": price,
        "size_usd": size_usd,
        "btc_amount": btc_amount,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_confidence": signal.get("confidence"),
        "reasoning": signal.get("reasoning"),
        "pnl": None,
    }
    ledger["trades"].append(buy_record)

    print(f"  OPENED LONG: {btc_amount} BTC @ ${price:,.2f} (${size_usd:,.2f})")
    print(f"  Stop-loss: ${stop_loss:,.2f} | Take-profit: ${take_profit:,.2f}")


def close_position(ledger: dict, price: float, reason: str, signal: dict | None = None):
    position = ledger["position"]
    if not position:
        return

    proceeds = round(position["btc_amount"] * price, 2)
    cost = position["size_usd"]
    pnl = round(proceeds - cost, 2)
    pnl_pct = round((proceeds - cost) / cost * 100, 2)

    ledger["capital"] += proceeds
    ledger["position"] = None

    trade_id = len(ledger["trades"]) + 1
    sell_record = {
        "id": trade_id,
        "type": "SELL",
        "reason": reason,
        "price": price,
        "size_usd": proceeds,
        "btc_amount": position["btc_amount"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_confidence": signal.get("confidence") if signal else None,
        "reasoning": signal.get("reasoning") if signal else reason,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "entry_price": position["entry_price"],
        "entry_trade_id": position["trade_id"],
    }
    ledger["trades"].append(sell_record)

    emoji = "PROFIT" if pnl >= 0 else "LOSS"
    print(f"  CLOSED ({reason}): {position['btc_amount']} BTC @ ${price:,.2f}")
    print(f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%) — {emoji}")
    print(f"  New capital: ${ledger['capital']:,.2f}")


def main():
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] Trade engine running...")

    ledger = load_ledger()
    signal = ledger.get("last_signal")

    if not signal:
        print("No signal found in ledger. Run analyze.py first.")
        sys.exit(1)

    price = fetch_current_price()
    sig_price = signal.get("price_at_signal", price)

    # Use the price from the signal (fresher than refetching for consistency)
    # but also check current price for stop/take-profit
    position = ledger.get("position")

    print(f"  Current price: ${price:,.2f}")
    print(f"  Signal: {signal['signal']} (confidence: {signal['confidence']}%)")
    print(f"  Capital: ${ledger['capital']:,.2f} | Position: {'LONG' if position else 'None'}")

    action_taken = "HOLD"

    if position:
        entry = position["entry_price"]
        change_pct = (price - entry) / entry

        # Check stop-loss
        if price <= position["stop_loss_price"]:
            print(f"\n  STOP-LOSS triggered ({change_pct*100:+.2f}% from entry)")
            close_position(ledger, price, "STOP_LOSS")
            action_taken = "STOP_LOSS"

        # Check take-profit
        elif price >= position["take_profit_price"]:
            print(f"\n  TAKE-PROFIT triggered ({change_pct*100:+.2f}% from entry)")
            close_position(ledger, price, "TAKE_PROFIT")
            action_taken = "TAKE_PROFIT"

        # Sell signal with sufficient confidence
        elif signal["signal"] == "SELL" and signal["confidence"] >= SELL_CONFIDENCE_MIN:
            print(f"\n  SELL signal ({signal['confidence']}% confidence)")
            close_position(ledger, price, "SIGNAL_SELL", signal)
            action_taken = "SIGNAL_SELL"

        else:
            print(f"\n  Holding position (change: {change_pct*100:+.2f}% from entry)")

    else:
        # Buy signal with sufficient confidence
        if signal["signal"] == "BUY" and signal["confidence"] >= BUY_CONFIDENCE_MIN:
            if ledger["capital"] < 100:
                print("\n  Insufficient capital to open position.")
            else:
                risk, reward, rr_ratio, sl, tp = compute_rr(price, signal)
                required_rr = min_rr_for_confidence(signal["confidence"])
                print(f"\n  BUY signal ({signal['confidence']}% confidence)")
                print(f"  R:R check — risk: ${risk:,.2f}, reward: ${reward:,.2f}, ratio: {rr_ratio:.2f}:1 (min {required_rr:.1f}:1 for {signal['confidence']}% conf)")
                if rr_ratio >= required_rr:
                    open_position(ledger, price, signal, sl, tp)
                    action_taken = "BUY"
                else:
                    print(f"  REJECTED — R:R {rr_ratio:.2f}:1 below minimum {required_rr:.1f}:1, skipping trade")
                    action_taken = "RR_REJECTED"
        else:
            reason = f"signal={signal['signal']}, confidence={signal['confidence']}% (min {BUY_CONFIDENCE_MIN}%)"
            print(f"\n  No trade — {reason}")

    signal["last_trade_action"] = action_taken
    ledger["last_signal"] = signal
    save_ledger(ledger)
    print(f"\n  Action: {action_taken} | Ledger saved.")


if __name__ == "__main__":
    main()
