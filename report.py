#!/usr/bin/env python3
"""
P&L Report — reads ledger.json and prints a full trading summary.
"""

import json
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

LEDGER_PATH = os.path.join(os.path.dirname(__file__), "ledger.json")
STARTING_CAPITAL = 10000.0


def fetch_current_price() -> float:
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        resp = requests.get(url, params={"symbol": "BTCUSDT"}, timeout=8)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        return None


def load_ledger() -> dict:
    if not os.path.exists(LEDGER_PATH):
        print("No ledger.json found. Run analyze.py and trade.py first.")
        return None
    with open(LEDGER_PATH) as f:
        return json.load(f)


def bar(pct: float, width: int = 20) -> str:
    filled = int(abs(pct) / 100 * width)
    filled = min(filled, width)
    return ("+" if pct >= 0 else "-") + "#" * filled + "." * (width - filled)


def main():
    ledger = load_ledger()
    if not ledger:
        return

    price = fetch_current_price()

    capital = ledger["capital"]
    position = ledger.get("position")
    trades = ledger.get("trades", [])
    last_signal = ledger.get("last_signal")

    # Closed trades only
    closed = [t for t in trades if t["type"] == "SELL" and t.get("pnl") is not None]
    wins = [t for t in closed if t["pnl"] >= 0]
    losses = [t for t in closed if t["pnl"] < 0]

    total_realized_pnl = sum(t["pnl"] for t in closed)

    # Unrealized P&L
    unrealized_pnl = 0.0
    unrealized_pct = 0.0
    if position and price:
        unrealized_pnl = round((price - position["entry_price"]) * position["btc_amount"], 2)
        unrealized_pct = round((price - position["entry_price"]) / position["entry_price"] * 100, 2)

    total_value = capital + (position["btc_amount"] * price if position and price else 0)
    total_return = round((total_value - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2)

    print("\n" + "=" * 56)
    print("         BTC AI ANALYST — PAPER TRADING REPORT")
    print(f"         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 56)

    print(f"\n  Starting capital:  ${STARTING_CAPITAL:>10,.2f}")
    print(f"  Cash available:    ${capital:>10,.2f}")
    if position and price:
        btc_val = position["btc_amount"] * price
        print(f"  Position value:    ${btc_val:>10,.2f}  ({position['btc_amount']} BTC)")
    print(f"  Total portfolio:   ${total_value:>10,.2f}")
    print(f"  Total return:      {total_return:>+10.2f}%  {bar(total_return)}")

    if position:
        print(f"\n  OPEN POSITION:")
        print(f"    Entry: ${position['entry_price']:,.2f}  |  Current: ${price:,.2f}" if price else f"    Entry: ${position['entry_price']:,.2f}")
        print(f"    Unrealized P&L: ${unrealized_pnl:+,.2f} ({unrealized_pct:+.2f}%)")
        print(f"    Stop-loss:  ${position['stop_loss_price']:,.2f}")
        print(f"    Take-profit: ${position['take_profit_price']:,.2f}")
        print(f"    Opened: {position['entry_time'][:16].replace('T', ' ')} UTC")
    else:
        print("\n  OPEN POSITION: None (flat)")

    print(f"\n  CLOSED TRADES: {len(closed)}")
    print(f"    Realized P&L:   ${total_realized_pnl:+,.2f}")
    print(f"    Win rate:        {len(wins)}/{len(closed)} ({round(len(wins)/len(closed)*100) if closed else 0}%)")

    if closed:
        best = max(closed, key=lambda t: t["pnl"])
        worst = min(closed, key=lambda t: t["pnl"])
        print(f"    Best trade:     ${best['pnl']:+,.2f} ({best.get('pnl_pct', 0):+.2f}%)")
        print(f"    Worst trade:    ${worst['pnl']:+,.2f} ({worst.get('pnl_pct', 0):+.2f}%)")

    if last_signal:
        print(f"\n  LAST SIGNAL ({last_signal.get('timestamp', '')[:16].replace('T', ' ')} UTC):")
        print(f"    {last_signal['signal']} | Confidence: {last_signal['confidence']}% | Risk: {last_signal.get('risk', 'N/A')}")
        print(f"    {last_signal.get('reasoning', '')[:80]}...")
        print(f"    Targets — 1d: {last_signal.get('target_1d_pct', 0):+.1f}%  2d: {last_signal.get('target_2d_pct', 0):+.1f}%  3d: {last_signal.get('target_3d_pct', 0):+.1f}%")
        print(f"    Support: ${last_signal.get('key_level_support', 0):,.2f}  |  Resistance: ${last_signal.get('key_level_resistance', 0):,.2f}")

    if closed:
        print(f"\n  TRADE HISTORY:")
        for t in closed[-5:]:
            pnl_str = f"${t['pnl']:+,.2f} ({t.get('pnl_pct', 0):+.2f}%)"
            ts = t["timestamp"][:10]
            reason = t.get("reason", "SIGNAL")
            print(f"    [{ts}] SELL @ ${t['price']:,.2f} | {pnl_str} | {reason}")

    print("\n" + "=" * 56 + "\n")


if __name__ == "__main__":
    main()
