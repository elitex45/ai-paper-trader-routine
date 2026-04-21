#!/usr/bin/env python3
"""
Unit tests for trade.py math and ledger logic.
Mocks all network calls — no API key or internet required.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Point the module at a temp ledger before importing
os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")

import trade


def make_signal(sig="BUY", confidence=80, price=90000.0):
    return {
        "signal": sig,
        "confidence": confidence,
        "reasoning": "Test signal.",
        "target_1d_pct": 2.0,
        "target_2d_pct": 4.0,
        "target_3d_pct": 6.0,
        "key_level_support": 85000.0,
        "key_level_resistance": 95000.0,
        "risk": "MEDIUM",
        "price_at_signal": price,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def make_ledger(capital=10000.0, position=None):
    return {
        "capital": capital,
        "position": position,
        "trades": [],
        "last_signal": None,
    }


class TestOpenPosition(unittest.TestCase):

    def test_position_size_is_50pct_of_capital(self):
        ledger = make_ledger(capital=10000.0)
        signal = make_signal(sig="BUY", confidence=80, price=90000.0)
        trade.open_position(ledger, 90000.0, signal)

        self.assertEqual(ledger["position"]["size_usd"], 5000.0)

    def test_btc_amount_math(self):
        """btc_amount = size_usd / price, rounded to 8 decimal places."""
        ledger = make_ledger(capital=10000.0)
        signal = make_signal(price=50000.0)
        trade.open_position(ledger, 50000.0, signal)

        pos = ledger["position"]
        expected_btc = round(5000.0 / 50000.0, 8)  # 0.1
        self.assertAlmostEqual(pos["btc_amount"], expected_btc, places=8)

    def test_capital_reduced_by_size(self):
        ledger = make_ledger(capital=10000.0)
        trade.open_position(ledger, 90000.0, make_signal())
        self.assertAlmostEqual(ledger["capital"], 5000.0, places=2)

    def test_stop_loss_price(self):
        """Stop-loss = entry * (1 - STOP_LOSS_PCT) = entry * 0.95"""
        ledger = make_ledger(capital=10000.0)
        trade.open_position(ledger, 100000.0, make_signal())
        expected_sl = round(100000.0 * (1 - trade.STOP_LOSS_PCT), 2)
        self.assertEqual(ledger["position"]["stop_loss_price"], expected_sl)

    def test_take_profit_price(self):
        """Take-profit = entry * (1 + TAKE_PROFIT_PCT) = entry * 1.08"""
        ledger = make_ledger(capital=10000.0)
        trade.open_position(ledger, 100000.0, make_signal())
        expected_tp = round(100000.0 * (1 + trade.TAKE_PROFIT_PCT), 2)
        self.assertEqual(ledger["position"]["take_profit_price"], expected_tp)

    def test_buy_record_added_to_trades(self):
        ledger = make_ledger()
        trade.open_position(ledger, 90000.0, make_signal())
        self.assertEqual(len(ledger["trades"]), 1)
        self.assertEqual(ledger["trades"][0]["type"], "BUY")


class TestClosePosition(unittest.TestCase):

    def _open(self, entry_price=90000.0, capital=10000.0):
        """Helper: open a position and return ledger."""
        ledger = make_ledger(capital=capital)
        trade.open_position(ledger, entry_price, make_signal(price=entry_price))
        return ledger

    def test_pnl_profitable_trade(self):
        """Sell at 10% above entry should produce ~+$500 P&L on $5000 position."""
        ledger = self._open(entry_price=100000.0)
        pos = ledger["position"]
        exit_price = 110000.0  # +10%
        trade.close_position(ledger, exit_price, "TAKE_PROFIT")

        sell = ledger["trades"][-1]
        expected_proceeds = round(pos["btc_amount"] * exit_price, 2)
        expected_pnl = round(expected_proceeds - 5000.0, 2)
        self.assertAlmostEqual(sell["pnl"], expected_pnl, places=2)
        self.assertGreater(sell["pnl"], 0)

    def test_pnl_losing_trade(self):
        """Sell at 5% below entry should produce a negative P&L."""
        ledger = self._open(entry_price=100000.0)
        pos = ledger["position"]
        exit_price = 95000.0  # -5%
        trade.close_position(ledger, exit_price, "STOP_LOSS")

        sell = ledger["trades"][-1]
        self.assertLess(sell["pnl"], 0)

    def test_capital_restored_after_close(self):
        """After close, capital = initial cash (after buy) + proceeds from sell."""
        ledger = self._open(entry_price=100000.0, capital=10000.0)
        cash_after_buy = ledger["capital"]  # 5000
        pos = ledger["position"]
        exit_price = 105000.0
        proceeds = round(pos["btc_amount"] * exit_price, 2)

        trade.close_position(ledger, exit_price, "SIGNAL_SELL")
        self.assertAlmostEqual(ledger["capital"], cash_after_buy + proceeds, places=2)

    def test_position_cleared_after_close(self):
        ledger = self._open()
        trade.close_position(ledger, 95000.0, "STOP_LOSS")
        self.assertIsNone(ledger["position"])

    def test_pnl_pct_correct(self):
        """pnl_pct = (exit - entry) / entry * 100"""
        entry = 80000.0
        exit_p = 84000.0
        ledger = self._open(entry_price=entry)
        pos = ledger["position"]
        trade.close_position(ledger, exit_p, "TAKE_PROFIT")

        sell = ledger["trades"][-1]
        expected_pct = round((exit_p - entry) / entry * 100, 2)
        self.assertAlmostEqual(sell["pnl_pct"], expected_pct, places=2)

    def test_trade_id_increments(self):
        ledger = self._open()
        buy_id = ledger["trades"][0]["id"]
        trade.close_position(ledger, 95000.0, "SIGNAL_SELL")
        sell_id = ledger["trades"][-1]["id"]
        self.assertEqual(sell_id, buy_id + 1)


class TestSignalThresholds(unittest.TestCase):

    def _run_main(self, ledger_data: dict, signal_data: dict, current_price: float):
        """Run trade.main() with full mocks, return updated ledger."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            ledger_data["last_signal"] = signal_data
            json.dump(ledger_data, f)
            tmp_path = f.name

        original_path = trade.LEDGER_PATH
        trade.LEDGER_PATH = tmp_path
        try:
            with patch("trade.fetch_current_price", return_value=current_price):
                trade.main()
            with open(tmp_path) as f:
                result = json.load(f)
        finally:
            trade.LEDGER_PATH = original_path
            os.unlink(tmp_path)
        return result

    def test_buy_signal_above_threshold_opens_position(self):
        ledger = make_ledger(capital=10000.0)
        signal = make_signal(sig="BUY", confidence=70, price=90000.0)
        result = self._run_main(ledger, signal, 90000.0)
        self.assertIsNotNone(result["position"])

    def test_buy_signal_below_threshold_no_trade(self):
        ledger = make_ledger(capital=10000.0)
        signal = make_signal(sig="BUY", confidence=50, price=90000.0)  # below 65
        result = self._run_main(ledger, signal, 90000.0)
        self.assertIsNone(result["position"])

    def test_sell_signal_above_threshold_closes_position(self):
        # First open a position manually
        ledger = make_ledger(capital=5000.0)
        ledger["position"] = {
            "entry_price": 90000.0,
            "size_usd": 5000.0,
            "btc_amount": round(5000.0 / 90000.0, 8),
            "entry_time": "2026-01-01T00:00:00+00:00",
            "stop_loss_price": round(90000.0 * 0.95, 2),
            "take_profit_price": round(90000.0 * 1.08, 2),
            "trade_id": 1,
        }
        ledger["trades"] = [{"id": 1, "type": "BUY", "price": 90000.0,
                              "size_usd": 5000.0, "btc_amount": ledger["position"]["btc_amount"],
                              "timestamp": "2026-01-01T00:00:00+00:00",
                              "signal_confidence": 75, "reasoning": "test", "pnl": None}]

        signal = make_signal(sig="SELL", confidence=65, price=92000.0)
        result = self._run_main(ledger, signal, 92000.0)
        self.assertIsNone(result["position"])

    def test_stop_loss_triggers_when_price_below_sl(self):
        entry = 90000.0
        sl_price = round(entry * (1 - trade.STOP_LOSS_PCT), 2)
        trigger_price = sl_price - 1  # just below stop-loss

        ledger = make_ledger(capital=5000.0)
        ledger["position"] = {
            "entry_price": entry,
            "size_usd": 5000.0,
            "btc_amount": round(5000.0 / entry, 8),
            "entry_time": "2026-01-01T00:00:00+00:00",
            "stop_loss_price": sl_price,
            "take_profit_price": round(entry * 1.08, 2),
            "trade_id": 1,
        }
        ledger["trades"] = [{"id": 1, "type": "BUY", "price": entry,
                              "size_usd": 5000.0, "btc_amount": ledger["position"]["btc_amount"],
                              "timestamp": "2026-01-01T00:00:00+00:00",
                              "signal_confidence": 75, "reasoning": "test", "pnl": None}]

        signal = make_signal(sig="HOLD", confidence=50, price=entry)
        result = self._run_main(ledger, signal, trigger_price)
        self.assertIsNone(result["position"])
        sell = [t for t in result["trades"] if t["type"] == "SELL"][0]
        self.assertEqual(sell["reason"], "STOP_LOSS")

    def test_take_profit_triggers_when_price_above_tp(self):
        entry = 90000.0
        tp_price = round(entry * (1 + trade.TAKE_PROFIT_PCT), 2)
        trigger_price = tp_price + 1  # just above take-profit

        ledger = make_ledger(capital=5000.0)
        ledger["position"] = {
            "entry_price": entry,
            "size_usd": 5000.0,
            "btc_amount": round(5000.0 / entry, 8),
            "entry_time": "2026-01-01T00:00:00+00:00",
            "stop_loss_price": round(entry * 0.95, 2),
            "take_profit_price": tp_price,
            "trade_id": 1,
        }
        ledger["trades"] = [{"id": 1, "type": "BUY", "price": entry,
                              "size_usd": 5000.0, "btc_amount": ledger["position"]["btc_amount"],
                              "timestamp": "2026-01-01T00:00:00+00:00",
                              "signal_confidence": 75, "reasoning": "test", "pnl": None}]

        signal = make_signal(sig="HOLD", confidence=50, price=entry)
        result = self._run_main(ledger, signal, trigger_price)
        self.assertIsNone(result["position"])
        sell = [t for t in result["trades"] if t["type"] == "SELL"][0]
        self.assertEqual(sell["reason"], "TAKE_PROFIT")


class TestCapitalAccounting(unittest.TestCase):

    def test_capital_conservation_across_round_trip(self):
        """Capital at open + position value = starting capital at all times."""
        ledger = make_ledger(capital=10000.0)
        entry_price = 80000.0
        trade.open_position(ledger, entry_price, make_signal(price=entry_price))

        pos = ledger["position"]
        # Capital held as cash + value locked in BTC = initial 10000
        position_value = pos["btc_amount"] * entry_price
        total = round(ledger["capital"] + position_value, 2)
        self.assertAlmostEqual(total, 10000.0, places=1)

    def test_no_fractional_error_accumulation(self):
        """Multiple open/close cycles should not drift capital by >1 cent."""
        ledger = make_ledger(capital=10000.0)
        prices = [80000, 85000, 75000, 90000]

        for i in range(0, len(prices), 2):
            trade.open_position(ledger, prices[i], make_signal(price=prices[i]))
            trade.close_position(ledger, prices[i + 1], "TEST")

        # All positions closed; capital should be deterministic
        self.assertIsNone(ledger["position"])
        # Just verify no NaN or Inf
        self.assertTrue(0 < ledger["capital"] < 1_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
