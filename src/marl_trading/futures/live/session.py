"""A live, controllable simulation session for the browser viewer.

The session owns a ``PerpetualMarket`` and advances it in a background thread
at ``speed`` simulated seconds per real second. All access goes through a lock
so the HTTP handlers always see a consistent market.
"""

from __future__ import annotations

import threading
import time as wallclock
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional

from ..analysis import market_summary, stylized_facts
from ..config import MarketConfig, format_time
from ..market import PerpetualMarket
from ..types import Side, Transparency
from .. import scenarios


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


class LiveSession:
    def __init__(self, scenario: str = "default", seed: int = 1, transparency: Optional[str] = None, speed: float = 10.0, duration: Optional[float] = None, config: Optional[MarketConfig] = None) -> None:
        self.lock = threading.RLock()
        self.speed = float(speed)
        self.playing = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._summary_cache: Dict[str, Any] = {}
        self._summary_time = -1.0
        self._custom_config = config
        self.scenario = scenario
        self.duration = duration
        self._build(scenario, seed, transparency)

    # ------------------------------------------------------------- lifecycle

    def _build(self, scenario: str, seed: int, transparency: Optional[str]) -> None:
        overrides: Dict[str, Any] = {"seed": int(seed)}
        if self.duration is not None:
            overrides["duration"] = float(self.duration)
        if self._custom_config is not None:
            config = replace(self._custom_config, **overrides)
        else:
            config = scenarios.get(scenario, **overrides)
        if transparency:
            config = replace(config, transparency=Transparency(transparency))
        self.scenario = scenario
        self.market = PerpetualMarket(config)
        self.market.run_until(0.0)
        self._summary_time = -1.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        tick = 0.05
        last = wallclock.perf_counter()
        while not self._stop.is_set():
            now = wallclock.perf_counter()
            elapsed = now - last
            last = now
            with self.lock:
                if self.playing and not self.market.finished:
                    target = min(self.market.time + elapsed * self.speed, self.market.config.duration)
                    self._advance_to(target, budget=0.25)
                    if self.market.finished:
                        self.playing = False
            wallclock.sleep(tick)

    def _advance_to(self, target: float, budget: float) -> None:
        """Advance the market toward ``target`` but never hold the lock longer than ``budget`` seconds."""
        deadline = wallclock.perf_counter() + budget
        market = self.market
        while not market.finished and market._queue and market._queue[0][0] <= target:
            market.step()
            if wallclock.perf_counter() > deadline:
                return
        if not market.finished:
            market.time = max(market.time, target)

    # --------------------------------------------------------------- controls

    def control(self, payload: Dict[str, Any]) -> None:
        action = str(payload.get("action", "")).lower()
        with self.lock:
            if action == "play":
                self.playing = True
            elif action == "pause":
                self.playing = False
            elif action == "step":
                self.playing = False
                seconds = float(payload.get("seconds", 1.0))
                self._advance_to(self.market.time + seconds, budget=10.0)
            elif action == "finish":
                self.playing = False
                while not self.market.finished:
                    self.market.step()
            elif action == "speed":
                self.speed = max(0.1, float(payload.get("speed", self.speed)))
            elif action == "reset":
                self.playing = False
                self._build(
                    str(payload.get("scenario") or self.scenario),
                    int(payload.get("seed", self.market.seed)),
                    payload.get("transparency") or None,
                )
            else:
                raise ValueError(f"unknown action {action!r}")

    # ------------------------------------------------------------------ state

    def _book(self, depth: int = 18) -> Dict[str, List[Dict[str, Any]]]:
        market = self.market
        tick = market.tick_size
        out: Dict[str, List[Dict[str, Any]]] = {}
        for side, key in ((Side.BUY, "bids"), (Side.SELL, "asks")):
            levels = []
            orders_by_price: Dict[int, Dict[str, int]] = {}
            for order in market.book.orders(side, depth):
                owners = orders_by_price.setdefault(order.price, {})
                owners[order.agent_id] = owners.get(order.agent_id, 0) + order.remaining
            for price, quantity, count in market.book.levels(side, depth):
                owners = sorted(orders_by_price.get(price, {}).items(), key=lambda kv: -kv[1])
                levels.append(
                    {
                        "price": round(price * tick, 6),
                        "quantity": quantity,
                        "orders": count,
                        "owners": [{"id": agent_id, "type": market.accounts[agent_id].agent_type, "quantity": q} for agent_id, q in owners[:6]],
                    }
                )
            out[key] = levels
        return out

    def _trades(self, n: int = 60) -> List[Dict[str, Any]]:
        market = self.market
        rows = []
        for t in list(market.recent_trades)[-n:]:
            rows.append(
                {
                    "id": t.trade_id,
                    "time": round(t.time, 3),
                    "price": t.price,
                    "quantity": t.quantity,
                    "aggressor": t.aggressor.value,
                    "buyer": t.buyer,
                    "seller": t.seller,
                    "buyer_type": market.accounts[t.buyer].agent_type,
                    "seller_type": market.accounts[t.seller].agent_type,
                    "liquidation": t.liquidation,
                    "adl": t.adl,
                }
            )
        return rows

    def _accounts(self) -> List[Dict[str, Any]]:
        market = self.market
        rows = []
        for snap in market.snapshots():
            row = asdict(snap)
            for key in ("collateral", "entry_price", "unrealized_pnl", "realized_pnl", "equity", "notional", "leverage", "fees_paid", "funding_paid", "total_pnl", "return_pct", "volume", "max_drawdown_pct", "max_leverage", "maker_share"):
                row[key] = _r(row[key], 4)
            row["liquidation_price"] = _r(row["liquidation_price"], 4)
            rows.append(row)
        return rows

    def _groups(self, accounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, float]] = {}
        for a in accounts:
            g = groups.setdefault(a["agent_type"], {"agent_type": a["agent_type"], "agents": 0, "active": 0, "initial": 0.0, "pnl": 0.0, "long": 0, "short": 0, "volume": 0.0, "liquidations": 0})
            g["agents"] += 1
            g["active"] += 1 if a["active"] else 0
            g["initial"] += a["initial_collateral"]
            g["pnl"] += a["total_pnl"]
            g["volume"] += a["volume"]
            g["liquidations"] += a["liquidations"]
            if a["position"] > 0:
                g["long"] += a["position"]
            else:
                g["short"] += -a["position"]
        rows = list(groups.values())
        for g in rows:
            g["return_pct"] = round(100.0 * g["pnl"] / g["initial"], 3) if g["initial"] else 0.0
            g["pnl"] = round(g["pnl"], 2)
            g["volume"] = round(g["volume"], 0)
        rows.sort(key=lambda g: -g["return_pct"])
        return rows

    def _bars_since(self, start: int) -> Dict[str, List[Any]]:
        bars = self.market.recorder.bars
        keys = ("time", "open", "high", "low", "close", "volume", "mark", "index", "open_interest", "spread", "predicted_funding")
        return {key: [(_r(v, 5) if isinstance(v, float) else v) for v in bars[key][start:]] for key in keys}

    def _summary(self) -> Dict[str, Any]:
        market = self.market
        if market.time - self._summary_time >= 5.0 or market.finished or not self._summary_cache:
            bars = market.recorder.bars
            summary = market_summary(bars, market.config.record_interval)
            facts = stylized_facts(bars["close"], market.config.record_interval) if len(bars["close"]) > 400 else {}
            index_facts = stylized_facts(bars["index"], market.config.record_interval) if len(bars["index"]) > 400 else {}
            self._summary_cache = {"market": summary, "stylized_facts": facts, "index_stylized_facts": index_facts}
            self._summary_time = market.time
        return self._summary_cache

    def state(self, bars_since: int = 0) -> Dict[str, Any]:
        with self.lock:
            market = self.market
            config = market.config
            accounts = self._accounts()
            bid, ask = market.best_bid_price(), market.best_ask_price()
            liquidations = [
                {
                    "time": round(l.time, 2),
                    "agent_id": l.agent_id,
                    "agent_type": market.accounts[l.agent_id].agent_type,
                    "side": "long" if l.side_closed is Side.SELL else "short",
                    "quantity": l.quantity,
                    "mark_price": _r(l.mark_price),
                    "avg_fill_price": _r(l.avg_fill_price),
                    "adl_quantity": l.adl_quantity,
                    "bad_debt": _r(l.bad_debt, 2),
                    "penalty": _r(l.penalty, 2),
                    "deactivated": l.deactivated,
                }
                for l in market.recorder.liquidations[-100:]
            ]
            orders = [
                {
                    "time": round(e.time, 2),
                    "kind": e.kind,
                    "agent_id": e.agent_id,
                    "agent_type": market.accounts[e.agent_id].agent_type,
                    "side": e.side.value,
                    "type": e.order_type,
                    "quantity": e.quantity,
                    "price": _r(e.price),
                    "reason": e.reason,
                }
                for e in list(market.recorder.recent_orders)[-80:]
            ]
            return {
                "session": {
                    "scenario": self.scenario,
                    "scenarios": scenarios.names(),
                    "seed": market.seed,
                    "playing": self.playing,
                    "finished": market.finished,
                    "speed": self.speed,
                    "time": round(market.time, 3),
                    "clock": format_time(market.time),
                    "duration": config.duration,
                    "transparency": config.transparency.value,
                    "symbol": config.contract.symbol,
                    "tick_size": config.contract.tick_size,
                    "max_leverage": config.contract.max_leverage,
                    "maker_fee": config.contract.maker_fee,
                    "taker_fee": config.contract.taker_fee,
                    "events": market.events_processed,
                },
                "prices": {
                    "last": market.last_price,
                    "mark": _r(market.mark_price, 5),
                    "index": _r(market.index_price, 5),
                    "best_bid": bid,
                    "best_ask": ask,
                    "mid": _r(market.mid_price(), 5),
                    "spread": _r(None if bid is None or ask is None else ask - bid, 5),
                    "open": config.contract.initial_price,
                },
                "stats": {
                    "open_interest": market.open_interest(),
                    "volume": int(sum(market.recorder.bars["volume"])),
                    "trades": len(market.recorder.trades),
                    "funding_rate": market.last_funding_rate,
                    "predicted_funding": market.predicted_funding_rate(),
                    "next_funding": round(market.next_funding_time, 1),
                    "insurance_fund": round(market.insurance_fund, 2),
                    "fees_collected": round(market.fees_collected, 2),
                    "liquidations": market.liquidation_count,
                    "active_agents": sum(1 for a in market.accounts.values() if a.active),
                    "agents": len(market.accounts),
                    "conservation_error": market.conservation_error(),
                },
                "bars_offset": bars_since,
                "bars": self._bars_since(bars_since),
                "book": self._book(),
                "trades": self._trades(),
                "accounts": accounts,
                "groups": self._groups(accounts),
                "liquidations": liquidations,
                "funding": [
                    {"time": round(f.time, 1), "rate": f.rate, "premium": f.premium, "mark": _r(f.mark_price), "index": _r(f.index_price), "paid_by_longs": round(f.total_paid_by_longs, 2)}
                    for f in market.recorder.funding
                ],
                "news": [{"time": round(n.time, 1), "headline": n.headline, "log_jump": n.log_jump} for n in market.recorder.news[-50:]],
                "orders": orders,
                "summary": self._summary(),
            }

    def account_detail(self, agent_id: str) -> Dict[str, Any]:
        with self.lock:
            market = self.market
            if agent_id not in market.accounts:
                raise KeyError(agent_id)
            account = market.accounts[agent_id]
            agent = market.agents[agent_id]
            recorder = market.recorder
            trades = [t for t in recorder.trades[-5000:] if t.buyer == agent_id or t.seller == agent_id][-60:]
            params = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in vars(agent).items() if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
            return {
                "account": asdict(account.snapshot(market.mark_price, market.contract.mmr)),
                "params": params,
                "private_signals": agent_id in market._signal_subscribers,
                "orders": [
                    {"order_id": o.order_id, "side": o.side.value, "price": round(o.price * market.tick_size, 6), "remaining": o.remaining, "quantity": o.quantity, "time": round(o.time, 2), "tag": o.tag}
                    for o in market.book.orders_of(agent_id)
                ],
                "trades": [
                    {
                        "time": round(t.time, 2),
                        "side": "buy" if t.buyer == agent_id else "sell",
                        "price": t.price,
                        "quantity": t.quantity,
                        "counterparty": t.seller if t.buyer == agent_id else t.buyer,
                        "maker": (t.aggressor is Side.SELL) == (t.buyer == agent_id),
                        "liquidation": t.liquidation,
                    }
                    for t in reversed(trades)
                ],
                "equity_series": {"time": recorder.account_times, "equity": [round(v, 2) for v in recorder.account_equity.get(agent_id, [])], "position": recorder.account_position.get(agent_id, [])},
            }
