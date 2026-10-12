"""A live, controllable simulation session for the browser viewer.

The session owns a ``PerpetualMarket`` and advances it in a background thread
at ``speed`` simulated seconds per real second. All access goes through a lock
so the HTTP handlers always see a consistent market.

With ``replay`` the session plays back a saved run (see ``replay.py``) instead
of a new market: the same controls work, and at the end the session checks
that the playback reproduced the original run. Because runs are deterministic,
seeking backwards rebuilds the market and replays it up to the requested time.
"""

from __future__ import annotations

import threading
import time as wallclock
from dataclasses import asdict, replace
from typing import Any, Optional

from .. import scenarios
from ..analysis import market_summary, stylized_facts
from ..config import MarketConfig, format_time
from ..market import PerpetualMarket
from ..replay import Replay, ReplayRun
from ..types import Side, Transparency


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Round for JSON; ``None`` passes through."""
    return None if value is None else round(float(value), digits)


class LiveSession:
    """A market the viewer can play, pause, step and reset.

    Every public method takes the lock so HTTP handlers never read a
    half-updated market; the background loop releases it often enough
    (``_advance_to``) that the controls stay responsive.
    """

    def __init__(
        self,
        scenario: str = "default",
        seed: int = 1,
        transparency: Optional[str] = None,
        speed: float = 10.0,
        duration: Optional[float] = None,
        config: Optional[MarketConfig] = None,
        rl_checkpoints: Optional[list[str]] = None,
        replay: Optional[Replay] = None,
    ) -> None:
        self.lock = threading.RLock()
        self.speed = float(speed)
        self.playing = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._summary_cache: dict[str, Any] = {}
        self._summary_time = -1.0
        self._custom_config = config
        self.rl_checkpoints = list(rl_checkpoints or [])
        self.replay = replay
        self.runner: Optional[ReplayRun] = None
        self.verification: Optional[dict[str, Any]] = None
        self.replay_description = replay.describe() if replay is not None else None
        self.scenario = scenario
        self.duration = duration
        self._build(scenario, seed, transparency)

    # ------------------------------------------------------------- lifecycle

    def _build(self, scenario: str, seed: int, transparency: Optional[str]) -> None:
        """Create a fresh market (and optional RL participants) for a scenario, seed and transparency level."""
        self._settings = (scenario, seed, transparency)
        self._summary_time = -1.0
        if self.replay is not None:
            self.runner = self.replay.build()
            self.market = self.runner.market
            self.scenario = self.market.config.name
            self.verification = None
            self.runner.run_until(0.0)
            return
        overrides: dict[str, Any] = {"seed": int(seed)}
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
        for i, checkpoint in enumerate(self.rl_checkpoints, start=1):
            from ..rl.policy import load_run_config, policy_agent_from_checkpoint

            agent = policy_agent_from_checkpoint(checkpoint)
            agent.agent_id = f"rl_{i:02d}"
            env = load_run_config(checkpoint).get("env", {})
            self.market.add_agent(agent, float(env.get("collateral", 100_000.0)), private_signals=bool(env.get("private_signals")))
        self.market.run_until(0.0)
        self._summary_time = -1.0

    def start(self) -> None:
        """Start the background clock."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background clock and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        """Advance the market in proportion to real elapsed time while playing."""
        tick = 0.05
        last = wallclock.perf_counter()
        while not self._stop.is_set():
            now = wallclock.perf_counter()
            elapsed = now - last
            last = now
            with self.lock:
                if self.playing and not self.finished:
                    target = min(self.market.time + elapsed * self.speed, self.duration_seconds)
                    self._advance_to(target, budget=0.25)
                    if self.finished:
                        self.playing = False
            wallclock.sleep(tick)

    def _advance_to(self, target: float, budget: float) -> None:
        """Advance the market toward ``target`` but never hold the lock longer than ``budget`` seconds."""
        deadline = wallclock.perf_counter() + budget
        if self.runner is not None:
            self.runner.run_until(target, deadline)
            self._check_replay()
            return
        market = self.market
        while not market.finished and market._queue and market._queue[0][0] <= target:
            market.step()
            if wallclock.perf_counter() > deadline:
                return
        if not market.finished:
            market.time = max(market.time, target)

    @property
    def finished(self) -> bool:
        """End of the session: the market's end, or the point where a replayed run was saved."""
        return self.runner.done if self.runner is not None else self.market.finished

    @property
    def duration_seconds(self) -> float:
        return self.runner.end_time if self.runner is not None else self.market.config.duration

    def _check_replay(self) -> None:
        """Once a replay is complete, compare it with the original run (once)."""
        if self.runner is not None and self.runner.done and self.verification is None:
            self.verification = self.runner.verify()

    def _seek(self, target: float) -> None:
        """Jump to simulated ``target``: forward by running, backward by rebuilding and replaying (runs are deterministic)."""
        target = min(max(0.0, target), self.duration_seconds)
        if target < self.market.time:
            self._build(*self._settings)
        self._advance_to(target, budget=120.0)

    # --------------------------------------------------------------- controls

    def control(self, payload: dict[str, Any]) -> None:
        """Apply a viewer command: play, pause, step, finish, speed or reset."""
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
                if self.runner is not None:
                    while self.runner.step():
                        pass
                    self._check_replay()
                else:
                    while not self.market.finished:
                        self.market.step()
            elif action == "seek":
                self.playing = False
                self._seek(float(payload.get("time", self.market.time)))
            elif action == "speed":
                self.speed = max(0.1, float(payload.get("speed", self.speed)))
            elif action == "reset" and self.replay is not None:
                self.playing = False
                self._build(*self._settings)
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

    def _book(self, depth: int = 18) -> dict[str, list[dict[str, Any]]]:
        """Aggregated book levels with their owners, for the depth panel."""
        market = self.market
        tick = market.tick_size
        out: dict[str, list[dict[str, Any]]] = {}
        for side, key in ((Side.BUY, "bids"), (Side.SELL, "asks")):
            levels = []
            orders_by_price: dict[int, dict[str, int]] = {}
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

    def _trades(self, n: int = 60) -> list[dict[str, Any]]:
        """The recent trade tape with participant types."""
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

    def _accounts(self) -> list[dict[str, Any]]:
        """Every account's snapshot, rounded for JSON."""
        market = self.market
        rows = []
        for snap in market.snapshots():
            row = asdict(snap)
            for key in (
                "collateral",
                "entry_price",
                "unrealized_pnl",
                "realized_pnl",
                "equity",
                "notional",
                "leverage",
                "fees_paid",
                "funding_paid",
                "total_pnl",
                "return_pct",
                "volume",
                "max_drawdown_pct",
                "max_leverage",
                "maker_share",
            ):
                row[key] = _r(row[key], 4)
            row["liquidation_price"] = _r(row["liquidation_price"], 4)
            rows.append(row)
        return rows

    def _groups(self, accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Per-type aggregates of the account rows."""
        groups: dict[str, dict[str, float]] = {}
        for a in accounts:
            g = groups.setdefault(
                a["agent_type"],
                {"agent_type": a["agent_type"], "agents": 0, "active": 0, "initial": 0.0, "pnl": 0.0, "long": 0, "short": 0, "volume": 0.0, "liquidations": 0},
            )
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

    def _bars_since(self, start: int) -> dict[str, list[Any]]:
        """Bars from index ``start`` on, so the client only fetches what it has not seen."""
        bars = self.market.recorder.bars
        keys = ("time", "open", "high", "low", "close", "volume", "mark", "index", "open_interest", "spread", "predicted_funding")
        return {key: [(_r(v, 5) if isinstance(v, float) else v) for v in bars[key][start:]] for key in keys}

    def _summary(self) -> dict[str, Any]:
        """Market-quality statistics, recomputed at most every 5 simulated seconds because they are the expensive part of the state."""
        market = self.market
        if market.time - self._summary_time >= 5.0 or market.finished or not self._summary_cache:
            bars = market.recorder.bars
            summary = market_summary(bars, market.config.record_interval)
            facts = stylized_facts(bars["close"], market.config.record_interval) if len(bars["close"]) > 400 else {}
            index_facts = stylized_facts(bars["index"], market.config.record_interval) if len(bars["index"]) > 400 else {}
            self._summary_cache = {"market": summary, "stylized_facts": facts, "index_stylized_facts": index_facts}
            self._summary_time = market.time
        return self._summary_cache

    def state(self, bars_since: int = 0) -> dict[str, Any]:
        """Everything the viewer needs for one refresh."""
        with self.lock:
            market = self.market
            config = market.config
            accounts = self._accounts()
            bid, ask = market.best_bid_price(), market.best_ask_price()
            liquidations = [
                {
                    "time": round(liq.time, 2),
                    "agent_id": liq.agent_id,
                    "agent_type": market.accounts[liq.agent_id].agent_type,
                    "side": "long" if liq.side_closed is Side.SELL else "short",
                    "quantity": liq.quantity,
                    "mark_price": _r(liq.mark_price),
                    "avg_fill_price": _r(liq.avg_fill_price),
                    "adl_quantity": liq.adl_quantity,
                    "bad_debt": _r(liq.bad_debt, 2),
                    "penalty": _r(liq.penalty, 2),
                    "deactivated": liq.deactivated,
                }
                for liq in market.recorder.liquidations[-100:]
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
                    "finished": self.finished,
                    "mode": "replay" if self.replay is not None else "live",
                    "replay": None
                    if self.replay is None
                    else {"description": self.replay_description, "source": self.replay.source, "verification": self.verification},
                    "speed": self.speed,
                    "time": round(market.time, 3),
                    "clock": format_time(market.time),
                    "duration": self.duration_seconds,
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
                    {
                        "time": round(f.time, 1),
                        "rate": f.rate,
                        "premium": f.premium,
                        "mark": _r(f.mark_price),
                        "index": _r(f.index_price),
                        "paid_by_longs": round(f.total_paid_by_longs, 2),
                    }
                    for f in market.recorder.funding
                ],
                "news": [{"time": round(n.time, 1), "headline": n.headline, "log_jump": n.log_jump} for n in market.recorder.news[-50:]],
                "orders": orders,
                "summary": self._summary(),
            }

    def account_detail(self, agent_id: str) -> dict[str, Any]:
        """Drill-down for one account: snapshot, strategy parameters, orders, fills and equity path."""
        with self.lock:
            market = self.market
            if agent_id not in market.accounts:
                raise KeyError(agent_id)
            account = market.accounts[agent_id]
            agent = market.agents[agent_id]
            recorder = market.recorder
            trades = [t for t in recorder.trades[-5000:] if t.buyer == agent_id or t.seller == agent_id][-60:]
            params = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in vars(agent).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))
            }
            if hasattr(agent, "action_distribution"):
                params.update({f"action {name}": share for name, share in agent.action_distribution().items()})
            return {
                "account": asdict(account.snapshot(market.mark_price, market.contract.mmr)),
                "params": params,
                "private_signals": agent_id in market._signal_subscribers,
                "orders": [
                    {
                        "order_id": o.order_id,
                        "side": o.side.value,
                        "price": round(o.price * market.tick_size, 6),
                        "remaining": o.remaining,
                        "quantity": o.quantity,
                        "time": round(o.time, 2),
                        "tag": o.tag,
                    }
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
                "equity_series": {
                    "time": recorder.account_times,
                    "equity": [round(v, 2) for v in recorder.account_equity.get(agent_id, [])],
                    "position": recorder.account_position.get(agent_id, []),
                },
            }
