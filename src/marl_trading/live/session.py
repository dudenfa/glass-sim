from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import threading
import time
from typing import Any

from marl_trading.agents.base import ScriptedAgent
from marl_trading.analysis import EventType, MarketEvent, OrderBookSnapshot
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.core.config import SimulationConfig
from marl_trading.market.simulator import SyntheticMarketSimulator, _analysis_snapshot_from_exchange_snapshot
from marl_trading.rl.live import PPOPolicyAdapter, RuntimePolicyControlledAgent


@dataclass
class _PnLTracker:
    starting_cash: float
    starting_inventory: float
    starting_midpoint: float
    inventory: float
    cost_basis: float
    realized_pnl: float = 0.0


@dataclass(frozen=True)
class _RuntimeRLConfig:
    checkpoint_path: Path
    learning_agent_id: str
    learning_agent_starting_inventory: float


class LiveMarketSession:
    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        horizon: int | None = None,
        history_limit: int = 600,
        event_limit: int = 300,
        step_delay_seconds: float = 0.35,
        autoplay: bool = True,
        checkpoint_path: str | Path | None = None,
        learning_agent_id: str | None = None,
        learning_agent_starting_inventory: float = 0.0,
    ) -> None:
        self.base_config = config or default_simulation_config()
        self.horizon = int(horizon if horizon is not None else self.base_config.market.event_horizon)
        self.history_limit = int(history_limit)
        self.event_limit = int(event_limit)
        self.step_delay_seconds = max(float(step_delay_seconds), 0.0)
        self.autoplay = bool(autoplay)
        self._runtime_rl = self._build_runtime_rl_config(
            checkpoint_path=checkpoint_path,
            learning_agent_id=learning_agent_id,
            learning_agent_starting_inventory=learning_agent_starting_inventory,
        )
        self._ppo_model = None if self._runtime_rl is None else self._load_ppo_policy(self._runtime_rl.checkpoint_path)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._reset_count = 0
        self._processed_event_count = 0
        self._pnl_trackers: dict[str, _PnLTracker] = {}
        self._trade_history: list[dict[str, Any]] = []
        self._last_action_by_agent: dict[str, dict[str, Any]] = {}
        self._event_type_counts: dict[str, int] = defaultdict(int)
        self._first_event_timestamp: int | None = None
        self._last_event_timestamp: int | None = None
        self._max_news_severity: float | None = None
        self._latest_state: dict[str, Any] | None = None
        self.playing = False
        self.finished = False
        self.simulator = self._make_simulator(seed=self.base_config.seed, horizon=self.horizon)
        self._bootstrap_trackers()
        self._ingest_new_events()
        self._capture_state()
        if self.autoplay:
            self.play()

    @property
    def steps_per_second(self) -> float:
        if self.step_delay_seconds <= 0:
            return 0.0
        return 1.0 / self.step_delay_seconds

    def _make_simulator(self, *, seed: int, horizon: int) -> SyntheticMarketSimulator:
        updated_config = SimulationConfig(
            simulation_id=self.base_config.simulation_id,
            market=self.base_config.market,
            agents=self.base_config.agents,
            seed=seed,
            enable_news=self.base_config.enable_news,
            enable_private_signals=self.base_config.enable_private_signals,
            public_tape_enabled=self.base_config.public_tape_enabled,
        )
        simulator = SyntheticMarketSimulator(updated_config, horizon=horizon)
        self._attach_runtime_rl_agent(simulator)
        simulator._start_session()  # noqa: SLF001
        return simulator

    def _build_runtime_rl_config(
        self,
        *,
        checkpoint_path: str | Path | None,
        learning_agent_id: str | None,
        learning_agent_starting_inventory: float,
    ) -> _RuntimeRLConfig | None:
        if checkpoint_path is None:
            return None
        agent_id = str(learning_agent_id or "").strip()
        if not agent_id:
            raise ValueError("A learning_agent_id is required when checkpoint_path is provided.")
        return _RuntimeRLConfig(
            checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
            learning_agent_id=agent_id,
            learning_agent_starting_inventory=float(learning_agent_starting_inventory),
        )

    def _load_ppo_policy(self, checkpoint_path: Path):
        adapter, status = PPOPolicyAdapter.try_load(
            checkpoint_path,
            device="cpu",
            deterministic=True,
        )
        if adapter is None:
            raise RuntimeError(
                status.reason or f"Unable to load PPO checkpoint: {status.checkpoint_path}"
            )
        return adapter

    def _attach_runtime_rl_agent(self, simulator: SyntheticMarketSimulator) -> None:
        if self._runtime_rl is None:
            return
        agent_id = self._runtime_rl.learning_agent_id
        if agent_id not in simulator.agents:
            raise KeyError(f"Unknown runtime learning agent id: {agent_id}")
        original = simulator.agents[agent_id]
        simulator.agents[agent_id] = RuntimePolicyControlledAgent(
            agent_id=agent_id,
            policy=self._ppo_model,
            fallback_agent=original if isinstance(original, ScriptedAgent) else None,
            agent_type="rl_agent",
            max_resting_orders=getattr(original, "max_resting_orders", 1),
            delegate_bootstrap=False,
        )
        portfolio = simulator.portfolios.get(agent_id)
        overridden_inventory = float(self._runtime_rl.learning_agent_starting_inventory)
        portfolio.starting_inventory = overridden_inventory
        portfolio.inventory = overridden_inventory
        portfolio.reserved_inventory = 0.0

    def _analysis_snapshot(self, depth: int, timestamp_ns: int) -> OrderBookSnapshot:
        exchange_snapshot = self.simulator.exchange.snapshot(depth=depth, timestamp=timestamp_ns)
        return _analysis_snapshot_from_exchange_snapshot(exchange_snapshot, self.simulator.tick_size)

    def _event_to_dict(self, event: MarketEvent) -> dict[str, Any]:
        order_book = None
        if event.order_book is not None:
            order_book = {
                "timestamp_ns": int(event.order_book.timestamp),
                "bids": [{"price": float(level.price), "quantity": float(level.quantity)} for level in event.order_book.bids],
                "asks": [{"price": float(level.price), "quantity": float(level.quantity)} for level in event.order_book.asks],
            }
        return {
            "sequence": int(event.sequence),
            "timestamp_ns": int(event.timestamp),
            "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
            "action": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
            "agent_id": event.agent_id,
            "order_id": event.order_id,
            "side": None if event.side is None else (event.side.value if hasattr(event.side, "value") else str(event.side)),
            "order_type": None if event.order_type is None else (event.order_type.value if hasattr(event.order_type, "value") else str(event.order_type)),
            "price": None if event.price is None else float(event.price),
            "quantity": None if event.quantity is None else float(event.quantity),
            "payload": dict(event.payload),
            "order_book": order_book,
        }

    def _bootstrap_trackers(self) -> None:
        midpoint = float(self.simulator.start_midpoint)
        for agent_id, portfolio in self.simulator.portfolios.portfolios.items():
            starting_inventory = float(portfolio.starting_inventory)
            self._pnl_trackers[agent_id] = _PnLTracker(
                starting_cash=float(portfolio.starting_cash),
                starting_inventory=starting_inventory,
                starting_midpoint=midpoint,
                inventory=float(portfolio.inventory),
                cost_basis=starting_inventory * midpoint,
            )

    def _record_trade(self, event: MarketEvent) -> None:
        payload = dict(event.payload)
        buy_agent_id = payload.get("buy_agent_id")
        sell_agent_id = payload.get("sell_agent_id")
        if buy_agent_id is None or sell_agent_id is None or event.price is None or event.quantity is None:
            return

        price = float(event.price)
        quantity = float(event.quantity)
        mark = float(self.simulator.start_midpoint)

        buyer = self._pnl_trackers.setdefault(
            str(buy_agent_id),
            _PnLTracker(starting_cash=0.0, starting_inventory=0.0, starting_midpoint=mark, inventory=0.0, cost_basis=0.0),
        )
        seller = self._pnl_trackers.setdefault(
            str(sell_agent_id),
            _PnLTracker(starting_cash=0.0, starting_inventory=0.0, starting_midpoint=mark, inventory=0.0, cost_basis=0.0),
        )

        buyer.inventory += quantity
        buyer.cost_basis += quantity * price

        if seller.inventory > 1e-12:
            average_cost = seller.cost_basis / seller.inventory
        else:
            average_cost = price
        seller.realized_pnl += quantity * (price - average_cost)
        seller.cost_basis = max(seller.cost_basis - average_cost * quantity, 0.0)
        seller.inventory = max(seller.inventory - quantity, 0.0)

        self._trade_history.append(
            {
                "timestamp_ns": int(event.timestamp),
                "side": "buy" if str(event.agent_id) == str(buy_agent_id) else "sell",
                "price": price,
                "quantity": quantity,
                "agent_id": str(event.agent_id) if event.agent_id is not None else None,
                "buy_agent_id": str(buy_agent_id),
                "sell_agent_id": str(sell_agent_id),
                "note": str(payload.get("annotation") or payload.get("reason") or payload.get("note") or "trade"),
            }
        )

    def _ingest_new_events(self) -> None:
        events = self.simulator.event_log.events
        if self._processed_event_count >= len(events):
            return

        for event in events[self._processed_event_count :]:
            event_dict = self._event_to_dict(event)
            event_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
            self._event_type_counts[event_type] += 1
            timestamp_ns = int(event.timestamp)
            if self._first_event_timestamp is None:
                self._first_event_timestamp = timestamp_ns
            self._last_event_timestamp = timestamp_ns
            if event_type == EventType.TRADE.value:
                self._record_trade(event)
            elif event_type == EventType.NEWS.value:
                severity = event.payload.get("severity")
                if severity is not None:
                    severity_value = float(severity)
                    if self._max_news_severity is None or severity_value > self._max_news_severity:
                        self._max_news_severity = severity_value
            elif event_type in {
                EventType.LIMIT_ORDER.value,
                EventType.MARKET_ORDER.value,
                EventType.CANCEL_ORDER.value,
            } and event.agent_id is not None:
                self._last_action_by_agent[str(event.agent_id)] = event_dict
            self._processed_event_count += 1

    def _build_line(self, *, start_index: int | None = None) -> list[dict[str, Any]]:
        price_history = self.simulator.price_history
        if not price_history:
            return []

        if start_index is None:
            history_limit = max(int(self.history_limit), 1)
            start_index = max(0, len(price_history) - history_limit)
        else:
            start_index = max(0, int(start_index))
        fundamental_history = self.simulator.fundamental_history
        midpoint_history = self.simulator.midpoint_history
        fallback_fundamental = float(self.simulator.fundamental.current_value)

        return [
            {
                "step_index": index,
                "timestamp_ns": index,
                "price": float(price_history[index]),
                "midpoint": midpoint_history[index] if index < len(midpoint_history) else None,
                "fundamental": float(fundamental_history[index]) if index < len(fundamental_history) else fallback_fundamental,
            }
            for index in range(start_index, len(price_history))
        ]

    def _build_candles(
        self,
        line: list[dict[str, Any]],
        bucket_size: int,
        *,
        recent_trades: list[dict[str, Any]] | None = None,
        recent_events: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        candles: list[dict[str, Any]] = []
        recent_trades = list(recent_trades or [])
        recent_events = list(recent_events or [])
        if bucket_size <= 0:
            return candles

        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for point in line:
            step_index = int(point["step_index"])
            bucket_index = step_index // bucket_size
            buckets[bucket_index].append(point)

        for bucket_index in sorted(buckets):
            chunk = buckets[bucket_index]
            price_values = [
                float(point["price"])
                for point in chunk
                if point.get("price") is not None
            ]
            if not price_values:
                continue
            fundamental_values = [float(point["fundamental"]) for point in chunk if point.get("fundamental") is not None]
            start_step = int(chunk[0]["step_index"])
            end_step = int(chunk[-1]["step_index"])
            candles.append(
                {
                    "bucket_index": bucket_index,
                    "start_step": start_step,
                    "end_step": end_step,
                    "open": price_values[0],
                    "high": max(price_values),
                    "low": min(price_values),
                    "close": price_values[-1],
                    "fundamental_open": fundamental_values[0] if fundamental_values else None,
                    "fundamental_close": fundamental_values[-1] if fundamental_values else None,
                    "trade_count": int(sum(1 for trade in recent_trades if start_step <= int(trade["timestamp_ns"]) <= end_step)),
                    "news_count": int(sum(1 for event in recent_events if event.get("event_type") == EventType.NEWS.value and start_step <= int(event["timestamp_ns"]) <= end_step)),
                }
            )
        return candles

    def _event_type(self, event: MarketEvent) -> str:
        return event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)

    def _build_agent_states(self, mark_price: float) -> list[dict[str, Any]]:
        agents: list[dict[str, Any]] = []
        for agent_id, portfolio in self.simulator.portfolios.portfolios.items():
            tracker = self._pnl_trackers.get(
                agent_id,
                _PnLTracker(
                    starting_cash=float(portfolio.starting_cash),
                    starting_inventory=float(portfolio.starting_inventory),
                    starting_midpoint=float(self.simulator.start_midpoint),
                    inventory=float(portfolio.inventory),
                    cost_basis=float(portfolio.inventory) * float(self.simulator.start_midpoint),
                ),
            )
            unrealized_pnl = tracker.inventory * float(mark_price) - tracker.cost_basis
            summary = portfolio.summary(mark_price)
            last_action_event = self._last_action_by_agent.get(agent_id)
            last_action = None
            if last_action_event is not None:
                payload = dict(last_action_event.get("payload", {}))
                last_action = {
                    "action_kind": last_action_event.get("event_type"),
                    "order_type": last_action_event.get("order_type"),
                    "order_id": last_action_event.get("order_id"),
                    "side": last_action_event.get("side"),
                    "quantity": last_action_event.get("quantity"),
                    "limit_price": last_action_event.get("price"),
                    "annotation": payload.get("annotation"),
                    "reason": payload.get("reason"),
                    "timestamp_ns": last_action_event.get("timestamp_ns"),
                    "event_type": last_action_event.get("event_type"),
                }
            rl_diagnostics = None
            runtime_agent = self.simulator.agents[agent_id]
            if hasattr(runtime_agent, "diagnostics"):
                diagnostics = runtime_agent.diagnostics()
                if diagnostics:
                    rl_diagnostics = diagnostics
            agents.append(
                {
                    **summary,
                    "agent_type": runtime_agent.agent_type,
                    "open_orders": len(self.simulator.open_orders.get(agent_id, [])),
                    "starting_cash": tracker.starting_cash,
                    "starting_inventory": tracker.starting_inventory,
                    "starting_midpoint": tracker.starting_midpoint,
                    "realized_pnl": float(tracker.realized_pnl),
                    "unrealized_pnl": float(unrealized_pnl),
                    "total_pnl": float(tracker.realized_pnl + unrealized_pnl),
                    "last_action": last_action,
                    "last_action_event": last_action_event,
                    "rl_diagnostics": rl_diagnostics,
                    "active": bool(portfolio.active),
                }
            )
        agents.sort(key=lambda item: float(item.get("equity", 0.0)), reverse=True)
        return agents

    def _build_state(self, *, full_book: bool = True, depth: int = 10, candle_window: int = 5) -> dict[str, Any]:
        self._ingest_new_events()
        current_step = int(self.simulator.current_step_index)
        top_snapshot = self.simulator._current_book_snapshot(timestamp_ns=current_step)  # noqa: SLF001
        full_depth = max(depth, self.simulator.depth_levels)
        if full_book:
            full_depth = max(
                full_depth,
                len(self.simulator.exchange.book._bids),  # noqa: SLF001
                len(self.simulator.exchange.book._asks),  # noqa: SLF001
            )
        full_snapshot = self._analysis_snapshot(depth=max(full_depth, 1), timestamp_ns=current_step) if full_book else top_snapshot
        mark_price = float(top_snapshot.midpoint() or self.simulator.fundamental.current_value)
        recent_events = [self._event_to_dict(event) for event in self.simulator.event_log.events[-self.event_limit :]]
        recent_trades = [trade for trade in self._trade_history[-self.event_limit :]]
        line = self._build_line()
        candle_start_index = int(line[0]["step_index"]) if line else 0
        if candle_window > 0:
            candle_start_index = max(0, candle_start_index - (candle_start_index % candle_window))
        candle_line = self._build_line(start_index=candle_start_index)
        candles = self._build_candles(candle_line, candle_window, recent_trades=recent_trades, recent_events=recent_events)
        recent_news = []
        for event in recent_events:
            if event["event_type"] != EventType.NEWS.value:
                continue
            payload = dict(event.get("payload", {}))
            recent_news.append(
                {
                    "time": int(event["timestamp_ns"]),
                    "headline": str(payload.get("headline") or "news"),
                    "severity": float(payload.get("severity", 0.0)),
                    "impact": float(payload.get("impact", 0.0)),
                }
            )
        recent_actions = []
        for event in recent_events:
            if event["event_type"] not in {
                EventType.LIMIT_ORDER.value,
                EventType.MARKET_ORDER.value,
                EventType.CANCEL_ORDER.value,
            }:
                continue
            payload = dict(event.get("payload", {}))
            original_side = payload.get("original_side")
            original_quantity = payload.get("original_quantity")
            original_price = payload.get("original_price")
            side = event.get("side")
            quantity = event.get("quantity")
            price = event.get("price")
            if event["event_type"] == EventType.CANCEL_ORDER.value:
                if original_side is not None:
                    side = original_side
                if original_quantity is not None:
                    quantity = original_quantity
                if original_price is not None:
                    price = original_price
            recent_actions.append(
                {
                    "time": int(event["timestamp_ns"]),
                    "agent_id": event.get("agent_id"),
                    "action": event["action"],
                    "event_type": event.get("event_type"),
                    "order_id": event.get("order_id"),
                    "order_type": event.get("order_type"),
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "original_side": original_side,
                    "original_quantity": original_quantity,
                    "original_price": original_price,
                    "original_order_id": event.get("order_id"),
                    "note": str(payload.get("annotation") or payload.get("reason") or payload.get("note") or ""),
                    "payload": payload,
                }
            )
        agents = self._build_agent_states(mark_price)
        rl_agents = [agent for agent in agents if str(agent.get("agent_type")) == "rl_agent"]
        rl_diagnostics = []
        for agent in rl_agents:
            recent_agent_actions = [
                action
                for action in recent_actions
                if str(action.get("agent_id") or "") == str(agent.get("agent_id") or "")
            ][-12:]
            rl_diagnostics.append(
                {
                    "agent_id": agent.get("agent_id"),
                    "decision_count": int((agent.get("rl_diagnostics") or {}).get("decision_count", 0)),
                    "action_counts": dict((agent.get("rl_diagnostics") or {}).get("action_counts", {})),
                    "last_action_type": (agent.get("rl_diagnostics") or {}).get("last_action_type"),
                    "last_failure_reason": (agent.get("rl_diagnostics") or {}).get("last_failure_reason"),
                    "recent_order_events": recent_agent_actions,
                    "open_orders": int(agent.get("open_orders", 0)),
                    "inventory": float(agent.get("inventory", 0.0)),
                    "cash": float(agent.get("cash", 0.0)),
                    "equity": float(agent.get("equity", 0.0)),
                    "realized_pnl": float(agent.get("realized_pnl", 0.0)),
                    "unrealized_pnl": float(agent.get("unrealized_pnl", 0.0)),
                }
            )
        summary = {
            "event_count": len(self.simulator.event_log.events),
            "trade_count": int(self._event_type_counts.get(EventType.TRADE.value, 0)),
            "news_count": int(self._event_type_counts.get(EventType.NEWS.value, 0)),
            "snapshot_count": int(self._event_type_counts.get(EventType.SNAPSHOT.value, 0)),
            "fundamental_point_count": len(self.simulator.fundamental_history),
            "unique_agent_count": len(self.simulator.agents),
            "first_timestamp": self._first_event_timestamp,
            "last_timestamp": self._last_event_timestamp,
            "final_midpoint": top_snapshot.midpoint(),
            "news_severity_max": self._max_news_severity,
            "has_order_book_snapshots": bool(self._event_type_counts.get(EventType.SNAPSHOT.value, 0)),
        }

        market = {
            "symbol": self.simulator.market_config.symbol.value,
            "step_index": current_step,
            "timestamp_ns": current_step,
            "last_price": float(self.simulator.last_trade_price),
            "midpoint": top_snapshot.midpoint(),
            "fundamental": float(self.simulator.fundamental.current_value),
            "spread": top_snapshot.spread(),
            "best_bid": top_snapshot.best_bid(),
            "best_ask": top_snapshot.best_ask(),
            "active_agents": len(self.simulator.portfolios.active_portfolios()),
            "total_equity": float(sum(portfolio.equity(mark_price) for portfolio in self.simulator.portfolios.portfolios.values())),
            "order_book": {
                "timestamp_ns": int(top_snapshot.timestamp),
                "best_bid": top_snapshot.best_bid(),
                "best_ask": top_snapshot.best_ask(),
                "spread": top_snapshot.spread(),
                "midpoint": top_snapshot.midpoint(),
                "bids": [{"price": float(level.price), "quantity": float(level.quantity)} for level in top_snapshot.bids],
                "asks": [{"price": float(level.price), "quantity": float(level.quantity)} for level in top_snapshot.asks],
                "depth_levels": len(top_snapshot.bids),
            },
            "full_order_book": {
                "timestamp_ns": int(full_snapshot.timestamp),
                "best_bid": full_snapshot.best_bid(),
                "best_ask": full_snapshot.best_ask(),
                "spread": full_snapshot.spread(),
                "midpoint": full_snapshot.midpoint(),
                "bids": [{"price": float(level.price), "quantity": float(level.quantity)} for level in full_snapshot.bids],
                "asks": [{"price": float(level.price), "quantity": float(level.quantity)} for level in full_snapshot.asks],
                "depth_levels": len(full_snapshot.bids),
            },
            "line": line,
            "candles": candles,
        }

        state = {
            "session": {
                "simulation_id": self.simulator.config.simulation_id.value,
                "seed": self.simulator.config.seed,
                "horizon": self.horizon,
                "step_index": current_step,
                "last_timestamp_ns": current_step,
                "playing": self.playing,
                "finished": self.finished or self.simulator.is_finished,
                "status": "finished"
                if self.finished or self.simulator.is_finished
                else ("playing" if self.playing else "paused"),
                "reset_count": self._reset_count,
                "steps_per_second": self.steps_per_second,
                "speed": self.steps_per_second,
                "step_delay_seconds": self.step_delay_seconds,
                "runtime_policy": {
                    "enabled": self._runtime_rl is not None,
                    "learning_agent_id": None if self._runtime_rl is None else self._runtime_rl.learning_agent_id,
                    "checkpoint_path": None if self._runtime_rl is None else str(self._runtime_rl.checkpoint_path),
                    "learning_agent_starting_inventory": None if self._runtime_rl is None else float(self._runtime_rl.learning_agent_starting_inventory),
                },
            },
            "market": market,
            "history": list(line),
            "recent_events": recent_events,
            "recent_trades": recent_trades,
            "recent_news": recent_news,
            "recent_actions": recent_actions,
            "agents": agents,
            "summary": {
                **summary,
                "history_point_count": len(line),
                "active_agent_count": len(self.simulator.portfolios.active_portfolios()),
            },
            "tape": recent_trades,
            "actions": recent_actions,
            "news": recent_news,
            "portfolios": agents,
            "rl_diagnostics": rl_diagnostics,
        }
        self._latest_state = state
        return state

    def _capture_state(self) -> dict[str, Any]:
        state = self._build_state()
        self._latest_state = state
        return state

    def _advance_one_step(self) -> bool:
        if self.finished or self.simulator.is_finished:
            self.finished = True
            self.playing = False
            return False
        record = self.simulator.step()
        self.finished = bool(self.simulator.is_finished)
        if self.finished:
            self.playing = False
        return record is not None

    def _shutdown_worker(self) -> None:
        self.playing = False
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def reset(self, *, seed: int | None = None, horizon: int | None = None) -> dict[str, Any]:
        with self._lock:
            self._shutdown_worker()
            self._reset_count += 1
            self.horizon = int(self.horizon if horizon is None else horizon)
            next_seed = self.base_config.seed if seed is None else int(seed)
            self.simulator = self._make_simulator(seed=next_seed, horizon=self.horizon)
            self._processed_event_count = 0
            self._pnl_trackers = {}
            self._trade_history = []
            self._last_action_by_agent = {}
            self._event_type_counts = defaultdict(int)
            self._first_event_timestamp = None
            self._last_event_timestamp = None
            self._max_news_severity = None
            self._bootstrap_trackers()
            self._ingest_new_events()
            self.playing = False
            self.finished = False
            return self.state()

    def play(self) -> dict[str, Any]:
        with self._lock:
            if self.finished or self.simulator.is_finished:
                self.playing = False
                return self.state()
            self.playing = True
            self._stop_event.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()
            return self.state()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self.playing = False
            return self.state()

    def set_speed(self, steps_per_second: float) -> dict[str, Any]:
        with self._lock:
            requested = max(float(steps_per_second), 0.1)
            self.step_delay_seconds = 1.0 / requested
            return self.state()

    def step(self, steps: int = 1) -> dict[str, Any]:
        with self._lock:
            self.playing = False
            for _ in range(max(int(steps), 1)):
                if not self._advance_one_step():
                    break
            return self.state()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                playing = self.playing
                finished = self.finished or self.simulator.is_finished
                delay = self.step_delay_seconds
                if finished:
                    self.finished = True
                    self.playing = False
            if finished:
                time.sleep(0.05)
                continue
            if not playing:
                time.sleep(0.05)
                continue
            with self._lock:
                progressed = self._advance_one_step()
                state_finished = self.finished
            if not progressed or state_finished:
                time.sleep(0.05)
                continue
            time.sleep(max(delay, 0.01))

    def stop(self) -> None:
        with self._lock:
            self._shutdown_worker()

    def state(self) -> dict[str, Any]:
        with self._lock:
            state = self._capture_state()
            self.finished = bool(state["session"]["finished"])
            return state
