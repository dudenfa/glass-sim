from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

from marl_trading.analysis import EventType, MarketEvent, OrderBookSnapshot, summarize_event_log
from marl_trading.configs.defaults import default_simulation_config
from marl_trading.core.config import SimulationConfig
from marl_trading.market.simulator import SyntheticMarketSimulator, _analysis_snapshot_from_exchange_snapshot


@dataclass
class _PnLTracker:
    starting_cash: float
    starting_inventory: float
    starting_midpoint: float
    inventory: float
    cost_basis: float
    realized_pnl: float = 0.0


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
    ) -> None:
        self.base_config = config or default_simulation_config()
        self.horizon = int(horizon if horizon is not None else self.base_config.market.event_horizon)
        self.history_limit = int(history_limit)
        self.event_limit = int(event_limit)
        self.step_delay_seconds = max(float(step_delay_seconds), 0.0)
        self.autoplay = bool(autoplay)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._reset_count = 0
        self._processed_event_count = 0
        self._pnl_trackers: dict[str, _PnLTracker] = {}
        self._trade_history: list[dict[str, Any]] = []
        self._last_action_by_agent: dict[str, dict[str, Any]] = {}
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
        simulator._start_session()  # noqa: SLF001
        return simulator

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
            if event_type == EventType.TRADE.value:
                self._record_trade(event)
            elif event_type in {
                EventType.LIMIT_ORDER.value,
                EventType.MARKET_ORDER.value,
                EventType.CANCEL_ORDER.value,
            } and event.agent_id is not None:
                self._last_action_by_agent[str(event.agent_id)] = event_dict
            self._processed_event_count += 1

    def _build_line(self) -> list[dict[str, Any]]:
        return [
            {
                "step_index": index,
                "timestamp_ns": index,
                "midpoint": float(midpoint),
                "fundamental": float(self.simulator.fundamental_history[index]) if index < len(self.simulator.fundamental_history) else float(self.simulator.fundamental.current_value),
            }
            for index, midpoint in enumerate(self.simulator.price_history)
        ]

    def _build_candles(self, line: list[dict[str, Any]], bucket_size: int) -> list[dict[str, Any]]:
        candles: list[dict[str, Any]] = []
        for bucket_index, start in enumerate(range(0, len(line), bucket_size)):
            chunk = line[start : start + bucket_size]
            if not chunk:
                continue
            midpoint_values = [float(point["midpoint"]) for point in chunk if point.get("midpoint") is not None]
            if not midpoint_values:
                continue
            fundamental_values = [float(point["fundamental"]) for point in chunk if point.get("fundamental") is not None]
            candles.append(
                {
                    "bucket_index": bucket_index,
                    "start_step": int(chunk[0]["step_index"]),
                    "end_step": int(chunk[-1]["step_index"]),
                    "open": midpoint_values[0],
                    "high": max(midpoint_values),
                    "low": min(midpoint_values),
                    "close": midpoint_values[-1],
                    "fundamental_open": fundamental_values[0] if fundamental_values else None,
                    "fundamental_close": fundamental_values[-1] if fundamental_values else None,
                    "trade_count": int(sum(1 for trade in self._trade_history if start <= int(trade["timestamp_ns"]) <= int(chunk[-1]["step_index"]))),
                    "news_count": int(sum(1 for event in self.simulator.event_log.events if self._event_type(event) == EventType.NEWS.value and start <= int(event.timestamp) <= int(chunk[-1]["step_index"]))),
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
            agents.append(
                {
                    **summary,
                    "agent_type": self.simulator.agents[agent_id].agent_type,
                    "open_orders": len(self.simulator.open_orders.get(agent_id, [])),
                    "starting_cash": tracker.starting_cash,
                    "starting_inventory": tracker.starting_inventory,
                    "starting_midpoint": tracker.starting_midpoint,
                    "realized_pnl": float(tracker.realized_pnl),
                    "unrealized_pnl": float(unrealized_pnl),
                    "total_pnl": float(tracker.realized_pnl + unrealized_pnl),
                    "last_action": last_action,
                    "last_action_event": last_action_event,
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
        line = self._build_line()
        candles = self._build_candles(line, candle_window)
        recent_events = [self._event_to_dict(event) for event in self.simulator.event_log.events[-self.event_limit :]]
        recent_trades = [trade for trade in self._trade_history[-self.event_limit :]]
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
        summary = summarize_event_log(self.simulator.event_log)

        market = {
            "symbol": self.simulator.market_config.symbol.value,
            "step_index": current_step,
            "timestamp_ns": current_step,
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
