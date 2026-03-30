from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping, Optional

import numpy as np

from marl_trading.agents import (
    InformedTraderAgent,
    MarketMakerAgent,
    MarketObservation,
    NoiseTraderAgent,
    OrderIntent,
    ScriptedAgent,
    TrendFollowerAgent,
)
from marl_trading.analysis import EventLog, EventType, MarketEvent, OrderBookLevel, OrderBookSnapshot, OrderSide, OrderType, summarize_event_log
from marl_trading.core.config import AgentConfig, MarketConfig, SimulationConfig
from marl_trading.exchange import ExchangeKernel
from marl_trading.exchange.models import Order as ExchangeOrder
from marl_trading.exchange.models import OrderStatus, OrderType as ExchangeOrderType, Side as ExchangeSide
from marl_trading.portfolio import PortfolioManager, SpotPortfolio

from .processes import FundamentalProcess, NewsEvent, PublicNewsProcess


@dataclass(frozen=True)
class MarketStepRecord:
    step_index: int
    timestamp_ns: int
    midpoint: float | None
    fundamental: float
    spread: float | None
    best_bid: float | None
    best_ask: float | None
    trade_count: int
    news_headline: str | None
    news_severity: float | None
    active_agents: int
    total_equity: float


@dataclass(frozen=True)
class MarketRunResult:
    event_log: EventLog
    step_records: list[MarketStepRecord]
    summary: dict[str, object]
    final_portfolios: dict[str, dict[str, object]]
    final_fundamental: float


def _price_to_ticks(price: float, tick_size: float) -> int:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive.")
    return int(round(price / tick_size))


def _ticks_to_price(ticks: int, tick_size: float) -> float:
    return float(ticks * tick_size)


def _round_to_tick(price: float, tick_size: float) -> float:
    return _ticks_to_price(_price_to_ticks(price, tick_size), tick_size)


def _analysis_snapshot_from_exchange_snapshot(snapshot, tick_size: float) -> OrderBookSnapshot:
    bids = tuple(OrderBookLevel(price=_ticks_to_price(level.price, tick_size), quantity=float(level.quantity)) for level in snapshot.bids)
    asks = tuple(OrderBookLevel(price=_ticks_to_price(level.price, tick_size), quantity=float(level.quantity)) for level in snapshot.asks)
    return OrderBookSnapshot(
        timestamp=float(snapshot.timestamp),
        bids=bids,
        asks=asks,
    )


class SyntheticMarketSimulator:
    def __init__(
        self,
        config: SimulationConfig,
        *,
        horizon: int | None = None,
        depth_levels: int | None = None,
    ) -> None:
        self.config = config
        self.market_config: MarketConfig = config.market
        self.horizon = int(horizon if horizon is not None else self.market_config.event_horizon)
        self.depth_levels = int(depth_levels if depth_levels is not None else self.market_config.max_order_levels)
        self._runtime_seed = int(config.seed)
        self.price_scale = max(int(round(1.0 / self.market_config.tick_size)), 1)
        self.tick_size = float(self.market_config.tick_size)
        self.start_midpoint = float(self.market_config.starting_mid_price)
        self.start_midpoint_ticks = _price_to_ticks(self.start_midpoint, self.tick_size)
        self._simulation_started = False
        self._simulation_finished = False
        self._session_start_logged = False
        self._session_end_logged = False
        self._current_step_index = 0
        self._reset_runtime_state()

    def _initialize_agents(self) -> None:
        for agent_cfg in self.config.agents:
            portfolio = SpotPortfolio(
                agent_id=agent_cfg.agent_id,
                symbol=self.market_config.symbol,
                starting_cash=agent_cfg.starting_cash,
                starting_inventory=self._starting_inventory(agent_cfg),
                ruin_threshold=agent_cfg.ruin_threshold,
            )
            self.portfolios.register(portfolio)
            agent = self._build_agent(agent_cfg)
            self.agents[portfolio.agent_id] = agent
            self.open_orders[portfolio.agent_id] = deque()

    def _reset_runtime_state(self) -> None:
        self.rng = np.random.default_rng(self._runtime_seed)
        self.exchange = ExchangeKernel()
        self.portfolios = PortfolioManager()
        self.agents = {}
        self.open_orders = {}
        self._order_counter = 0
        self._sequence_counter = 0
        self.event_log = EventLog()
        self.recent_midpoints = deque(maxlen=12)
        self.recent_news = (None, None)
        self.step_records = []
        self.price_history: list[float] = []
        self.fundamental_history: list[float] = []
        self.fundamental = FundamentalProcess(current_value=self.start_midpoint)
        self.news_process = PublicNewsProcess(interval_steps=self._news_interval_steps())
        self._simulation_started = False
        self._simulation_finished = False
        self._session_start_logged = False
        self._session_end_logged = False
        self._current_step_index = 0
        self._initialize_agents()

    def _starting_inventory(self, agent_cfg: AgentConfig) -> float:
        inventory_map = {
            "market_maker": 40.0,
            "noise_trader": 20.0,
            "trend_follower": 16.0,
            "informed_trader": 18.0,
        }
        return float(inventory_map.get(agent_cfg.agent_type, 10.0))

    def _news_interval_steps(self) -> int:
        # Keep long-running live sessions visibly active without making news too frequent.
        return max(20, min(60, self.horizon // 8))

    def _build_agent(self, agent_cfg: AgentConfig) -> ScriptedAgent:
        if agent_cfg.agent_type == "market_maker":
            return MarketMakerAgent(
                agent_id=agent_cfg.agent_id,
                max_resting_orders=agent_cfg.max_resting_orders,
                inventory_anchor=self._starting_inventory(agent_cfg),
            )
        if agent_cfg.agent_type == "noise_trader":
            return NoiseTraderAgent(
                agent_id=agent_cfg.agent_id,
                max_resting_orders=agent_cfg.max_resting_orders,
            )
        if agent_cfg.agent_type == "trend_follower":
            return TrendFollowerAgent(
                agent_id=agent_cfg.agent_id,
                max_resting_orders=agent_cfg.max_resting_orders,
            )
        if agent_cfg.agent_type == "informed_trader":
            return InformedTraderAgent(
                agent_id=agent_cfg.agent_id,
                max_resting_orders=agent_cfg.max_resting_orders,
                signal_noise=max(0.05, 0.5 - agent_cfg.private_signal_strength * 0.2),
            )
        return NoiseTraderAgent(agent_id=agent_cfg.agent_id, max_resting_orders=agent_cfg.max_resting_orders)

    def _next_sequence(self) -> int:
        self._sequence_counter += 1
        return self._sequence_counter

    def _next_order_id(self, agent_id: str, step_index: int) -> str:
        self._order_counter += 1
        return f"{agent_id}_{step_index}_{self._order_counter}"

    def _current_book_snapshot(self, timestamp_ns: int) -> OrderBookSnapshot:
        snapshot = self.exchange.snapshot(depth=self.depth_levels, timestamp=timestamp_ns)
        return _analysis_snapshot_from_exchange_snapshot(snapshot, self.tick_size)

    def _recent_returns_bps(self) -> tuple[float, ...]:
        if len(self.recent_midpoints) < 2:
            return ()
        points = list(self.recent_midpoints)
        returns: list[float] = []
        for previous, current in zip(points[:-1], points[1:]):
            if previous <= 0:
                continue
            returns.append(((current - previous) / previous) * 10_000.0)
        return tuple(returns)

    def _make_observation(
        self,
        *,
        agent_id: str,
        step_index: int,
        timestamp_ns: int,
        news: NewsEvent | None,
        portfolio: SpotPortfolio,
        snapshot: OrderBookSnapshot,
    ) -> MarketObservation:
        midpoint = snapshot.midpoint()
        spread = snapshot.spread()
        return MarketObservation(
            timestamp_ns=timestamp_ns,
            symbol=self.market_config.symbol.value,
            tick_size=self.tick_size,
            best_bid=snapshot.best_bid(),
            best_ask=snapshot.best_ask(),
            midpoint=midpoint,
            spread=spread,
            latent_fundamental=self.fundamental.current_value,
            recent_midpoints=tuple(self.recent_midpoints),
            recent_returns_bps=self._recent_returns_bps(),
            news_headline=None if news is None else news.headline,
            news_severity=None if news is None else news.severity,
            agent_cash=portfolio.cash,
            agent_inventory=portfolio.inventory,
            agent_equity=portfolio.equity(midpoint if midpoint is not None else self.fundamental.current_value),
            open_orders=len(self.open_orders[agent_id]),
            active_agents=len(self.portfolios.active_portfolios()),
            portfolio_active=portfolio.active,
            agent_type=self.agents[agent_id].agent_type,
            public_note=f"step={step_index}",
        )

    def _log_event(
        self,
        *,
        event_type: EventType,
        timestamp_ns: int,
        payload: dict[str, object] | None = None,
        agent_id: str | None = None,
        order_id: str | None = None,
        side: OrderSide | None = None,
        order_type: OrderType | None = None,
        price: float | None = None,
        quantity: float | None = None,
        order_book: OrderBookSnapshot | None = None,
    ) -> None:
        self.event_log.append(
            MarketEvent(
                sequence=self._next_sequence(),
                timestamp=float(timestamp_ns),
                event_type=event_type,
                agent_id=agent_id,
                order_id=order_id,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity,
                payload=dict(payload or {}),
                order_book=order_book,
            )
        )

    def _sync_open_orders(self, agent_id: str) -> None:
        resting_ids = set(self.exchange.book.resting_orders.keys())
        queue = self.open_orders[agent_id]
        filtered = deque(order_id for order_id in queue if order_id in resting_ids)
        self.open_orders[agent_id] = filtered

    def _cancel_event_payload(self, canceled: ExchangeOrder, *, reason: str, canceled_status: str | None = None, extra: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "reason": reason,
            "original_side": canceled.side.value if hasattr(canceled.side, "value") else str(canceled.side),
            "original_quantity": float(canceled.quantity),
            "original_remaining_quantity": float(canceled.remaining_quantity),
            "original_order_type": canceled.order_type.value if hasattr(canceled.order_type, "value") else str(canceled.order_type),
            "original_price": None if canceled.price is None else _ticks_to_price(int(canceled.price), self.tick_size),
        }
        if canceled_status is not None:
            payload["canceled_status"] = canceled_status
        if extra:
            payload.update(extra)
        return payload

    def _cancel_oldest_order(self, agent_id: str, timestamp_ns: int) -> None:
        queue = self.open_orders[agent_id]
        if not queue:
            return
        order_id = queue.popleft()
        try:
            canceled = self.exchange.cancel_order(order_id, timestamp=timestamp_ns)
        except Exception:
            return
        portfolio = self.portfolios.get(agent_id)
        if order_id in portfolio.reservations:
            portfolio.release_order(order_id)
        self._log_event(
            event_type=EventType.CANCEL_ORDER,
            timestamp_ns=timestamp_ns,
            agent_id=agent_id,
            order_id=order_id,
            order_type=OrderType.CANCEL,
            payload=self._cancel_event_payload(canceled, reason="capacity_trim", canceled_status=canceled.status.value),
            order_book=self._current_book_snapshot(timestamp_ns),
        )

    def _bootstrap_market(self, timestamp_ns: int) -> None:
        snapshot = self._current_book_snapshot(timestamp_ns)
        for agent_id, agent in self.agents.items():
            if agent.agent_type != "market_maker":
                continue
            observation = self._make_observation(
                agent_id=agent_id,
                step_index=0,
                timestamp_ns=timestamp_ns,
                news=None,
                portfolio=self.portfolios.get(agent_id),
                snapshot=snapshot,
            )
            for intent in agent.bootstrap(observation, self.rng):
                self._submit_intent(
                    agent_id=agent_id,
                    intent=intent,
                    timestamp_ns=timestamp_ns,
                    step_index=0,
                    reason="bootstrap",
                )
                snapshot = self._current_book_snapshot(timestamp_ns)
        bootstrap_midpoint = snapshot.midpoint() or self.fundamental.current_value
        self.recent_midpoints.append(bootstrap_midpoint)
        self.price_history.append(float(bootstrap_midpoint))
        self.fundamental_history.append(float(self.fundamental.current_value))
        self._log_event(
            event_type=EventType.SNAPSHOT,
            timestamp_ns=timestamp_ns,
            payload={
                "latent_fundamental": self.fundamental.current_value,
                "note": "bootstrap",
                "active_agents": len(self.portfolios.active_portfolios()),
            },
            order_book=snapshot,
        )

    def _reservation_price(self, intent: OrderIntent, snapshot: OrderBookSnapshot) -> float:
        midpoint = snapshot.midpoint() or self.fundamental.current_value
        if intent.order_type is ExchangeOrderType.MARKET:
            if intent.side is ExchangeSide.BUY:
                return max(snapshot.best_ask() or midpoint * 1.01, midpoint)
            return max(snapshot.best_bid() or midpoint, self.tick_size)
        if intent.limit_price is not None:
            return float(intent.limit_price)
        return float(midpoint)

    def _submit_intent(
        self,
        *,
        agent_id: str,
        intent: OrderIntent,
        timestamp_ns: int,
        step_index: int,
        reason: str = "agent_action",
    ) -> None:
        portfolio = self.portfolios.get(agent_id)
        if not portfolio.active:
            return

        snapshot = self._current_book_snapshot(timestamp_ns)
        if len(self.open_orders[agent_id]) >= self.agents[agent_id].max_resting_orders:
            self._cancel_oldest_order(agent_id, timestamp_ns)

        order_id = self._next_order_id(agent_id, step_index)
        reservation_price = self._reservation_price(intent, snapshot)
        try:
            portfolio.reserve_order(
                order_id=order_id,
                side=intent.side,
                quantity=intent.quantity,
                reservation_price=reservation_price,
            )
        except Exception as exc:
            self._log_event(
                event_type=EventType.CANCEL_ORDER,
                timestamp_ns=timestamp_ns,
                agent_id=agent_id,
                order_id=order_id,
                order_type=OrderType.CANCEL,
                payload={"reason": "reservation_rejected", "error": str(exc), "intent": intent.annotation, "mode": reason},
                order_book=snapshot,
            )
            return

        exchange_order = ExchangeOrder(
            order_id=order_id,
            agent_id=agent_id,
            side=intent.side,
            order_type=intent.order_type,
            quantity=int(intent.quantity),
            price=None if intent.order_type is ExchangeOrderType.MARKET else _price_to_ticks(intent.limit_price or reservation_price, self.tick_size),
            timestamp=timestamp_ns,
        )

        self._log_event(
            event_type=EventType.MARKET_ORDER if intent.order_type is ExchangeOrderType.MARKET else EventType.LIMIT_ORDER,
            timestamp_ns=timestamp_ns,
            agent_id=agent_id,
            order_id=order_id,
            side=OrderSide(intent.side.value),
            order_type=OrderType(intent.order_type.value),
            price=reservation_price,
            quantity=float(intent.quantity),
            payload={"annotation": intent.annotation, "reason": reason},
            order_book=snapshot,
        )

        trades = self.exchange.submit_order(exchange_order)
        for trade in trades:
            execution_price = _ticks_to_price(trade.price, self.tick_size)
            self.portfolios.apply_trade(trade, execution_price=execution_price)
            trade_snapshot = self._current_book_snapshot(timestamp_ns)
            taker_agent_id = trade.buy_agent_id if trade.aggressor_side is ExchangeSide.BUY else trade.sell_agent_id
            self._log_event(
                event_type=EventType.TRADE,
                timestamp_ns=timestamp_ns,
                agent_id=taker_agent_id,
                order_id=trade.taker_order_id,
                side=OrderSide(trade.aggressor_side.value),
                order_type=OrderType.MARKET if exchange_order.order_type is ExchangeOrderType.MARKET else OrderType.LIMIT,
                price=execution_price,
                quantity=float(trade.quantity),
                payload={
                    "buy_agent_id": trade.buy_agent_id,
                    "sell_agent_id": trade.sell_agent_id,
                    "buy_order_id": trade.buy_order_id,
                    "sell_order_id": trade.sell_order_id,
                    "maker_order_id": trade.maker_order_id,
                    "taker_order_id": trade.taker_order_id,
                    "annotation": intent.annotation,
                },
                order_book=trade_snapshot,
            )
            for participant_id in (trade.buy_agent_id, trade.sell_agent_id):
                self._sync_open_orders(str(participant_id))

        if exchange_order.order_type is ExchangeOrderType.LIMIT and exchange_order.remaining_quantity > 0 and exchange_order.status in {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}:
            self.open_orders[agent_id].append(order_id)
        elif exchange_order.order_type is ExchangeOrderType.MARKET or exchange_order.status in {OrderStatus.FILLED, OrderStatus.EXPIRED}:
            if order_id in portfolio.reservations:
                portfolio.release_order(order_id)

        if exchange_order.status is OrderStatus.EXPIRED and order_id in portfolio.reservations:
            portfolio.release_order(order_id)

    def _deactivate_ruined_agents(self, timestamp_ns: int) -> None:
        snapshot = self._current_book_snapshot(timestamp_ns)
        mark_price = snapshot.midpoint() or self.fundamental.current_value
        deactivated = self.portfolios.deactivate_ruined(mark_price=mark_price, timestamp_ns=timestamp_ns)
        for portfolio in deactivated:
            agent_id = portfolio.agent_id
            for order_id in list(self.open_orders.get(agent_id, [])):
                canceled = None
                try:
                    canceled = self.exchange.cancel_order(order_id, timestamp=timestamp_ns)
                except Exception:
                    pass
                if order_id in portfolio.reservations:
                    try:
                        portfolio.release_order(order_id)
                    except Exception:
                        pass
                self._log_event(
                    event_type=EventType.CANCEL_ORDER,
                    timestamp_ns=timestamp_ns,
                    agent_id=agent_id,
                    order_id=order_id,
                    order_type=OrderType.CANCEL,
                    payload=self._cancel_event_payload(canceled, reason="ruin_deactivation") if canceled is not None else {"reason": "ruin_deactivation"},
                    order_book=self._current_book_snapshot(timestamp_ns),
                )
            self.open_orders[agent_id].clear()
            self._log_event(
                event_type=EventType.SNAPSHOT,
                timestamp_ns=timestamp_ns,
                agent_id=agent_id,
                payload={
                    "agent_annotation": "deactivated",
                    "deactivation_reason": portfolio.deactivated_reason,
                    "latent_fundamental": self.fundamental.current_value,
                    "equity": portfolio.equity(mark_price),
                },
                    order_book=self._current_book_snapshot(timestamp_ns),
                )

    def _start_session(self) -> None:
        self._log_event(
            event_type=EventType.SESSION_START,
            timestamp_ns=0,
            payload={
                "symbol": self.market_config.symbol.value,
                "seed": self._runtime_seed,
                "horizon": self.horizon,
                "agent_count": len(self.agents),
                "tick_size": self.tick_size,
            },
        )
        self._bootstrap_market(timestamp_ns=0)
        self._simulation_started = True
        self._session_start_logged = True

    def reset(self, seed: int | None = None, horizon: int | None = None) -> dict[str, object]:
        if seed is not None:
            self._runtime_seed = int(seed)
        if horizon is not None:
            self.horizon = int(horizon)
        self._reset_runtime_state()
        self._start_session()
        return self.snapshot_state()

    @property
    def current_step_index(self) -> int:
        return self._current_step_index

    @property
    def is_started(self) -> bool:
        return self._simulation_started

    @property
    def is_finished(self) -> bool:
        return self._simulation_finished

    def _finalize_session(self) -> None:
        if self._session_end_logged:
            return
        self._log_event(
            event_type=EventType.SESSION_END,
            timestamp_ns=self._current_step_index + 1,
            payload={
                "final_fundamental": self.fundamental.current_value,
                "active_agents": len(self.portfolios.active_portfolios()),
            },
            order_book=self._current_book_snapshot(self._current_step_index + 1),
        )
        self._session_end_logged = True
        self._simulation_finished = True

    def _advance_one_step(self) -> MarketStepRecord | None:
        if not self._simulation_started:
            self._start_session()
        if self._simulation_finished or self._current_step_index >= self.horizon:
            self._finalize_session()
            return None

        step_index = self._current_step_index + 1
        timestamp_ns = step_index
        news = self.news_process.maybe_emit(step_index, self.rng)
        news_event: NewsEvent | None = news
        if news_event is not None:
            self.recent_news = (news_event.headline, news_event.severity)
            self._log_event(
                event_type=EventType.NEWS,
                timestamp_ns=timestamp_ns,
                payload={
                    "headline": news_event.headline,
                    "severity": news_event.severity,
                    "impact": news_event.impact,
                },
                order_book=self._current_book_snapshot(timestamp_ns),
            )

        self.fundamental.advance(self.rng, news_impact=0.0 if news_event is None else news_event.impact)
        snapshot = self._current_book_snapshot(timestamp_ns)
        active_ids = [agent_id for agent_id, portfolio in self.portfolios.portfolios.items() if portfolio.active]
        if not active_ids:
            self._log_event(
                event_type=EventType.SNAPSHOT,
                timestamp_ns=timestamp_ns,
                payload={
                    "latent_fundamental": self.fundamental.current_value,
                    "active_agents": 0,
                    "news_headline": None if news_event is None else news_event.headline,
                },
                order_book=snapshot,
            )
            midpoint = snapshot.midpoint() or self.fundamental.current_value
            self.recent_midpoints.append(midpoint)
            self.price_history.append(float(midpoint))
            self.fundamental_history.append(float(self.fundamental.current_value))
            record = MarketStepRecord(
                step_index=step_index,
                timestamp_ns=timestamp_ns,
                midpoint=snapshot.midpoint(),
                fundamental=self.fundamental.current_value,
                spread=snapshot.spread(),
                best_bid=snapshot.best_bid(),
                best_ask=snapshot.best_ask(),
                trade_count=0,
                news_headline=None if news_event is None else news_event.headline,
                news_severity=None if news_event is None else news_event.severity,
                active_agents=0,
                total_equity=0.0,
            )
            self.step_records.append(record)
            self._current_step_index = step_index
            if self._current_step_index >= self.horizon:
                self._finalize_session()
            return record

        chosen_agent_id = active_ids[(step_index - 1) % len(active_ids)]
        chosen_agent = self.agents[chosen_agent_id]
        chosen_portfolio = self.portfolios.get(chosen_agent_id)
        observation = self._make_observation(
            agent_id=chosen_agent_id,
            step_index=step_index,
            timestamp_ns=timestamp_ns,
            news=news_event,
            portfolio=chosen_portfolio,
            snapshot=snapshot,
        )
        intent = chosen_agent.decide(observation, self.rng)
        if intent is not None:
            self._submit_intent(
                agent_id=chosen_agent_id,
                intent=intent,
                timestamp_ns=timestamp_ns,
                step_index=step_index,
            )

        self._deactivate_ruined_agents(timestamp_ns=timestamp_ns)
        snapshot = self._current_book_snapshot(timestamp_ns)
        midpoint = snapshot.midpoint() or self.fundamental.current_value
        self.recent_midpoints.append(midpoint)
        self.price_history.append(float(midpoint))
        self.fundamental_history.append(float(self.fundamental.current_value))
        total_equity = sum(portfolio.equity(midpoint) for portfolio in self.portfolios.portfolios.values())
        self._log_event(
            event_type=EventType.SNAPSHOT,
            timestamp_ns=timestamp_ns,
            payload={
                "latent_fundamental": self.fundamental.current_value,
                "active_agents": len(self.portfolios.active_portfolios()),
                "news_headline": None if news_event is None else news_event.headline,
                "news_severity": None if news_event is None else news_event.severity,
                "total_equity": total_equity,
            },
            order_book=snapshot,
        )
        record = MarketStepRecord(
            step_index=step_index,
            timestamp_ns=timestamp_ns,
            midpoint=midpoint,
            fundamental=self.fundamental.current_value,
            spread=snapshot.spread(),
            best_bid=snapshot.best_bid(),
            best_ask=snapshot.best_ask(),
            trade_count=sum(1 for event in self.event_log.events if event.event_type == EventType.TRADE and event.timestamp == float(timestamp_ns)),
            news_headline=None if news_event is None else news_event.headline,
            news_severity=None if news_event is None else news_event.severity,
            active_agents=len(self.portfolios.active_portfolios()),
            total_equity=total_equity,
        )
        self.step_records.append(record)
        self._current_step_index = step_index
        if self._current_step_index >= self.horizon:
            self._finalize_session()
        return record

    def step(self) -> MarketStepRecord | None:
        return self._advance_one_step()

    def _book_snapshot_dict(self, snapshot: OrderBookSnapshot) -> dict[str, object]:
        return {
            "timestamp_ns": int(snapshot.timestamp),
            "best_bid": snapshot.best_bid(),
            "best_ask": snapshot.best_ask(),
            "spread": snapshot.spread(),
            "mid_price": snapshot.midpoint(),
            "bids": [
                {"price": float(level.price), "quantity": float(level.quantity)}
                for level in snapshot.bids
            ],
            "asks": [
                {"price": float(level.price), "quantity": float(level.quantity)}
                for level in snapshot.asks
            ],
        }

    def _trade_event_dict(self, event: MarketEvent) -> dict[str, object]:
        payload = dict(event.payload)
        return {
            "sequence": int(event.sequence),
            "timestamp_ns": int(event.timestamp),
            "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
            "agent_id": event.agent_id,
            "order_id": event.order_id,
            "side": event.side.value if hasattr(event.side, "value") else event.side,
            "order_type": event.order_type.value if hasattr(event.order_type, "value") else event.order_type,
            "price": event.price,
            "quantity": event.quantity,
            "payload": payload,
        }

    def _candle_series(self, candle_window: int) -> list[dict[str, object]]:
        if candle_window <= 0:
            raise ValueError("candle_window must be positive.")
        if not self.price_history:
            return []

        candles: list[dict[str, object]] = []
        total_points = len(self.price_history)
        for start_index in range(0, total_points, candle_window):
            end_index = min(start_index + candle_window, total_points)
            price_slice = self.price_history[start_index:end_index]
            fundamental_slice = self.fundamental_history[start_index:end_index]
            if not price_slice:
                continue
            step_slice = self.step_records[start_index : max(end_index - 1, start_index)]
            candles.append(
                {
                    "bucket_index": len(candles),
                    "start_step": start_index,
                    "end_step": max(end_index - 1, start_index),
                    "open": float(price_slice[0]),
                    "high": float(max(price_slice)),
                    "low": float(min(price_slice)),
                    "close": float(price_slice[-1]),
                    "fundamental_open": float(fundamental_slice[0]) if fundamental_slice else None,
                    "fundamental_close": float(fundamental_slice[-1]) if fundamental_slice else None,
                    "trade_count": int(sum(record.trade_count for record in step_slice)),
                    "news_count": int(sum(1 for record in step_slice if record.news_headline is not None)),
                }
            )
        return candles

    def snapshot_state(
        self,
        *,
        depth: int = 10,
        full_book: bool = False,
        recent_limit: int = 25,
        candle_window: int = 5,
    ) -> dict[str, object]:
        top_depth = max(depth, 1)
        full_depth = top_depth
        if full_book:
            current_bids = len(self.exchange.book._bids)
            current_asks = len(self.exchange.book._asks)
            full_depth = max(full_depth, current_bids, current_asks)
        top_snapshot = _analysis_snapshot_from_exchange_snapshot(
            self.exchange.snapshot(depth=top_depth, timestamp=self._current_step_index),
            self.tick_size,
        )
        full_snapshot = top_snapshot
        if full_book:
            full_snapshot = _analysis_snapshot_from_exchange_snapshot(
                self.exchange.snapshot(depth=full_depth, timestamp=self._current_step_index),
                self.tick_size,
            )
        current_midpoint = top_snapshot.midpoint() or self.fundamental.current_value
        recent_events = list(self.event_log.events[-max(recent_limit, 1):])
        visible_event_types = {
            EventType.LIMIT_ORDER.value,
            EventType.MARKET_ORDER.value,
            EventType.CANCEL_ORDER.value,
            EventType.TRADE.value,
            EventType.NEWS.value,
            EventType.SNAPSHOT.value,
        }
        action_events = [
            self._trade_event_dict(event)
            for event in recent_events
            if (event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type))
            in visible_event_types
        ]
        news_events = [
            self._trade_event_dict(event)
            for event in recent_events
            if (event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)) == EventType.NEWS.value
        ]
        trade_events = [
            self._trade_event_dict(event)
            for event in recent_events
            if (event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)) == EventType.TRADE.value
        ]
        portfolios = [
            {
                **portfolio.summary(current_midpoint),
                "agent_type": self.agents[agent_id].agent_type,
                "open_orders": len(self.open_orders.get(agent_id, [])),
                "active": portfolio.active,
            }
            for agent_id, portfolio in self.portfolios.portfolios.items()
        ]
        return {
            "session": {
                "seed": self._runtime_seed,
                "horizon": self.horizon,
                "started": self._simulation_started,
                "finished": self._simulation_finished,
                "current_step_index": self._current_step_index,
                "agent_count": len(self.agents),
                "active_agent_count": len(self.portfolios.active_portfolios()),
            },
            "market": {
                "timestamp_ns": int(self._current_step_index),
                "midpoint": top_snapshot.midpoint(),
                "fundamental": self.fundamental.current_value,
                "spread": top_snapshot.spread(),
                "best_bid": top_snapshot.best_bid(),
                "best_ask": top_snapshot.best_ask(),
                "line": [
                    {
                        "step_index": index,
                        "midpoint": float(midpoint),
                        "fundamental": float(self.fundamental_history[index]) if index < len(self.fundamental_history) else float(self.fundamental.current_value),
                    }
                    for index, midpoint in enumerate(self.price_history)
                ],
                "candles": self._candle_series(candle_window),
                "order_book": self._book_snapshot_dict(top_snapshot),
                "full_order_book": self._book_snapshot_dict(full_snapshot) if full_book else None,
            },
            "tape": trade_events,
            "actions": action_events,
            "news": news_events,
            "portfolios": portfolios,
            "summary": summarize_event_log(self.event_log),
        }

    def run(self, horizon: int | None = None) -> MarketRunResult:
        total_steps = int(horizon if horizon is not None else self.horizon)
        self.reset(horizon=total_steps)
        while not self.is_finished:
            self.step()

        summary = summarize_event_log(self.event_log)
        summary.update(
            {
                "horizon": total_steps,
                "final_fundamental": self.fundamental.current_value,
                "active_agent_count": len(self.portfolios.active_portfolios()),
                "final_midpoint": summary.get("final_midpoint") or self.fundamental.current_value,
            }
        )
        final_portfolios = {
            agent_id: portfolio.summary(self.fundamental.current_value)
            for agent_id, portfolio in self.portfolios.portfolios.items()
        }
        return MarketRunResult(
            event_log=self.event_log,
            step_records=self.step_records,
            summary=summary,
            final_portfolios=final_portfolios,
            final_fundamental=self.fundamental.current_value,
        )


def run_market_demo(config: SimulationConfig | None = None, horizon: int | None = None, seed: int | None = None) -> MarketRunResult:
    from marl_trading.configs.defaults import default_simulation_config

    base_config = config or default_simulation_config()
    if seed is None and horizon is None:
        simulator = SyntheticMarketSimulator(base_config)
    else:
        updated_config = SimulationConfig(
            simulation_id=base_config.simulation_id,
            market=base_config.market,
            agents=base_config.agents,
            seed=base_config.seed if seed is None else seed,
            enable_news=base_config.enable_news,
            enable_private_signals=base_config.enable_private_signals,
            public_tape_enabled=base_config.public_tape_enabled,
        )
        simulator = SyntheticMarketSimulator(updated_config, horizon=horizon)
    return simulator.run(horizon=horizon)
