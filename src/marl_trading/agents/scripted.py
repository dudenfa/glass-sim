from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from marl_trading.exchange.models import OrderType, Side

from .base import MarketObservation, OrderIntent, ScriptedAgent, _clamp_price, _midpoint_or_fallback


@dataclass
class MarketMakerAgent(ScriptedAgent):
    inventory_anchor: float = 40.0
    quote_size: int = 3
    quote_padding_ticks: int = 1

    def __init__(
        self,
        agent_id: str,
        max_resting_orders: int = 3,
        inventory_anchor: float = 40.0,
        quote_size: int = 3,
        quote_padding_ticks: int = 1,
    ) -> None:
        super().__init__(agent_id=agent_id, agent_type="market_maker", max_resting_orders=max_resting_orders)
        self.inventory_anchor = float(inventory_anchor)
        self.quote_size = int(quote_size)
        self.quote_padding_ticks = int(quote_padding_ticks)
        self._last_side: Side = Side.BUY

    def bootstrap(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        midpoint = _midpoint_or_fallback(observation)
        tick = observation.tick_size
        bid = _clamp_price(midpoint - max(tick, self.quote_padding_ticks * tick), tick)
        ask = _clamp_price(midpoint + max(tick, self.quote_padding_ticks * tick), tick)
        return (
            OrderIntent(side=Side.BUY, order_type=OrderType.LIMIT, quantity=self.quote_size, limit_price=bid, annotation="bootstrap_bid"),
            OrderIntent(side=Side.SELL, order_type=OrderType.LIMIT, quantity=self.quote_size, limit_price=ask, annotation="bootstrap_ask"),
        )

    def decide(self, observation: MarketObservation, rng) -> Optional[OrderIntent]:
        if not observation.portfolio_active:
            return None

        midpoint = _midpoint_or_fallback(observation)
        tick = observation.tick_size
        target_inventory = self.inventory_anchor
        inventory_gap = observation.agent_inventory - target_inventory

        if abs(inventory_gap) < max(1.0, 0.1 * target_inventory):
            side = Side.BUY if self._last_side is Side.SELL else Side.SELL
        else:
            side = Side.SELL if inventory_gap > 0 else Side.BUY

        self._last_side = side
        quantity = self.quote_size if abs(inventory_gap) < target_inventory else max(1, self.quote_size - 1)
        spread = observation.spread if observation.spread is not None else 2.0 * tick
        padding = max(tick, 0.5 * spread)
        if side is Side.BUY:
            price = _clamp_price(midpoint - padding, tick)
        else:
            price = _clamp_price(midpoint + padding, tick)

        return OrderIntent(
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=price,
            annotation="inventory_skew_quote",
        )


@dataclass
class NoiseTraderAgent(ScriptedAgent):
    aggressiveness: float = 0.55
    market_order_probability: float = 0.7

    def __init__(
        self,
        agent_id: str,
        max_resting_orders: int = 2,
        aggressiveness: float = 0.55,
        market_order_probability: float = 0.7,
    ) -> None:
        super().__init__(agent_id=agent_id, agent_type="noise_trader", max_resting_orders=max_resting_orders)
        self.aggressiveness = float(aggressiveness)
        self.market_order_probability = float(market_order_probability)

    def decide(self, observation: MarketObservation, rng) -> Optional[OrderIntent]:
        if not observation.portfolio_active:
            return None

        if rng.random() > self.aggressiveness:
            return None

        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        if side is Side.SELL and observation.agent_inventory < 1.0:
            return None

        quantity = 1 if rng.random() < 0.75 else 2
        if rng.random() < self.market_order_probability:
            order_type = OrderType.MARKET
            limit_price = None
        else:
            tick = observation.tick_size
            midpoint = _midpoint_or_fallback(observation)
            if side is Side.BUY:
                limit_price = _clamp_price((observation.best_bid or midpoint) - tick, tick)
            else:
                limit_price = _clamp_price((observation.best_ask or midpoint) + tick, tick)
            order_type = OrderType.LIMIT

        return OrderIntent(
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            annotation="noise_trade",
        )


@dataclass
class TrendFollowerAgent(ScriptedAgent):
    threshold_bps: float = 1.5
    market_order_probability: float = 0.5

    def __init__(
        self,
        agent_id: str,
        max_resting_orders: int = 2,
        threshold_bps: float = 1.5,
        market_order_probability: float = 0.5,
    ) -> None:
        super().__init__(agent_id=agent_id, agent_type="trend_follower", max_resting_orders=max_resting_orders)
        self.threshold_bps = float(threshold_bps)
        self.market_order_probability = float(market_order_probability)

    def decide(self, observation: MarketObservation, rng) -> Optional[OrderIntent]:
        if not observation.portfolio_active or not observation.recent_returns_bps:
            return None

        signal = float(np.mean(observation.recent_returns_bps[-3:]))
        if abs(signal) < self.threshold_bps:
            return None

        side = Side.BUY if signal > 0 else Side.SELL
        if side is Side.SELL and observation.agent_inventory < 1.0:
            return None

        quantity = 1 if abs(signal) < 3.0 else 2
        if rng.random() < self.market_order_probability:
            order_type = OrderType.MARKET
            limit_price = None
        else:
            tick = observation.tick_size
            midpoint = _midpoint_or_fallback(observation)
            if side is Side.BUY:
                limit_price = _clamp_price((observation.best_ask or midpoint) + tick, tick)
            else:
                limit_price = _clamp_price((observation.best_bid or midpoint) - tick, tick)
            order_type = OrderType.LIMIT

        return OrderIntent(
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            annotation="trend_follow",
        )


@dataclass
class InformedTraderAgent(ScriptedAgent):
    signal_noise: float = 0.15
    news_bias: float = 1.25
    threshold_bps: float = 1.0

    def __init__(
        self,
        agent_id: str,
        max_resting_orders: int = 2,
        signal_noise: float = 0.15,
        news_bias: float = 1.25,
        threshold_bps: float = 1.0,
    ) -> None:
        super().__init__(agent_id=agent_id, agent_type="informed_trader", max_resting_orders=max_resting_orders)
        self.signal_noise = float(signal_noise)
        self.news_bias = float(news_bias)
        self.threshold_bps = float(threshold_bps)

    def decide(self, observation: MarketObservation, rng) -> Optional[OrderIntent]:
        if not observation.portfolio_active:
            return None

        midpoint = observation.midpoint if observation.midpoint is not None else observation.latent_fundamental
        raw_edge = float(observation.latent_fundamental - midpoint)
        if observation.news_severity is not None:
            raw_edge += self.news_bias * float(observation.news_severity)
        raw_edge += float(rng.normal(0.0, self.signal_noise))
        edge_bps = raw_edge / max(midpoint, 1e-6) * 10_000.0

        if abs(edge_bps) < self.threshold_bps:
            return None

        side = Side.BUY if edge_bps > 0 else Side.SELL
        if side is Side.SELL and observation.agent_inventory < 1.0:
            return None

        quantity = 1 if abs(edge_bps) < 3.0 else 2
        if edge_bps > 0:
            limit_price = _clamp_price((observation.best_ask or midpoint) + observation.tick_size, observation.tick_size)
        else:
            limit_price = _clamp_price((observation.best_bid or midpoint) - observation.tick_size, observation.tick_size)

        return OrderIntent(
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            annotation="informed_signal",
        )
