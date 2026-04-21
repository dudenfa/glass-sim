from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl_trading.exchange.models import OrderType, Side

from .base import MarketObservation, OrderIntent, ScriptedAgent, _clamp_price, _midpoint_or_fallback


def _single_intent(intent: OrderIntent | None) -> tuple[OrderIntent, ...]:
    if intent is None:
        return ()
    return (intent,)


@dataclass
class MarketMakerAgent(ScriptedAgent):
    inventory_anchor: float = 40.0
    quote_size: int = 3
    quote_padding_ticks: int = 1
    inventory_tolerance: float = 4.0
    min_quote_size: int = 1
    max_quote_size: int = 3
    bid_padding_ticks: int = 1
    ask_padding_ticks: int = 1
    inventory_skew_strength: float = 0.75
    inventory_size_decay: float = 0.5
    empty_side_padding_ticks: int = 1

    def __init__(
        self,
        agent_id: str,
        max_resting_orders: int = 3,
        inventory_anchor: float = 40.0,
        quote_size: int = 3,
        quote_padding_ticks: int = 1,
        inventory_tolerance: float | None = None,
        min_quote_size: int | None = None,
        max_quote_size: int | None = None,
        bid_padding_ticks: int | None = None,
        ask_padding_ticks: int | None = None,
        inventory_skew_strength: float = 0.75,
        inventory_size_decay: float = 0.5,
        empty_side_padding_ticks: int = 1,
    ) -> None:
        super().__init__(agent_id=agent_id, agent_type="market_maker", max_resting_orders=max_resting_orders)
        self.inventory_anchor = float(inventory_anchor)
        self.quote_size = max(1, int(quote_size))
        self.quote_padding_ticks = max(1, int(quote_padding_ticks))
        self.inventory_tolerance = float(
            inventory_tolerance if inventory_tolerance is not None else max(1.0, 0.1 * abs(self.inventory_anchor))
        )
        self.min_quote_size = max(1, int(min_quote_size if min_quote_size is not None else 1))
        configured_max_quote_size = int(max_quote_size if max_quote_size is not None else self.quote_size)
        self.max_quote_size = max(self.min_quote_size, configured_max_quote_size)
        self.bid_padding_ticks = max(1, int(bid_padding_ticks if bid_padding_ticks is not None else self.quote_padding_ticks))
        self.ask_padding_ticks = max(1, int(ask_padding_ticks if ask_padding_ticks is not None else self.quote_padding_ticks))
        self.inventory_skew_strength = max(0.0, float(inventory_skew_strength))
        self.inventory_size_decay = max(0.0, float(inventory_size_decay))
        self.empty_side_padding_ticks = max(1, int(empty_side_padding_ticks))

    def bootstrap(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        return self._build_quotes(observation, prefer_two_sided=True)

    def decide(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        if not observation.portfolio_active:
            return ()
        return self._build_quotes(observation, prefer_two_sided=False)

    def _build_quotes(self, observation: MarketObservation, *, prefer_two_sided: bool) -> tuple[OrderIntent, ...]:
        anchor = self._anchor_price(observation)
        inventory_gap = float(observation.agent_inventory - self.inventory_anchor)
        can_restore_bid = self._can_quote_bid(observation, anchor, self.min_quote_size)
        can_restore_ask = self._can_quote_ask(observation, self.min_quote_size)
        ask_missing = observation.best_ask is None
        bid_missing = observation.best_bid is None
        open_orders = max(0, int(observation.open_orders))

        # When one side disappears publicly, immediately restore it if the maker has resources.
        if ask_missing and can_restore_ask:
            ask_intent = self._quote_for_side(
                observation,
                anchor=anchor,
                side=Side.SELL,
                inventory_gap=inventory_gap,
                override_padding=self.empty_side_padding_ticks,
                emergency=True,
            )
            if ask_intent is not None and bid_missing and can_restore_bid and (prefer_two_sided or open_orders < self.max_resting_orders):
                bid_intent = self._quote_for_side(
                    observation,
                    anchor=anchor,
                    side=Side.BUY,
                    inventory_gap=inventory_gap,
                    override_padding=self.empty_side_padding_ticks,
                    emergency=True,
                )
                if bid_intent is not None and self.max_resting_orders >= 2:
                    return (bid_intent, ask_intent)
            return _single_intent(ask_intent)

        if bid_missing and can_restore_bid:
            bid_intent = self._quote_for_side(
                observation,
                anchor=anchor,
                side=Side.BUY,
                inventory_gap=inventory_gap,
                override_padding=self.empty_side_padding_ticks,
                emergency=True,
            )
            if ask_missing and can_restore_ask and (prefer_two_sided or open_orders < self.max_resting_orders):
                ask_intent = self._quote_for_side(
                    observation,
                    anchor=anchor,
                    side=Side.SELL,
                    inventory_gap=inventory_gap,
                    override_padding=self.empty_side_padding_ticks,
                    emergency=True,
                )
                if ask_intent is not None and self.max_resting_orders >= 2:
                    return (bid_intent, ask_intent)
            return _single_intent(bid_intent)

        # If we have room or are bootstrapping, maintain both sides when resources allow.
        if self.max_resting_orders >= 2 and (prefer_two_sided or open_orders == 0):
            bid_intent = self._quote_for_side(observation, anchor=anchor, side=Side.BUY, inventory_gap=inventory_gap)
            ask_intent = self._quote_for_side(observation, anchor=anchor, side=Side.SELL, inventory_gap=inventory_gap)
            intents = tuple(intent for intent in (bid_intent, ask_intent) if intent is not None)
            if intents:
                return intents

        # With partial existing state, refresh the side that best rebalances inventory.
        preferred_side = self._preferred_inventory_side(inventory_gap)
        primary_intent = self._quote_for_side(observation, anchor=anchor, side=preferred_side, inventory_gap=inventory_gap)
        if primary_intent is not None:
            if open_orders <= 1 and self.max_resting_orders >= 2:
                opposite_intent = self._quote_for_side(
                    observation,
                    anchor=anchor,
                    side=Side.SELL if preferred_side is Side.BUY else Side.BUY,
                    inventory_gap=inventory_gap,
                )
                intents = tuple(intent for intent in (primary_intent, opposite_intent) if intent is not None)
                if len(intents) == 2:
                    return intents
            return (primary_intent,)

        opposite_side = Side.SELL if preferred_side is Side.BUY else Side.BUY
        return _single_intent(self._quote_for_side(observation, anchor=anchor, side=opposite_side, inventory_gap=inventory_gap))

    def _anchor_price(self, observation: MarketObservation) -> float:
        if observation.midpoint is not None:
            return float(observation.midpoint)
        if observation.best_bid is not None:
            return float(observation.best_bid)
        if observation.best_ask is not None:
            return float(observation.best_ask)
        return float(observation.latent_fundamental)

    def _preferred_inventory_side(self, inventory_gap: float) -> Side:
        if inventory_gap > self.inventory_tolerance:
            return Side.SELL
        if inventory_gap < -self.inventory_tolerance:
            return Side.BUY
        return Side.BUY if inventory_gap < 0 else Side.SELL

    def _imbalance_scale(self, inventory_gap: float) -> float:
        tolerance = max(self.inventory_tolerance, 1.0)
        return min(abs(inventory_gap) / tolerance, 4.0)

    def _size_for_side(self, side: Side, inventory_gap: float) -> int:
        scale = self._imbalance_scale(inventory_gap)
        size_adjust = int(round(scale * self.inventory_size_decay * max(self.quote_size, 1)))
        quantity = self.quote_size
        if inventory_gap > self.inventory_tolerance:
            if side is Side.SELL:
                quantity += size_adjust
            else:
                quantity -= size_adjust
        elif inventory_gap < -self.inventory_tolerance:
            if side is Side.BUY:
                quantity += size_adjust
            else:
                quantity -= size_adjust
        return max(self.min_quote_size, min(self.max_quote_size, int(quantity)))

    def _padding_ticks_for_side(self, side: Side, inventory_gap: float, override_padding: int | None = None) -> int:
        if override_padding is not None:
            return max(1, int(override_padding))
        scale = self._imbalance_scale(inventory_gap)
        skew_ticks = int(round(scale * self.inventory_skew_strength))
        if side is Side.BUY:
            padding = self.bid_padding_ticks
            if inventory_gap > self.inventory_tolerance:
                return padding + skew_ticks
            if inventory_gap < -self.inventory_tolerance:
                return max(1, padding - skew_ticks)
            return padding
        padding = self.ask_padding_ticks
        if inventory_gap > self.inventory_tolerance:
            return max(1, padding - skew_ticks)
        if inventory_gap < -self.inventory_tolerance:
            return padding + skew_ticks
        return padding

    def _can_quote_bid(self, observation: MarketObservation, anchor: float, quantity: int) -> bool:
        padding = self.bid_padding_ticks * observation.tick_size
        reservation_price = max(observation.tick_size, anchor - padding)
        required_cash = float(quantity) * reservation_price
        return observation.agent_cash + 1e-9 >= required_cash

    def _can_quote_ask(self, observation: MarketObservation, quantity: int) -> bool:
        return observation.agent_inventory + 1e-9 >= float(quantity)

    def _quote_for_side(
        self,
        observation: MarketObservation,
        *,
        anchor: float,
        side: Side,
        inventory_gap: float,
        override_padding: int | None = None,
        emergency: bool = False,
    ) -> OrderIntent | None:
        quantity = self._size_for_side(side, inventory_gap)
        if side is Side.SELL and not self._can_quote_ask(observation, quantity):
            return None
        if side is Side.BUY and not self._can_quote_bid(observation, anchor, quantity):
            return None

        tick = observation.tick_size
        padding_ticks = self._padding_ticks_for_side(side, inventory_gap, override_padding=override_padding)
        padding = max(tick, padding_ticks * tick)
        if side is Side.BUY:
            price = _clamp_price(anchor - padding, tick)
        else:
            price = _clamp_price(anchor + padding, tick)
        annotation = "restore_empty_side" if emergency else "two_sided_inventory_quote"
        return OrderIntent(
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=price,
            annotation=annotation,
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

    def decide(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        if not observation.portfolio_active:
            return ()

        if rng.random() > self.aggressiveness:
            return ()

        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        if side is Side.SELL and observation.agent_inventory < 1.0:
            return ()

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

        return (
            OrderIntent(
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                annotation="noise_trade",
            ),
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

    def decide(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        if not observation.portfolio_active or not observation.recent_returns_bps:
            return ()

        signal = float(np.mean(observation.recent_returns_bps[-3:]))
        if abs(signal) < self.threshold_bps:
            return ()

        side = Side.BUY if signal > 0 else Side.SELL
        if side is Side.SELL and observation.agent_inventory < 1.0:
            return ()

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

        return (
            OrderIntent(
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                annotation="trend_follow",
            ),
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

    def decide(self, observation: MarketObservation, rng) -> tuple[OrderIntent, ...]:
        if not observation.portfolio_active:
            return ()

        midpoint = observation.midpoint if observation.midpoint is not None else observation.latent_fundamental
        raw_edge = float(observation.latent_fundamental - midpoint)
        if observation.news_severity is not None:
            raw_edge += self.news_bias * float(observation.news_severity)
        raw_edge += float(rng.normal(0.0, self.signal_noise))
        edge_bps = raw_edge / max(midpoint, 1e-6) * 10_000.0

        if abs(edge_bps) < self.threshold_bps:
            return ()

        side = Side.BUY if edge_bps > 0 else Side.SELL
        if side is Side.SELL and observation.agent_inventory < 1.0:
            return ()

        quantity = 1 if abs(edge_bps) < 3.0 else 2
        if edge_bps > 0:
            limit_price = _clamp_price((observation.best_ask or midpoint) + observation.tick_size, observation.tick_size)
        else:
            limit_price = _clamp_price((observation.best_bid or midpoint) - observation.tick_size, observation.tick_size)

        return (
            OrderIntent(
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                limit_price=limit_price,
                annotation="informed_signal",
            ),
        )
