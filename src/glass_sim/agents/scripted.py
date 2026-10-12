"""Scripted perpetual-futures traders.

Each class models a recognisable participant type of real crypto-perpetual
markets. They are deliberately simple and interpretable, parameterised so that
populations can be made heterogeneous (see ``AgentSpec.randomize``).

==================  ===============================================================
agent               role in the ecology
==================  ===============================================================
MarketMaker         posts multi-level two-sided quotes around a fair price; widens
                    with volatility and news; skews quotes to shed inventory
NoiseTrader         retail flow: random direction with a mild trend-chasing bias,
                    often high leverage, takes profit / holds too long / gets
                    liquidated
MomentumTrader      trend follower on fast/slow moving averages of the trade price
BasisArbitrageur    trades the gap between the perpetual and the index price,
                    which anchors the perp to the index (the funding-arb desk)
InformedTrader      receives noisy private signals about news before it is public
                    and trades ahead of it
Whale               large account running multi-hour accumulation / unwind
                    campaigns with TWAP child orders
FlowFollower        ATTRIBUTED or FULL: follows the account with the strongest
                    one-directional flow on the attributed trade tape
CopyTrader          FULL transparency only: mirrors the direction of the most
                    profitable visible account
LiquidationHunter   FULL transparency only: pushes the price toward clusters of
                    visible liquidation prices and takes profit in the cascade
==================  ===============================================================
"""

from __future__ import annotations

import math
from typing import Optional

from ..types import Action, Cancel, CancelAll, Fill, LimitOrder, MarketOrder, Side, TimeInForce, Transparency
from ..views import MarketView
from .base import Agent
from .common import capacity, clamp, ema, move_toward, realized_vol_per_second, safe_log_ratio, sign


def _floor_tick(price: float, tick: float) -> float:
    """Round down to the tick grid (never below one tick)."""
    return max(tick, math.floor(price / tick + 1e-9) * tick)


def _ceil_tick(price: float, tick: float) -> float:
    """Round up to the tick grid (never below one tick)."""
    return max(tick, math.ceil(price / tick - 1e-9) * tick)


# =============================================================================
# Market maker
# =============================================================================


class MarketMaker(Agent):
    """Multi-level two-sided quoting in the spirit of Avellaneda-Stoikov.

    - fair price: blend of book mid and the index price (``index_weight``);
    - half-spread: ``max(min_half_spread_bps, vol_multiplier * sigma * sqrt(vol_horizon))``,
      capped at ``max_half_spread_bps``, with sigma estimated from the index price;
    - size: each level quotes ``level_fraction`` of the maker's capacity at
      ``target_leverage`` (at least ``base_size``), growing with depth;
    - inventory skew: the reservation price moves against the inventory by up
      to ``skew_bps`` at the inventory limit, and quote sizes shrink on the side
      that would add inventory;
    - after public news the maker pulls all quotes and quotes wider for a while;
    - quotes are post-only, and only orders whose price is no longer wanted are
      canceled, so resting orders keep their queue priority.
    """

    agent_type = "market_maker"
    default_prefix = "mm"

    def __init__(
        self,
        wake_interval: float = 1.5,
        levels: int = 4,
        base_size: int = 5,
        level_fraction: float = 0.015,
        min_half_spread_bps: float = 2.0,
        level_spacing_bps: float = 2.5,
        vol_multiplier: float = 1.0,
        vol_horizon: float = 10.0,
        max_half_spread_bps: float = 60.0,
        index_weight: float = 0.5,
        target_leverage: float = 3.0,
        skew_bps: float = 12.0,
        news_widen: float = 3.0,
        news_widen_seconds: float = 30.0,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.levels = int(levels)
        self.base_size = int(base_size)
        self.level_fraction = float(level_fraction)
        self.min_half_spread_bps = float(min_half_spread_bps)
        self.level_spacing_bps = float(level_spacing_bps)
        self.vol_multiplier = float(vol_multiplier)
        self.vol_horizon = float(vol_horizon)
        self.max_half_spread_bps = float(max_half_spread_bps)
        self.index_weight = float(index_weight)
        self.target_leverage = float(target_leverage)
        self.skew_bps = float(skew_bps)
        self.news_widen = float(news_widen)
        self.news_widen_seconds = float(news_widen_seconds)
        self._widen_until = -1.0

    def on_start(self, view, rng):
        """Quote from scratch at time 0; there is nothing to reconcile yet."""
        return self._quotes(view, reconcile=False)

    def on_wakeup(self, view, rng):
        """Re-quote, keeping orders whose price is still wanted so they keep their queue priority."""
        return self._quotes(view, reconcile=True)

    def on_news(self, news, view, rng):
        """Pull every quote and re-quote wider: stale quotes right after a jump are free options for others."""
        self._widen_until = view.time + self.news_widen_seconds
        return [CancelAll()] + self._quotes(view, reconcile=False)

    def desired_quotes(self, view: MarketView) -> list[tuple[Side, float, int]]:
        """The ladder of ``(side, price, size)`` the maker wants right now.

        Kept separate from order management so the pricing logic can be read
        and tuned without touching how resting orders are reconciled.
        """
        tick = view.tick_size
        mark = view.mark_price
        mid = view.mid
        # Perp market makers reference the spot index (where they hedge) as well as their own book.
        anchor = view.index_price
        fair = (1.0 - self.index_weight) * mid + self.index_weight * anchor if mid is not None else 0.5 * (mark + anchor)
        # Volatility from the index, not from perp trades: trade prices bounce between bid and
        # ask, which would feed the maker's own spread back into its volatility estimate.
        vol = realized_vol_per_second(view.index_history(300))
        half_frac = max(self.min_half_spread_bps / 1e4, self.vol_multiplier * vol * math.sqrt(self.vol_horizon))
        half_frac = min(half_frac, self.max_half_spread_bps / 1e4)
        if view.time < self._widen_until:
            half_frac *= self.news_widen
        max_pos = max(1, capacity(view, self.target_leverage))
        q = clamp(view.position / max_pos, -1.0, 1.0)
        reservation = fair * (1.0 - q * self.skew_bps / 1e4)
        half = fair * half_frac
        spacing = max(tick, fair * self.level_spacing_bps / 1e4)
        best_bid, best_ask = view.best_bid, view.best_ask
        level_size = max(self.base_size, self.level_fraction * max_pos)
        quotes: list[tuple[Side, float, int]] = []
        for i in range(self.levels):
            growth = 1.0 + 0.5 * i
            bid_size = int(round(level_size * growth * (1.0 - q)))
            ask_size = int(round(level_size * growth * (1.0 + q)))
            bid = _floor_tick(reservation - half - i * spacing, tick)
            ask = _ceil_tick(reservation + half + i * spacing, tick)
            if best_ask is not None and bid >= best_ask:
                bid = _floor_tick(best_ask - tick - i * spacing, tick)
            if best_bid is not None and ask <= best_bid:
                ask = _ceil_tick(best_bid + tick + i * spacing, tick)
            if bid >= ask:
                continue
            if bid_size > 0:
                quotes.append((Side.BUY, round(bid, 10), bid_size))
            if ask_size > 0:
                quotes.append((Side.SELL, round(ask, 10), ask_size))
        return quotes

    def _quotes(self, view: MarketView, reconcile: bool) -> list[Action]:
        """Turn desired quotes into actions, canceling only orders that are no longer wanted when ``reconcile`` is set."""
        desired = self.desired_quotes(view)
        tick = view.tick_size
        actions: list[Action] = []
        if reconcile:
            wanted: dict[tuple[Side, int], int] = {(side, int(round(price / tick))): size for side, price, size in desired}
            existing = set()
            for order in view.my_orders():
                key = (order.side, int(round(order.price / tick)))
                if key in wanted and key not in existing:
                    existing.add(key)
                else:
                    actions.append(Cancel(order.order_id))
            for side, price, size in desired:
                if (side, int(round(price / tick))) not in existing:
                    actions.append(LimitOrder(side, size, price, tif=TimeInForce.POST_ONLY, tag="quote"))
        else:
            for side, price, size in desired:
                actions.append(LimitOrder(side, size, price, tif=TimeInForce.POST_ONLY, tag="quote"))
        return actions


# =============================================================================
# Retail / noise trader
# =============================================================================


class NoiseTrader(Agent):
    """Retail-style flow.

    When flat it opens a position with probability ``trade_probability``. The
    side is random with a mild bias toward the recent price move (retail
    chases). Size uses a random fraction of equity at ``leverage``. The
    position is closed after a random holding time, at a take-profit on margin
    (ROE), or at a stop-loss if one is set; without a stop the trader may hold
    until liquidated, like real retail.
    """

    agent_type = "retail"
    default_prefix = "retail"

    def __init__(
        self,
        wake_interval: float = 90.0,
        leverage: float = 5.0,
        trade_probability: float = 0.6,
        market_order_probability: float = 0.7,
        mean_holding: float = 900.0,
        take_profit: float = 0.3,
        stop_loss: Optional[float] = None,
        size_fraction: tuple[float, float] = (0.3, 1.0),
        chase_bias: float = 0.25,
        stale_order_seconds: float = 120.0,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.leverage = float(leverage)
        self.trade_probability = float(trade_probability)
        self.market_order_probability = float(market_order_probability)
        self.mean_holding = float(mean_holding)
        self.take_profit = float(take_profit)
        self.stop_loss = None if stop_loss is None else float(stop_loss)
        self.size_fraction = size_fraction
        self.chase_bias = float(chase_bias)
        self.stale_order_seconds = float(stale_order_seconds)
        self._entry_time: Optional[float] = None
        self._close_time: Optional[float] = None

    def on_fill(self, fill: Fill, view: MarketView) -> None:
        """Start the holding clock at the first fill, not at the order, so limit entries that fill late are not closed early."""
        if view.position == 0:
            self._entry_time = None
            self._close_time = None
        elif self._entry_time is None:
            self._entry_time = fill.time

    def on_wakeup(self, view, rng):
        """Manage the open position first; otherwise maybe open one.

        Stale resting entries are canceled so an old limit order cannot fill at
        a price that no longer reflects what the trader wanted.
        """
        actions: list[Action] = []
        stale = [o for o in view.my_orders() if view.time - o.time > self.stale_order_seconds]
        actions.extend(Cancel(o.order_id) for o in stale)
        position = view.position
        if position != 0:
            if self._close_time is None:
                self._close_time = view.time + float(rng.exponential(self.mean_holding))
            me = view.me
            margin_used = max(abs(position) * me.entry_price * view.contract_size / self.leverage, 1e-9)
            roe = me.unrealized_pnl / margin_used
            if view.time >= self._close_time or roe >= self.take_profit or (self.stop_loss is not None and roe <= -self.stop_loss):
                side = Side.SELL if position > 0 else Side.BUY
                actions.append(MarketOrder(side, abs(position), reduce_only=True, tag="close"))
            return actions
        if view.my_orders() or rng.random() >= self.trade_probability:
            return actions
        prices = view.price_history(120)
        move_bps = safe_log_ratio(prices[-1], prices[0]) * 1e4 if len(prices) >= 2 else 0.0
        p_buy = 0.5 + self.chase_bias * math.tanh(move_bps / 25.0)
        side = Side.BUY if rng.random() < p_buy else Side.SELL
        fraction = float(rng.uniform(*self.size_fraction))
        quantity = int(capacity(view, self.leverage) * fraction)
        if quantity < 1:
            return actions
        if rng.random() < self.market_order_probability:
            actions.append(MarketOrder(side, quantity, tag="open"))
        else:
            touch = view.best_bid if side is Side.BUY else view.best_ask
            if touch is None:
                return actions
            offset = float(rng.uniform(0.0, 8.0)) * 1e-4 * touch
            price = touch - offset if side is Side.BUY else touch + offset
            actions.append(LimitOrder(side, quantity, view.round_price(price), tag="open"))
        return actions


# =============================================================================
# Momentum
# =============================================================================


class MomentumTrader(Agent):
    """Trend follower: position proportional to a fast/slow EMA crossover signal."""

    agent_type = "momentum"
    default_prefix = "momentum"

    def __init__(
        self,
        wake_interval: float = 30.0,
        fast_seconds: float = 180.0,
        slow_seconds: float = 1200.0,
        entry_bps: float = 15.0,
        full_signal_bps: float = 80.0,
        leverage: float = 2.0,
        clip_fraction: float = 0.1,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.fast_seconds = float(fast_seconds)
        self.slow_seconds = float(slow_seconds)
        self.entry_bps = float(entry_bps)
        self.full_signal_bps = float(full_signal_bps)
        self.leverage = float(leverage)
        self.clip_fraction = float(clip_fraction)

    def signal_bps(self, view: MarketView) -> Optional[float]:
        """Fast-minus-slow EMA gap in bps, or ``None`` until enough history exists (a half-formed average is just noise)."""
        prices = view.price_history(int(self.slow_seconds))
        if len(prices) < self.slow_seconds / 4:
            return None
        fast = ema(prices, self.fast_seconds)
        slow = ema(prices, self.slow_seconds)
        if not fast or not slow:
            return None
        return (fast - slow) / slow * 1e4

    def on_wakeup(self, view, rng):
        """Scale the target with the signal; the dead band between entry and exit avoids flip-flopping around the threshold."""
        signal = self.signal_bps(view)
        if signal is None:
            return []
        cap = capacity(view, self.leverage)
        position = view.position
        if abs(signal) >= self.entry_bps:
            target = int(sign(signal) * cap * min(1.0, abs(signal) / self.full_signal_bps))
        elif abs(signal) < 0.5 * self.entry_bps:
            target = 0
        else:
            target = position
        return move_toward(view, target, max(1, int(cap * self.clip_fraction)), tag="trend")


# =============================================================================
# Basis arbitrage
# =============================================================================


class BasisArbitrageur(Agent):
    """Trades the perpetual against the index (as if hedged on spot elsewhere).

    It sells the perp when the best bid is at least ``entry_bps`` above the
    index and buys when the best ask is at least ``entry_bps`` below, using
    immediate-or-cancel limit orders so it never trades at a price without
    edge. It scales in with the size of the gap and closes (again with price
    limits) once the perp has converged back to within ``exit_bps`` of the
    index. Expected funding is added to the edge: a positive predicted funding
    rate makes shorting more attractive, as for a real funding-arb desk. This
    is the force that keeps the perpetual close to the index.

    The spot hedge is not simulated, so the arbitrageur's PnL also contains
    index moves on its open position.
    """

    agent_type = "arbitrageur"
    default_prefix = "arb"

    def __init__(
        self,
        wake_interval: float = 4.0,
        entry_bps: float = 12.0,
        exit_bps: float = 2.0,
        full_bps: float = 40.0,
        leverage: float = 3.0,
        clip_fraction: float = 0.05,
        funding_weight: float = 0.5,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.entry_bps = float(entry_bps)
        self.exit_bps = float(exit_bps)
        self.full_bps = float(full_bps)
        self.leverage = float(leverage)
        self.clip_fraction = float(clip_fraction)
        self.funding_weight = float(funding_weight)

    def on_wakeup(self, view, rng):
        """Compare the executable edge at the touch with the entry threshold and trade only behind a price limit."""
        index = view.index_price
        tick = view.tick_size
        funding_bps = self.funding_weight * view.predicted_funding_rate * 1e4
        bid, ask = view.best_bid, view.best_ask
        cap = capacity(view, self.leverage)
        clip = max(1, int(cap * self.clip_fraction))
        position = view.position
        # Executable edge versus the index if the arb hit the touch now.
        sell_edge = (bid - index) / index * 1e4 + funding_bps if bid is not None else None
        buy_edge = (index - ask) / index * 1e4 - funding_bps if ask is not None else None

        if sell_edge is not None and sell_edge >= self.entry_bps:
            target = -int(cap * clamp(sell_edge / self.full_bps, 0.2, 1.0))
            if target < position:
                floor = _ceil_tick(index * (1 + (self.entry_bps - funding_bps) / 1e4), tick)
                return [LimitOrder(Side.SELL, min(clip, position - target), floor, tif=TimeInForce.IOC, tag="basis_open")]
            return []
        if buy_edge is not None and buy_edge >= self.entry_bps:
            target = int(cap * clamp(buy_edge / self.full_bps, 0.2, 1.0))
            if target > position:
                ceiling = _floor_tick(index * (1 - (self.entry_bps + funding_bps) / 1e4), tick)
                return [LimitOrder(Side.BUY, min(clip, target - position), ceiling, tif=TimeInForce.IOC, tag="basis_open")]
            return []
        if position < 0 and (sell_edge is None or sell_edge < self.exit_bps):
            ceiling = _floor_tick(index * (1 + self.exit_bps / 1e4), tick)
            return [LimitOrder(Side.BUY, min(clip, -position), ceiling, tif=TimeInForce.IOC, reduce_only=True, tag="basis_close")]
        if position > 0 and (buy_edge is None or buy_edge < self.exit_bps):
            floor = _ceil_tick(index * (1 - self.exit_bps / 1e4), tick)
            return [LimitOrder(Side.SELL, min(clip, position), floor, tif=TimeInForce.IOC, reduce_only=True, tag="basis_close")]
        return []


# =============================================================================
# Informed trader
# =============================================================================


class InformedTrader(Agent):
    """Trades on private signals about upcoming news.

    Requires ``AgentSpec(private_signals=True)``. When a signal arrives the
    trader builds a position in the expected direction (the more confident the
    signal, the larger), holds it through the public release and closes it
    ``hold_after_news`` seconds later. Its buying and selling before the news
    is what makes prices move before information becomes public.
    """

    agent_type = "informed"
    default_prefix = "informed"

    def __init__(
        self,
        wake_interval: float = 3.0,
        leverage: float = 3.0,
        min_signal_bps: float = 15.0,
        full_signal_bps: float = 80.0,
        hold_after_news: float = 30.0,
        clip_fraction: float = 0.1,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.leverage = float(leverage)
        self.min_signal_bps = float(min_signal_bps)
        self.full_signal_bps = float(full_signal_bps)
        self.hold_after_news = float(hold_after_news)
        self.clip_fraction = float(clip_fraction)
        self._target = 0
        self._exit_time: Optional[float] = None

    def on_signal(self, signal, view, rng):
        """Size the position by signal confidence and schedule the exit relative to the public release."""
        expected_bps = signal.expected_log_jump * 1e4
        if abs(expected_bps) < self.min_signal_bps:
            return []
        cap = capacity(view, self.leverage)
        strength = clamp(abs(expected_bps) / self.full_signal_bps, 0.25, 1.0)
        self._target = int(sign(expected_bps) * cap * strength)
        self._exit_time = signal.public_time + self.hold_after_news
        return move_toward(view, self._target, max(1, int(cap * self.clip_fraction)), tag="informed")

    def on_wakeup(self, view, rng):
        """Keep working toward the target until the exit time, then flatten; flatten leftovers when no signal is pending."""
        if self._exit_time is None:
            return move_toward(view, 0, max(1, abs(view.position)), tag="flatten") if view.position else []
        cap = capacity(view, self.leverage)
        clip = max(1, int(cap * self.clip_fraction))
        if view.time >= self._exit_time:
            self._target = 0
            self._exit_time = None
            return move_toward(view, 0, clip, tag="exit")
        return move_toward(view, self._target, clip, tag="informed")


# =============================================================================
# Whale
# =============================================================================


class Whale(Agent):
    """A large account executing slow accumulation and unwind campaigns.

    State machine: idle -> accumulating -> holding -> unwinding -> idle.
    Accumulation and unwinding are split into child orders over the campaign
    duration (TWAP), half passive at the touch and half aggressive. Its large
    position is very visible under FULL transparency.
    """

    agent_type = "whale"
    default_prefix = "whale"

    def __init__(
        self,
        wake_interval: float = 15.0,
        leverage: float = 3.0,
        start_probability: float = 0.03,
        target_fraction: tuple[float, float] = (0.3, 0.8),
        campaign_seconds: tuple[float, float] = (900.0, 2700.0),
        hold_seconds: tuple[float, float] = (600.0, 1800.0),
        passive_fraction: float = 0.5,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.leverage = float(leverage)
        self.start_probability = float(start_probability)
        self.target_fraction = target_fraction
        self.campaign_seconds = campaign_seconds
        self.hold_seconds = hold_seconds
        self.passive_fraction = float(passive_fraction)
        self.state = "idle"
        self._target = 0
        self._phase_end = 0.0

    def on_wakeup(self, view, rng):
        """Advance the campaign state machine and send the next TWAP slice.

        Resting child orders are canceled every wakeup so unfilled passive
        slices do not pile up behind a moving price.
        """
        actions: list[Action] = [Cancel(o.order_id) for o in view.my_orders()]
        now = view.time
        if self.state == "idle":
            if rng.random() < self.start_probability:
                direction = 1 if rng.random() < 0.5 else -1
                self._target = direction * int(capacity(view, self.leverage) * rng.uniform(*self.target_fraction))
                self._phase_end = now + float(rng.uniform(*self.campaign_seconds))
                self.state = "accumulating"
            return actions
        if self.state == "accumulating" and (now >= self._phase_end or view.position == self._target):
            self.state = "holding"
            self._phase_end = now + float(rng.uniform(*self.hold_seconds))
        elif self.state == "holding" and now >= self._phase_end:
            self.state = "unwinding"
            self._target = 0
            self._phase_end = now + float(rng.uniform(*self.campaign_seconds))
        elif self.state == "unwinding" and view.position == 0:
            self.state = "idle"
            return actions
        if self.state == "holding":
            return actions
        remaining = self._target - view.position
        if remaining == 0:
            return actions
        slices_left = max(1.0, (self._phase_end - now) / self.wake_interval)
        child = max(1, int(math.ceil(abs(remaining) / slices_left)))
        side = Side.BUY if remaining > 0 else Side.SELL
        if rng.random() < self.passive_fraction:
            touch = view.best_bid if side is Side.BUY else view.best_ask
            if touch is not None:
                actions.append(LimitOrder(side, child, touch, tag="twap_passive"))
                return actions
        actions.extend(move_toward(view, view.position + child * side.sign, child, tag="twap"))
        return actions


# =============================================================================
# Transparency-aware agents
# =============================================================================


class CopyTrader(Agent):
    """Mirrors the most profitable visible account (FULL transparency only).

    Every ``reselect_every`` wakeups it picks the active account with the
    highest total PnL among those currently holding at least ``min_notional``
    at a leverage of at least ``min_leader_leverage`` (so it follows traders
    taking directional risk rather than market makers carrying inventory), then
    targets the same direction at its own leverage. In markets with lower
    transparency it cannot see positions and stays flat.
    """

    agent_type = "copy_trader"
    default_prefix = "copy"

    def __init__(
        self,
        wake_interval: float = 30.0,
        leverage: float = 3.0,
        min_notional: float = 5_000.0,
        min_leader_leverage: float = 1.0,
        reselect_every: int = 10,
        clip_fraction: float = 0.34,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.leverage = float(leverage)
        self.min_notional = float(min_notional)
        self.min_leader_leverage = float(min_leader_leverage)
        self.reselect_every = int(reselect_every)
        self.clip_fraction = float(clip_fraction)
        self.leader: Optional[str] = None
        self._wakeups = 0

    def on_wakeup(self, view, rng):
        """Pick a leader only every few wakeups (re-selecting each time would churn the position) and mirror its direction."""
        if not view.can_see(Transparency.FULL):
            return []
        accounts = {a.public_id: a for a in view.accounts() if a.active}
        self._wakeups += 1
        if self.leader not in accounts or self._wakeups % self.reselect_every == 1:
            candidates = [a for a in accounts.values() if a.notional >= self.min_notional and a.leverage >= self.min_leader_leverage and a.total_pnl > 0]
            self.leader = max(candidates, key=lambda a: a.total_pnl).public_id if candidates else None
        cap = capacity(view, self.leverage)
        leader = accounts.get(self.leader) if self.leader else None
        target = 0 if leader is None else int(sign(leader.position) * cap)
        return move_toward(view, target, max(1, int(cap * self.clip_fraction)), tag="copy")


class FlowFollower(Agent):
    """Follows the account with the strongest one-directional flow (ATTRIBUTED or FULL).

    It reads the public trade tape, which at ATTRIBUTED transparency carries
    the buyer and seller of every trade, and sums each account's net bought
    contracts over the last ``window`` seconds. If one account has been
    persistently buying or selling at least ``min_flow`` contracts (a whale
    working a large order, say), it trades in the same direction, betting
    that the flow will continue to push the price. This is "wallet watching"
    as practised on on-chain exchanges. In an anonymous market it stays idle.

    The window is capped by what the market remembers: it reads at most
    ``MarketConfig.recent_trades_kept`` trades (2,000 by default, about
    15 minutes in the reference ecology), so in a much busier market the
    effective window is shorter than ``window`` unless that setting is raised.
    """

    agent_type = "flow_follower"
    default_prefix = "flow"

    def __init__(
        self,
        wake_interval: float = 15.0,
        leverage: float = 3.0,
        window: float = 600.0,
        min_flow: int = 1_000,
        full_flow: int = 5_000,
        clip_fraction: float = 0.25,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.leverage = float(leverage)
        self.window = float(window)
        self.min_flow = int(min_flow)
        self.full_flow = int(full_flow)
        self.clip_fraction = float(clip_fraction)
        self.following: Optional[str] = None

    def net_flows(self, view: MarketView) -> dict[str, int]:
        """Net contracts bought per visible account over the window; the agent's own trades are excluded."""
        flows: dict[str, int] = {}
        start = view.time - self.window
        for trade in view.recent_trades(10_000):
            if trade.time < start or trade.buyer is None:
                continue
            flows[trade.buyer] = flows.get(trade.buyer, 0) + trade.quantity
            flows[trade.seller] = flows.get(trade.seller, 0) - trade.quantity
        flows.pop(self.public_id, None)
        return flows

    def on_wakeup(self, view, rng):
        """Follow the strongest one-directional flow when it exceeds the threshold, otherwise go flat."""
        if not view.can_see(Transparency.ATTRIBUTED):
            return []
        flows = self.net_flows(view)
        cap = capacity(view, self.leverage)
        target = 0
        self.following = None
        if flows:
            leader, flow = max(flows.items(), key=lambda kv: (abs(kv[1]), kv[0]))
            if abs(flow) >= self.min_flow:
                self.following = leader
                target = int(sign(flow) * cap * clamp(abs(flow) / self.full_flow, 0.25, 1.0))
        return move_toward(view, target, max(1, int(cap * self.clip_fraction)), tag="flow")


class LiquidationHunter(Agent):
    """Pushes the price into visible liquidation clusters (FULL transparency only).

    Every wakeup it looks at every public account's liquidation price. For the
    positions whose liquidation price is within ``search_bps`` of the mark on
    one side, it compares

    - ``forced``: the contracts that would be force-closed if the price got there,
    - ``push``: the contracts it would have to trade to walk the book to that price.

    When ``forced >= min_ratio * push`` (the liquidations would overwhelm the
    cost of getting there) it trades ``push`` contracts toward the cluster and
    rests orders at the cluster price on the other side, so the forced flow of
    the cascade closes its position at a profit. Unfilled positions are closed
    gradually after ``max_hold`` seconds. In less transparent markets it cannot
    see liquidation prices and stays idle.

    Because the rule is economic, the hunter is mostly quiet in calm markets:
    makers and arbitrageurs absorb a push, and a trader at 20x is liquidated
    about 3.75% away, outside the default ``search_bps`` band, so only the
    most levered accounts are ever in reach. It hunts about once per 4-hour
    session in the reference ecology (or not at all) and several times in the
    ``volatile`` scenario. That is intended: hunting pays in thin, volatile
    moments, and the thresholds are deliberately not tuned to make it trade.
    """

    agent_type = "liquidation_hunter"
    default_prefix = "hunter"

    def __init__(
        self,
        wake_interval: float = 5.0,
        leverage: float = 5.0,
        search_bps: float = 120.0,
        min_ratio: float = 1.5,
        min_forced: int = 200,
        max_hold: float = 300.0,
        exit_clip_fraction: float = 0.1,
        cooldown: float = 120.0,
    ) -> None:
        super().__init__(wake_interval=wake_interval)
        self.leverage = float(leverage)
        self.search_bps = float(search_bps)
        self.min_ratio = float(min_ratio)
        self.min_forced = int(min_forced)
        self.max_hold = float(max_hold)
        self.exit_clip_fraction = float(exit_clip_fraction)
        self.cooldown = float(cooldown)
        self.hunting = False
        self._started = 0.0
        self._cooldown_until = 0.0
        self.hunts = 0

    def clusters(self, view: MarketView) -> dict[int, tuple[int, Optional[float]]]:
        """``{-1: (forced_contracts, far_price) below mark, +1: (...) above mark}``.

        ``far_price`` is the liquidation price furthest from the mark within the
        search window, i.e. where the whole cluster would have been triggered.
        """
        mark = view.mark_price
        out: dict[int, tuple[int, Optional[float]]] = {-1: (0, None), 1: (0, None)}
        for account in view.accounts():
            price = account.liquidation_price
            if price is None or not account.active or account.position == 0:
                continue
            distance_bps = (price - mark) / mark * 1e4
            if account.position > 0 and -self.search_bps <= distance_bps < 0:
                forced, far = out[-1]
                out[-1] = (forced + account.position, price if far is None else min(far, price))
            elif account.position < 0 and 0 < distance_bps <= self.search_bps:
                forced, far = out[1]
                out[1] = (forced - account.position, price if far is None else max(far, price))
        return out

    @staticmethod
    def push_cost(view: MarketView, direction: int, target_price: float) -> Optional[int]:
        """Contracts needed to trade the book through ``target_price``; ``None`` if the book is too thin to tell."""
        bids, asks = view.book(depth=200)
        levels = asks if direction > 0 else bids
        needed = 0
        for level in levels:
            if (direction > 0 and level.price > target_price) or (direction < 0 and level.price < target_price):
                return needed
            needed += level.quantity
        return None

    def on_wakeup(self, view, rng):
        """Hunt only from flat and after the cooldown, and only when the forced flow outweighs the cost of pushing the price there."""
        if not view.can_see(Transparency.FULL):
            return []
        now = view.time
        position = view.position
        if self.hunting:
            if position == 0 and not view.my_orders():
                self.hunting = False
                self._cooldown_until = now + self.cooldown
                return []
            if now - self._started > self.max_hold:
                clip = max(1, int(capacity(view, self.leverage) * self.exit_clip_fraction))
                actions: list[Action] = [CancelAll()]
                actions.extend(move_toward(view, 0, clip, tag="hunt_exit"))
                if position == 0:
                    self.hunting = False
                    self._cooldown_until = now + self.cooldown
                return actions
            return []
        if position != 0:
            clip = max(1, int(capacity(view, self.leverage) * self.exit_clip_fraction))
            return move_toward(view, 0, clip, tag="hunt_flatten")
        if now < self._cooldown_until:
            return []
        cap = capacity(view, self.leverage)
        best: Optional[tuple[float, int, int, float]] = None
        for direction, (forced, far_price) in self.clusters(view).items():
            if far_price is None or forced < self.min_forced:
                continue
            push = self.push_cost(view, direction, far_price)
            if push is None or push == 0 or push > cap:
                continue
            ratio = forced / push
            if ratio >= self.min_ratio and (best is None or ratio > best[0]):
                best = (ratio, direction, push, far_price)
        if best is None:
            return []
        _, direction, push, far_price = best
        self.hunting = True
        self.hunts += 1
        self._started = now
        side = Side.BUY if direction > 0 else Side.SELL
        # Push, then rest the exit at the cluster where the forced orders will arrive.
        return [
            MarketOrder(side, push, tag="hunt_push"),
            LimitOrder(side.opposite, push, view.round_price(far_price), tag="hunt_exit"),
        ]
