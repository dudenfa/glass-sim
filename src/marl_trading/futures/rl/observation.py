"""Observation encoding: what a learning agent sees, as a fixed-size vector.

The encoder turns a ``MarketView`` into ``np.float32`` features scaled to
roughly [-5, 5]. It has three blocks:

- **market**: always visible (book, prices, returns, volatility, funding,
  open interest, recent news, trade flow);
- **account**: the agent's own state (position, leverage, PnL, margin,
  distance to liquidation, resting orders);
- **transparency**: what the market's transparency level reveals about other
  agents (aggregate positioning, liquidation clusters, strongest account flow).
  Hidden features are zero, and two flags say which parts are visible, so the
  vector has the same size at every level and one network can be trained and
  compared across levels;
- **signal** (optional): the agent's pending private news signal, used to test
  that the learning pipeline can pick up a known edge.

``feature_names`` documents every position in the vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..types import Side, Transparency
from ..views import MarketView

CLIP = 5.0


def _ret_bps(series: Sequence[float], lag: int) -> float:
    if len(series) <= lag or not series[-1] or not series[-1 - lag]:
        return 0.0
    return (series[-1] / series[-1 - lag] - 1.0) * 1e4


def _vol_bps(series: Sequence[float]) -> float:
    clean = [p for p in series if p]
    if len(clean) < 5:
        return 0.0
    returns = np.diff(np.log(np.asarray(clean, dtype=float)))
    return float(returns.std() * 1e4)


@dataclass(frozen=True)
class ObservationConfig:
    include_transparency: bool = True
    include_signal: bool = False
    flow_window: float = 600.0
    liquidation_band_bps: float = 100.0


MARKET_FEATURES = [
    "spread_bps",
    "mid_vs_index_bps",
    "mark_vs_index_bps",
    "return_10s_bps",
    "return_60s_bps",
    "return_300s_bps",
    "index_return_60s_bps",
    "volatility_60s_bps",
    "book_imbalance_5",
    "log_depth_bid",
    "log_depth_ask",
    "signed_taker_flow_60s",
    "predicted_funding_bps",
    "time_to_funding",
    "open_interest_change_300s",
    "seconds_since_news",
    "last_news_jump_bps",
]
ACCOUNT_FEATURES = [
    "signed_leverage",
    "unrealized_pnl_pct",
    "total_return_pct",
    "available_margin_frac",
    "liquidation_distance",
    "has_bid",
    "has_ask",
]
TRANSPARENCY_FEATURES = [
    "sees_owners",
    "sees_positions",
    "crowd_net_positioning",
    "high_leverage_net_positioning",
    "long_liquidations_below",
    "short_liquidations_above",
    "largest_account_signed_leverage",
    "strongest_flow_signed",
]
SIGNAL_FEATURES = ["signal_expected_jump_bps", "signal_seconds_to_public"]


class ObservationEncoder:
    def __init__(self, config: Optional[ObservationConfig] = None) -> None:
        self.config = config or ObservationConfig()
        names = MARKET_FEATURES + ACCOUNT_FEATURES
        if self.config.include_transparency:
            names = names + TRANSPARENCY_FEATURES
        if self.config.include_signal:
            names = names + SIGNAL_FEATURES
        self.feature_names: List[str] = names

    @property
    def size(self) -> int:
        return len(self.feature_names)

    # ----------------------------------------------------------------- blocks

    def _market(self, view: MarketView) -> List[float]:
        index = view.index_price
        mark = view.mark_price
        mid = view.mid if view.mid is not None else view.last_price
        spread_bps = (view.spread / mid * 1e4) if view.spread is not None and mid else 50.0
        prices = view.price_history(300)
        index_hist = view.index_history(60)
        bids, asks = view.book(depth=5)
        bid_qty = sum(level.quantity for level in bids)
        ask_qty = sum(level.quantity for level in asks)
        imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) else 0.0
        flow = 0
        for trade in view.recent_trades(300):
            if trade.time >= view.time - 60.0:
                flow += trade.quantity if trade.aggressor is Side.BUY else -trade.quantity
        depth = max(1.0, (bid_qty + ask_qty) / 2.0)
        oi_hist = view.open_interest_history(300)
        oi_change = (oi_hist[-1] - oi_hist[0]) / max(1.0, oi_hist[-1]) if len(oi_hist) > 1 else 0.0
        news = view.recent_news(1)
        since_news = min(1.0, (view.time - news[-1].time) / 600.0) if news else 1.0
        news_jump = news[-1].log_jump * 1e4 if news and view.time - news[-1].time < 600.0 else 0.0
        return [
            spread_bps / 10.0,
            (mid - index) / index * 1e4 / 20.0,
            (mark - index) / index * 1e4 / 20.0,
            _ret_bps(prices, 10) / 10.0,
            _ret_bps(prices, 60) / 20.0,
            _ret_bps(prices, 299) / 50.0,
            _ret_bps(index_hist, len(index_hist) - 1) / 20.0 if len(index_hist) > 1 else 0.0,
            _vol_bps(view.mark_history(60)) / 5.0,
            imbalance,
            math.log1p(bid_qty) / 5.0,
            math.log1p(ask_qty) / 5.0,
            flow / depth,
            view.predicted_funding_rate * 1e4 / 5.0,
            max(0.0, view.next_funding_time - view.time) / 3600.0,
            oi_change * 10.0,
            since_news,
            news_jump / 50.0,
        ]

    def _account(self, view: MarketView) -> List[float]:
        me = view.me
        mark = view.mark_price
        equity = max(me.equity, 1e-9)
        signed_leverage = me.position * mark * view.contract_size / equity
        liq = me.liquidation_price
        if liq is None or me.position == 0:
            liq_distance = 1.0
        else:
            liq_distance = min(1.0, abs(mark - liq) / mark / 0.10)
        orders = view.my_orders()
        return [
            signed_leverage / 5.0,
            me.unrealized_pnl / me.initial_collateral * 100.0 / 2.0,
            (me.equity / me.initial_collateral - 1.0) * 100.0 / 10.0,
            max(-1.0, min(1.0, me.available_margin / equity)),
            liq_distance,
            1.0 if any(o.side is Side.BUY for o in orders) else 0.0,
            1.0 if any(o.side is Side.SELL for o in orders) else 0.0,
        ]

    def _transparency(self, view: MarketView) -> List[float]:
        sees_owners = view.can_see(Transparency.ATTRIBUTED)
        sees_positions = view.can_see(Transparency.FULL)
        crowd = high_lev = liq_below = liq_above = largest = 0.0
        flow_signal = 0.0
        bids, asks = view.book(depth=10)
        depth = max(1.0, (sum(l.quantity for l in bids) + sum(l.quantity for l in asks)) / 2.0)
        if sees_positions:
            mark = view.mark_price
            accounts = [a for a in view.accounts() if a.active and a.position != 0]
            gross = sum(abs(a.position) for a in accounts)
            if gross:
                crowd = sum(a.position for a in accounts) / gross
            levered = [a for a in accounts if a.leverage >= 10.0]
            levered_gross = sum(abs(a.position) for a in levered)
            if levered_gross:
                high_lev = sum(a.position for a in levered) / levered_gross
            band = self.config.liquidation_band_bps / 1e4
            below = sum(a.position for a in accounts if a.position > 0 and a.liquidation_price and mark * (1 - band) <= a.liquidation_price < mark)
            above = sum(-a.position for a in accounts if a.position < 0 and a.liquidation_price and mark < a.liquidation_price <= mark * (1 + band))
            liq_below = math.log1p(below / depth)
            liq_above = math.log1p(above / depth)
            if accounts:
                top = max(accounts, key=lambda a: a.notional)
                largest = (1 if top.position > 0 else -1) * min(top.leverage, 20.0) / 5.0
        if sees_owners:
            flows = {}
            start = view.time - self.config.flow_window
            for trade in view.recent_trades(2000):
                if trade.time < start:
                    continue
                flows[trade.buyer] = flows.get(trade.buyer, 0) + trade.quantity
                flows[trade.seller] = flows.get(trade.seller, 0) - trade.quantity
            flows.pop(view.me.public_id, None)
            if flows:
                _, strongest = max(flows.items(), key=lambda kv: abs(kv[1]))
                flow_signal = strongest / depth
        return [1.0 if sees_owners else 0.0, 1.0 if sees_positions else 0.0, crowd, high_lev, liq_below, liq_above, largest, flow_signal]

    def _signal(self, view: MarketView) -> List[float]:
        pending = [s for s in view.signals() if s.public_time > view.time]
        if not pending:
            return [0.0, 0.0]
        signal = min(pending, key=lambda s: s.public_time)
        return [signal.expected_log_jump * 1e4 / 50.0, (signal.public_time - view.time) / 90.0]

    # ------------------------------------------------------------------ main

    def encode(self, view: MarketView) -> np.ndarray:
        values = self._market(view) + self._account(view)
        if self.config.include_transparency:
            values += self._transparency(view)
        if self.config.include_signal:
            values += self._signal(view)
        vector = np.asarray(values, dtype=np.float32)
        np.nan_to_num(vector, copy=False, nan=0.0, posinf=CLIP, neginf=-CLIP)
        return np.clip(vector, -CLIP, CLIP)
