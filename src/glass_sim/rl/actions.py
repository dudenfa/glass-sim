"""Discrete action menu for learning agents, with validity masks.

Sizes are expressed in units of the agent's own equity, so the same policy
works for any capital: a "small" trade adds ``small_leverage`` × equity of
notional, a "large" one ``large_leverage`` × equity. The agent's total
exposure, counting resting orders, is capped at ``max_leverage``.

=====  ====================  ============================================
index  name                  effect
=====  ====================  ============================================
0      hold                  do nothing
1      buy_small             market buy, small size
2      buy_large             market buy, large size
3      sell_small            market sell, small size
4      sell_large            market sell, large size
5      bid_at_touch          replace own bids with one post-only bid at the best bid (small size)
6      ask_at_touch          replace own asks with one post-only ask at the best ask (small size)
7      cancel_all            cancel all resting orders
8      close_position        reduce-only market order closing the whole position
=====  ====================  ============================================

The mask marks an action invalid when it cannot work: no liquidity on the
side it needs, it would exceed the exposure cap, there is nothing to cancel or
nothing to close. Masked actions are never sampled by MaskablePPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Action, Cancel, LimitOrder, MarketOrder, Side, TimeInForce
from ..views import MarketView

ACTION_NAMES = [
    "hold",
    "buy_small",
    "buy_large",
    "sell_small",
    "sell_large",
    "bid_at_touch",
    "ask_at_touch",
    "cancel_all",
    "close_position",
]


@dataclass(frozen=True)
class ActionConfig:
    """Order sizes as multiples of equity, the exposure cap, and which action menu to use.

    ``kind="basic"`` is the 9-action menu below. ``kind="ladder"`` is the richer
    menu of ``LadderActionSpace`` (several price offsets and sizes per side);
    its shape is set by ``ladder_offsets_bps``, ``ladder_sizes`` and
    ``max_orders_per_side``.
    """

    small_leverage: float = 0.5
    large_leverage: float = 2.0
    max_leverage: float = 10.0
    kind: str = "basic"
    ladder_offsets_bps: tuple = (0.0, 2.0, 5.0, 10.0)
    ladder_sizes: tuple = (0.5, 2.0)
    max_orders_per_side: int = 1


class ActionSpace:
    """Builds orders from action indices and tells which actions make sense in the current state."""

    names = ACTION_NAMES

    def __init__(self, config: ActionConfig = ActionConfig()) -> None:
        self.config = config

    @property
    def n(self) -> int:
        """Number of actions."""
        return len(ACTION_NAMES)

    # ---------------------------------------------------------------- sizing

    def _size(self, view: MarketView, leverage: float) -> int:
        """Contracts worth ``leverage`` times equity at the mark (at least one)."""
        equity = view.me.equity
        price = view.mark_price * view.contract_size
        if equity <= 0 or price <= 0:
            return 0
        return max(1, int(equity * leverage / price))

    def _cap(self, view: MarketView) -> int:
        """Largest worst-case position allowed."""
        return self._size(view, self.config.max_leverage)

    def _fits(self, view: MarketView, side: Side, quantity: int, replace_side: bool = False) -> bool:
        """Whether adding ``quantity`` on ``side`` keeps worst-case exposure within the cap.

        Orders that reduce the absolute position are always allowed.
        """
        me = view.me
        pos = me.position
        if side is Side.BUY and pos < 0 and quantity <= -pos and not replace_side:
            return True
        if side is Side.SELL and pos > 0 and quantity <= pos and not replace_side:
            return True
        open_buy = 0 if (replace_side and side is Side.BUY) else me.open_buy_quantity
        open_sell = 0 if (replace_side and side is Side.SELL) else me.open_sell_quantity
        extra_buy = quantity if side is Side.BUY else 0
        extra_sell = quantity if side is Side.SELL else 0
        worst = max(abs(pos + open_buy + extra_buy), abs(pos - open_sell - extra_sell))
        return worst <= self._cap(view)

    # ------------------------------------------------------------------ masks

    def mask(self, view: MarketView) -> np.ndarray:
        """Validity of every action right now; ``hold`` is always valid so the policy can never be stuck."""
        valid = np.zeros(self.n, dtype=bool)
        valid[0] = True
        if not view.me.active or view.me.equity <= 0:
            return valid
        small = self._size(view, self.config.small_leverage)
        large = self._size(view, self.config.large_leverage)
        has_ask, has_bid = view.best_ask is not None, view.best_bid is not None
        valid[1] = has_ask and self._fits(view, Side.BUY, small)
        valid[2] = has_ask and self._fits(view, Side.BUY, large)
        valid[3] = has_bid and self._fits(view, Side.SELL, small)
        valid[4] = has_bid and self._fits(view, Side.SELL, large)
        valid[5] = has_bid and self._fits(view, Side.BUY, small, replace_side=True)
        valid[6] = has_ask and self._fits(view, Side.SELL, small, replace_side=True)
        valid[7] = bool(view.my_orders())
        valid[8] = view.position != 0 and (has_bid if view.position > 0 else has_ask)
        return valid

    # ---------------------------------------------------------------- decode

    def decode(self, index: int, view: MarketView) -> list[Action]:
        """Orders for action ``index`` (the caller checks the mask first)."""
        name = ACTION_NAMES[int(index)]
        small = self._size(view, self.config.small_leverage)
        large = self._size(view, self.config.large_leverage)
        if name == "hold":
            return []
        if name == "buy_small":
            return [MarketOrder(Side.BUY, small, tag="rl")]
        if name == "buy_large":
            return [MarketOrder(Side.BUY, large, tag="rl")]
        if name == "sell_small":
            return [MarketOrder(Side.SELL, small, tag="rl")]
        if name == "sell_large":
            return [MarketOrder(Side.SELL, large, tag="rl")]
        if name in ("bid_at_touch", "ask_at_touch"):
            side = Side.BUY if name == "bid_at_touch" else Side.SELL
            price = view.best_bid if side is Side.BUY else view.best_ask
            if price is None:
                return []
            actions: list[Action] = [Cancel(o.order_id) for o in view.my_orders() if o.side is side]
            actions.append(LimitOrder(side, small, price, tif=TimeInForce.POST_ONLY, tag="rl"))
            return actions
        if name == "cancel_all":
            return [Cancel(o.order_id) for o in view.my_orders()]
        if name == "close_position":
            position = view.position
            if position == 0:
                return []
            return [MarketOrder(Side.SELL if position > 0 else Side.BUY, abs(position), reduce_only=True, tag="rl")]
        raise ValueError(f"unknown action index {index}")


def make_action_space(config: ActionConfig = ActionConfig()) -> ActionSpace:
    """The action menu selected by ``config.kind`` ("basic" or "ladder")."""
    if config.kind == "basic":
        return ActionSpace(config)
    if config.kind == "ladder":
        from .ladder import LadderActionSpace

        return LadderActionSpace(config)
    raise ValueError(f"unknown action kind {config.kind!r} (expected 'basic' or 'ladder')")
