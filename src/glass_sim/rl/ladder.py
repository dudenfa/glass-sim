"""A richer discrete action menu: quotes at several price offsets and sizes.

The basic menu (``ActionSpace``) places one order per side, at the best price,
with a fixed size. ``LadderActionSpace`` also lets the agent choose the
distance from the best price (``ladder_offsets_bps``) and the size
(``ladder_sizes``, multiples of equity). Actions: hold, cancel_all,
close_position, a market buy and sell per size, and a post-only bid and ask
per (offset, size). With ``max_orders_per_side=1`` a new quote replaces the
agent's quote on that side; with more, quotes accumulate and the oldest is
canceled when the side is full. Default settings give 23 actions.
"""

from __future__ import annotations

from ..types import Side
from .actions import ActionConfig, ActionSpace


def _fmt(value: float) -> str:
    return f"{value:g}"


class LadderActionSpace(ActionSpace):
    """Market orders and post-only quotes at several offsets and sizes."""

    def __init__(self, config: ActionConfig = ActionConfig(kind="ladder")) -> None:
        super().__init__(config)
        if not config.ladder_offsets_bps or not config.ladder_sizes or config.max_orders_per_side < 1:
            raise ValueError("ladder needs offsets, sizes and max_orders_per_side >= 1")
        # Each entry: (name, kind, side, offset in bps, size as multiple of equity).
        entries = [("hold", "hold", None, 0.0, 0.0), ("cancel_all", "cancel", None, 0.0, 0.0), ("close_position", "close", None, 0.0, 0.0)]
        for size in config.ladder_sizes:
            entries.append((f"market_buy_{_fmt(size)}x", "market", Side.BUY, 0.0, float(size)))
            entries.append((f"market_sell_{_fmt(size)}x", "market", Side.SELL, 0.0, float(size)))
        for side, label in ((Side.BUY, "bid"), (Side.SELL, "ask")):
            for offset in config.ladder_offsets_bps:
                for size in config.ladder_sizes:
                    entries.append((f"{label}_{_fmt(offset)}bps_{_fmt(size)}x", "quote", side, float(offset), float(size)))
        self.entries = entries
        self.names = [e[0] for e in entries]

    @property
    def n(self) -> int:
        return len(self.entries)

    def _anchor(self, view, kind: str, side: Side):
        """Price the action needs: the opposite touch for market orders, the own-side touch for quotes."""
        if kind == "market":
            return view.best_ask if side is Side.BUY else view.best_bid
        return view.best_bid if side is Side.BUY else view.best_ask

    def mask(self, view):
        import numpy as np

        valid = np.zeros(self.n, dtype=bool)
        valid[0] = True
        if not view.me.active or view.me.equity <= 0:
            return valid
        has_orders = bool(view.my_orders())
        replace = self.config.max_orders_per_side == 1
        for i, (_name, kind, side, _offset, size) in enumerate(self.entries):
            if kind == "cancel":
                valid[i] = has_orders
            elif kind == "close":
                pos = view.position
                valid[i] = pos != 0 and (view.best_bid if pos > 0 else view.best_ask) is not None
            elif kind in ("market", "quote"):
                qty = self._size(view, size)
                valid[i] = self._anchor(view, kind, side) is not None and self._fits(view, side, qty, replace_side=(kind == "quote" and replace))
        return valid

    def decode(self, index: int, view) -> list:
        from ..types import Cancel, LimitOrder, MarketOrder, TimeInForce

        _name, kind, side, offset, size = self.entries[int(index)]
        if kind == "hold":
            return []
        if kind == "cancel":
            return [Cancel(o.order_id) for o in view.my_orders()]
        if kind == "close":
            pos = view.position
            if pos == 0:
                return []
            return [MarketOrder(Side.SELL if pos > 0 else Side.BUY, abs(pos), reduce_only=True, tag="rl")]
        qty = self._size(view, size)
        if kind == "market":
            return [MarketOrder(side, qty, tag="rl")]
        anchor = self._anchor(view, kind, side)
        if anchor is None:
            return []
        shift = offset / 1e4
        price = view.round_price(anchor * (1 - shift) if side is Side.BUY else anchor * (1 + shift))
        same_side = [o for o in view.my_orders() if o.side is side]
        if self.config.max_orders_per_side == 1:
            stale = same_side
        elif len(same_side) >= self.config.max_orders_per_side:
            stale = [min(same_side, key=lambda o: o.time)]
        else:
            stale = []
        actions = [Cancel(o.order_id) for o in stale]
        actions.append(LimitOrder(side, qty, price, tif=TimeInForce.POST_ONLY, tag="rl"))
        return actions
