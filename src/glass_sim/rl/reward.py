"""Reward for learning agents.

Design:

The base reward is the change, between two consecutive decisions, of the
account's **exit value**: what the account would be worth if it closed its
whole position right now by walking the real order book and paying the taker
fee::

    exit_value = collateral + position_value_at_exit - closing_fee

``collateral`` already contains realized PnL, every fee paid, every funding
payment and every liquidation penalty, so all of them flow into the reward
automatically at the moment they happen. Valuing the open position at its
executable exit (not at the mark) means:

- opening a position costs reward immediately (half the spread plus fees);
- a position too large for the book, or held in an illiquid market, is worth
  less than its mark value, so hoarding illiquid inventory is not rewarded
  (the V1 failure mode) without any arbitrary inventory penalty.

If the book cannot absorb the whole position, the unfillable remainder is
valued at the mark minus ``unfillable_haircut``.

On top of the base reward:

- ``liquidation_penalty``: an extra fraction of initial capital subtracted each
  time the agent is liquidated (the economic loss is already in the base
  reward; this teaches avoidance);
- ``leverage_penalty``: optional, quadratic in leverage above
  ``leverage_soft_cap``, per decision; 0 by default;
- ``inactivity_penalty``: optional, per decision without a position or resting
  orders; 0 by default (V1 showed it distorts behaviour).

The reward is scaled by ``scale / initial_collateral`` so that it is
comparable across capital sizes (default: 1 unit = 1% of initial capital).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..types import Side

if TYPE_CHECKING:  # pragma: no cover
    from ..market import PerpetualMarket


@dataclass(frozen=True)
class RewardConfig:
    """Reward scale and penalties; the module docstring explains each."""

    scale: float = 100.0
    liquidation_penalty: float = 0.05
    leverage_penalty: float = 0.0
    leverage_soft_cap: float = 5.0
    inactivity_penalty: float = 0.0
    unfillable_haircut: float = 0.10


def exit_value(market: PerpetualMarket, agent_id: str, haircut: float = 0.10) -> float:
    """Account value if the whole position were closed now against the book."""
    account = market.accounts[agent_id]
    position = account.position
    if position == 0:
        return account.collateral
    side = Side.SELL if position > 0 else Side.BUY
    quantity = abs(position)
    filled, notional_ticks = market.book.estimate_fill(side, quantity, agent_id)
    cs = account.contract_size
    notional = notional_ticks * market.tick_size * cs
    remainder = quantity - filled
    if remainder:
        worst = market.mark_price * (1 - haircut) if side is Side.SELL else market.mark_price * (1 + haircut)
        notional += remainder * worst * cs
    closing_fee = market.contract.taker_fee * notional
    cost_basis = quantity * account.entry_price * cs
    pnl = notional - cost_basis if position > 0 else cost_basis - notional
    return account.collateral + pnl - closing_fee


@dataclass
class RewardTracker:
    """Computes the reward between consecutive decisions of one agent."""

    market: PerpetualMarket
    agent_id: str
    config: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Re-baseline on the current state (episode start)."""
        account = self.market.accounts[self.agent_id]
        self.initial = account.initial_collateral
        self._value = exit_value(self.market, self.agent_id, self.config.unfillable_haircut)
        self._liquidations = account.liquidations
        self._fees = account.fees_paid
        self._funding = account.funding_paid
        self._realized = account.realized_pnl

    def step(self) -> tuple[float, dict[str, float]]:
        """Reward since the previous call, plus a breakdown for logging."""
        cfg = self.config
        account = self.market.accounts[self.agent_id]
        value = exit_value(self.market, self.agent_id, cfg.unfillable_haircut)
        delta = value - self._value
        new_liquidations = account.liquidations - self._liquidations
        mark = self.market.mark_price
        leverage = account.leverage(mark) if account.position else 0.0
        penalty = new_liquidations * cfg.liquidation_penalty * self.initial
        if cfg.leverage_penalty and leverage > cfg.leverage_soft_cap:
            penalty += cfg.leverage_penalty * (leverage - cfg.leverage_soft_cap) ** 2 * self.initial / 1000.0
        if cfg.inactivity_penalty and account.position == 0 and account.open_orders == 0:
            penalty += cfg.inactivity_penalty * self.initial / 1000.0
        reward = (delta - penalty) * cfg.scale / self.initial
        info = {
            "reward": reward,
            "exit_value": value,
            "value_change": delta,
            "penalty": penalty,
            "fees_change": account.fees_paid - self._fees,
            "funding_change": account.funding_paid - self._funding,
            "realized_change": account.realized_pnl - self._realized,
            "liquidations": float(new_liquidations),
            "position": float(account.position),
            "leverage": leverage,
            "equity": account.equity(mark),
        }
        self._value = value
        self._liquidations = account.liquidations
        self._fees = account.fees_paid
        self._funding = account.funding_paid
        self._realized = account.realized_pnl
        return reward, info
