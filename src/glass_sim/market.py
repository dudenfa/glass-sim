"""The perpetual-futures market simulator.

``PerpetualMarket`` is a discrete-event simulation in continuous time:

- agents wake up at random times (each agent has its own Poisson clock), look
  at a ``MarketView`` and send actions;
- actions reach the exchange after a latency, then are matched in a
  price-time priority order book;
- every fill updates two margin accounts; fees go to the exchange;
- the index (oracle) price evolves on its own clock and jumps on news;
- the mark price is a robust median of book mid, last trade and index-plus-basis;
- funding is settled periodically between longs and shorts;
- accounts whose equity falls below maintenance margin are liquidated, with
  auto-deleveraging and an insurance fund covering what the book cannot absorb.

Typical use::

    from glass_sim import PerpetualMarket, scenarios

    market = PerpetualMarket(scenarios.default(seed=1))
    results = market.run()
    print(results.summary())

Events at the same timestamp are processed in a fixed priority order (exogenous
updates, then order arrivals, then agent wakeups, then recording), and ties are
broken by insertion sequence, so runs are exactly reproducible for a seed.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from itertools import count
from statistics import median
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .account import AccountSnapshot, MarginAccount
from .agents.base import Agent, ExternalAgent, actions_list
from .book import MatchResult, OrderBook, RestingOrder
from .config import AgentSpec, MarketConfig
from .processes import IndexProcess, NewsItem, PrivateSignal, generate_news, generate_signals, sample_collateral
from .recorder import Recorder
from .types import (
    Action,
    Cancel,
    CancelAll,
    Fill,
    FundingRecord,
    LimitOrder,
    LiquidationRecord,
    MarketOrder,
    NewsRecord,
    OrderEvent,
    Side,
    TimeInForce,
    TradeRecord,
)
from .views import AccountView, MarketView, PublicAccount

if TYPE_CHECKING:  # pragma: no cover
    from .results import SimulationResult

# Event priorities for equal timestamps (lower runs first).
_P_NEWS = 0
_P_INDEX = 1
_P_SIGNAL = 2
_P_FUNDING = 3
_P_ARRIVAL = 4
_P_WAKEUP = 5
_P_RECORD = 6

MARKET_ORDER_SLIPPAGE = 0.05
LIQUIDATION_SLIPPAGE = 0.10
BASIS_EMA_SECONDS = 150.0
ACCOUNT_SAMPLE_EVERY = 10  # record-interval multiples


def _stable_hash(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


class PerpetualMarket:
    """The exchange, the clock and the participants of one simulated session.

    Agents never touch this object: they observe through a ``MarketView`` and
    their actions travel through the event queue, so a seed always replays the
    same session. Researchers use the direct controls (``execute``,
    ``shock_index``, ``run_until``) for interventions.
    """

    def __init__(self, config: MarketConfig, agents: Optional[Sequence[tuple[Agent, float]]] = None, *, keep_order_events: bool = False) -> None:
        """Create a market.

        ``agents`` optionally overrides ``config.agents`` with already-built
        ``(agent, collateral)`` pairs; agent ids must be set on the instances.
        """
        self.config = config
        self.tick_size = float(config.contract.tick_size)
        self.contract = config.contract
        self._imr = config.contract.initial_margin_ratio
        self._mmr = config.contract.mmr
        self.keep_order_events = keep_order_events
        self._explicit_agents = list(agents) if agents is not None else None
        self.reset(config.seed)

    # ================================================================ setup

    def reset(self, seed: Optional[int] = None) -> None:
        """Rebuild the market from its configuration (optionally with a new seed)."""
        config = self.config
        self.seed = int(config.seed if seed is None else seed)
        root = np.random.SeedSequence(self.seed)
        exo_index, exo_news, exo_signals, setup = root.spawn(4)
        self._index_rng = np.random.default_rng(exo_index)
        self._setup_rng = np.random.default_rng(setup)

        self.time = 0.0
        self.book = OrderBook()
        self.accounts: dict[str, MarginAccount] = {}
        self.agents: dict[str, Agent] = {}
        self._agent_rngs: dict[str, np.random.Generator] = {}
        self._latency_rngs: dict[str, np.random.Generator] = {}
        self._signal_subscribers: set = set()
        self._external_ids: set = set()
        self._public_ids: dict[str, str] = {}
        self._views: dict[str, MarketView] = {}

        self.index = IndexProcess(config.index, config.contract.initial_price, self._index_rng)
        self.index_price = self.index.price
        self.news_items: list[NewsItem] = generate_news(config.news, config.duration, np.random.default_rng(exo_news))
        self.signals: list[PrivateSignal] = generate_signals(config.news, self.news_items, np.random.default_rng(exo_signals))
        self.public_news: list[NewsItem] = []
        self._released_news: set = set()
        self._delivered_signals: list[PrivateSignal] = []

        self.last_price = float(config.contract.initial_price)
        self.mark_price = float(config.contract.initial_price)
        self._basis_ema = 0.0
        self.last_funding_rate = 0.0
        self.next_funding_time = config.funding.interval
        self._premium_samples: list[float] = []

        self.insurance_fund = float(config.liquidation.insurance_fund)
        self.fees_collected = 0.0
        self.total_deposits = 0.0

        self.recorder = Recorder(keep_order_events=self.keep_order_events)
        self.recent_trades: deque[TradeRecord] = deque(maxlen=config.recent_trades_kept)
        self._trade_ids = count(1)
        self._order_ids = count(1)
        self._event_seq = count(1)
        # Heap of (time, priority, sequence, kind, payload) tuples; see _push().
        self._queue: list[tuple] = []

        self._bar: dict[str, float] = {}
        self._reset_bar()
        self._price_samples: deque[float] = deque(maxlen=3600)
        self._mark_samples: deque[float] = deque(maxlen=3600)
        self._index_samples: deque[float] = deque(maxlen=3600)
        self._record_index = 0

        self._pending_decision: Optional[str] = None
        self._started = False
        self._finished = False
        self.liquidation_count = 0
        self.events_processed = 0
        # What a replay needs beyond the config: agents not built from ``config.agents``
        # and every outside call (act, execute, shock_index), in order (see ``replay.py``).
        self.added_agents: list[dict[str, Any]] = []
        self.control_log: list[dict[str, Any]] = []
        self._building_from_specs = False

        if self._explicit_agents is not None:
            for agent, collateral in self._explicit_agents:
                self.add_agent(agent, collateral)
        else:
            self._build_agents_from_specs(config.agents)

    def _build_agents_from_specs(self, specs: Iterable[AgentSpec]) -> None:
        """Instantiate ``AgentSpec`` groups, sampling randomized parameters and collateral from the setup stream."""
        used_prefix_counts: dict[str, int] = {}
        self._building_from_specs = True
        for spec in specs:
            prefix = spec.prefix or getattr(spec.agent, "default_prefix", "agent")
            for _ in range(spec.count):
                used_prefix_counts[prefix] = used_prefix_counts.get(prefix, 0) + 1
                agent_id = f"{prefix}_{used_prefix_counts[prefix]:02d}"
                params = dict(spec.params)
                for key in spec.randomize:
                    low, high = params[key]
                    if isinstance(low, int) and isinstance(high, int):
                        params[key] = int(self._setup_rng.integers(low, high + 1))
                    else:
                        params[key] = float(self._setup_rng.uniform(low, high))
                agent = spec.agent(**params)
                agent.agent_id = agent_id
                collateral = sample_collateral(spec.collateral, self._setup_rng)
                self.add_agent(agent, collateral, private_signals=spec.private_signals, external=spec.external)
        self._building_from_specs = False

    def add_agent(self, agent: Agent, collateral: float, *, private_signals: bool = False, external: bool = False) -> None:
        """Add a participant. Must be called before the simulation starts."""
        if self._started:
            raise RuntimeError("agents must be added before the simulation starts")
        if not agent.agent_id:
            agent.agent_id = f"{agent.default_prefix}_{len(self.agents) + 1:02d}"
        agent_id = agent.agent_id
        if agent_id in self.accounts:
            raise ValueError(f"duplicate agent id {agent_id}")
        public_id = self._make_public_id(agent_id)
        agent.public_id = public_id
        self._public_ids[agent_id] = public_id
        self.accounts[agent_id] = MarginAccount(
            agent_id,
            collateral,
            public_id=public_id,
            agent_type=agent.agent_type,
            contract_size=self.contract.contract_size,
        )
        self.total_deposits += float(collateral)
        self.agents[agent_id] = agent
        base = [self.seed, _stable_hash(agent_id)]
        self._agent_rngs[agent_id] = np.random.default_rng(np.random.SeedSequence(base + [1]))
        self._latency_rngs[agent_id] = np.random.default_rng(np.random.SeedSequence(base + [2]))
        self._views[agent_id] = MarketView(self, agent_id)
        if private_signals:
            self._signal_subscribers.add(agent_id)
        if external or (isinstance(agent, ExternalAgent) and agent.policy is None):
            self._external_ids.add(agent_id)
        if not self._building_from_specs:
            self.added_agents.append({"agent": agent, "collateral": float(collateral), "private_signals": private_signals, "external": external})

    def _make_public_id(self, agent_id: str) -> str:
        """Wallet-like hash per seed, so agents cannot infer a type from a name like ``whale_01``."""
        if not self.config.pseudonymous_ids:
            return agent_id
        digest = hashlib.sha256(f"{self.seed}:{agent_id}".encode()).hexdigest()
        return f"0x{digest[:4]}…{digest[-4:]}"

    def public_id(self, agent_id: str) -> str:
        """The id other agents see for ``agent_id`` (hashed when ``pseudonymous_ids`` is on)."""
        return self._public_ids.get(agent_id, agent_id)

    def view(self, agent_id: str) -> MarketView:
        """The ``MarketView`` through which ``agent_id`` observes the market."""
        return self._views[agent_id]

    # ============================================================ scheduling

    def _push(self, time: float, priority: int, kind: str, payload: object = None) -> None:
        """Queue an event; the sequence number breaks ties so equal timestamps replay in insertion order."""
        heapq.heappush(self._queue, (time, priority, next(self._event_seq), kind, payload))

    def _start(self) -> None:
        """Schedule the exogenous processes, the first wakeups and the end of the session.

        Runs lazily on the first ``step()`` so agents can still be added after
        construction.
        """
        if self._started:
            return
        self._started = True
        cfg = self.config
        if cfg.index.enabled:
            self._push(cfg.index.update_interval, _P_INDEX, "index")
        for item in self.news_items:
            self._push(item.time, _P_NEWS, "news", item)
        if self._signal_subscribers:
            for signal in self.signals:
                self._push(signal.arrival_time, _P_SIGNAL, "signal", signal)
        if cfg.funding.enabled:
            self._push(cfg.funding.sample_interval, _P_FUNDING, "funding_sample")
            self._push(cfg.funding.interval, _P_FUNDING, "funding")
        self._push(cfg.record_interval, _P_RECORD, "record")
        self._push(cfg.duration, _P_RECORD + 1, "end")
        self._record_bar(initial=True)
        for agent_id, agent in self.agents.items():
            rng = self._agent_rngs[agent_id]
            if agent_id not in self._external_ids:
                self._send(agent_id, agent.on_start(self._views[agent_id], rng))
            self._push(self.time + agent.next_wakeup_delay(rng), _P_WAKEUP, "wakeup", agent_id)

    def _send(self, agent_id: str, actions: Optional[Sequence[Action]]) -> None:
        """Queue ``actions`` to reach the exchange after the agent's latency.

        Latency is what lets the market move between a decision and its
        execution; without it every agent would trade on a perfectly fresh book.
        """
        batch = actions_list(actions)
        if not batch:
            return
        latency = self.config.latency
        delay = latency.base
        if latency.jitter > 0:
            delay += float(self._latency_rngs[agent_id].uniform(0.0, latency.jitter))
        self._push(self.time + delay, _P_ARRIVAL, "arrival", (agent_id, batch))

    # ================================================================ running

    @property
    def finished(self) -> bool:
        """True once the end-of-session event has run."""
        return self._finished

    @property
    def pending_decision(self) -> Optional[str]:
        """Id of the external agent waiting for ``act()``, or ``None``."""
        return self._pending_decision

    def step(self) -> Optional[str]:
        """Process the next event. Returns its kind, or ``None`` when finished."""
        if self._pending_decision is not None:
            raise RuntimeError(f"waiting for external decision from {self._pending_decision}; call act()")
        self._start()
        if self._finished or not self._queue:
            self._finished = True
            return None
        time, _, _, kind, payload = heapq.heappop(self._queue)
        self.time = time
        self.events_processed += 1
        handler = getattr(self, f"_on_{kind}")
        handler(payload)
        return kind

    def run_until(self, time: float) -> None:
        """Advance until simulated ``time`` (or the end, or an external decision)."""
        self._start()
        while not self._finished and self._pending_decision is None and self._queue and self._queue[0][0] <= time:
            self.step()
        if not self._finished and self._pending_decision is None:
            self.time = max(self.time, min(time, self.config.duration))

    def run(self, progress: Optional[Callable[[float], None]] = None) -> SimulationResult:
        """Run to the end and return the results."""
        from .results import SimulationResult

        self._start()
        next_report = 0.0
        while not self._finished:
            if self._pending_decision is not None:
                raise RuntimeError(f"agent {self._pending_decision} is external without a policy; drive it with run_until_decision()/act()")
            self.step()
            if progress is not None and self.time >= next_report:
                progress(self.time)
                next_report += self.config.duration / 100.0
        return SimulationResult.from_market(self)

    def _log_control(self, kind: str, **details: Any) -> None:
        """Record an outside call with the exact point (event count and time) at which it happened."""
        self.control_log.append({"kind": kind, "events": self.events_processed, "time": self.time, **details})

    def replay(self):
        """A ``Replay`` that rebuilds this run exactly (raises ``ReplayUnavailable`` with the reason if it cannot)."""
        from .replay import Replay

        return Replay.from_market(self)

    def run_until_decision(self) -> Optional[str]:
        """Advance until an external agent must decide. Returns its id, or ``None`` at the end."""
        while not self._finished and self._pending_decision is None:
            self.step()
        return self._pending_decision

    def act(self, agent_id: str, actions: Sequence[Action]) -> None:
        """Answer a pending external decision."""
        if self._pending_decision != agent_id:
            raise RuntimeError(f"no pending decision for {agent_id}")
        self._log_control("act", agent_id=agent_id, actions=list(actions))
        self._pending_decision = None
        self._send(agent_id, actions)
        self._push(self.time + self.agents[agent_id].next_wakeup_delay(self._agent_rngs[agent_id]), _P_WAKEUP, "wakeup", agent_id)

    def execute(self, agent_id: str, actions: Sequence[Action]) -> None:
        """Process actions for ``agent_id`` immediately, bypassing latency.

        Intended for tests and deliberate interventions (e.g. injecting a large
        order at a chosen time). Normal agents act through their wakeups.
        """
        self._start()
        self._log_control("execute", agent_id=agent_id, actions=list(actions))
        for action in actions:
            self._process_action(agent_id, action)
        self._after_market_change()

    def shock_index(self, log_jump: float, headline: str = "Exogenous shock") -> None:
        """Apply an immediate exogenous jump to the index price (a controlled experiment)."""
        self._start()
        self._log_control("shock", log_jump=float(log_jump), headline=headline)
        self.index_price = self.index.jump(log_jump)
        self.recorder.news.append(NewsRecord(time=self.time, headline=headline, log_jump=log_jump))
        self._after_market_change()

    def results(self) -> SimulationResult:
        """Snapshot of the session so far (valid before the end as well)."""
        from .results import SimulationResult

        return SimulationResult.from_market(self)

    # ======================================================== event handlers

    def _on_index(self, _payload) -> None:
        """Advance the index diffusion by one update interval."""
        interval = self.config.index.update_interval
        self.index_price = self.index.advance(interval)
        self._push(self.time + interval, _P_INDEX, "index")
        self._after_market_change()

    def _on_news(self, item: NewsItem) -> None:
        """Apply the jump to the index, publish the item and let active internal agents react."""
        self.index_price = self.index.jump(item.log_jump)
        self.public_news.append(item)
        self._released_news.add(item.news_id)
        self.recorder.news.append(NewsRecord(time=self.time, headline=item.headline, log_jump=item.log_jump))
        self._after_market_change()
        for agent_id, agent in self.agents.items():
            if agent_id in self._external_ids or not self.accounts[agent_id].active:
                continue
            self._send(agent_id, agent.on_news(item, self._views[agent_id], self._agent_rngs[agent_id]))

    def _on_signal(self, signal: PrivateSignal) -> None:
        """Deliver a private signal to its subscribers while the news is still unpublished."""
        self._delivered_signals.append(signal)
        for agent_id in sorted(self._signal_subscribers):
            if agent_id in self._external_ids or not self.accounts[agent_id].active:
                continue
            agent = self.agents[agent_id]
            self._send(agent_id, agent.on_signal(signal, self._views[agent_id], self._agent_rngs[agent_id]))

    def _on_wakeup(self, agent_id: str) -> None:
        """An agent's decision point.

        External agents pause the market here until ``act()`` answers; internal
        agents answer at once and draw their next wakeup from their own stream.
        """
        account = self.accounts[agent_id]
        if not account.active:
            return
        agent = self.agents[agent_id]
        rng = self._agent_rngs[agent_id]
        if agent_id in self._external_ids:
            self._pending_decision = agent_id
            return
        self._send(agent_id, agent.on_wakeup(self._views[agent_id], rng))
        self._push(self.time + agent.next_wakeup_delay(rng), _P_WAKEUP, "wakeup", agent_id)

    def _on_arrival(self, payload) -> None:
        """Execute a batch of actions that has reached the exchange."""
        agent_id, actions = payload
        for action in actions:
            self._process_action(agent_id, action)
        self._after_market_change()

    def _on_funding_sample(self, _payload) -> None:
        """Sample the premium; the settlement uses the mean of the samples."""
        if self.index_price > 0:
            self._premium_samples.append((self.mark_price - self.index_price) / self.index_price)
        self._push(self.time + self.config.funding.sample_interval, _P_FUNDING, "funding_sample")

    def _on_funding(self, _payload) -> None:
        """Settle funding from the sampled premium.

        Payments move collateral, so an account can end up under water here
        without any trade: hence the liquidation check at the end.
        """
        cfg = self.config.funding
        premium = float(np.mean(self._premium_samples)) if self._premium_samples else 0.0
        rate = max(-cfg.cap, min(cfg.cap, premium + cfg.interest_rate))
        self._premium_samples.clear()
        paid_by_longs = 0.0
        cs = self.contract.contract_size
        for account in self.accounts.values():
            if account.position == 0:
                continue
            payment = account.position * cs * self.index_price * rate
            account.apply_funding(payment)
            if account.position > 0:
                paid_by_longs += payment
        self.last_funding_rate = rate
        self.recorder.funding.append(
            FundingRecord(
                time=self.time, rate=rate, premium=premium, mark_price=self.mark_price, index_price=self.index_price, total_paid_by_longs=paid_by_longs
            )
        )
        self.next_funding_time = self.time + cfg.interval
        self._push(self.next_funding_time, _P_FUNDING, "funding")
        self._run_liquidations()

    def _on_record(self, _payload) -> None:
        """Close the current bar and sample account paths."""
        self._update_basis_ema()
        self._record_bar()
        self._push(self.time + self.config.record_interval, _P_RECORD, "record")

    def _on_end(self, _payload) -> None:
        """Stop the session; anything still queued is discarded."""
        self._finished = True
        self._queue.clear()

    # ============================================================= prices

    def best_bid_price(self) -> Optional[float]:
        """Best bid in price units (``None`` when the bid side is empty)."""
        bid = self.book.best_bid()
        return None if bid is None else bid * self.tick_size

    def best_ask_price(self) -> Optional[float]:
        """Best ask in price units (``None`` when the ask side is empty)."""
        ask = self.book.best_ask()
        return None if ask is None else ask * self.tick_size

    def mid_price(self) -> Optional[float]:
        """Book mid in price units (``None`` when a side is empty)."""
        mid = self.book.mid()
        return None if mid is None else mid * self.tick_size

    def _to_ticks(self, price: float) -> int:
        return max(1, int(round(price / self.tick_size)))

    def _update_basis_ema(self) -> None:
        """Track the perp-index basis slowly so the mark anchor follows a persistent premium but ignores a single swing."""
        mid = self.mid_price()
        if mid is None:
            return
        alpha = 1.0 - math.exp(-self.config.record_interval / BASIS_EMA_SECONDS)
        self._basis_ema += alpha * ((mid - self.index_price) - self._basis_ema)

    def _compute_mark(self) -> float:
        """Median of three price sources, so no single one (a thin book, a stale trade, a slow index) can drag the mark alone."""
        anchored = self.index_price + self._basis_ema
        mid = self.mid_price()
        candidates = [anchored, self.last_price, mid if mid is not None else self.index_price]
        return float(median(candidates))

    def _after_market_change(self) -> None:
        """Refresh the mark and liquidate whoever it puts under water; called after every event that can move prices."""
        self.mark_price = self._compute_mark()
        self._run_liquidations()

    def predicted_funding_rate(self) -> float:
        """The rate the next settlement would apply if the premium stayed where it is."""
        cfg = self.config.funding
        premium = float(np.mean(self._premium_samples)) if self._premium_samples else 0.0
        return max(-cfg.cap, min(cfg.cap, premium + cfg.interest_rate))

    def open_interest(self) -> int:
        """Total long contracts (equal to total short contracts by construction)."""
        return sum(account.position for account in self.accounts.values() if account.position > 0)

    def recent_prices(self, seconds: int) -> list[float]:
        """Last ``seconds`` of per-interval trade prices, oldest first."""
        n = max(1, int(seconds / self.config.record_interval))
        return list(self._price_samples)[-n:]

    def recent_marks(self, seconds: int) -> list[float]:
        """Last ``seconds`` of per-interval mark prices, oldest first."""
        n = max(1, int(seconds / self.config.record_interval))
        return list(self._mark_samples)[-n:]

    def recent_open_interest(self, seconds: int) -> list[float]:
        """Last ``seconds`` of per-interval open interest, oldest first."""
        n = max(1, int(seconds / self.config.record_interval))
        return list(self.recorder.bars["open_interest"][-n:])

    def recent_index(self, seconds: int) -> list[float]:
        """Last ``seconds`` of per-interval index prices, oldest first."""
        n = max(1, int(seconds / self.config.record_interval))
        return list(self._index_samples)[-n:]

    # ============================================================ orders

    def _order_event(
        self, kind: str, order_id: str, agent_id: str, side: Side, order_type: str, quantity: int, price: Optional[float], reason: str = ""
    ) -> None:
        """Record an order lifecycle event and keep the per-account counters in sync."""
        account = self.accounts[agent_id]
        if kind == "accepted":
            account.orders_accepted += 1
        elif kind == "rejected":
            account.orders_rejected += 1
        elif kind == "canceled":
            account.orders_canceled += 1
        self.recorder.add_order_event(
            OrderEvent(
                time=self.time, kind=kind, order_id=order_id, agent_id=agent_id, side=side, order_type=order_type, quantity=quantity, price=price, reason=reason
            )
        )

    def _new_order_id(self, agent_id: str) -> str:
        return f"{agent_id}-{next(self._order_ids)}"

    def _margin_ok(self, account: MarginAccount, side: Side, quantity: int, est_price: float) -> bool:
        """Whether adding ``quantity`` on ``side`` keeps initial margin covered.

        Orders that do not increase the worst-case position are always allowed,
        so an under-margined account can still reduce risk.
        """
        extra_buy = quantity if side is Side.BUY else 0
        extra_sell = quantity if side is Side.SELL else 0
        if account.worst_case_position(extra_buy, extra_sell) <= account.worst_case_position():
            return True
        mark = self.mark_price
        required = account.initial_margin(mark, self._imr, extra_buy, extra_sell)
        fee = self.contract.taker_fee * quantity * est_price * self.contract.contract_size
        return account.equity(mark) - fee >= required

    def _reduce_only_cap(self, account: MarginAccount, side: Side, quantity: int) -> int:
        """How much of a reduce-only order can actually reduce the position (0 when it would add to it)."""
        pos = account.position
        if pos == 0 or (pos > 0) == (side is Side.BUY):
            return 0
        return min(quantity, abs(pos))

    def _process_action(self, agent_id: str, action: Action) -> None:
        """Dispatch one action; cancels are honoured even for deactivated accounts so nothing stays stuck in the book."""
        account = self.accounts[agent_id]
        if isinstance(action, Cancel):
            self._cancel(agent_id, action.order_id, "user")
            return
        if isinstance(action, CancelAll):
            for order in self.book.orders_of(agent_id):
                if action.side is None or order.side is action.side:
                    self._cancel(agent_id, order.order_id, "user")
            return
        if not account.active:
            return
        if isinstance(action, LimitOrder):
            self._limit(agent_id, account, action)
        elif isinstance(action, MarketOrder):
            self._market(agent_id, account, action)
        else:
            raise TypeError(f"unknown action {action!r}")

    def _limit(self, agent_id: str, account: MarginAccount, action: LimitOrder) -> None:
        """Validate, match and rest a limit order. Post-only is checked before margin because a crossing post-only order is simply rejected."""
        order_id = self._new_order_id(agent_id)
        price_ticks = self._to_ticks(action.price)
        price = price_ticks * self.tick_size
        quantity = int(action.quantity)
        if action.tif is TimeInForce.POST_ONLY and self.book.would_cross(action.side, price_ticks):
            self._order_event("rejected", order_id, agent_id, action.side, "limit", quantity, price, "post_only_would_cross")
            return
        if action.reduce_only:
            quantity = self._reduce_only_cap(account, action.side, quantity)
            if quantity <= 0:
                self._order_event("rejected", order_id, agent_id, action.side, "limit", int(action.quantity), price, "reduce_only_no_position")
                return
        elif not self._margin_ok(account, action.side, quantity, price):
            self._order_event("rejected", order_id, agent_id, action.side, "limit", quantity, price, "insufficient_margin")
            return
        self._order_event("accepted", order_id, agent_id, action.side, "limit", quantity, price)
        result = self.book.match(action.side, quantity, agent_id, limit=price_ticks)
        self._apply_match(agent_id, order_id, action.side, result)
        remaining = quantity - result.filled
        if remaining > 0:
            if action.tif is TimeInForce.IOC:
                self._order_event("expired", order_id, agent_id, action.side, "limit", remaining, price, "ioc_remainder")
            else:
                resting = RestingOrder(order_id, agent_id, action.side, price_ticks, remaining, self.time, next(self._event_seq), action.tif, action.tag)
                self.book.add(resting)
                account.reserve_order(action.side, remaining)

    def _market(self, agent_id: str, account: MarginAccount, action: MarketOrder) -> None:
        """Execute a market order as an aggressive limit a few percent from the mark, so a thin book cannot be swept to absurd prices."""
        order_id = self._new_order_id(agent_id)
        quantity = int(action.quantity)
        if action.reduce_only:
            quantity = self._reduce_only_cap(account, action.side, quantity)
            if quantity <= 0:
                self._order_event("rejected", order_id, agent_id, action.side, "market", int(action.quantity), None, "reduce_only_no_position")
                return
        reference = self.mark_price
        limit_price = reference * (1 + MARKET_ORDER_SLIPPAGE) if action.side is Side.BUY else reference * (1 - MARKET_ORDER_SLIPPAGE)
        limit_ticks = self._to_ticks(limit_price)
        fillable, notional_ticks = self.book.estimate_fill(action.side, quantity, agent_id, limit=limit_ticks)
        if fillable == 0:
            self._order_event("rejected", order_id, agent_id, action.side, "market", quantity, None, "no_liquidity")
            return
        est_price = notional_ticks * self.tick_size / fillable
        if not action.reduce_only and not self._margin_ok(account, action.side, fillable, est_price):
            self._order_event("rejected", order_id, agent_id, action.side, "market", quantity, None, "insufficient_margin")
            return
        self._order_event("accepted", order_id, agent_id, action.side, "market", quantity, None)
        result = self.book.match(action.side, fillable, agent_id, limit=limit_ticks)
        self._apply_match(agent_id, order_id, action.side, result)
        if result.filled < quantity:
            self._order_event("expired", order_id, agent_id, action.side, "market", quantity - result.filled, None, "insufficient_liquidity")

    def _cancel(self, agent_id: str, order_id: str, reason: str) -> bool:
        """Cancel one order if it belongs to ``agent_id``; returns whether anything was canceled."""
        order = self.book.get(order_id)
        if order is None or order.agent_id != agent_id:
            return False
        self.book.cancel(order_id)
        self.accounts[agent_id].reduce_reservation(order.side, order.remaining, closes_order=True)
        self._order_event("canceled", order_id, agent_id, order.side, "limit", order.remaining, order.price * self.tick_size, reason)
        return True

    def _cancel_all(self, agent_id: str, reason: str) -> None:
        for order in self.book.orders_of(agent_id):
            self._cancel(agent_id, order.order_id, reason)

    def _apply_match(self, taker_id: str, taker_order_id: str, taker_side: Side, result: MatchResult, *, liquidation: bool = False) -> None:
        """Settle every fill of a match: fees, both accounts, the trade record and the agents' ``on_fill`` hooks."""
        for canceled in result.self_trade_canceled:
            self.accounts[canceled.agent_id].reduce_reservation(canceled.side, canceled.remaining, closes_order=True)
            self._order_event(
                "canceled",
                canceled.order_id,
                canceled.agent_id,
                canceled.side,
                "limit",
                canceled.remaining,
                canceled.price * self.tick_size,
                "self_trade_prevention",
            )
        cs = self.contract.contract_size
        taker = self.accounts[taker_id]
        for match in result.matches:
            price = match.price * self.tick_size
            qty = match.quantity
            maker_order = match.maker
            maker = self.accounts[maker_order.agent_id]
            maker_fee = self.contract.maker_fee * qty * price * cs
            taker_fee = self.contract.taker_fee * qty * price * cs
            maker.reduce_reservation(maker_order.side, qty, closes_order=maker_order.remaining == 0)
            maker_realized = maker.apply_fill(maker_order.side, qty, price, maker_fee)
            taker_realized = taker.apply_fill(taker_side, qty, price, taker_fee)
            maker.maker_quantity += qty
            taker.taker_quantity += qty
            self.fees_collected += maker_fee + taker_fee
            self.last_price = price
            buyer, seller = (taker_id, maker_order.agent_id) if taker_side is Side.BUY else (maker_order.agent_id, taker_id)
            trade = TradeRecord(
                trade_id=next(self._trade_ids),
                time=self.time,
                price=price,
                quantity=qty,
                buyer=buyer,
                seller=seller,
                aggressor=taker_side,
                liquidation=liquidation,
            )
            self._record_trade(trade)
            self._notify_fill(
                maker_order.agent_id,
                Fill(self.time, maker_order.order_id, maker_order.agent_id, maker_order.side, price, qty, maker_fee, True, maker_realized, liquidation),
            )
            self._notify_fill(taker_id, Fill(self.time, taker_order_id, taker_id, taker_side, price, qty, taker_fee, False, taker_realized, liquidation))

    def _record_trade(self, trade: TradeRecord) -> None:
        """Append the trade to the tape and fold it into the current bar."""
        self.recorder.trades.append(trade)
        self.recent_trades.append(trade)
        bar = self._bar
        if bar["open"] is None:
            bar["open"] = trade.price
            bar["high"] = trade.price
            bar["low"] = trade.price
        else:
            bar["high"] = max(bar["high"], trade.price)
            bar["low"] = min(bar["low"], trade.price)
        bar["close"] = trade.price
        bar["volume"] += trade.quantity
        bar["trades"] += 1

    def _notify_fill(self, agent_id: str, fill: Fill) -> None:
        """Tell the owner about a fill (bookkeeping only; agents cannot act from this hook)."""
        agent = self.agents.get(agent_id)
        if agent is not None:
            agent.on_fill(fill, self._views[agent_id])

    # ======================================================= liquidations

    def _run_liquidations(self) -> None:
        """Liquidate every under-water account, worst first, until none is left.

        The loop is bounded by the number of accounts: every pass closes at
        least one position, so even a full cascade terminates.
        """
        if not self.config.liquidation.enabled:
            return
        # Liquidation fills move the mark, which can trigger further liquidations (a cascade).
        for _ in range(len(self.accounts) + 1):
            mark = self.mark_price
            underwater = [
                account for account in self.accounts.values() if account.position != 0 and account.equity(mark) < account.maintenance_margin(mark, self._mmr)
            ]
            if not underwater:
                return
            underwater.sort(key=lambda a: (a.equity(mark) - a.maintenance_margin(mark, self._mmr), a.agent_id))
            for account in underwater:
                self._liquidate(account)
            self.mark_price = self._compute_mark()

    def _liquidate(self, account: MarginAccount) -> None:
        """Force-close one account: cancel, sell into the book, auto-deleverage the rest, charge the penalty, settle bad debt.

        Equity and path statistics are captured first so the record shows what
        the account looked like at the moment it failed.
        """
        cfg = self.config.liquidation
        mark = self.mark_price
        agent_id = account.agent_id
        equity_before = account.equity(mark)
        account.update_path(mark)
        if account.first_liquidation_time is None:
            account.first_liquidation_time = self.time
        self._cancel_all(agent_id, "liquidation")
        position = account.position
        if position == 0:
            return
        close_side = Side.SELL if position > 0 else Side.BUY
        quantity = abs(position)
        closed_notional = quantity * mark * self.contract.contract_size
        limit_price = mark * (1 - LIQUIDATION_SLIPPAGE) if close_side is Side.SELL else mark * (1 + LIQUIDATION_SLIPPAGE)
        order_id = f"{agent_id}-liq-{next(self._order_ids)}"
        result = self.book.match(close_side, quantity, agent_id, limit=self._to_ticks(limit_price))
        self._apply_match(agent_id, order_id, close_side, result, liquidation=True)
        filled_notional = sum(m.price * m.quantity for m in result.matches) * self.tick_size
        avg_fill = filled_notional / result.filled if result.filled else None
        residual = abs(account.position)
        if residual:
            self._auto_deleverage(account, residual, mark)
        penalty = 0.0
        if account.collateral > 0 and cfg.penalty_rate > 0:
            penalty = min(account.collateral, cfg.penalty_rate * closed_notional)
            account.collateral -= penalty
            self.insurance_fund += penalty
        bad_debt = 0.0
        if account.collateral < 0:
            bad_debt = -account.collateral
            self.insurance_fund -= bad_debt
            account.collateral = 0.0
        account.liquidations += 1
        self.liquidation_count += 1
        deactivated = account.collateral < cfg.deactivate_below_fraction * account.initial_collateral
        if deactivated:
            account.active = False
            account.deactivation_reason = "liquidated"
        self.recorder.liquidations.append(
            LiquidationRecord(
                time=self.time,
                agent_id=agent_id,
                side_closed=close_side,
                quantity=quantity,
                mark_price=mark,
                avg_fill_price=avg_fill,
                adl_quantity=residual,
                bad_debt=bad_debt,
                penalty=penalty,
                equity_before=equity_before,
                deactivated=deactivated,
            )
        )
        agent = self.agents.get(agent_id)
        if agent is not None:
            agent.on_liquidation(self._views[agent_id])

    def _auto_deleverage(self, account: MarginAccount, quantity: int, mark: float) -> None:
        """Close ``quantity`` of the account's position against opposite positions at the mark.

        Counterparties are ranked by unrealized-PnL ratio times leverage, the
        usual exchange ADL priority: the most profitable, most levered go first.
        """
        liquidated_long = account.position > 0
        candidates = [other for other in self.accounts.values() if other is not account and other.position != 0 and (other.position < 0) == liquidated_long]

        def priority(other: MarginAccount) -> float:
            """ADL rank: unrealized-PnL ratio times leverage; bankrupt accounts go last."""
            equity = other.equity(mark)
            if equity <= 0:
                return -math.inf
            pnl_ratio = other.unrealized_pnl(mark) / max(abs(other.position) * other.entry_price * other.contract_size, 1e-9)
            return pnl_ratio * other.leverage(mark)

        candidates.sort(key=lambda other: (-priority(other), other.agent_id))
        remaining = quantity
        close_side = Side.SELL if liquidated_long else Side.BUY
        for other in candidates:
            if remaining == 0:
                break
            take = min(remaining, abs(other.position))
            account.apply_fill(close_side, take, mark, 0.0)
            other.apply_fill(close_side.opposite, take, mark, 0.0)
            remaining -= take
            buyer, seller = (other.agent_id, account.agent_id) if liquidated_long else (account.agent_id, other.agent_id)
            trade = TradeRecord(
                trade_id=next(self._trade_ids),
                time=self.time,
                price=mark,
                quantity=take,
                buyer=buyer,
                seller=seller,
                aggressor=close_side,
                liquidation=True,
                adl=True,
            )
            self._record_trade(trade)

    # ============================================================ recording

    def _reset_bar(self) -> None:
        self._bar = {"open": None, "high": None, "low": None, "close": None, "volume": 0, "trades": 0}

    def _record_bar(self, initial: bool = False) -> None:
        """Write one bar from the trades since the last record, plus the book and risk state at this instant."""
        bar = self._bar
        close = bar["close"] if bar["close"] is not None else self.last_price
        bid, ask = self.best_bid_price(), self.best_ask_price()
        self.recorder.add_bar(
            time=self.time,
            open=bar["open"] if bar["open"] is not None else close,
            high=bar["high"] if bar["high"] is not None else close,
            low=bar["low"] if bar["low"] is not None else close,
            close=close,
            volume=bar["volume"],
            trades=bar["trades"],
            mark=self.mark_price,
            index=self.index_price,
            mid=self.mid_price(),
            best_bid=bid,
            best_ask=ask,
            spread=None if bid is None or ask is None else ask - bid,
            bid_depth=self.book.depth_within(Side.BUY, 50),
            ask_depth=self.book.depth_within(Side.SELL, 50),
            open_interest=self.open_interest(),
            predicted_funding=self.predicted_funding_rate(),
            insurance_fund=self.insurance_fund,
            active_agents=sum(1 for a in self.accounts.values() if a.active),
            liquidations=self.liquidation_count,
        )
        self._price_samples.append(close)
        self._mark_samples.append(self.mark_price)
        self._index_samples.append(self.index_price)
        self._reset_bar()
        mark = self.mark_price
        for account in self.accounts.values():
            account.update_path(mark)
        if self._record_index % ACCOUNT_SAMPLE_EVERY == 0:
            mark = self.mark_price
            self.recorder.add_account_sample(
                self.time,
                {agent_id: account.equity(mark) for agent_id, account in self.accounts.items()},
                {agent_id: account.position for agent_id, account in self.accounts.items()},
            )
        self._record_index += 1

    # ========================================================= state access

    def pending_signals(self, agent_id: str) -> list[PrivateSignal]:
        """Signals delivered to ``agent_id`` whose news is still unpublished (empty for non-subscribers)."""
        if agent_id not in self._signal_subscribers:
            return []
        return [s for s in self._delivered_signals if s.news_id not in self._released_news]

    def account_view(self, agent_id: str) -> AccountView:
        """The private picture an agent gets of its own account."""
        account = self.accounts[agent_id]
        mark = self.mark_price
        return AccountView(
            public_id=account.public_id,
            collateral=account.collateral,
            position=account.position,
            entry_price=account.entry_price,
            equity=account.equity(mark),
            unrealized_pnl=account.unrealized_pnl(mark),
            realized_pnl=account.realized_pnl,
            available_margin=account.available_margin(mark, self._imr),
            leverage=account.leverage(mark) if account.position else 0.0,
            liquidation_price=account.liquidation_price(self._mmr),
            open_buy_quantity=account.open_buy_qty,
            open_sell_quantity=account.open_sell_qty,
            initial_collateral=account.initial_collateral,
            active=account.active,
        )

    def public_accounts(self, exclude: Optional[str] = None) -> list[PublicAccount]:
        """What FULL transparency reveals about every account except ``exclude``."""
        mark = self.mark_price
        out = []
        for agent_id, account in self.accounts.items():
            if agent_id == exclude:
                continue
            equity = account.equity(mark)
            out.append(
                PublicAccount(
                    public_id=account.public_id,
                    position=account.position,
                    entry_price=account.entry_price,
                    notional=account.notional(mark),
                    equity=equity,
                    unrealized_pnl=account.unrealized_pnl(mark),
                    leverage=account.leverage(mark) if account.position else 0.0,
                    liquidation_price=account.liquidation_price(self._mmr),
                    total_pnl=equity - account.initial_collateral,
                    return_pct=100.0 * (equity - account.initial_collateral) / account.initial_collateral if account.initial_collateral else 0.0,
                    active=account.active,
                )
            )
        return out

    def snapshots(self) -> list[AccountSnapshot]:
        """Frozen per-account statistics at the current mark, for results and the viewer."""
        mark = self.mark_price
        return [account.snapshot(mark, self._mmr) for account in self.accounts.values()]

    def conservation_error(self) -> float:
        """Total equity + insurance fund + fees minus deposits and initial fund. Should be ~0."""
        mark = self.mark_price
        total = sum(account.equity(mark) for account in self.accounts.values())
        return total + self.insurance_fund + self.fees_collected - self.total_deposits - self.config.liquidation.insurance_fund
