import numpy as np
import pytest

pytest.importorskip("gymnasium")

from marl_trading.futures import PerpetualMarket, scenarios
from marl_trading.futures.rl import ACTION_NAMES, EnvConfig, ObservationConfig, ObservationEncoder, PerpTradingEnv, exit_value
from marl_trading.futures.rl.policy import HoldPolicy, PolicyAgent, RandomPolicy
from marl_trading.futures.types import Transparency


def small_env(**overrides):
    return PerpTradingEnv(EnvConfig(scenario="small", episode_duration=600.0, **overrides))


def test_env_passes_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(small_env(), skip_render_check=True)


def test_rewards_sum_to_exit_value_change():
    env = small_env()
    env.reset()
    start = exit_value(env.market, "rl_01")
    rng = np.random.default_rng(1)
    total, penalties, done = 0.0, 0.0, False
    while not done:
        action = rng.choice(np.flatnonzero(env.action_masks()))
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        penalties += info["penalty"]
        done = terminated or truncated
    initial = env.market.accounts["rl_01"].initial_collateral
    scale = env.config.reward.scale
    assert total * initial / scale + penalties == pytest.approx(exit_value(env.market, "rl_01") - start, abs=1e-6)
    assert abs(env.market.conservation_error()) < 1e-6


def test_masks_block_impossible_actions():
    env = small_env()
    env.reset()
    mask = env.action_masks()
    assert mask[ACTION_NAMES.index("hold")]
    assert not mask[ACTION_NAMES.index("close_position")], "flat agent cannot close"
    assert not mask[ACTION_NAMES.index("cancel_all")], "no orders to cancel"
    env.step(ACTION_NAMES.index("buy_large"))
    mask = env.action_masks()
    assert mask[ACTION_NAMES.index("close_position")]


def test_exposure_cap_is_enforced_by_masks():
    env = small_env()
    env.reset()
    for _ in range(40):
        mask = env.action_masks()
        if not mask[ACTION_NAMES.index("buy_large")]:
            break
        env.step(ACTION_NAMES.index("buy_large"))
    account = env.market.accounts["rl_01"]
    cap = env.config.actions.max_leverage
    assert account.leverage(env.market.mark_price) <= cap * 1.15


def test_observation_hides_what_transparency_hides():
    encoder = ObservationEncoder(ObservationConfig())
    names = encoder.feature_names
    for level, owners, positions in [(Transparency.ANONYMOUS, 0.0, 0.0), (Transparency.ATTRIBUTED, 1.0, 0.0), (Transparency.FULL, 1.0, 1.0)]:
        market = PerpetualMarket(scenarios.small(seed=1, duration=300.0, transparency=level))
        market.run_until(200.0)
        obs = encoder.encode(market.view(next(iter(market.accounts))))
        assert obs[names.index("sees_owners")] == owners
        assert obs[names.index("sees_positions")] == positions
        if level is Transparency.ANONYMOUS:
            for name in ("crowd_net_positioning", "long_liquidations_below", "strongest_flow_signed"):
                assert obs[names.index(name)] == 0.0
    assert np.all(np.abs(obs) <= 5.0)


def test_policy_agent_runs_inside_a_market():
    for policy, expect_trades in ((HoldPolicy(), False), (RandomPolicy(), True)):
        agent = PolicyAgent(policy)
        agent.agent_id = "rl_01"
        market = PerpetualMarket(scenarios.small(seed=2, duration=300.0))
        market.add_agent(agent, 100_000.0)
        result = market.run()
        snap = next(a for a in result.accounts if a.agent_id == "rl_01")
        assert (snap.trade_count > 0) == expect_trades
        assert agent.decisions > 0


def test_parallel_workers_use_disjoint_seeds():
    seeds = tuple(range(100, 124))
    used = []
    for rank in range(4):
        env = PerpTradingEnv(EnvConfig(scenario="small", episode_duration=60.0, train_seeds=seeds))
        env.seed_offset, env.seed_stride = rank, 4
        for _ in range(6):
            used.append(env.episode_seed())
            env._episode += 1
    assert sorted(used) == list(seeds)
