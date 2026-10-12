import numpy as np
import pytest

pytest.importorskip("gymnasium")

from glass_sim import PerpetualMarket, scenarios
from glass_sim.rl import ACTION_NAMES, EnvConfig, ObservationConfig, ObservationEncoder, PerpTradingEnv, exit_value
from glass_sim.rl.policy import HoldPolicy, PolicyAgent, RandomPolicy
from glass_sim.types import Transparency


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


def test_multi_agent_market_rewards_add_up_per_learner():
    from glass_sim.rl.marl import MultiAgentConfig, MultiAgentMarket

    env = MultiAgentMarket(MultiAgentConfig(scenario="small", n_learners=3, episode_duration=600.0))
    request = env.reset(seed=5)
    start = {agent_id: exit_value(env.market, agent_id) for agent_id in env.ids}
    totals = {agent_id: 0.0 for agent_id in env.ids}
    penalties = {agent_id: 0.0 for agent_id in env.ids}
    rng = np.random.default_rng(0)
    decisions = {agent_id: 0 for agent_id in env.ids}
    while True:
        for agent_id, reward, _terminal, info in request["closed"]:
            totals[agent_id] += reward
            penalties[agent_id] += info["penalty"]
        if request["done"]:
            break
        decisions[request["agent"]] += 1
        request = env.step(request["agent"], int(rng.choice(np.flatnonzero(request["mask"]))))
    assert all(n > 20 for n in decisions.values())
    for agent_id in env.ids:
        initial = env.market.accounts[agent_id].initial_collateral
        scale = env.config.reward.scale
        assert totals[agent_id] * initial / scale + penalties[agent_id] == pytest.approx(exit_value(env.market, agent_id) - start[agent_id], abs=1e-6)
    assert abs(env.market.conservation_error()) < 1e-6


def test_ppo_update_handles_a_single_sample_last_minibatch():
    import torch

    from glass_sim.rl.marl import _make_model, ppo_update

    model = _make_model(32, 9, [16, 16], 3e-4, seed=0)
    n = 257  # 256 + 1: the last minibatch has one sample
    rng = np.random.default_rng(0)
    data = {
        "obs": rng.normal(size=(n, 32)).astype(np.float32),
        "masks": np.ones((n, 9), dtype=bool),
        "actions": rng.integers(0, 9, n),
        "log_probs": np.full(n, np.log(1 / 9), dtype=np.float32),
        "values": np.zeros(n, dtype=np.float32),
        "advantages": rng.normal(size=n).astype(np.float32),
        "returns": rng.normal(size=n).astype(np.float32),
    }
    ppo_update(model.policy, data, epochs=2, batch_size=256, clip=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())


def test_truncated_transition_never_bootstraps_from_the_next_episode():
    from glass_sim.rl.marl import Learner, _Pending

    class Model:
        class action_space:
            n = 9

        policy = None

    learner = Learner("x", Model, n_workers=1, gamma=0.99, gae_lambda=0.95)
    obs, mask = np.zeros(3, dtype=np.float32), np.ones(9, dtype=bool)
    # Episode 1: one decision, then the time limit (a truncation, not a termination).
    learner.pending[0] = _Pending(obs, mask, 0, 0.0, value=1.0)
    learner.record_close(0, reward=0.5, terminal=False, truncated=True)
    learner.record_final_value(0, 3.0)  # value of the state the episode ended in
    learner.end_episode(0)
    # Episode 2 starts on the same worker: its first decision must not touch the old transition.
    learner.record_value_of_next(0, 100.0)
    learner.pending[0] = _Pending(obs, mask, 0, 0.0, value=2.0)
    learner.record_close(0, reward=1.0, terminal=False)
    learner.record_value_of_next(0, 4.0)
    first, second = learner.streams[0]
    assert first.truncated and first.has_next and first.next_value == 3.0
    assert second.has_next and second.next_value == 4.0
    data = learner.batch()
    # The first advantage is exactly its own TD error: nothing of episode 2 leaks into it.
    assert data["advantages"][0] == pytest.approx(0.5 + 0.99 * 3.0 - 1.0)
    assert data["advantages"][1] == pytest.approx(1.0 + 0.99 * 4.0 - 2.0)


def test_record_value_of_next_still_bootstraps_within_an_episode():
    from glass_sim.rl.marl import Learner, _Pending

    class Model:
        class action_space:
            n = 9

        policy = None

    learner = Learner("x", Model, n_workers=1, gamma=0.99, gae_lambda=0.95)
    obs, mask = np.zeros(3, dtype=np.float32), np.ones(9, dtype=bool)
    learner.pending[0] = _Pending(obs, mask, 0, 0.0, value=1.0)
    learner.record_close(0, reward=0.0, terminal=False)
    learner.record_value_of_next(0, 5.0)
    assert learner.streams[0][-1].has_next and learner.streams[0][-1].next_value == 5.0
    learner.record_final_value(0, 9.0)  # only truncated transitions take a final value
    assert learner.streams[0][-1].next_value == 5.0


def test_episode_end_reports_the_final_observation_of_truncated_learners():
    from glass_sim.rl.marl import MultiAgentConfig, MultiAgentMarket

    env = MultiAgentMarket(MultiAgentConfig(scenario="small", n_learners=2, episode_duration=300.0))
    request = env.reset(seed=3)
    while not request["done"]:
        request = env.step(request["agent"], 0)
    truncated = {agent_id for agent_id, _reward, terminal, _info in request["closed"] if not terminal}
    assert truncated == set(request["final_obs"])
    for obs in request["final_obs"].values():
        assert obs.shape == (env.encoder.size,) and obs.dtype == np.float32
