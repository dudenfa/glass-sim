import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from glass_sim.rl import ActionConfig, exit_value
from glass_sim.rl.marl import MultiAgentConfig
from glass_sim.rl.pettingzoo_env import GlassAECEnv


def small_config(**overrides):
    settings = {"scenario": "small", "n_learners": 3, "episode_duration": 600.0, **overrides}
    return MultiAgentConfig(**settings)


def test_passes_pettingzoo_api_test():
    from pettingzoo.test import api_test

    api_test(GlassAECEnv(small_config()), num_cycles=100)
    api_test(GlassAECEnv(small_config(n_learners=2), mask_in_observation=False), num_cycles=100)


def test_same_seed_same_episode():
    from pettingzoo.test import seed_test

    seed_test(lambda: GlassAECEnv(small_config()), num_cycles=50)


def play(env, seed, rng):
    """Random valid actions until every agent is done; returns per-agent reward sums, penalties and turns."""
    env.reset(seed=seed)
    start = {agent: exit_value(env.market, agent) for agent in env.possible_agents}
    totals = {agent: 0.0 for agent in env.possible_agents}
    penalties = {agent: 0.0 for agent in env.possible_agents}
    turns = {agent: 0 for agent in env.possible_agents}
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        totals[agent] += reward
        penalties[agent] += info.get("penalty", 0.0)
        if terminated or truncated:
            env.step(None)
            continue
        turns[agent] += 1
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    return start, totals, penalties, turns


def test_rewards_add_up_to_exit_value_change():
    env = GlassAECEnv(small_config())
    start, totals, penalties, turns = play(env, seed=5, rng=np.random.default_rng(0))
    assert env.agents == [], "every agent is removed at the end"
    assert all(n > 20 for n in turns.values())
    scale = env.config.reward.scale
    for agent in env.possible_agents:
        initial = env.market.accounts[agent].initial_collateral
        assert totals[agent] * initial / scale + penalties[agent] == pytest.approx(exit_value(env.market, agent) - start[agent], abs=1e-6)
    assert abs(env.market.conservation_error()) < 1e-6


def test_only_the_selected_agent_may_act():
    env = GlassAECEnv(small_config())
    env.reset(seed=2)
    while env.observe(env.agent_selection)["action_mask"].sum() == 1:  # the book fills up after the start
        env.step(0)
    selected = env.agent_selection
    for agent in env.agents:
        mask = env.observe(agent)["action_mask"]
        if agent == selected:
            assert mask.sum() > 1
            assert np.array_equal(mask, env.infos[agent]["action_mask"])
        else:
            assert mask.tolist() == [1] + [0] * (len(mask) - 1), "waiting agents can only hold"


def test_ladder_menu_and_custom_reward():
    class Constant:
        def __init__(self, market, agent_id):
            pass

        def reset(self):
            pass

        def step(self):
            return 1.0, {}

    env = GlassAECEnv(small_config(actions=ActionConfig(kind="ladder")), reward=Constant)
    assert env.action_space(env.possible_agents[0]).n == 23
    env.reset(seed=3)
    for _ in range(30):
        obs, reward, terminated, truncated, _info = env.last()
        if terminated or truncated:
            break
        env.step(0)
    assert reward == 1.0


def test_liquidated_agent_is_terminated_and_the_rest_continue():
    env = GlassAECEnv(small_config())
    buy = env.action_names.index("buy_large")
    env.reset(seed=4)
    victim, drained, ended = "marl_01", False, {}
    for agent in env.agent_iter():
        obs, _reward, terminated, truncated, _info = env.last()
        if terminated or truncated:
            ended[agent] = ("terminated" if terminated else "truncated", env.market.time)
            env.step(None)
            continue
        account = env.market.accounts[victim]
        if not drained and account.position != 0:
            account.collateral = 1.0  # wipe out the margin: the exchange liquidates it
            drained = True
        env.step(buy if agent == victim and obs["action_mask"][buy] else 0)
    assert ended[victim][0] == "terminated" and ended[victim][1] < 600.0
    assert all(ended[agent][0] == "truncated" for agent in ("marl_02", "marl_03"))
    assert env.agents == []
