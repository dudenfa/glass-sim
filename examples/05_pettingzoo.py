"""Several learning agents in one market through the PettingZoo AEC API.

    pip install -e ".[rl]"
    python examples/05_pettingzoo.py

Three learners share a market with the scripted population. Here they are
driven by simple hand-written policies so the loop is easy to read; a MARL
library plugs its own policies in at the same place (``choose``). Turns follow
the market: whoever wakes up next decides, so the order is not a fixed
rotation.
"""

import numpy as np

from glass_sim.rl.marl import MultiAgentConfig
from glass_sim.rl.pettingzoo_env import GlassAECEnv

env = GlassAECEnv(MultiAgentConfig(scenario="small", n_learners=3, episode_duration=1800.0))
names = env.action_names
rng = np.random.default_rng(0)


def choose(agent: str, obs: dict) -> int:
    """marl_01 makes markets, marl_02 trades at random, marl_03 only holds."""
    valid = np.flatnonzero(obs["action_mask"])
    if agent == "marl_01":
        # Quote on both sides, alternating; flatten when the position is not zero.
        position = env.market.accounts[agent].position
        if position != 0 and obs["action_mask"][names.index("close_position")]:
            return names.index("close_position")
        side = "bid_at_touch" if rng.random() < 0.5 else "ask_at_touch"
        return names.index(side) if obs["action_mask"][names.index(side)] else 0
    if agent == "marl_02":
        return int(rng.choice(valid))
    return 0


env.reset(seed=11)
returns = {agent: 0.0 for agent in env.possible_agents}
turns = {agent: 0 for agent in env.possible_agents}
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    returns[agent] += reward
    if terminated or truncated:
        env.step(None)
        continue
    turns[agent] += 1
    env.step(choose(agent, obs))

print(f"market seed {env.market_seed}, {len(names)} actions per turn")
for agent in env.possible_agents:
    account = env.market.accounts[agent]
    print(f"  {agent}: {turns[agent]:4d} decisions, reward {returns[agent]:+.3f} (% of capital), {account.liquidations} liquidations")
