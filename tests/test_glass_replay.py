import json

import numpy as np
import pytest

from glass_sim import Agent, MarketMaker, MarketOrder, PerpetualMarket, Side, scenarios
from glass_sim.replay import Replay, ReplayError, decode


def small(seed=1, duration=600.0):
    return scenarios.small(seed=seed, duration=duration)


def assert_identical(replay, original_result):
    again = replay.run()
    assert replay.verify(again)["identical"]
    assert again.summary() == original_result.summary()
    assert again.agent_table() == original_result.agent_table()


def test_saved_run_replays_identically(tmp_path):
    result = PerpetualMarket(small(seed=3)).run()
    path = result.save(tmp_path / "run")
    assert (path / "replay.json").stat().st_size < 20_000, "a replay stores the recipe, not the states"
    assert_identical(Replay.load(path), result)


def test_interventions_and_partial_runs_replay_identically(tmp_path):
    market = PerpetualMarket(scenarios.default(seed=2, duration=1800.0))
    market.run_until(300.0)
    market.shock_index(np.log(0.98), "crash")
    market.run_until(400.0)
    market.execute("retail_01", [MarketOrder(Side.BUY, 50)])
    market.run_until(900.0)  # saved before the end of the configured duration
    result = market.results()
    replay = Replay.load(result.replay.save(tmp_path / "partial.json"))
    assert replay.end_time == pytest.approx(900.0)
    assert_identical(replay, result)


def test_replay_can_pause_and_continue():
    result = PerpetualMarket(small(seed=4)).run()
    run = result.replay.build()
    assert run.run_until(200.0)
    assert run.market.time == pytest.approx(200.0) and run.market.events_processed > 0
    assert not run.done
    run.run()
    assert run.done and run.verify()["identical"]


def test_external_decisions_are_replayed():
    pytest.importorskip("gymnasium")
    from glass_sim.rl import ActionConfig, EnvConfig, PerpTradingEnv

    env = PerpTradingEnv(EnvConfig(scenario="small", episode_duration=600.0, actions=ActionConfig(kind="ladder")))
    env.reset(options={"market_seed": 4})
    rng = np.random.default_rng(1)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        done = terminated or truncated
    result = env.market.results()
    assert sum(1 for e in result.replay.data["control_log"] if e["kind"] == "act") > 50
    assert_identical(result.replay, result)


def test_multi_agent_episode_is_replayed():
    pytest.importorskip("gymnasium")
    from glass_sim.rl.marl import MultiAgentConfig, MultiAgentMarket

    env = MultiAgentMarket(MultiAgentConfig(scenario="small", n_learners=2, episode_duration=600.0))
    request = env.reset(seed=6)
    rng = np.random.default_rng(0)
    while not request["done"]:
        request = env.step(request["agent"], int(rng.choice(np.flatnonzero(request["mask"]))))
    result = env.market.results()
    assert_identical(result.replay, result)


def test_agents_added_by_hand_are_rebuilt():
    market = PerpetualMarket(small(seed=5))
    control = MarketMaker(wake_interval=1.5)
    control.agent_type = "control_mm"  # changed after construction, as the RL evaluation does
    control.agent_id = "ctrl_01"
    market.add_agent(control, 200_000.0)
    result = market.run()
    assert result.replay is not None
    assert_identical(result.replay, result)


def test_builtin_policy_agent_is_rebuilt():
    pytest.importorskip("gymnasium")
    from glass_sim.rl.policy import PolicyAgent, RandomPolicy

    market = PerpetualMarket(small(seed=7))
    market.add_agent(PolicyAgent(RandomPolicy()), 100_000.0)
    result = market.run()
    assert_identical(result.replay, result)


def test_explicitly_built_markets_replay():
    agents = [(MarketMaker(wake_interval=2.0), 300_000.0), (MarketMaker(wake_interval=3.0), 300_000.0)]
    for i, (agent, _) in enumerate(agents, start=1):
        agent.agent_id = f"mm_{i:02d}"
    result = PerpetualMarket(small(seed=8), agents=agents).run()
    assert_identical(result.replay, result)


def test_unimportable_agents_explain_why(tmp_path):
    class Local(Agent):
        def on_wakeup(self, view, rng):
            return []

    market = PerpetualMarket(small(seed=1, duration=60.0))
    market.add_agent(Local(), 1_000.0)
    result = market.run()
    assert result.replay is None
    assert "inside a function" in result.replay_unavailable
    path = result.save(tmp_path / "run")
    assert not (path / "replay.json").exists()


@pytest.mark.parametrize(
    "entry",
    [
        {"__agent_class__": "subprocess:Popen"},
        {"__dataclass__": "subprocess:CompletedProcess", "fields": {}},
        {"__enum__": "signal:Signals", "value": 9},
    ],
)
def test_loading_never_builds_foreign_objects(entry):
    with pytest.raises(ReplayError):
        decode(entry)


def test_tampered_file_is_refused(tmp_path):
    result = PerpetualMarket(small(seed=1, duration=60.0)).run()
    path = result.replay.save(tmp_path / "replay.json")
    data = json.loads(path.read_text())
    data["config"]["fields"]["agents"]["__tuple__"][0]["fields"]["agent"] = {"__agent_class__": "os:system"}
    path.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="not an Agent subclass"):
        Replay.load(path)


def test_viewer_session_replays_seeks_and_verifies(tmp_path):
    from glass_sim.live import LiveSession

    market = PerpetualMarket(small(seed=9))
    market.run_until(200.0)
    market.shock_index(np.log(0.97), "crash")
    result = market.run()
    session = LiveSession(replay=Replay.load(result.save(tmp_path / "run")))
    state = session.state()["session"]
    assert state["mode"] == "replay" and state["replay"]["verification"] is None
    session.control({"action": "seek", "time": 400.0})
    assert session.market.time == pytest.approx(400.0)
    session.control({"action": "seek", "time": 100.0})  # backwards: rebuilt and replayed
    assert session.market.time == pytest.approx(100.0)
    session.control({"action": "finish"})
    state = session.state()["session"]
    assert state["finished"] and state["replay"]["verification"]["identical"]


def test_cli_replay_checks_the_run(tmp_path, capsys):
    from glass_sim.__main__ import main

    PerpetualMarket(small(seed=2, duration=120.0)).run().save(tmp_path / "run")
    assert main(["replay", str(tmp_path / "run")]) == 0
    assert "identical to the original run" in capsys.readouterr().out
