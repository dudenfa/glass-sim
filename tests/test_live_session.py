from __future__ import annotations

import time
from pathlib import Path

from marl_trading.live.session import LiveMarketSession


class _FakePPOModel:
    def __init__(self, action=(0, 0, 0)) -> None:
        self.action = action
        self.predict_calls = 0

    def predict(self, observation, deterministic=True):  # noqa: ARG002
        self.predict_calls += 1
        return self.action, None


def test_live_session_steps_and_state() -> None:
    session = LiveMarketSession(horizon=12, autoplay=False, step_delay_seconds=0.01)
    state = session.state()
    assert state["session"]["step_index"] == 0
    assert state["session"]["status"] == "paused"
    assert state["market"]["order_book"]["bids"]
    assert state["market"]["order_book"]["asks"]

    next_state = session.step()
    assert next_state["session"]["step_index"] == 1
    assert next_state["history"]
    assert "agents" in next_state


def test_live_session_controls() -> None:
    session = LiveMarketSession(horizon=8, autoplay=False, step_delay_seconds=0.01)
    session.set_speed(10.0)
    assert session.state()["session"]["steps_per_second"] == 10.0
    session.play()
    time.sleep(0.05)
    session.pause()
    assert session.state()["session"]["step_index"] >= 1
    session.pause()
    assert session.state()["session"]["status"] == "paused"


def test_live_session_exposes_live_pnl_payload() -> None:
    session = LiveMarketSession(horizon=24, autoplay=False, step_delay_seconds=0.01)

    initial_state = session.state()
    assert initial_state["agents"]
    assert all("realized_pnl" in agent and "unrealized_pnl" in agent for agent in initial_state["agents"])
    assert all("open_orders" in agent for agent in initial_state["agents"])

    updated_state = session.step(12)
    assert updated_state["agents"]
    assert any(
        abs(float(agent["realized_pnl"])) > 1e-9 or abs(float(agent["unrealized_pnl"])) > 1e-9
        for agent in updated_state["agents"]
    )
    assert any(agent["last_action"] for agent in updated_state["agents"])
    assert any(int(agent["open_orders"]) > 0 for agent in updated_state["agents"])


def test_live_session_tape_accumulates_across_snapshot_window() -> None:
    session = LiveMarketSession(horizon=120, event_limit=1, autoplay=False, step_delay_seconds=0.01)

    for _ in range(80):
        session.step()

    state = session.state()
    assert state["summary"]["trade_count"] > 1
    assert len(state["tape"]) == len(state["recent_trades"]) == 1
    assert len(state["recent_trades"]) == len(state["tape"])


def test_live_session_bounded_history_window_preserves_step_indices() -> None:
    session = LiveMarketSession(
        horizon=180,
        history_limit=24,
        event_limit=24,
        autoplay=False,
        step_delay_seconds=0.01,
    )

    for _ in range(80):
        session.step()

    state = session.state()
    history = state["history"]
    market_line = state["market"]["line"]
    candles = state["market"]["candles"]

    assert 0 < len(history) <= 24
    assert len(market_line) == len(history)
    assert len(candles) <= len(history)
    assert history[-1]["step_index"] == state["session"]["step_index"]
    assert history[0]["step_index"] == history[-1]["step_index"] - len(history) + 1
    assert state["summary"]["history_point_count"] == len(history)


def test_live_session_candles_stay_aligned_to_absolute_step_buckets() -> None:
    session = LiveMarketSession(
        horizon=180,
        history_limit=24,
        event_limit=24,
        autoplay=False,
        step_delay_seconds=0.01,
    )

    for _ in range(80):
        session.step()

    state = session.state()
    candles = state["market"]["candles"]

    assert candles
    assert all(int(candle["start_step"]) % 5 == 0 for candle in candles)
    assert all(int(candle["end_step"]) >= int(candle["start_step"]) for candle in candles)


def test_live_session_recent_actions_include_order_metadata() -> None:
    session = LiveMarketSession(horizon=80, event_limit=40, autoplay=False, step_delay_seconds=0.01)

    for _ in range(25):
        session.step()

    state = session.state()
    recent_actions = state["recent_actions"]
    assert recent_actions
    assert any(action["order_id"] for action in recent_actions if action["event_type"] != "snapshot")
    assert any(action["order_type"] for action in recent_actions if action["event_type"] != "snapshot")
    assert all("event_type" in action for action in recent_actions)


def test_live_session_cancel_actions_expose_original_order_details() -> None:
    session = LiveMarketSession(horizon=20, event_limit=40, autoplay=False, step_delay_seconds=0.01)
    session.reset(seed=7, horizon=20)

    for _ in range(6):
        session.step()

    cancel_action = next(action for action in session.state()["recent_actions"] if action["event_type"] == "cancel_order")
    assert cancel_action["original_side"] in {"buy", "sell"}
    assert float(cancel_action["original_quantity"]) > 0.0
    assert float(cancel_action["original_price"]) > 0.0
    assert float(cancel_action["quantity"]) > 0.0
    assert float(cancel_action["price"]) > 0.0


def test_live_session_reset_restores_initial_state() -> None:
    session = LiveMarketSession(horizon=8, autoplay=False, step_delay_seconds=0.01)
    session.step(3)
    reset_state = session.reset(seed=11, horizon=6)
    assert reset_state["session"]["step_index"] == 0
    assert reset_state["session"]["reset_count"] >= 1
    assert reset_state["session"]["status"] == "paused"


def test_live_session_runtime_ppo_replaces_requested_slot(monkeypatch) -> None:
    fake_model = _FakePPOModel()
    monkeypatch.setattr(LiveMarketSession, "_load_ppo_model", lambda self, path: fake_model)

    session = LiveMarketSession(
        horizon=12,
        autoplay=False,
        step_delay_seconds=0.01,
        checkpoint_path=Path("/tmp/fake_model.zip"),
        learning_agent_id="trend_01",
    )
    state = session.state()
    trend_state = next(agent for agent in state["agents"] if agent["agent_id"] == "trend_01")

    assert trend_state["agent_type"] == "rl_agent"
    for _ in range(6):
        session.step()
        if fake_model.predict_calls >= 1:
            break
    assert fake_model.predict_calls >= 1


def test_live_session_runtime_ppo_can_override_starting_inventory(monkeypatch) -> None:
    fake_model = _FakePPOModel()
    monkeypatch.setattr(LiveMarketSession, "_load_ppo_model", lambda self, path: fake_model)

    session = LiveMarketSession(
        horizon=12,
        autoplay=False,
        step_delay_seconds=0.01,
        checkpoint_path=Path("/tmp/fake_model.zip"),
        learning_agent_id="trend_01",
        learning_agent_starting_inventory=0.0,
    )
    state = session.state()
    trend_state = next(agent for agent in state["agents"] if agent["agent_id"] == "trend_01")

    assert trend_state["inventory"] == 0.0
    assert state["session"]["runtime_policy"]["learning_agent_starting_inventory"] == 0.0


def test_live_session_requires_learning_agent_id_when_checkpoint_enabled() -> None:
    try:
        LiveMarketSession(
            horizon=12,
            autoplay=False,
            step_delay_seconds=0.01,
            checkpoint_path=Path("/tmp/fake_model.zip"),
        )
    except ValueError as exc:
        assert "learning_agent_id" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError when checkpoint is provided without learning_agent_id.")
