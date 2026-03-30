from __future__ import annotations

import time

from marl_trading.live.session import LiveMarketSession


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
    assert len(state["tape"]) == state["summary"]["trade_count"]
    assert len(state["recent_trades"]) == len(state["tape"])
    assert len(state["tape"]) > 1


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
