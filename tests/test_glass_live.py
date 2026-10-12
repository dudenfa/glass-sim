import json
import urllib.request

from glass_sim.live import FuturesViewer, LiveSession


def test_live_session_controls_and_state():
    session = LiveSession(scenario="small", seed=1, duration=600.0)
    state = session.state()
    assert state["session"]["scenario"] == "small"
    session.control({"action": "step", "seconds": 60})
    state = session.state(bars_since=0)
    assert state["session"]["time"] >= 60
    assert len(state["bars"]["time"]) >= 60
    assert state["book"]["bids"] and state["book"]["asks"]
    agent_id = state["accounts"][0]["agent_id"]
    detail = session.account_detail(agent_id)
    assert detail["account"]["agent_id"] == agent_id
    session.control({"action": "reset", "seed": 2, "transparency": "anonymous"})
    state = session.state()
    assert state["session"]["seed"] == 2 and state["session"]["transparency"] == "anonymous"
    session.control({"action": "finish"})
    assert session.state()["session"]["finished"]


def test_viewer_serves_page_and_api():
    session = LiveSession(scenario="small", seed=1, duration=300.0)
    viewer = FuturesViewer(session, port=0)
    viewer.start()
    try:
        html = urllib.request.urlopen(viewer.url + "/").read().decode()
        assert "GLASS Viewer" in html
        assert urllib.request.urlopen(viewer.url + "/static/app.js").status == 200
        state = json.loads(urllib.request.urlopen(viewer.url + "/api/state").read())
        assert "prices" in state and "accounts" in state
        request = urllib.request.Request(
            viewer.url + "/api/control", data=json.dumps({"action": "step", "seconds": 5}).encode(), method="POST", headers={"Content-Type": "application/json"}
        )
        assert json.loads(urllib.request.urlopen(request).read())["ok"]
    finally:
        viewer.stop()


def test_reset_keeps_settings_that_are_not_given_and_rejects_bad_ones():
    import pytest

    session = LiveSession(scenario="small", seed=1, transparency="anonymous", duration=60.0)
    session.control({"action": "finish"})
    session.control({"action": "reset"})  # same session again
    state = session.state()["session"]
    assert (state["scenario"], state["transparency"], state["seed"]) == ("small", "anonymous", 1)
    assert not state["finished"]
    session.control({"action": "reset", "seed": 2})  # only the seed changes
    state = session.state()["session"]
    assert (state["scenario"], state["transparency"], state["seed"]) == ("small", "anonymous", 2)
    session.control({"action": "reset", "scenario": "default"})  # a new scenario brings its own level
    state = session.state()["session"]
    assert (state["scenario"], state["transparency"], state["seed"]) == ("default", "full", 2)
    with pytest.raises(ValueError):
        session.control({"action": "reset", "transparency": "bogus"})
    with pytest.raises(ValueError):
        session.control({"action": "reset", "scenario": "nope"})
    state = session.state()["session"]
    assert (state["scenario"], state["transparency"], state["seed"]) == ("default", "full", 2), "a bad request leaves the session untouched"
