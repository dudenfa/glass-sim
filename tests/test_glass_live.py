import json
import threading
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
        request = urllib.request.Request(viewer.url + "/api/control", data=json.dumps({"action": "step", "seconds": 5}).encode(), method="POST", headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(request).read())["ok"]
    finally:
        viewer.stop()
