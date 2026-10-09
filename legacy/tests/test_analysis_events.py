from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from marl_trading.analysis import (
    EventLog,
    EventType,
    MarketEvent,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderSide,
)


def build_sample_log() -> EventLog:
    snapshot = OrderBookSnapshot(
        timestamp=0.0,
        bids=(OrderBookLevel(99.0, 10.0), OrderBookLevel(98.5, 12.0)),
        asks=(OrderBookLevel(100.0, 8.0), OrderBookLevel(100.5, 9.0)),
    )
    return EventLog(
        events=[
            MarketEvent(sequence=1, timestamp=0.0, event_type=EventType.SESSION_START, payload={"session": "demo"}),
            MarketEvent(sequence=2, timestamp=1.0, event_type=EventType.SNAPSHOT, order_book=snapshot),
            MarketEvent(
                sequence=3,
                timestamp=2.0,
                event_type=EventType.TRADE,
                agent_id="agent_1",
                side=OrderSide.BUY,
                price=100.0,
                quantity=2.0,
            ),
            MarketEvent(sequence=4, timestamp=3.0, event_type=EventType.NEWS, payload={"headline": "shock"}),
        ]
    )


def test_event_log_round_trip_json(tmp_path: Path) -> None:
    log = build_sample_log()
    path = tmp_path / "events.json"
    log.to_json(path)

    loaded = EventLog.from_json(path)
    assert len(loaded) == 4
    assert loaded.events[1].order_book is not None
    assert loaded.events[1].order_book.best_bid() == 99.0
    assert loaded.events[1].order_book.best_ask() == 100.0
    assert loaded.events[2].side == OrderSide.BUY.value or loaded.events[2].side == OrderSide.BUY


def test_event_log_round_trip_jsonl(tmp_path: Path) -> None:
    log = build_sample_log()
    path = tmp_path / "events.jsonl"
    log.to_jsonl(path)

    loaded = EventLog.from_jsonl(path)
    assert len(loaded.trades()) == 1
    assert len(loaded.news_events()) == 1
    assert len(loaded.snapshots()) == 1


def test_event_log_generic_load_and_save(tmp_path: Path) -> None:
    log = build_sample_log()
    path = tmp_path / "events.json"
    log.save(path)
    loaded = EventLog.load(path)
    assert loaded.to_dict()["events"][0]["event_type"] == "session_start"
    assert json.loads(path.read_text(encoding="utf-8"))["events"][3]["payload"]["headline"] == "shock"


def test_event_log_from_dict_accepts_plain_event_list_with_nested_snapshot() -> None:
    raw_events = [
        {
            "sequence": 1,
            "timestamp": 1.0,
            "event_type": "snapshot",
            "depth_snapshot": {
                "timestamp": 1.0,
                "bids": [{"price": 99.0, "quantity": 10.0}],
                "asks": [{"price": 100.0, "quantity": 8.0}],
            },
            "payload": {"latent_fundamental": 99.7},
        },
        {
            "sequence": 2,
            "timestamp": 2.0,
            "event_type": "news",
            "payload": {"headline": "shock", "severity": 0.8},
        },
    ]

    loaded = EventLog.from_dict(raw_events)
    assert len(loaded) == 2
    assert loaded.events[0].order_book is not None
    assert loaded.events[0].order_book.best_bid() == 99.0
    assert loaded.events[1].payload["severity"] == 0.8
