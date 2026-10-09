from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
import json


class EventType(str, Enum):
    LIMIT_ORDER = "limit_order"
    MARKET_ORDER = "market_order"
    CANCEL_ORDER = "cancel_order"
    TRADE = "trade"
    NEWS = "news"
    SNAPSHOT = "snapshot"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: float

    def to_dict(self) -> dict[str, float]:
        return {"price": float(self.price), "quantity": float(self.quantity)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderBookLevel":
        return cls(price=float(data["price"]), quantity=float(data["quantity"]))


@dataclass(frozen=True)
class OrderBookSnapshot:
    timestamp: float
    bids: tuple[OrderBookLevel, ...] = ()
    asks: tuple[OrderBookLevel, ...] = ()

    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    def midpoint(self) -> float | None:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return 0.5 * (best_bid + best_ask)

    def spread(self) -> float | None:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "bids": [level.to_dict() for level in self.bids],
            "asks": [level.to_dict() for level in self.asks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderBookSnapshot":
        if "bids" not in data and "asks" not in data:
            depth = data.get("depth_snapshot") or data.get("order_book") or data.get("book") or {}
            if isinstance(depth, dict):
                data = depth
        return cls(
            timestamp=float(data["timestamp"]),
            bids=tuple(OrderBookLevel.from_dict(level) for level in data.get("bids", [])),
            asks=tuple(OrderBookLevel.from_dict(level) for level in data.get("asks", [])),
        )


@dataclass(frozen=True)
class MarketEvent:
    sequence: int
    timestamp: float
    event_type: EventType | str
    agent_id: str | None = None
    order_id: str | None = None
    side: OrderSide | str | None = None
    order_type: OrderType | str | None = None
    price: float | None = None
    quantity: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    order_book: OrderBookSnapshot | None = None

    def payload_value(self, *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in self.payload:
                return self.payload[key]
        return default

    def to_dict(self) -> dict[str, Any]:
        data = {
            "sequence": int(self.sequence),
            "timestamp": float(self.timestamp),
            "event_type": self.event_type.value if isinstance(self.event_type, Enum) else str(self.event_type),
            "agent_id": self.agent_id,
            "order_id": self.order_id,
            "side": self.side.value if isinstance(self.side, Enum) else self.side,
            "order_type": self.order_type.value if isinstance(self.order_type, Enum) else self.order_type,
            "price": None if self.price is None else float(self.price),
            "quantity": None if self.quantity is None else float(self.quantity),
            "payload": dict(self.payload),
            "order_book": self.order_book.to_dict() if self.order_book is not None else None,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketEvent":
        order_book_data = data.get("order_book")
        if order_book_data is None:
            order_book_data = data.get("depth_snapshot") or data.get("book_snapshot") or data.get("snapshot")
        payload = dict(data.get("payload", {}))
        if isinstance(order_book_data, dict) and "order_book" in payload and not payload.get("depth_snapshot"):
            nested = payload.get("order_book")
            if isinstance(nested, dict):
                order_book_data = nested
        return cls(
            sequence=int(data["sequence"]),
            timestamp=float(data["timestamp"]),
            event_type=data["event_type"],
            agent_id=data.get("agent_id"),
            order_id=data.get("order_id"),
            side=data.get("side"),
            order_type=data.get("order_type"),
            price=None if data.get("price") is None else float(data["price"]),
            quantity=None if data.get("quantity") is None else float(data["quantity"]),
            payload=payload,
            order_book=OrderBookSnapshot.from_dict(order_book_data) if order_book_data else None,
        )


@dataclass
class EventLog:
    events: list[MarketEvent] = field(default_factory=list)

    def append(self, event: MarketEvent) -> None:
        self.events.append(event)

    def extend(self, events: Iterable[MarketEvent]) -> None:
        self.events.extend(events)

    def __iter__(self) -> Iterator[MarketEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {"events": [event.to_dict() for event in self.events]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[dict[str, Any]]) -> "EventLog":
        if isinstance(data, list):
            return cls(events=[MarketEvent.from_dict(event) for event in data])
        return cls(events=[MarketEvent.from_dict(event) for event in data.get("events", [])])

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "EventLog":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EventLog":
        events: list[MarketEvent] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                events.append(MarketEvent.from_dict(json.loads(line)))
        return cls(events=events)

    @classmethod
    def load(cls, path: str | Path) -> "EventLog":
        resolved = Path(path)
        if resolved.suffix.lower() == ".jsonl":
            return cls.from_jsonl(resolved)
        return cls.from_json(resolved)

    def save(self, path: str | Path) -> None:
        resolved = Path(path)
        if resolved.suffix.lower() == ".jsonl":
            self.to_jsonl(resolved)
        else:
            self.to_json(resolved)

    def filter_by_type(self, event_type: EventType | str) -> list[MarketEvent]:
        value = event_type.value if isinstance(event_type, Enum) else str(event_type)
        return [event for event in self.events if (event.event_type.value if isinstance(event.event_type, Enum) else str(event.event_type)) == value]

    def trades(self) -> list[MarketEvent]:
        return self.filter_by_type(EventType.TRADE)

    def snapshots(self) -> list[MarketEvent]:
        return self.filter_by_type(EventType.SNAPSHOT)

    def news_events(self) -> list[MarketEvent]:
        return self.filter_by_type(EventType.NEWS)
