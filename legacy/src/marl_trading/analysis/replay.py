from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .events import EventLog, MarketEvent, OrderBookSnapshot, EventType


@dataclass(frozen=True)
class ReplayAnnotation:
    timestamp: float
    label: str
    agent_id: str | None = None
    severity: float | None = None


@dataclass(frozen=True)
class ReplaySeries:
    timestamps: list[float]
    best_bid: list[float | None]
    best_ask: list[float | None]
    midpoint: list[float | None]
    spread: list[float | None]
    fundamental_timestamps: list[float]
    fundamental_values: list[float]
    trade_timestamps: list[float]
    trade_prices: list[float]
    trade_sides: list[str]
    trade_quantities: list[float]
    news_timestamps: list[float]
    news_labels: list[str]
    news_severities: list[float | None]
    annotations: list[ReplayAnnotation]
    snapshot_timestamps: list[float]
    snapshots: list[OrderBookSnapshot]


def _as_events(events: EventLog | Sequence[MarketEvent] | Iterable[MarketEvent]) -> list[MarketEvent]:
    if isinstance(events, EventLog):
        return list(events.events)
    return list(events)


def _payload_text(payload: dict[str, object], *keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _payload_float(payload: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def build_replay_series(events: EventLog | Sequence[MarketEvent] | Iterable[MarketEvent]) -> ReplaySeries:
    event_list = _as_events(events)

    timestamps: list[float] = []
    best_bid: list[float | None] = []
    best_ask: list[float | None] = []
    midpoint: list[float | None] = []
    spread: list[float | None] = []
    fundamental_timestamps: list[float] = []
    fundamental_values: list[float] = []
    trade_timestamps: list[float] = []
    trade_prices: list[float] = []
    trade_sides: list[str] = []
    trade_quantities: list[float] = []
    news_timestamps: list[float] = []
    news_labels: list[str] = []
    news_severities: list[float | None] = []
    annotations: list[ReplayAnnotation] = []
    snapshot_timestamps: list[float] = []
    snapshots: list[OrderBookSnapshot] = []

    for event in event_list:
        timestamps.append(float(event.timestamp))
        payload = dict(event.payload)

        if event.order_book is not None:
            snapshot_timestamps.append(float(event.timestamp))
            snapshots.append(event.order_book)
            best_bid.append(event.order_book.best_bid())
            best_ask.append(event.order_book.best_ask())
            midpoint.append(event.order_book.midpoint())
            spread.append(event.order_book.spread())
        else:
            best_bid.append(None)
            best_ask.append(None)
            midpoint.append(None)
            spread.append(None)

        fundamental_value = _payload_float(
            payload,
            "latent_fundamental",
            "fundamental",
            "fair_value",
            "reference_value",
            "mark_price",
        )
        if fundamental_value is not None:
            fundamental_timestamps.append(float(event.timestamp))
            fundamental_values.append(fundamental_value)

        event_type = event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type)
        if event_type == EventType.TRADE.value:
            trade_timestamps.append(float(event.timestamp))
            trade_prices.append(float(event.price) if event.price is not None else float("nan"))
            trade_sides.append(str(event.side) if event.side is not None else "unknown")
            trade_quantities.append(float(event.quantity) if event.quantity is not None else 0.0)
        elif event_type == EventType.NEWS.value:
            news_timestamps.append(float(event.timestamp))
            label = _payload_text(payload, "headline", "message", "label", default="news") or "news"
            news_labels.append(label)
            news_severities.append(_payload_float(payload, "severity", "impact", "weight", "intensity"))

        annotation_label = _payload_text(
            payload,
            "agent_annotation",
            "annotation",
            "note",
            "comment",
            "agent_note",
            "agent_state",
            "strategy",
            "signal",
        )
        if annotation_label is not None:
            annotations.append(
                ReplayAnnotation(
                    timestamp=float(event.timestamp),
                    label=annotation_label,
                    agent_id=str(event.agent_id) if event.agent_id is not None else _payload_text(payload, "agent_id"),
                    severity=_payload_float(payload, "confidence", "strength", "score"),
                )
            )

    return ReplaySeries(
        timestamps=timestamps,
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        spread=spread,
        fundamental_timestamps=fundamental_timestamps,
        fundamental_values=fundamental_values,
        trade_timestamps=trade_timestamps,
        trade_prices=trade_prices,
        trade_sides=trade_sides,
        trade_quantities=trade_quantities,
        news_timestamps=news_timestamps,
        news_labels=news_labels,
        news_severities=news_severities,
        annotations=annotations,
        snapshot_timestamps=snapshot_timestamps,
        snapshots=snapshots,
    )


def summarize_event_log(events: EventLog | Sequence[MarketEvent] | Iterable[MarketEvent]) -> dict[str, object]:
    event_list = _as_events(events)
    series = build_replay_series(event_list)

    first_timestamp = series.timestamps[0] if series.timestamps else None
    last_timestamp = series.timestamps[-1] if series.timestamps else None
    final_midpoint = next((value for value in reversed(series.midpoint) if value is not None), None)
    active_agent_ids = {
        str(event.agent_id)
        for event in event_list
        if event.agent_id is not None
    }
    news_severities = [severity for severity in series.news_severities if severity is not None]
    annotation_agent_ids = {annotation.agent_id for annotation in series.annotations if annotation.agent_id}
    fundamental_values = series.fundamental_values

    return {
        "event_count": len(event_list),
        "trade_count": len(series.trade_timestamps),
        "news_count": len(series.news_timestamps),
        "snapshot_count": len(series.snapshot_timestamps),
        "fundamental_point_count": len(fundamental_values),
        "annotation_count": len(series.annotations),
        "unique_agent_count": len(active_agent_ids),
        "annotation_agent_count": len(annotation_agent_ids),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "final_midpoint": final_midpoint,
        "fundamental_min": min(fundamental_values) if fundamental_values else None,
        "fundamental_max": max(fundamental_values) if fundamental_values else None,
        "news_severity_max": max(news_severities) if news_severities else None,
        "news_labels_sample": list(dict.fromkeys(series.news_labels))[:5],
        "annotation_labels_sample": list(dict.fromkeys(annotation.label for annotation in series.annotations))[:5],
        "has_order_book_snapshots": bool(series.snapshots),
    }
