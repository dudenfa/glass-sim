from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .events import EventLog, MarketEvent
from .replay import ReplaySeries, build_replay_series


def _coerce_events(events: EventLog | Sequence[MarketEvent] | Iterable[MarketEvent]) -> list[MarketEvent]:
    if isinstance(events, EventLog):
        return list(events.events)
    return list(events)


def _plot_trade_markers(axis: plt.Axes, series: ReplaySeries) -> None:
    if not series.trade_timestamps:
        return

    colors = {"buy": "#1f7a4d", "sell": "#c0392b"}
    markers = {"buy": "^", "sell": "v"}

    for side in {"buy", "sell"}:
        indices = [index for index, value in enumerate(series.trade_sides) if value == side]
        if not indices:
            continue
        axis.scatter(
            [series.trade_timestamps[index] for index in indices],
            [series.trade_prices[index] for index in indices],
            color=colors[side],
            marker=markers[side],
            s=28,
            label=f"{side.title()} trade",
            zorder=4,
        )


def plot_market_replay(
    events: EventLog | Sequence[MarketEvent] | Iterable[MarketEvent],
    output_path: str | Path,
    depth_output_path: str | Path | None = None,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it to generate replay figures."
        ) from exc

    event_list = _coerce_events(events)
    series = build_replay_series(event_list)

    has_context = bool(series.fundamental_values or series.news_severities or series.annotations)
    nrows = 3 if has_context else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 4 + 2.6 * nrows), sharex=True)
    if nrows == 2:
        ax_price, ax_spread = axes
        ax_context = None
    else:
        ax_price, ax_spread, ax_context = axes

    if series.midpoint:
        ax_price.plot(series.timestamps, series.midpoint, color="#2c3e50", linewidth=1.5, label="Midpoint")
    if series.best_bid:
        ax_price.plot(series.timestamps, series.best_bid, color="#1f7a4d", alpha=0.7, linewidth=1.0, label="Best bid")
    if series.best_ask:
        ax_price.plot(series.timestamps, series.best_ask, color="#c0392b", alpha=0.7, linewidth=1.0, label="Best ask")
    if series.fundamental_values:
        ax_price.plot(
            series.fundamental_timestamps,
            series.fundamental_values,
            color="#f39c12",
            linestyle="--",
            linewidth=1.3,
            label="Latent fundamental",
        )

    _plot_trade_markers(ax_price, series)
    for index, timestamp in enumerate(series.news_timestamps):
        severity = series.news_severities[index] if index < len(series.news_severities) else None
        alpha = 0.18 if severity is None else min(0.65, 0.18 + abs(float(severity)) * 0.35)
        ax_price.axvline(timestamp, color="#8e44ad", alpha=alpha, linewidth=0.9)
        if index < 5:
            ax_price.annotate(
                series.news_labels[index],
                xy=(timestamp, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -6),
                textcoords="offset points",
                rotation=90,
                ha="right",
                va="top",
                fontsize=7,
                color="#6c3483",
            )
    ax_price.set_ylabel("Price")
    ax_price.set_title("Synthetic Market Replay")
    ax_price.legend(loc="best")
    ax_price.grid(True, alpha=0.25)

    if series.spread:
        ax_spread.plot(series.timestamps, series.spread, color="#8e44ad", linewidth=1.2, label="Spread")
    ax_spread.set_ylabel("Spread")
    ax_spread.set_xlabel("Event time")
    ax_spread.legend(loc="best")
    ax_spread.grid(True, alpha=0.25)

    if ax_context is not None:
        ax_context.axhline(0.0, color="#bdc3c7", linewidth=0.9)
        if series.news_severities:
            ctx_news_y = [
                0.0 if severity is None else float(severity)
                for severity in series.news_severities
            ]
            ax_context.scatter(
                series.news_timestamps,
                ctx_news_y,
                color="#8e44ad",
                s=24,
                marker="D",
                label="News severity",
                zorder=3,
            )
        if series.annotations:
            ax_context.scatter(
                [annotation.timestamp for annotation in series.annotations],
                [0.0] * len(series.annotations),
                color="#34495e",
                s=20,
                marker="x",
                label="Agent annotations",
                zorder=4,
            )
            for annotation in series.annotations[:5]:
                ax_context.annotate(
                    annotation.label,
                    xy=(annotation.timestamp, 0.0),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2c3e50",
                )
        if series.fundamental_values:
            ax_context.plot(
                series.fundamental_timestamps,
                series.fundamental_values,
                color="#f39c12",
                linewidth=1.1,
                label="Latent fundamental",
            )
        ax_context.set_ylabel("Context")
        ax_context.set_xlabel("Event time")
        ax_context.legend(loc="best")
        ax_context.grid(True, alpha=0.25)

    fig.suptitle(
        "Synthetic Market Replay"
        f" | events={len(series.timestamps)}"
        f" trades={len(series.trade_timestamps)}"
        f" news={len(series.news_timestamps)}"
        f" snapshots={len(series.snapshot_timestamps)}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    if depth_output_path is None and series.snapshots:
        depth_output_path = output_path.with_name(f"{output_path.stem}_depth.png")

    if depth_output_path is not None and series.snapshots:
        import numpy as np

        max_levels = max(max(len(snapshot.bids), len(snapshot.asks)) for snapshot in series.snapshots)
        bid_matrix = np.full((len(series.snapshots), max_levels), np.nan, dtype=float)
        ask_matrix = np.full((len(series.snapshots), max_levels), np.nan, dtype=float)
        for row_index, snapshot in enumerate(series.snapshots):
            for level_index, level in enumerate(snapshot.bids):
                bid_matrix[row_index, level_index] = level.quantity
            for level_index, level in enumerate(snapshot.asks):
                ask_matrix[row_index, level_index] = level.quantity

        fig_depth, (ax_bid, ax_ask) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        bid_im = ax_bid.imshow(bid_matrix, aspect="auto", origin="lower", cmap="Greens")
        ax_bid.set_title("Bid Depth Over Time")
        ax_bid.set_ylabel("Snapshot")
        fig_depth.colorbar(bid_im, ax=ax_bid, fraction=0.046, pad=0.04)

        ask_im = ax_ask.imshow(ask_matrix, aspect="auto", origin="lower", cmap="Reds")
        ax_ask.set_title("Ask Depth Over Time")
        ax_ask.set_ylabel("Snapshot")
        ax_ask.set_xlabel("Book level")
        fig_depth.colorbar(ask_im, ax=ax_ask, fraction=0.046, pad=0.04)

        fig_depth.tight_layout()
        depth_output_path = Path(depth_output_path)
        depth_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig_depth.savefig(depth_output_path, dpi=160)
        plt.close(fig_depth)
        return depth_output_path

    return output_path
