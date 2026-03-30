from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from .simulator import MarketRunResult


_BG = "#f6f8fb"
_PANEL = "#ffffff"
_GRID = "#d7dee8"
_TEXT = "#22303f"
_SUBTEXT = "#59687a"
_MIDPOINT = "#2c3e50"
_FUNDAMENTAL = "#f39c12"
_SPREAD = "#8e44ad"
_TRADE = "#7f8c8d"
_ACTIVE = "#1f7a4d"
_EQUITY = "#2166ac"
_NEWS = "#c0392b"
_BAR = "#4c78a8"
_BAR_DIM = "#95a5a6"


def _font(size: int = 14) -> ImageFont.ImageFont:
    # Pillow's default bitmap font is always available and keeps the demo dependency-light.
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: str = _TEXT) -> None:
    draw.text(xy, text, fill=fill, font=_font())


def _finite(values: Sequence[float | None]) -> list[float]:
    finite_values = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        finite_values.append(float(value))
    return finite_values


def _bounds(*series: Sequence[float | None], pad_ratio: float = 0.08, minimum_span: float = 1.0) -> tuple[float, float]:
    values = [value for seq in series for value in _finite(seq)]
    if not values:
        return 0.0, 1.0

    low = min(values)
    high = max(values)
    span = max(high - low, minimum_span)
    pad = span * pad_ratio
    return low - pad, high + pad


def _map_point(
    index: int,
    count: int,
    value: float,
    low: float,
    high: float,
    box: tuple[int, int, int, int],
) -> tuple[float, float]:
    left, top, right, bottom = box
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    x = left if count <= 1 else left + (index / (count - 1)) * width
    y = bottom - ((value - low) / max(high - low, 1e-9)) * height
    return float(x), float(y)


def _draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=18, fill=_PANEL, outline=_GRID, width=2)
    _draw_text(draw, (box[0] + 16, box[1] + 12), title)


def _draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xs: Sequence[float],
    series: Sequence[tuple[str, Sequence[float | None], str]],
    *,
    y_label: str,
    x_label: str | None = None,
    markers: Sequence[tuple[float, str, str]] | None = None,
    legend_origin: tuple[int, int] | None = None,
) -> None:
    if not xs:
        return

    chart_box = (box[0] + 52, box[1] + 32, box[2] - 16, box[3] - 36)
    left, top, right, bottom = chart_box
    y_min, y_max = _bounds(*(values for _, values, _ in series))
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - fraction * (bottom - top)
        draw.line([(left, y), (right, y)], fill=_GRID, width=1)
        tick_value = y_min + fraction * (y_max - y_min)
        _draw_text(draw, (box[0] + 8, y - 6), f"{tick_value:,.2f}", fill=_SUBTEXT)

    draw.line([(left, top), (left, bottom)], fill=_SUBTEXT, width=1)
    draw.line([(left, bottom), (right, bottom)], fill=_SUBTEXT, width=1)
    _draw_text(draw, (box[0] + 8, box[1] + 18), y_label, fill=_SUBTEXT)
    if x_label is not None:
        _draw_text(draw, (right - 48, bottom + 4), x_label, fill=_SUBTEXT)

    for label, values, color in series:
        points: list[tuple[float, float]] = []
        for index, value in enumerate(values):
            if value is None:
                if len(points) > 1:
                    draw.line(points, fill=color, width=3)
                points = []
                continue
            if isinstance(value, float) and math.isnan(value):
                if len(points) > 1:
                    draw.line(points, fill=color, width=3)
                points = []
                continue
            points.append(_map_point(index, len(xs), float(value), y_min, y_max, chart_box))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)

    if markers:
        for x_value, _label, color in markers:
            if len(xs) <= 1:
                x = left
            else:
                x = left + (x_value / max(xs[-1], 1.0)) * (right - left)
            draw.line([(x, top), (x, bottom)], fill=color, width=1)

    if legend_origin is None:
        legend_origin = (right - 130, box[1] + 14)
    legend_x, legend_y = legend_origin
    for index, (label, _, color) in enumerate(series):
        y = legend_y + index * 16
        draw.rectangle((legend_x, y + 2, legend_x + 10, y + 12), fill=color, outline=color)
        _draw_text(draw, (legend_x + 16, y), label, fill=_TEXT)


def _draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    *,
    y_label: str,
) -> None:
    if not labels:
        return

    chart_box = (box[0] + 28, box[1] + 30, box[2] - 18, box[3] - 34)
    left, top, right, bottom = chart_box
    low = min(0.0, min(values))
    high = max(values)
    if math.isclose(low, high):
        high = low + 1.0
    zero_y = bottom - ((0.0 - low) / max(high - low, 1e-9)) * (bottom - top)

    draw.line([(left, zero_y), (right, zero_y)], fill=_SUBTEXT, width=1)
    _draw_text(draw, (box[0] + 8, box[1] + 18), y_label, fill=_SUBTEXT)

    bar_area_width = right - left
    bar_width = max(int(bar_area_width / max(len(labels), 1) * 0.65), 8)
    gap = max(int(bar_area_width / max(len(labels), 1) * 0.35), 4)
    x = left + 8
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        bar_height = 0 if math.isclose(low, high) else abs((value - 0.0) / max(high - low, 1e-9)) * (bottom - top)
        bar_top = zero_y - bar_height if value >= 0 else zero_y
        bar_bottom = zero_y if value >= 0 else zero_y + bar_height
        draw.rectangle((x, bar_top, x + bar_width, bar_bottom), fill=color, outline=color)
        _draw_text(draw, (x, bottom + 4), label[:10], fill=_SUBTEXT)
        _draw_text(draw, (x, bar_top - 12 if value >= 0 else bar_bottom + 2), f"{value:,.0f}", fill=_TEXT)
        x += bar_width + gap


def plot_market_world(result: MarketRunResult, output_path: str | Path) -> Path:
    step_records = list(result.step_records)
    if not step_records:
        raise ValueError("MarketRunResult has no step records to plot.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1600, 1180
    image = Image.new("RGBA", (width, height), _BG)
    draw = ImageDraw.Draw(image)

    margin = 28
    gap = 18
    panel_w = (width - 2 * margin - gap) // 2
    panel_h = (height - 2 * margin - gap) // 2
    panels = [
        (margin, margin, margin + panel_w, margin + panel_h),
        (margin + panel_w + gap, margin, margin + panel_w * 2 + gap, margin + panel_h),
        (margin, margin + panel_h + gap, margin + panel_w, margin + panel_h * 2 + gap),
        (margin + panel_w + gap, margin + panel_h + gap, margin + panel_w * 2 + gap, margin + panel_h * 2 + gap),
    ]

    xs = [float(record.step_index) for record in step_records]
    midpoint = [record.midpoint for record in step_records]
    fundamental = [record.fundamental for record in step_records]
    spread = [record.spread if record.spread is not None else None for record in step_records]
    trade_count = [float(record.trade_count) for record in step_records]
    active_agents = [float(record.active_agents) for record in step_records]
    total_equity = [float(record.total_equity) for record in step_records]
    news_severity = [record.news_severity if record.news_severity is not None else 0.0 for record in step_records]
    news_markers = [
        (float(record.step_index), record.news_headline or "news", _NEWS)
        for record in step_records
        if record.news_headline
    ]

    _draw_panel(draw, panels[0], "Price Discovery")
    _draw_chart(
        draw,
        panels[0],
        xs,
        [("Midpoint", midpoint, _MIDPOINT), ("Fundamental", fundamental, _FUNDAMENTAL)],
        y_label="Price",
        x_label="Step",
        markers=news_markers,
    )

    _draw_panel(draw, panels[1], "Liquidity and Flow")
    top_box = (panels[1][0] + 8, panels[1][1] + 28, panels[1][2] - 8, panels[1][1] + panel_h // 2)
    bottom_box = (panels[1][0] + 8, panels[1][1] + panel_h // 2 + 4, panels[1][2] - 8, panels[1][3] - 8)
    _draw_chart(draw, top_box, xs, [("Spread", spread, _SPREAD)], y_label="Spread")
    _draw_chart(draw, bottom_box, xs, [("Trades", trade_count, _TRADE)], y_label="Trades", x_label="Step")

    _draw_panel(draw, panels[2], "Participation and Equity")
    top_box = (panels[2][0] + 8, panels[2][1] + 28, panels[2][2] - 8, panels[2][1] + panel_h // 2)
    bottom_box = (panels[2][0] + 8, panels[2][1] + panel_h // 2 + 4, panels[2][2] - 8, panels[2][3] - 8)
    _draw_chart(draw, top_box, xs, [("Active agents", active_agents, _ACTIVE)], y_label="Agents")
    _draw_chart(draw, bottom_box, xs, [("Total equity", total_equity, _EQUITY)], y_label="Equity", x_label="Step")

    _draw_panel(draw, panels[3], "Final Portfolio Snapshot")
    final_portfolios = list(result.final_portfolios.values())
    final_portfolios.sort(key=lambda data: float(data.get("equity", 0.0)), reverse=True)
    labels = [str(data.get("agent_id", f"agent_{index + 1}")) for index, data in enumerate(final_portfolios)]
    values = [float(data.get("equity", 0.0)) for data in final_portfolios]
    colors = [_BAR if str(data.get("status", "")).lower() == "active" else _BAR_DIM for data in final_portfolios]
    _draw_bar_chart(draw, panels[3], labels, values, colors, y_label="Equity")

    title = (
        "Synthetic Market Demo"
        f" | horizon={len(step_records)}"
        f" | trades={result.summary.get('trade_count', 0)}"
        f" | news={result.summary.get('news_count', 0)}"
        f" | final_fundamental={result.final_fundamental:.2f}"
    )
    _draw_text(draw, (margin, 8), title, fill=_TEXT)

    image.save(output, format="PNG")
    return output
