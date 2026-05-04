from __future__ import annotations

from pathlib import Path


def _static_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "marl_trading" / "live" / "static"


def test_live_ui_shell_contains_compact_panels() -> None:
    root = _static_root()
    html = (root / "index.html").read_text(encoding="utf-8")
    for selector in [
        "main-column",
        "connectionStatus",
        "modeStatus",
        "seedStatus",
        "marketTitle",
        "marketTimestamp",
        "marketPrice",
        "newsFeed",
        "marketStats",
        "priceChart",
        "chartModeCandles",
        "chartModeLine",
        "chartTimeframe",
        "orderBookTop",
        "orderBookFull",
        "tradesTape",
        "latestOrders",
        "agentPortfolios",
        "playButton",
        "pauseButton",
        "stepButton",
        "resetButton",
        "speedSelect",
        "top-strip",
        "bottom-strip",
        "book-panel",
        "trades-panel",
        "agents-panel",
    ]:
        assert selector in html

    for label in [
        "Candles",
        "Line",
        "RECENT TRADES",
        "Recent news",
        "Latest orders",
        "Capital map",
        "Participant cards",
    ]:
        assert label in html

    assert "Order flow and capital map" not in html
    assert "agentActions" not in html

    for token in [
        "panel-note",
        "main-column",
        "book-details",
        "chart-head",
        "chart-title-row",
        "chart-price-block",
        "chart-step-inline",
        "chart-mode-toggle",
        "chart-stage",
        "chart-market-strip",
        "chart-stats",
        "sr-only",
    ]:
        assert token in html


def test_live_ui_script_supports_live_polling_and_demo_fallback() -> None:
    root = _static_root()
    js = (root / "app.js").read_text(encoding="utf-8")
    for token in [
        '"/api/state"',
        '"/api/control"',
        '"/api/live/state"',
        '"/api/live/control"',
        "fetchJsonCandidates",
        "createDemoWorld",
        "renderOrderBook",
        "renderTrades",
        "renderActions",
        "renderPortfolios",
        "formatPortfolioLastAction",
        "visibleSeriesForMode",
        "buildSeriesForMode",
        "applyChartWindow",
        "getMaxChartViewOffset",
        "resolveSeriesIndex",
        "ResizeObserver",
        "renderRecentNews",
        "bindTradeContextTooltipEvents",
        "trade-context-tooltip",
        "order-context",
        'seed: Number(firstDefined(session, ["seed"], CONFIG.seed))',
        "els.seedStatus.textContent",
    ]:
        assert token in js

    assert 'lastAction: formatPortfolioLastAction(firstDefined(raw, ["last_action", "note"], ""))' in js
    assert 'String(firstDefined(raw, ["last_action", "note"], ""))' not in js


def test_live_ui_styles_include_market_layout() -> None:
    root = _static_root()
    css = (root / "styles.css").read_text(encoding="utf-8")
    for token in [
        ".workspace",
        ".main-column",
        ".chart-stage",
        ".book-header",
        ".trade-feed-columns",
        ".latest-orders-columns",
        ".latest-order-row .order-detail.buy",
        ".latest-order-row .order-detail.sell",
        ".book-row",
        ".book-spread",
        ".book-total",
        ".trade-row",
        "min-height: 15px",
        "grid-template-columns: 68px 44px minmax(136px, 1.6fr) minmax(110px, max-content) 18px",
        "grid-template-columns: 68px 42px minmax(108px, 0.95fr) minmax(124px, 1.35fr) 18px",
        "white-space: nowrap",
        ".chart-title-row",
        ".chart-price-block",
        ".chart-step-inline",
        ".chart-mode-toggle",
        ".legend-volume",
        ".chart-stage",
        ".chart-market-strip",
        ".chart-panel .news-feed",
        ".news-feed-head",
        ".news-feed-columns",
        ".news-feed-columns > .news-col-severity",
        ".news-feed-columns > .news-col-impact",
        ".news-row",
        ".news-headline",
        ".sr-only",
        ".trade-context",
        "min-width: 42px",
        ".book-row .side-chip",
        "#tradesTape",
        ".latest-orders",
        ".latest-order-row",
        ".orderbook-full",
        ".portfolio-grid",
        "#agentPortfolios",
        "overflow: visible",
        "max-height: none",
        ".feed-list",
        ".chart-legend",
        ".agent-summary",
        ".portfolio-score",
        ".feed-title",
        ".action-detail",
        ".scroll-panel",
        ".top-strip",
        ".bottom-strip",
        ".trade-row.empty",
        ".trade-context-tooltip",
        "@media (max-width: 1260px)",
        "@media (max-width: 760px)",
        "height: 282px",
    ]:
        assert token in css
