from __future__ import annotations

from pathlib import Path


def test_live_ui_assets_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "marl_trading" / "live" / "static"
    assert (root / "index.html").exists()
    assert (root / "app.js").exists()
    assert (root / "styles.css").exists()

    html = (root / "index.html").read_text(encoding="utf-8")
    for text in [
        "Synthetic market live view",
        "Marl Trading Exchange",
        "Compact live order book, tape, and agent ecology.",
        "Seed --",
        "RECENT TRADES",
        "Recent news",
        "Capital map",
        "Participant cards",
        "Expand full book",
        "Candles",
        "Line",
        "Price",
        "Step",
    ]:
        assert text in html

    app_js = (root / "app.js").read_text(encoding="utf-8")
    assert "/api/state" in app_js
    assert "/api/control" in app_js
    assert "priceChart" in app_js
    assert "renderMarketMeta" in app_js
    assert "renderAgentSummary" in app_js
    assert "renderOrderBook" in app_js
    assert "book-spread" in app_js
    assert "renderTrades" in app_js
    assert "renderRecentNews" in app_js
    assert "renderLatestOrders" in app_js
    assert "CHART_WINDOW_SIZE" in app_js
    assert "setTimeout(" in app_js
    assert "setInterval(" not in app_js
    assert "loopGeneration" in app_js
    assert "refreshInFlight" in app_js
    assert "controlInFlight" in app_js
    assert "enqueueBackendRequest" in app_js
    assert "stopPolling" in app_js
    assert "schedulePolling" in app_js
    assert "TRADES_VISIBLE_ROWS = 10" in app_js
    assert "LATEST_ORDERS_VISIBLE_ROWS = 10" in app_js
    assert "NEWS_VISIBLE_ROWS = 5" in app_js
    assert "ORDER_BOOK_FULL_LEVELS" in app_js
    assert "fmtOrderBookTotal" in app_js
    assert "fmtCompactOrderDetail" in app_js
    assert "fmtLatestOrderType" in app_js
    assert "compactOrderId" in app_js
    assert "originalSide" in app_js
    assert "originalQuantity" in app_js
    assert "originalPrice" in app_js
    assert "orderDetailClass" in app_js
    assert "orderId" in app_js
    assert "order-context" in app_js
    assert "trade-feed-columns" in app_js
    assert "latest-orders-columns" in app_js
    assert "book-header" in app_js
    assert "book-total" in app_js
    assert "volumeBandHeight" in app_js
    assert "No volume data" in app_js
    assert "resolveChartHoverIndex" in app_js
    assert "resolveChartFocus" in app_js
    assert "renderChartPrice" in app_js
    assert "els.seedStatus.textContent" in app_js
    assert "newsCount" in app_js
    assert 'firstDefined(summary, ["trade_count", "tradeCount", "total_trades"], trades.length)' in app_js
    assert 'rows(asks, "ask", ORDER_BOOK_TOP_LEVELS, "bottom")' in app_js
    assert 'rows(fullAsks, "ask", ORDER_BOOK_FULL_LEVELS, "bottom")' in app_js
    assert "xOffset = xSlots - visibleSeries.length" in app_js
