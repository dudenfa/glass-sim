const QUERY = new URLSearchParams(window.location.search);

const unique = (values) => [...new Set(values.filter((value) => value && String(value).trim()))];

const ENDPOINTS = {
  state: unique([QUERY.get("state"), "/api/state", "/api/live/state"]),
  control: unique([QUERY.get("control"), "/api/control", "/api/live/control"]),
};

const CONFIG = {
  mode: QUERY.get("mode") || "auto",
  seed: Number.parseInt(QUERY.get("seed") || "17", 10),
  pollMs: Math.max(250, Number.parseInt(QUERY.get("poll") || "1000", 10)),
  defaultSpeed: Number.parseFloat(QUERY.get("speed") || "1") || 1,
};

const CHART_WINDOW_SIZE = 120;
const TRADES_VISIBLE_ROWS = 10;
const LATEST_ORDERS_VISIBLE_ROWS = 10;
const NEWS_VISIBLE_ROWS = 5;
const ORDER_BOOK_TOP_LEVELS = 10;
const ORDER_BOOK_FULL_LEVELS = 14;
const CHART_LAYOUT = Object.freeze({
  padding: { left: 68, right: 84, top: 24, bottom: 42 },
});

const els = {
  connectionStatus: document.getElementById("connectionStatus"),
  modeStatus: document.getElementById("modeStatus"),
  marketTitle: document.getElementById("marketTitle"),
  marketTimestamp: document.getElementById("marketTimestamp"),
  marketPrice: document.getElementById("marketPrice"),
  newsFeed: document.getElementById("newsFeed"),
  marketStats: document.getElementById("marketStats"),
  agentSummary: document.getElementById("agentSummary"),
  priceChart: document.getElementById("priceChart"),
  chartModeCandles: document.getElementById("chartModeCandles"),
  chartModeLine: document.getElementById("chartModeLine"),
  orderBookTop: document.getElementById("orderBookTop"),
  orderBookFull: document.getElementById("orderBookFull"),
  tradesTape: document.getElementById("tradesTape"),
  latestOrders: document.getElementById("latestOrders"),
  agentActions: document.getElementById("agentActions"),
  agentPortfolios: document.getElementById("agentPortfolios"),
  playButton: document.getElementById("playButton"),
  pauseButton: document.getElementById("pauseButton"),
  stepButton: document.getElementById("stepButton"),
  resetButton: document.getElementById("resetButton"),
  speedSelect: document.getElementById("speedSelect"),
};

const state = {
  source: "demo",
  connected: false,
  autoplay: QUERY.get("autoplay") !== "0" && CONFIG.mode !== "paused",
  chartMode: "candles",
  backendStateUrl: null,
  backendControlUrl: null,
  timer: null,
  snapshot: null,
  chartHover: null,
  demoWorld: createDemoWorld(CONFIG.seed),
  demoIndex: 0,
  lastError: null,
};

let tradeContextTooltip = null;
let tradeContextAnchor = null;

function qs(selector, root = document) {
  const node = root.querySelector(selector);
  if (!node) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return node;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function fmtNumber(value, digits = 2) {
  if (!isFiniteNumber(value)) return "—";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtSigned(value, digits = 2) {
  if (!isFiniteNumber(value)) return "—";
  const prefix = value >= 0 ? "+" : "-";
  return `${prefix}${fmtNumber(Math.abs(value), digits)}`;
}

function fmtQty(value) {
  if (!isFiniteNumber(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function fmtOrderBookTotal(price, quantity) {
  const total = Number(price) * Number(quantity);
  if (!Number.isFinite(total)) return "--";
  return fmtNumber(total, 2);
}

function fmtCompactOrderDetail(action) {
  const side = String(action.side || "").trim().toLowerCase();
  const sideLabel = side === "buy" ? "BUY" : side === "sell" ? "SELL" : "";
  const quantity = Number(action.quantity);
  const price = Number(action.price);
  const orderType = fmtLatestOrderType(action);
  const orderId = String(firstDefined(action, ["orderId", "order_id", "canceled_order_id", "target_order_id"], "")).trim();
  const originalSide = String(firstDefined(action, ["originalSide", "original_side"], side)).trim().toLowerCase();
  const originalQuantity = Number(firstDefined(action, ["originalQuantity", "original_quantity"], quantity));
  const originalPrice = Number(firstDefined(action, ["originalPrice", "original_price"], price));
  const qtyText = Number.isFinite(quantity) && quantity > 0 ? fmtQty(quantity) : "--";
  const priceText = Number.isFinite(price) ? fmtNumber(price, 2) : "--";

  if (orderType === "CANCEL") {
    if (originalSide && Number.isFinite(originalQuantity) && originalQuantity > 0 && Number.isFinite(originalPrice)) {
      return `${originalSide.toUpperCase()} ${fmtQty(originalQuantity)} @ ${fmtNumber(originalPrice, 2)}`;
    }
    return orderId ? `cancel ${compactOrderId(orderId)}` : "cancel";
  }

  if (!sideLabel && qtyText === "--" && priceText === "--" && orderId) {
    return `order ${compactOrderId(orderId)}`;
  }

  if (!sideLabel && qtyText === "--" && priceText === "--") {
    return "--";
  }
  return sideLabel ? `${sideLabel} ${qtyText} @ ${priceText}` : `${qtyText} @ ${priceText}`;
}

function compactOrderId(orderId, maxLength = 10) {
  const trimmed = String(orderId || "").trim();
  if (!trimmed) return "";
  if (trimmed.length <= maxLength) return trimmed;

  const segments = trimmed.split("_").filter(Boolean);
  if (segments.length >= 2) {
    const suffix = segments.slice(-2).join("_");
    if (suffix.length < trimmed.length) {
      return suffix.length <= maxLength + 2 ? suffix : `…${suffix.slice(-maxLength)}`;
    }
  }

  const head = trimmed.slice(0, 4);
  const tail = trimmed.slice(-4);
  return `${head}…${tail}`;
}

function fmtLatestOrderType(action) {
  const candidates = [
    firstDefined(action, ["orderType", "order_type"], ""),
    firstDefined(action, ["eventType", "event_type"], ""),
    firstDefined(action, ["action", "type", "intent"], ""),
  ];

  for (const candidate of candidates) {
    const raw = String(candidate || "").trim().toLowerCase();
    if (!raw) continue;
    if (raw.includes("cancel")) return "CANCEL";
    if (raw.includes("market")) return "MARKET";
    if (raw.includes("limit")) return "LIMIT";
  }

  const price = Number(firstDefined(action, ["price", "limit_price"], NaN));
  const quantity = Number(firstDefined(action, ["quantity", "qty", "size"], NaN));

  if (Number.isFinite(price)) return "LIMIT";
  if (Number.isFinite(quantity)) return "MARKET";
  return "LIMIT";
}

function isOrderEvent(action) {
  const raw = String(firstDefined(action, ["orderType", "order_type", "eventType", "event_type", "action", "type", "intent"], "")).trim().toLowerCase();
  return /(limit|market|cancel|order)/.test(raw);
}

function fmtTime(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (!isFiniteNumber(value)) return String(value);
  const total = Math.max(0, Math.floor(value));
  const hours = String(Math.floor(total / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function toArray(value) {
  return Array.isArray(value) ? value : [];
}

function firstDefined(source, keys, fallback = undefined) {
  for (const key of keys) {
    const value = source?.[key];
    if (value !== undefined && value !== null) return value;
  }
  return fallback;
}

function normalizeLevel(level) {
  if (!level) return null;
  if (Array.isArray(level)) {
    return {
      price: Number(level[0]),
      quantity: Number(level[1]),
    };
  }
  return {
    price: Number(firstDefined(level, ["price", "p", "rate"], NaN)),
    quantity: Number(firstDefined(level, ["quantity", "qty", "size", "volume"], NaN)),
  };
}

function normalizeCandle(raw, fallbackStep = 0) {
  if (!raw) return null;
  if (Array.isArray(raw)) {
    const [step, open, high, low, close, volume, fundamental] = raw;
    return {
      step_index: Number(step ?? fallbackStep),
      open: Number(open),
      high: Number(high),
      low: Number(low),
      close: Number(close),
      volume: Number(volume ?? 0),
      fundamental: Number(fundamental ?? close),
    };
  }
  return {
    step_index: Number(firstDefined(raw, ["step_index", "start_step", "bucket_index", "step", "time", "timestamp_ns", "timestamp"], fallbackStep)),
    bucket_index: Number(firstDefined(raw, ["bucket_index"], fallbackStep)),
    start_step: Number(firstDefined(raw, ["start_step"], fallbackStep)),
    end_step: Number(firstDefined(raw, ["end_step"], fallbackStep)),
    open: Number(firstDefined(raw, ["open", "o"], NaN)),
    high: Number(firstDefined(raw, ["high", "h"], NaN)),
    low: Number(firstDefined(raw, ["low", "l"], NaN)),
    close: Number(firstDefined(raw, ["close", "c", "price", "midpoint", "last"], NaN)),
    volume: Number(firstDefined(raw, ["volume", "v", "trade_count"], 0)),
    fundamental: Number(firstDefined(raw, ["fundamental", "latent_fundamental", "fundamental_close", "fundamental_open"], NaN)),
  };
}

function normalizeTrade(raw, fallbackStep = 0) {
  if (!raw) return null;
  if (Array.isArray(raw)) {
    const [time, side, price, quantity, agent, note] = raw;
    return {
      time: Number(time ?? fallbackStep),
      side: String(side || "unknown").toLowerCase(),
      price: Number(price),
      quantity: Number(quantity),
      agent: String(agent || "—"),
      note: String(note || ""),
    };
  }
  return {
    time: Number(firstDefined(raw, ["time", "timestamp_ns", "timestamp", "step_index", "step"], fallbackStep)),
    side: String(firstDefined(raw, ["side", "aggressor_side", "direction"], "unknown")).toLowerCase(),
    price: Number(firstDefined(raw, ["price", "last_price", "execution_price", "midpoint"], NaN)),
    quantity: Number(firstDefined(raw, ["quantity", "qty", "size", "volume"], NaN)),
    agent: String(firstDefined(raw, ["agent", "agent_id", "taker_agent_id"], "—")),
    note: String(firstDefined(raw, ["note", "reason", "message"], "")),
  };
}

function normalizeAction(raw, fallbackStep = 0) {
  if (!raw) return null;
  if (Array.isArray(raw)) {
    const [time, agent, action, side, quantity, price, note] = raw;
    return {
      time: Number(time ?? fallbackStep),
      agent: String(agent || "—"),
      action: String(action || "observe"),
      orderType: "",
      eventType: "",
      side: String(side || "neutral").toLowerCase(),
      quantity: Number(quantity ?? 0),
      price: Number(price ?? NaN),
      originalSide: "",
      originalQuantity: NaN,
      originalPrice: NaN,
      note: String(note || ""),
    };
  }
  const payload = firstDefined(raw, ["payload"], {}) || {};
  return {
    time: Number(firstDefined(raw, ["time", "timestamp_ns", "timestamp", "step_index", "step"], fallbackStep)),
    agent: String(firstDefined(raw, ["agent", "agent_id", "id"], "—")),
    action: String(firstDefined(raw, ["action", "type", "intent"], "observe")),
    orderType: String(firstDefined(raw, ["order_type", "orderType", "kind"], "")),
    eventType: String(firstDefined(raw, ["event_type", "eventType"], "")),
    orderId: String(firstDefined(raw, ["order_id", "orderId", "canceled_order_id", "target_order_id"], "")),
    originalSide: String(firstDefined(raw, ["original_side", "originalSide"], firstDefined(payload, ["original_side", "originalSide"], ""))),
    originalQuantity: Number(firstDefined(raw, ["original_quantity", "originalQuantity"], firstDefined(payload, ["original_quantity", "originalQuantity"], NaN))),
    originalPrice: Number(firstDefined(raw, ["original_price", "originalPrice"], firstDefined(payload, ["original_price", "originalPrice"], NaN))),
    side: String(firstDefined(raw, ["side"], "neutral")).toLowerCase(),
    quantity: Number(firstDefined(raw, ["quantity", "qty", "size"], 0)),
    price: Number(firstDefined(raw, ["price", "limit_price"], NaN)),
    note: String(firstDefined(raw, ["note", "reason", "message"], "")),
  };
}

function normalizeNews(raw, fallbackStep = 0) {
  if (!raw) return null;
  if (Array.isArray(raw)) {
    const [time, headline, severity, impact] = raw;
    return {
      time: Number(time ?? fallbackStep),
      headline: String(headline || "news"),
      severity: Number(severity ?? 0),
      impact: Number(impact ?? 0),
    };
  }
  return {
    time: Number(firstDefined(raw, ["time", "timestamp_ns", "timestamp", "step_index", "step"], fallbackStep)),
    headline: String(firstDefined(raw, ["headline", "message", "label"], "news")),
    severity: Number(firstDefined(raw, ["severity", "impact"], 0)),
    impact: Number(firstDefined(raw, ["impact", "news_impact"], firstDefined(raw, ["severity"], 0))),
  };
}

function normalizeLatestOrders(actions) {
  const ordered = toArray(actions).filter(Boolean);
  const filtered = ordered.filter(isOrderEvent);
  return (filtered.length ? filtered : ordered).slice(-LATEST_ORDERS_VISIBLE_ROWS).slice().reverse();
}

function formatPortfolioLastAction(raw) {
  if (raw === null || raw === undefined) return "";
  if (typeof raw === "string") return raw.trim();
  if (typeof raw !== "object") return String(raw);

  const note = String(firstDefined(raw, ["annotation", "reason", "note"], "")).trim();
  const eventType = String(firstDefined(raw, ["event_type", "eventType", "action_kind", "actionKind"], "")).trim().toLowerCase();
  const orderType = String(firstDefined(raw, ["order_type", "orderType"], "")).trim().toLowerCase();
  const side = String(firstDefined(raw, ["side"], "")).trim().toLowerCase();
  const quantity = Number(firstDefined(raw, ["quantity", "qty", "size"], NaN));
  const price = Number(firstDefined(raw, ["limit_price", "price"], NaN));

  const parts = [];
  if (eventType.includes("cancel")) {
    parts.push("cancel");
  } else if (orderType.includes("market") || eventType.includes("market")) {
    parts.push("market");
  } else if (orderType.includes("limit") || eventType.includes("limit")) {
    parts.push("limit");
  }

  if (side === "buy" || side === "sell") {
    parts.push(side);
  }

  if (Number.isFinite(quantity) && quantity > 0) {
    parts.push(fmtQty(quantity));
  }

  if (Number.isFinite(price)) {
    parts.push(`@ ${fmtNumber(price, 2)}`);
  }

  const summary = parts.join(" ").trim();
  if (summary) return summary;
  if (note) return note;
  return "";
}

function normalizePortfolio(raw) {
  if (!raw) return null;
  return {
    id: String(firstDefined(raw, ["agent_id", "id", "name"], "—")),
    type: String(firstDefined(raw, ["agent_type", "type"], "agent")),
    status: String(firstDefined(raw, ["status"], firstDefined(raw, ["active"], true) ? "active" : "inactive")),
    cash: Number(firstDefined(raw, ["cash"], NaN)),
    inventory: Number(firstDefined(raw, ["inventory"], NaN)),
    equity: Number(firstDefined(raw, ["equity"], NaN)),
    realized: Number(firstDefined(raw, ["realized_pnl", "realized", "pnl_realized"], 0)),
    unrealized: Number(firstDefined(raw, ["unrealized_pnl", "unrealized", "pnl_unrealized"], 0)),
    availableCash: Number(firstDefined(raw, ["available_cash"], NaN)),
    availableInventory: Number(firstDefined(raw, ["available_inventory"], NaN)),
    freeEquity: Number(firstDefined(raw, ["free_equity"], NaN)),
    active: Boolean(firstDefined(raw, ["active", "is_active"], true)),
    deactivatedReason: String(firstDefined(raw, ["deactivated_reason", "deactivation_reason"], "")),
    deactivatedAtNs: Number(firstDefined(raw, ["deactivated_at_ns"], NaN)),
    ruinThreshold: Number(firstDefined(raw, ["ruin_threshold"], NaN)),
    openOrders: Number(firstDefined(raw, ["open_orders", "resting_orders"], 0)),
    lastAction: formatPortfolioLastAction(firstDefined(raw, ["last_action", "note"], "")),
  };
}

function normalizeOrderBook(raw) {
  if (!raw) {
    return { bids: [], asks: [], fullBids: [], fullAsks: [] };
  }
  const bids = toArray(firstDefined(raw, ["bids", "bid_levels", "buy_levels"], [])).map(normalizeLevel).filter(Boolean);
  const asks = toArray(firstDefined(raw, ["asks", "ask_levels", "sell_levels"], [])).map(normalizeLevel).filter(Boolean);
  const fullBids = toArray(firstDefined(raw, ["full_bids", "bids_full", "all_bids"], bids)).map(normalizeLevel).filter(Boolean);
  const fullAsks = toArray(firstDefined(raw, ["full_asks", "asks_full", "all_asks"], asks)).map(normalizeLevel).filter(Boolean);
  return { bids, asks, fullBids, fullAsks };
}

function visibleSeriesForMode(snapshot) {
  if (state.chartMode === "candles") {
    const candles = (snapshot.candles || []).slice(-CHART_WINDOW_SIZE);
    if (candles.length) {
      return {
        series: candles,
        fundamentals: candles.map((candle) => ({
          step_index: candle.step_index,
          value: Number.isFinite(candle.fundamental) ? candle.fundamental : candle.close,
        })),
      };
    }
    const lineSeries = (snapshot.lineSeries || []).slice(-CHART_WINDOW_SIZE);
    return {
      series: lineSeries.map((point) => ({
        step_index: point.step_index,
        open: point.midpoint,
        high: point.midpoint,
        low: point.midpoint,
        close: point.midpoint,
        fundamental: point.fundamental ?? snapshot.fundamental,
        volume: point.volume,
      })),
      fundamentals: lineSeries.map((point) => ({
        step_index: point.step_index,
        value: point.fundamental ?? snapshot.fundamental,
      })),
    };
  }
  const lineSeries = (snapshot.lineSeries || []).slice(-CHART_WINDOW_SIZE);
  return {
    series: lineSeries.length ? lineSeries : (snapshot.candles || []).slice(-CHART_WINDOW_SIZE).map((candle) => ({
      step_index: candle.step_index,
      midpoint: candle.close,
      fundamental: candle.fundamental ?? snapshot.fundamental,
      volume: candle.volume,
    })),
    fundamentals: (snapshot.fundamentalSeries || lineSeries.map((point) => ({
      step_index: point.step_index,
      value: point.fundamental ?? snapshot.fundamental,
    }))).slice(-CHART_WINDOW_SIZE),
  };
}

function resolveSeriesIndex(series, targetStep) {
  if (!series.length || !Number.isFinite(targetStep)) return -1;
  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (let index = 0; index < series.length; index += 1) {
    const point = series[index];
    const candidates = [
      point.step_index,
      point.start_step,
      point.end_step,
      point.bucket_index,
    ].filter((value) => Number.isFinite(value));

    if (candidates.some((value) => value === targetStep)) {
      return index;
    }

    for (const candidate of candidates) {
      const distance = Math.abs(candidate - targetStep);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    }
  }

  return bestIndex;
}

function getChartWidthPx() {
  if (!els.priceChart) return NaN;
  return els.priceChart.width / (window.devicePixelRatio || 1);
}

function resolveChartHoverIndex(snapshot, width, chartX) {
  if (!snapshot || !Number.isFinite(width) || !Number.isFinite(chartX)) return -1;
  const { series: visibleSeries, fundamentals } = visibleSeriesForMode(snapshot);
  if (!visibleSeries.length) return -1;
  const padding = CHART_LAYOUT.padding;
  const plotWidth = Math.max(1, width - padding.left - padding.right);
  const xSlots = Math.max(CHART_WINDOW_SIZE, visibleSeries.length, fundamentals.length, 1);
  const xStep = plotWidth / Math.max(1, xSlots - 1);
  const xOffset = xSlots - visibleSeries.length;
  const rawIndex = Math.round((chartX - padding.left) / xStep - xOffset);
  return Math.max(0, Math.min(visibleSeries.length - 1, rawIndex));
}

function resolveChartFocus(snapshot, width) {
  const { series: visibleSeries, fundamentals } = visibleSeriesForMode(snapshot);
  const latestPoint = visibleSeries.at(-1) || null;
  const latestFundamental = Number(fundamentals.at(-1)?.value ?? snapshot.fundamental);
  const hoveredIndex = resolveChartHoverIndex(snapshot, width, state.chartHover?.x);
  const focusIndex = hoveredIndex >= 0 ? hoveredIndex : visibleSeries.length - 1;
  const focusPoint = focusIndex >= 0 ? visibleSeries[focusIndex] || latestPoint : latestPoint;
  const focusFundamental = focusIndex >= 0
    ? Number(fundamentals[Math.min(focusIndex, fundamentals.length - 1)]?.value ?? latestFundamental)
    : latestFundamental;
  const fallbackPrice = Number(firstDefined(snapshot, ["midpoint", "price", "last_price"], NaN));
  const rawPrice = state.chartMode === "candles"
    ? Number(firstDefined(focusPoint || {}, ["close", "midpoint"], fallbackPrice))
    : Number(firstDefined(focusPoint || {}, ["midpoint", "close"], fallbackPrice));
  const price = Number.isFinite(rawPrice) ? rawPrice : fallbackPrice;
  const volume = Number.isFinite(Number(focusPoint?.volume)) ? Number(focusPoint.volume) : Number(snapshot.volume);
  return {
    hoveredIndex,
    focusIndex,
    focusPoint,
    price,
    fundamental: Number.isFinite(focusFundamental) ? focusFundamental : latestFundamental,
    volume: Number.isFinite(volume) ? volume : 0,
  };
}

function renderChartPrice(snapshot) {
  if (!els.marketPrice || !snapshot) return null;
  const focus = resolveChartFocus(snapshot, getChartWidthPx());
  els.marketPrice.textContent = fmtNumber(focus.price, 2);
  return focus;
}

function renderRecentNews(snapshot) {
  if (!els.newsFeed) return;

  const news = toArray(snapshot.news || snapshot.recent_news || [])
    .map((item, index) => normalizeNews(item, index))
    .filter((item) => item.headline)
    .slice(-NEWS_VISIBLE_ROWS)
    .slice()
    .reverse();
  const totalNews = Number(firstDefined(snapshot.stats || {}, ["newsCount", "news_count"], news.length));

  const headerRow = `
    <div class="news-feed-columns" aria-hidden="true">
      <span class="news-col-time">Time</span>
      <span class="news-col-headline">Headline</span>
      <span class="news-col-severity">Severity</span>
      <span class="news-col-impact">Impact</span>
    </div>
  `;

  const head = `
    <div class="news-feed-head">
      <span>Recent news</span>
    </div>
  `;

  if (!news.length) {
    els.newsFeed.innerHTML = [
      head,
      headerRow,
      `
        <article class="news-row empty">
          <span class="news-time">--:--:--</span>
          <span class="news-headline">Waiting for news</span>
          <span class="news-severity">—</span>
          <span class="news-impact">—</span>
        </article>
      `,
    ].join("");
    return;
  }

  els.newsFeed.innerHTML = [
    head,
    headerRow,
    ...news.map((item) => {
      const impact = Number(item.impact);
      const impactClass = impact > 0 ? "positive" : impact < 0 ? "negative" : "neutral";
      return `
        <article class="news-row">
          <span class="news-time">${escapeHtml(fmtTime(item.time))}</span>
          <span class="news-headline" title="${escapeHtml(item.headline)}">${escapeHtml(item.headline)}</span>
          <span class="news-severity">${fmtSigned(item.severity, 2)}</span>
          <span class="news-impact ${impactClass}">${fmtSigned(impact, 2)}</span>
        </article>
      `;
    }),
  ].join("");
}

function normalizeBackendState(raw) {
  const session = firstDefined(raw, ["session"], {}) || {};
  const market = firstDefined(raw, ["market"], {}) || {};
  const candles = toArray(firstDefined(market, ["candles"], firstDefined(raw, ["candles"], [])))
    .map((candle, index) => normalizeCandle(candle, index))
    .filter(Boolean);
  const lineSeries = toArray(firstDefined(market, ["line"], firstDefined(raw, ["history"], [])))
    .map((point, index) => {
      if (Array.isArray(point)) {
        return {
          step_index: Number(point[0] ?? index),
          midpoint: Number(point[1]),
          fundamental: Number(point[2] ?? point[1]),
          volume: Number(point[3] ?? NaN),
        };
      }
      return {
        step_index: Number(firstDefined(point, ["step_index", "step", "time", "timestamp"], index)),
        midpoint: Number(firstDefined(point, ["midpoint", "value", "close", "price"], NaN)),
        fundamental: Number(firstDefined(point, ["fundamental"], NaN)),
        volume: Number(firstDefined(point, ["volume", "v", "trade_volume"], NaN)),
      };
    })
    .filter((point) => Number.isFinite(point.midpoint));
  const orderBook = normalizeOrderBook(firstDefined(market, ["order_book", "orderBook", "book"], {}));
  const fullOrderBook = normalizeOrderBook(firstDefined(market, ["full_order_book", "fullOrderBook"], orderBook));
  const trades = toArray(firstDefined(raw, ["tape", "recent_trades"], []))
    .map((trade, index) => normalizeTrade(trade, index))
    .filter(Boolean);
  const actions = toArray(firstDefined(raw, ["actions", "recent_actions"], []))
    .map((action, index) => normalizeAction(action, index))
    .filter(Boolean);
  const news = toArray(firstDefined(raw, ["news", "recent_news"], []))
    .map((item, index) => normalizeNews(item, index))
    .filter((item) => item.headline);
  const portfolios = toArray(firstDefined(raw, ["portfolios", "agents"], []))
    .map(normalizePortfolio)
    .filter(Boolean);
  const summary = firstDefined(raw, ["summary"], {}) || {};
  const newsCount = Number(firstDefined(summary, ["news_count", "newsCount", "recent_news_count", "news_total"], news.length));

  const latestCandle = candles.at(-1) || null;
  const midpoint = Number(firstDefined(market, ["midpoint", "price", "last_price"], latestCandle?.close ?? lineSeries.at(-1)?.midpoint ?? NaN));
  const spread = Number(firstDefined(market, ["spread"], Number.isFinite(orderBook.bids[0]?.price) && Number.isFinite(orderBook.asks[0]?.price)
    ? orderBook.asks[0].price - orderBook.bids[0].price
    : NaN));

  return {
    source: "backend",
    marketName: String(firstDefined(raw, ["market_name"], firstDefined(market, ["symbol", "name"], "Synthetic market"))),
    timestamp: Number(firstDefined(session, ["current_step_index", "step_index"], firstDefined(market, ["timestamp_ns", "timestamp"], 0))),
    midpoint,
    fundamental: Number(firstDefined(market, ["fundamental"], lineSeries.at(-1)?.fundamental ?? midpoint)),
    spread,
    volume: Number(firstDefined(market, ["volume", "total_volume"], latestCandle?.volume ?? 0)),
    candles,
    lineSeries,
    fundamentalSeries: lineSeries
      .map((point) => ({ step_index: point.step_index, value: point.fundamental }))
      .filter((point) => Number.isFinite(point.value)),
    orderBook,
    fullOrderBook,
    trades,
    actions,
    news,
    portfolios,
    stats: {
      activeAgents: Number(firstDefined(session, ["active_agent_count"], portfolios.filter((portfolio) => portfolio.active).length)),
      tradeCount: trades.length,
      newsCount: Number.isFinite(newsCount) ? newsCount : news.length,
      eventCount: Number(firstDefined(raw, ["summary", "event_count"], trades.length + actions.length)),
    },
    raw,
  };
}

function makePortfolio(id, type, cash, inventory, avgCost = 100, ruinThreshold = 4000) {
  return {
    id,
    type,
    cash,
    inventory,
    avgCost,
    ruinThreshold,
    realized: 0,
    unrealized: 0,
    equity: cash + inventory * avgCost,
    active: true,
    deactivatedReason: "",
    openOrders: 0,
    lastAction: "boot",
  };
}

function applyBuy(portfolio, price, quantity) {
  const cost = price * quantity;
  const previousInventory = portfolio.inventory;
  const nextInventory = previousInventory + quantity;
  const basis = previousInventory > 0 ? portfolio.avgCost * previousInventory : 0;
  portfolio.avgCost = nextInventory > 0 ? (basis + cost) / nextInventory : 0;
  portfolio.inventory = nextInventory;
  portfolio.cash -= cost;
}

function applySell(portfolio, price, quantity) {
  const sellQty = Math.min(quantity, portfolio.inventory);
  if (sellQty <= 0) return 0;
  portfolio.cash += price * sellQty;
  portfolio.realized += sellQty * (price - portfolio.avgCost);
  portfolio.inventory -= sellQty;
  if (portfolio.inventory <= 1e-9) {
    portfolio.inventory = 0;
    portfolio.avgCost = 0;
  }
  return sellQty;
}

function updatePortfolioMark(portfolio, markPrice) {
  portfolio.unrealized = portfolio.inventory * (markPrice - portfolio.avgCost);
  portfolio.equity = portfolio.cash + portfolio.inventory * markPrice;
  if (portfolio.equity <= portfolio.ruinThreshold && portfolio.active) {
    portfolio.active = false;
    portfolio.deactivatedReason = "ruin_threshold_breached";
    portfolio.openOrders = 0;
    portfolio.lastAction = "ruined";
  }
}

function buildBookLevels(basePrice, tick, rng, count, skew = 0, bidBias = 0, askBias = 0) {
  const bids = [];
  const asks = [];
  for (let level = 1; level <= count; level += 1) {
    const bidPrice = Number((basePrice - 0.08 - (level - 1) * tick).toFixed(2));
    const askPrice = Number((basePrice + 0.08 + (level - 1) * tick).toFixed(2));
    bids.push({
      price: bidPrice,
      quantity: Math.max(1, Math.round(26 - level + bidBias + skew + gaussian(rng) * 4)),
    });
    asks.push({
      price: askPrice,
      quantity: Math.max(1, Math.round(24 - level + askBias - skew + gaussian(rng) * 4)),
    });
  }
  return { bids, asks, fullBids: bids, fullAsks: asks };
}

function createDemoWorld(seed = 17) {
  const rng = mulberry32(seed);
  const agents = [
    makePortfolio("maker_alpha", "market_maker", 16000, 52, 100.1, 5000),
    makePortfolio("noise_beta", "noise_trader", 12000, 24, 99.8, 4200),
    makePortfolio("trend_gamma", "trend_follower", 10000, 18, 100.3, 3800),
    makePortfolio("informed_delta", "informed_trader", 14000, 20, 100.6, 4500),
  ];

  const timeline = [];
  const trades = [];
  const actions = [];
  const news = [];

  let price = 100;
  let fundamental = 100.7;
  let previousClose = price;

  for (let index = 0; index < 240; index += 1) {
    const timestamp = 9 * 3600 + 30 * 60 + index * 45;
    const newsEvent = index > 0 && index % 48 === 0
      ? {
          time: timestamp,
          headline: [
            "Exchange flow imbalance",
            "Risk appetite shift",
            "Informed desks reprice risk",
            "Liquidity pocket opens",
          ][Math.floor(index / 48) % 4],
          severity: clamp(0.35 + rng() * 0.9, 0.2, 1.2),
        }
      : null;

    if (newsEvent) {
      news.push(newsEvent);
    }

    fundamental += 0.01 + gaussian(rng) * 0.04 + (newsEvent ? (newsEvent.severity - 0.5) * 0.18 : 0);
    const open = previousClose;
    const drift = (fundamental - price) * 0.08;
    const shock = gaussian(rng) * 0.12;
    const close = Math.max(1, price + drift + shock);
    const high = Math.max(open, close) + Math.abs(gaussian(rng)) * 0.12;
    const low = Math.min(open, close) - Math.abs(gaussian(rng)) * 0.12;
    const volume = Math.round(180 + Math.abs(gaussian(rng)) * 240 + (newsEvent ? newsEvent.severity * 180 : 0));
    price = close;
    previousClose = close;

    const book = buildBookLevels(close, 0.05, rng, 20, newsEvent ? newsEvent.severity : 0, newsEvent ? -2 : 3, newsEvent ? 2 : 0);
    const frameActions = [];
    const frameTrades = [];

    const maker = agents[0];
    const noise = agents[1];
    const trend = agents[2];
    const informed = agents[3];

    if (maker.active && Math.abs(maker.inventory - 52) > 8) {
      const side = maker.inventory > 52 ? "sell" : "buy";
      const qty = 2;
      const priceLevel = side === "buy" ? book.asks[0].price : book.bids[0].price;
      if (side === "buy") {
        applyBuy(maker, priceLevel, qty);
      } else {
        applySell(maker, priceLevel, qty);
      }
      maker.lastAction = `${side} ${qty}`;
      const action = { time: timestamp, agent: maker.id, action: "rebalance", side, quantity: qty, price: priceLevel, note: "inventory_skew_quote" };
      frameActions.push(action);
      actions.push(action);
      frameTrades.push({ time: timestamp, side, price: priceLevel, quantity: qty, agent: maker.id, note: "maker_rebalance" });
    }

    if (noise.active && rng() < 0.45) {
      const side = rng() < 0.5 ? "buy" : "sell";
      const qty = rng() < 0.8 ? 1 : 2;
      const priceLevel = side === "buy" ? book.asks[0].price : book.bids[0].price;
      if (side === "buy") {
        applyBuy(noise, priceLevel, qty);
      } else {
        applySell(noise, priceLevel, qty);
      }
      noise.lastAction = `${side} ${qty}`;
      const action = { time: timestamp, agent: noise.id, action: "noise", side, quantity: qty, price: priceLevel, note: "random_flow" };
      frameActions.push(action);
      actions.push(action);
      frameTrades.push({ time: timestamp, side, price: priceLevel, quantity: qty, agent: noise.id, note: "random_flow" });
    }

    const recentReturn = index > 2 ? close - timeline[index - 3].close : 0;
    if (trend.active && (Math.abs(recentReturn) > 0.15 || newsEvent)) {
      const side = recentReturn >= 0 ? "buy" : "sell";
      const qty = clamp(Math.round(1 + Math.abs(recentReturn) * 6), 1, 4);
      const priceLevel = side === "buy" ? book.asks[1].price : book.bids[1].price;
      if (side === "buy") {
        applyBuy(trend, priceLevel, qty);
      } else {
        applySell(trend, priceLevel, qty);
      }
      trend.lastAction = `${side} ${qty}`;
      const action = {
        time: timestamp,
        agent: trend.id,
        action: "trend_follow",
        side,
        quantity: qty,
        price: priceLevel,
        note: recentReturn >= 0 ? "momentum_bid" : "momentum_offer",
      };
      frameActions.push(action);
      actions.push(action);
      frameTrades.push({ time: timestamp, side, price: priceLevel, quantity: qty, agent: trend.id, note: action.note });
    }

    const edge = fundamental - close + (newsEvent ? newsEvent.severity * 0.08 : 0);
    if (informed.active && Math.abs(edge) > 0.1) {
      const side = edge > 0 ? "buy" : "sell";
      const qty = clamp(Math.round(2 + Math.abs(edge) * 6), 1, 5);
      const priceLevel = side === "buy" ? book.asks[0].price : book.bids[0].price;
      if (side === "buy") {
        applyBuy(informed, priceLevel, qty);
      } else {
        applySell(informed, priceLevel, qty);
      }
      informed.lastAction = `${side} ${qty}`;
      const action = {
        time: timestamp,
        agent: informed.id,
        action: "signal_trade",
        side,
        quantity: qty,
        price: priceLevel,
        note: newsEvent ? "news_signal" : "fundamental_edge",
      };
      frameActions.push(action);
      actions.push(action);
      frameTrades.push({ time: timestamp, side, price: priceLevel, quantity: qty, agent: informed.id, note: action.note });
    }

    for (const portfolio of agents) {
      updatePortfolioMark(portfolio, close);
    }

    timeline.push({
      time: timestamp,
      open,
      high,
      low,
      close,
      volume,
      fundamental,
      orderBook: book,
      trades: frameTrades,
      actions: frameActions,
      news: newsEvent ? [newsEvent] : [],
      portfolios: agents.map((portfolio) => ({
        id: portfolio.id,
        type: portfolio.type,
        status: portfolio.active ? "active" : "inactive",
        cash: portfolio.cash,
        inventory: portfolio.inventory,
        equity: portfolio.equity,
        realized: portfolio.realized,
        unrealized: portfolio.unrealized,
        active: portfolio.active,
        deactivatedReason: portfolio.deactivatedReason,
        openOrders: portfolio.openOrders,
        lastAction: portfolio.lastAction,
      })),
    });
  }

  return {
    seed,
    timeline,
    trades,
    actions,
    news,
    portfolios: agents,
    frameAt(index) {
      const safeIndex = clamp(index, 0, timeline.length - 1);
      const frame = timeline[safeIndex];
      const candleWindow = timeline.slice(Math.max(0, safeIndex - 119), safeIndex + 1);
      const tradeWindow = trades.slice(Math.max(0, trades.length - 12));
      const actionWindow = actions.slice(Math.max(0, actions.length - 12));
      const newsWindow = news.slice(Math.max(0, news.length - 6));
      return {
        source: "demo",
        marketName: "Synthetic spot market",
        timestamp: frame.time,
        midpoint: frame.close,
        fundamental: frame.fundamental,
        spread: frame.orderBook.asks[0].price - frame.orderBook.bids[0].price,
        volume: frame.volume,
        candles: candleWindow.map((candle) => ({
          step_index: candle.time,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          volume: candle.volume,
          fundamental: candle.fundamental,
        })),
        lineSeries: candleWindow.map((candle) => ({
          step_index: candle.time,
          midpoint: candle.close,
          fundamental: candle.fundamental,
          volume: candle.volume,
        })),
        fundamentalSeries: candleWindow.map((candle) => ({
          step_index: candle.time,
          value: candle.fundamental,
        })),
        orderBook: frame.orderBook,
        fullOrderBook: frame.orderBook,
        trades: tradeWindow,
        actions: actionWindow,
        news: newsWindow,
        portfolios: frame.portfolios,
        stats: {
          activeAgents: frame.portfolios.filter((portfolio) => portfolio.active).length,
          tradeCount: trades.length,
          newsCount: news.length,
          eventCount: timeline.length,
        },
      };
    },
  };
}

function setConnectionStatus(text, tone = "neutral") {
  els.connectionStatus.textContent = text;
  els.connectionStatus.dataset.tone = tone;
}

function setModeStatus(text) {
  els.modeStatus.textContent = text;
}

function updateModeButtons() {
  els.chartModeCandles.classList.toggle("chip-active", state.chartMode === "candles");
  els.chartModeLine.classList.toggle("chip-active", state.chartMode === "line");
}

function resizeCanvas() {
  const canvas = els.priceChart;
  const parent = canvas.parentElement;
  const width = Math.max(320, parent.clientWidth - 16);
  const height = Math.max(420, Math.round(width * 0.56));
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  if (state.snapshot) {
    drawChart(state.snapshot);
    renderChartPrice(state.snapshot);
  }
}

function drawChart(snapshot) {
  const canvas = els.priceChart;
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1018";
  ctx.fillRect(0, 0, width, height);

  const candles = (snapshot.candles || []).slice(-CHART_WINDOW_SIZE);
  const lineSeries = (snapshot.lineSeries || candles.map((candle, index) => ({
    step_index: candle.step_index ?? index,
    midpoint: candle.close,
    fundamental: candle.fundamental ?? snapshot.fundamental,
    volume: candle.volume,
  }))).slice(-CHART_WINDOW_SIZE);
  const { series: visibleSeries, fundamentals } = visibleSeriesForMode(snapshot);
  const candleVolumeByStep = new Map(candles.map((candle, index) => [Number.isFinite(candle.step_index) ? candle.step_index : index, Number(candle.volume || 0)]));
  const values = [];

  if (state.chartMode === "candles") {
    for (const candle of visibleSeries) {
      values.push(candle.open, candle.high, candle.low, candle.close);
    }
  } else {
    for (const point of visibleSeries) {
      values.push(point.midpoint);
    }
  }
  for (const point of fundamentals) {
    values.push(point.value);
  }
  if (!values.length) {
    ctx.fillStyle = "#9eb0c7";
    ctx.font = "15px Space Grotesk, Trebuchet MS, sans-serif";
    ctx.fillText("Waiting for market data...", 24, 36);
    return;
  }

  const minValue = Math.min(...values) * 0.995;
  const maxValue = Math.max(...values) * 1.005;
  const range = Math.max(0.0001, maxValue - minValue);
  const padding = CHART_LAYOUT.padding;
  const plotWidth = Math.max(1, width - padding.left - padding.right);
  const volumeBandHeight = Math.max(64, Math.round(height * 0.18));
  const priceBottom = Math.max(padding.top + 1, height - padding.bottom - volumeBandHeight);
  const plotHeight = Math.max(1, priceBottom - padding.top);
  const xSlots = Math.max(CHART_WINDOW_SIZE, visibleSeries.length, fundamentals.length, 1);
  const xStep = plotWidth / Math.max(1, xSlots - 1);
  const xOffset = xSlots - visibleSeries.length;
  const toY = (value) => padding.top + (1 - (value - minValue) / range) * plotHeight;
  const toX = (index) => padding.left + (xOffset + index) * xStep;
  const focus = resolveChartFocus(snapshot, width);

  ctx.strokeStyle = "rgba(164, 186, 215, 0.08)";
  ctx.lineWidth = 1;
  ctx.font = "12px IBM Plex Mono, SFMono-Regular, Consolas, monospace";
  ctx.fillStyle = "#9eb0c7";
  for (let index = 0; index <= 5; index += 1) {
    const y = padding.top + (plotHeight / 5) * index;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    const price = maxValue - (range * index) / 5;
    ctx.fillText(fmtNumber(price, 2), width - padding.right + 8, y + 4);
  }

  for (let index = 0; index < visibleSeries.length; index += 12) {
    const x = toX(index);
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, priceBottom);
    ctx.stroke();
  }

  if (state.chartMode === "candles") {
    const bodyWidth = Math.max(4, Math.min(12, xStep * 0.58));
    visibleSeries.forEach((candle, index) => {
      const x = toX(index);
      const openY = toY(candle.open);
      const closeY = toY(candle.close);
      const highY = toY(candle.high);
      const lowY = toY(candle.low);
      const rising = candle.close >= candle.open;
      ctx.strokeStyle = rising ? "#167a5a" : "#b3443c";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, highY);
      ctx.lineTo(x, lowY);
      ctx.stroke();
      ctx.fillStyle = rising ? "rgba(22, 122, 90, 0.74)" : "rgba(179, 68, 60, 0.74)";
      ctx.fillRect(x - bodyWidth / 2, Math.min(openY, closeY), bodyWidth, Math.max(1, Math.abs(closeY - openY)));
    });
  } else {
    ctx.strokeStyle = "rgba(115, 167, 255, 0.95)";
    ctx.lineWidth = 2.25;
    ctx.beginPath();
    visibleSeries.forEach((point, index) => {
      const x = toX(index);
      const y = toY(point.midpoint);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "rgba(115, 167, 255, 0.12)";
    ctx.beginPath();
    visibleSeries.forEach((point, index) => {
      const x = toX(index);
      const y = toY(point.midpoint);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(toX(Math.max(0, visibleSeries.length - 1)), height - padding.bottom);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.closePath();
    ctx.fill();
  }

  if (fundamentals.length) {
    ctx.strokeStyle = "rgba(186, 123, 36, 0.88)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    fundamentals.forEach((point, index) => {
      const x = toX(index);
      const y = toY(point.value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (focus.focusIndex >= 0) {
    const hoverX = toX(focus.focusIndex);
    ctx.save();
    ctx.strokeStyle = "rgba(164, 186, 215, 0.22)";
    ctx.setLineDash([3, 6]);
    ctx.beginPath();
    ctx.moveTo(hoverX, padding.top);
    ctx.lineTo(hoverX, height - padding.bottom);
    ctx.stroke();
    ctx.restore();
  }

  const guidePrice = focus.price;
  const guideY = toY(guidePrice);
  ctx.save();
  ctx.strokeStyle = focus.hoveredIndex >= 0 ? "rgba(164, 186, 215, 0.22)" : "rgba(164, 186, 215, 0.16)";
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.moveTo(padding.left, guideY);
  ctx.lineTo(width - padding.right, guideY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  if (focus.focusIndex >= 0) {
    const hoverX = toX(focus.focusIndex);
    ctx.save();
    ctx.fillStyle = "#edf4ff";
    ctx.strokeStyle = "#0b1018";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(hoverX, guideY, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  ctx.fillStyle = "#edf4ff";
  ctx.font = "13px IBM Plex Mono, SFMono-Regular, Consolas, monospace";
  ctx.fillText(`Last ${fmtNumber(focus.price, 2)}`, padding.left, padding.top + 14);
  ctx.fillStyle = "#f5a524";
  ctx.fillText(`Fundamental ${fmtNumber(focus.fundamental, 2)}`, padding.left, padding.top + 30);

  const volumeValues = visibleSeries.map((point, index) => {
    const directVolume = Number(point.volume);
    if (Number.isFinite(directVolume)) {
      return Math.max(0, directVolume);
    }
    const key = Number.isFinite(point.step_index) ? point.step_index : index;
    return Math.max(0, candleVolumeByStep.get(key) ?? 0);
  });
  const maxVolume = Math.max(0, ...volumeValues);
  const volumeTop = priceBottom + 12;
  const volumeHeight = Math.max(1, height - padding.bottom - volumeTop);
  ctx.fillStyle = "rgba(164, 186, 215, 0.07)";
  ctx.fillRect(padding.left, volumeTop, plotWidth, volumeHeight);
  ctx.strokeStyle = "rgba(164, 186, 215, 0.12)";
  ctx.beginPath();
  ctx.moveTo(padding.left, volumeTop);
  ctx.lineTo(width - padding.right, volumeTop);
  ctx.stroke();
  ctx.fillStyle = "#9eb0c7";
  ctx.font = "10px IBM Plex Mono, SFMono-Regular, Consolas, monospace";
  ctx.fillText(`VOL ${fmtQty(focus.volume)}`, padding.left, volumeTop + 11);
  if (maxVolume > 0) {
    const barWidth = Math.max(3, Math.min(12, xStep * 0.6));
    const barScale = Math.max(1, volumeHeight - 18);
    volumeValues.forEach((volume, index) => {
      if (!Number.isFinite(volume) || volume <= 0) return;
      const x = toX(index);
      const barHeight = Math.max(1, (volume / maxVolume) * barScale);
      const y = height - padding.bottom - barHeight;
      const rising = state.chartMode === "candles" && (visibleSeries[index]?.close ?? 0) >= (visibleSeries[index]?.open ?? 0);
      ctx.fillStyle = rising ? "rgba(45, 212, 191, 0.45)" : "rgba(115, 167, 255, 0.45)";
      ctx.fillRect(x - barWidth / 2, y, barWidth, barHeight);
    });
  } else {
    ctx.fillStyle = "rgba(164, 186, 215, 0.35)";
    ctx.fillText("No volume data", padding.left + 36, volumeTop + 10);
  }
}

function renderMarketMeta(snapshot) {
  renderChartPrice(snapshot);
  els.marketTitle.textContent = snapshot.marketName || "Synthetic market";
  els.marketTimestamp.textContent = fmtTime(snapshot.timestamp);

  const stats = [
    { label: "Agents", value: snapshot.stats?.activeAgents ?? snapshot.portfolios?.filter((portfolio) => portfolio.active).length ?? 0 },
    { label: "Trades", value: snapshot.stats?.tradeCount ?? snapshot.trades?.length ?? 0 },
    { label: "News", value: snapshot.stats?.newsCount ?? (Array.isArray(snapshot.news) ? snapshot.news.length : snapshot.news ? 1 : 0) },
  ];
  els.marketStats.innerHTML = stats
    .map((item) => `<span class="stat-chip">${escapeHtml(item.label)}: <strong>${escapeHtml(String(item.value))}</strong></span>`)
    .join("");
}

function renderAgentSummary(snapshot) {
  const portfolios = snapshot.portfolios || snapshot.agents || [];
  const active = portfolios.filter((portfolio) => portfolio.active).length;
  const inactive = Math.max(0, portfolios.length - active);
  const totalEquity = portfolios.reduce((sum, portfolio) => sum + Number(portfolio.equity || 0), 0);
  const totalRealized = portfolios.reduce((sum, portfolio) => sum + Number(portfolio.realized || 0), 0);
  const totalOpenOrders = portfolios.reduce((sum, portfolio) => sum + Number(portfolio.openOrders || 0), 0);
  const topAgent = [...portfolios].sort((left, right) => Number(right.equity || 0) - Number(left.equity || 0))[0];

  els.agentSummary.innerHTML = [
    { label: "Active", value: active },
    { label: "Inactive", value: inactive },
    { label: "Total equity", value: fmtNumber(totalEquity, 2) },
    { label: "Open orders", value: totalOpenOrders },
    { label: "Realized PnL", value: fmtSigned(totalRealized, 2) },
    { label: "Leader", value: topAgent ? `${topAgent.id} (${fmtNumber(topAgent.equity, 2)})` : "—" },
  ]
    .map((item) => `
      <div class="summary-card">
        <span>${escapeHtml(item.label)}</span>
        <strong>${escapeHtml(String(item.value))}</strong>
      </div>
    `)
    .join("");
}

function renderOrderBook(snapshot) {
  const book = snapshot.fullOrderBook || snapshot.orderBook || { bids: [], asks: [], fullBids: [], fullAsks: [] };
  const bids = [...(book.bids || [])].sort((left, right) => right.price - left.price).slice(0, ORDER_BOOK_TOP_LEVELS);
  const asks = [...(book.asks || [])].sort((left, right) => left.price - right.price).slice(0, ORDER_BOOK_TOP_LEVELS);
  const fullBids = [...(book.fullBids || bids)].sort((left, right) => right.price - left.price).slice(0, ORDER_BOOK_FULL_LEVELS);
  const fullAsks = [...(book.fullAsks || asks)].sort((left, right) => left.price - right.price).slice(0, ORDER_BOOK_FULL_LEVELS);
  const spreadValue = Number.isFinite(snapshot.spread)
    ? Number(snapshot.spread)
    : (Number.isFinite(asks[0]?.price) && Number.isFinite(bids[0]?.price)
      ? asks[0].price - bids[0].price
      : NaN);
  const maxQty = Math.max(
    1,
    ...bids.map((level) => level.quantity || 0),
    ...asks.map((level) => level.quantity || 0),
    ...fullBids.map((level) => level.quantity || 0),
    ...fullAsks.map((level) => level.quantity || 0),
  );

  const padLevels = (levels, count, align = "top") => {
    const visibleLevels = levels.slice(0, count);
    const orderedLevels = align === "bottom" ? [...visibleLevels].reverse() : visibleLevels;
    const blanks = Array.from({ length: Math.max(0, count - visibleLevels.length) }, () => null);
    return align === "bottom" ? [...blanks, ...orderedLevels] : [...orderedLevels, ...blanks];
  };

  const spreadRow = (value) => `
    <div class="book-spread">
      <span>Spread</span>
      <strong>${fmtNumber(value, 2)}</strong>
    </div>
  `;

  const headerRow = `
    <div class="book-header" aria-hidden="true">
      <span>Side</span>
      <span>Price</span>
      <span>Size</span>
      <span>Total</span>
    </div>
  `;

  const rows = (levels, side, count, align = "top") => padLevels(levels, count, align).map((level) => {
    if (!level) {
      return `
        <div class="book-row empty ${side}">
          <span class="side-chip">${side === "buy" ? "Bid" : "Ask"}</span>
          <span class="book-price">--</span>
          <span class="book-qty">--</span>
          <span class="book-total">--</span>
        </div>
      `;
    }
    const fill = clamp((level.quantity / maxQty) * 100, 4, 100);
    const total = fmtOrderBookTotal(level.price, level.quantity);
    return `
      <div class="book-row ${side}" style="--fill:${fill}%">
        <span class="side-chip ${side === "buy" ? "buy" : "sell"}">${side === "buy" ? "Bid" : "Ask"}</span>
        <span class="book-price">${fmtNumber(level.price, 2)}</span>
        <span class="book-qty">${fmtQty(level.quantity)}</span>
        <span class="book-total">${total}</span>
      </div>
    `;
  }).join("");

  els.orderBookTop.innerHTML = `
    <div class="book-side">
      <h3>Asks</h3>
      ${headerRow}
      ${rows(asks, "ask", ORDER_BOOK_TOP_LEVELS, "bottom")}
      ${spreadRow(spreadValue)}
    </div>
    <div class="book-side">
      <h3>Bids</h3>
      ${headerRow}
      ${rows(bids, "buy", ORDER_BOOK_TOP_LEVELS)}
    </div>
  `;

  els.orderBookFull.innerHTML = `
    <div class="book-side">
      <h3>Full asks</h3>
      ${headerRow}
      ${rows(fullAsks, "ask", ORDER_BOOK_FULL_LEVELS, "bottom")}
      ${spreadRow(spreadValue)}
    </div>
    <div class="book-side">
      <h3>Full bids</h3>
      ${headerRow}
      ${rows(fullBids, "buy", ORDER_BOOK_FULL_LEVELS)}
    </div>
  `;
}

function renderTrades(snapshot) {
  const trades = (snapshot.trades || snapshot.recent_trades || []).slice(-TRADES_VISIBLE_ROWS).slice().reverse();
  const headerRow = `
    <div class="trade-feed-columns" aria-hidden="true">
      <span class="trade-col-time">Time</span>
      <span class="trade-col-chip">Side</span>
      <span class="trade-col-agent">Agent</span>
      <span class="trade-col-price">Price × Qty</span>
      <span class="trade-col-note">Note</span>
    </div>
  `;

  if (!trades.length) {
    els.tradesTape.innerHTML = [
      headerRow,
      Array.from({ length: TRADES_VISIBLE_ROWS }, (_, index) => `
      <article class="trade-row empty">
        <span class="trade-time">--:--:--</span>
        <span class="trade-chip">—</span>
        <span class="trade-agent">slot ${index + 1}</span>
        <span class="trade-price">—</span>
        <span class="trade-note">Waiting for trades</span>
      </article>
    `).join(""),
    ].join("");
    return;
  }

  const rows = trades.map((trade) => {
    const buy = trade.side === "buy";
    const context = String(trade.note || "").trim();
    const contextBadge = context
      ? `<button type="button" class="trade-context" data-context="${escapeHtml(context)}" aria-label="Trade context">${"i"}</button>`
      : `<span class="trade-context trade-context-empty" aria-hidden="true"></span>`;
    return `
      <article class="trade-row">
        <span class="trade-time">${escapeHtml(fmtTime(trade.time))}</span>
        <span class="trade-chip ${buy ? "buy" : "sell"}">${buy ? "BUY" : "SELL"}</span>
        <span class="trade-agent">${escapeHtml(trade.agent)}</span>
        <span class="trade-price">${fmtNumber(trade.price, 2)} × ${fmtQty(trade.quantity)}</span>
        ${contextBadge}
      </article>
    `;
  });
  const emptyRows = Array.from({ length: Math.max(0, TRADES_VISIBLE_ROWS - rows.length) }, (_, index) => `
    <article class="trade-row empty">
      <span class="trade-time">--:--:--</span>
      <span class="trade-chip">—</span>
      <span class="trade-agent">slot ${rows.length + index + 1}</span>
      <span class="trade-price">—</span>
      <span class="trade-context trade-context-empty" aria-hidden="true"></span>
    </article>
  `);
  els.tradesTape.innerHTML = [headerRow, ...rows, ...emptyRows].join("");
}

function renderLatestOrders(snapshot) {
  if (!els.latestOrders) {
    return;
  }

  const orders = normalizeLatestOrders(snapshot.actions || snapshot.recent_actions || []);
  const headerRow = `
    <div class="latest-orders-columns" aria-hidden="true">
      <span class="order-col-time">Time</span>
      <span class="order-col-chip">Type</span>
      <span class="order-col-agent">Agent</span>
      <span class="order-col-detail">Detail</span>
      <span class="order-col-info">Info</span>
    </div>
  `;

  els.latestOrders.innerHTML = `
    <div class="latest-orders-head">
      <span>Recent Orders</span>
    </div>
    <div class="latest-orders-list">
      ${headerRow}
      ${orders.length
        ? orders.map((action) => {
          const orderType = fmtLatestOrderType(action);
          const orderDetail = fmtCompactOrderDetail(action);
          const orderDetailClass = orderType === "CANCEL"
            ? "order-detail neutral"
            : String(action.side || "").toLowerCase() === "buy"
              ? "order-detail buy"
              : String(action.side || "").toLowerCase() === "sell"
                ? "order-detail sell"
                : "order-detail neutral";
          const note = String(action.note || "").trim();
          const noteBadge = note
            ? `<button type="button" class="trade-context order-context" data-context="${escapeHtml(note)}" aria-label="Order note">${"i"}</button>`
            : `<span class="trade-context trade-context-empty" aria-hidden="true"></span>`;
          return `
            <article class="latest-order-row">
              <span class="order-time">${escapeHtml(fmtTime(action.time))}</span>
              <span class="order-chip">${escapeHtml(orderType)}</span>
              <span class="order-agent">${escapeHtml(action.agent || "—")}</span>
              <span class="${orderDetailClass}" title="${escapeHtml(orderDetail)}">${escapeHtml(orderDetail)}</span>
              ${noteBadge}
            </article>
          `;
        }).join("")
        : Array.from({ length: LATEST_ORDERS_VISIBLE_ROWS }, (_, index) => `
            <article class="latest-order-row empty">
              <span class="order-time">--:--:--</span>
              <span class="order-chip">—</span>
              <span class="order-agent">slot ${index + 1}</span>
              <span class="order-detail">--</span>
              <span class="trade-context trade-context-empty" aria-hidden="true"></span>
            </article>
          `).join("")}
    </div>
  `;
}

function ensureTradeContextTooltip() {
  if (tradeContextTooltip) {
    return tradeContextTooltip;
  }

  const tooltip = document.createElement("div");
  tooltip.className = "trade-context-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.setAttribute("aria-hidden", "true");
  tooltip.hidden = true;
  document.body.appendChild(tooltip);
  tradeContextTooltip = tooltip;
  return tooltip;
}

function hideTradeContextTooltip() {
  if (!tradeContextTooltip) {
    return;
  }

  tradeContextAnchor = null;
  tradeContextTooltip.classList.remove("is-visible");
  tradeContextTooltip.hidden = true;
  tradeContextTooltip.setAttribute("aria-hidden", "true");
}

function positionTradeContextTooltip(anchor) {
  if (!tradeContextTooltip || tradeContextTooltip.hidden) {
    return;
  }

  const anchorRect = anchor.getBoundingClientRect();
  const tooltipRect = tradeContextTooltip.getBoundingClientRect();
  const margin = 12;
  const gap = 10;
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const tooltipWidth = Math.min(320, Math.max(220, tooltipRect.width || 260));
  const tooltipHeight = Math.max(28, tooltipRect.height || 0);
  const placeBelow = anchorRect.top - tooltipHeight - gap < margin;
  const left = clamp(anchorRect.left + (anchorRect.width / 2) - (tooltipWidth / 2), margin, Math.max(margin, viewportWidth - tooltipWidth - margin));
  const top = placeBelow
    ? clamp(anchorRect.bottom + gap, margin, Math.max(margin, viewportHeight - tooltipHeight - margin))
    : clamp(anchorRect.top - tooltipHeight - gap, margin, Math.max(margin, viewportHeight - tooltipHeight - margin));

  tradeContextTooltip.dataset.placement = placeBelow ? "bottom" : "top";
  tradeContextTooltip.style.left = `${left}px`;
  tradeContextTooltip.style.top = `${top}px`;
}

function showTradeContextTooltip(anchor, context) {
  const trimmed = String(context || "").trim();
  if (!trimmed) {
    hideTradeContextTooltip();
    return;
  }

  const tooltip = ensureTradeContextTooltip();
  tradeContextAnchor = anchor;
  tooltip.textContent = trimmed;
  tooltip.hidden = false;
  tooltip.setAttribute("aria-hidden", "false");
  tooltip.classList.add("is-visible");
  positionTradeContextTooltip(anchor);
}

function bindTradeContextTooltipEvents() {
  const getTrigger = (target) => (target instanceof Element ? target.closest(".trade-context[data-context]") : null);

  for (const root of [els.tradesTape, els.latestOrders].filter(Boolean)) {
    root.addEventListener("pointerover", (event) => {
      const trigger = getTrigger(event.target);
      if (!trigger) return;
      showTradeContextTooltip(trigger, trigger.dataset.context);
    });

    root.addEventListener("pointerout", (event) => {
      const trigger = getTrigger(event.target);
      if (!trigger || trigger !== tradeContextAnchor) return;
      const related = event.relatedTarget;
      if (related && (tradeContextTooltip?.contains(related) || trigger.contains(related))) return;
      hideTradeContextTooltip();
    });

    root.addEventListener("focusin", (event) => {
      const trigger = getTrigger(event.target);
      if (!trigger) return;
      showTradeContextTooltip(trigger, trigger.dataset.context);
    });

    root.addEventListener("focusout", (event) => {
      const trigger = getTrigger(event.target);
      if (!trigger || trigger !== tradeContextAnchor) return;
      const related = event.relatedTarget;
      if (related && (tradeContextTooltip?.contains(related) || trigger.contains(related))) return;
      hideTradeContextTooltip();
    });

    root.addEventListener("click", (event) => {
      const trigger = getTrigger(event.target);
      if (!trigger) return;
      event.preventDefault();
      showTradeContextTooltip(trigger, trigger.dataset.context);
    });
  }

  window.addEventListener("scroll", () => {
    if (tradeContextAnchor) {
      positionTradeContextTooltip(tradeContextAnchor);
    }
  }, true);

  window.addEventListener("resize", () => {
    if (tradeContextAnchor) {
      positionTradeContextTooltip(tradeContextAnchor);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideTradeContextTooltip();
    }
  });
}

function renderActions(snapshot) {
  if (!els.agentActions) return;

  const actions = (snapshot.actions || snapshot.recent_actions || []).slice(-12).slice().reverse();
  if (!actions.length) {
    els.agentActions.innerHTML = `<div class="empty-state">No agent actions recorded yet.</div>`;
    return;
  }

  els.agentActions.innerHTML = actions.map((action) => {
    const buy = action.side === "buy";
    const sideLabel = buy ? "BUY" : action.side === "sell" ? "SELL" : action.side ? action.side.toUpperCase() : "OBSERVE";
    const actionTitle = action.action || "observe";
    const actionDetail = action.quantity ? `${fmtQty(action.quantity)} @ ${fmtNumber(action.price, 2)}` : "No executable size";
    return `
      <article class="feed-item">
        <header>
          <div class="feed-title">
            <strong>${escapeHtml(action.agent)}</strong>
            <span class="feed-subtitle">${escapeHtml(actionTitle)}</span>
          </div>
          <span class="action-chip ${buy ? "buy" : "sell"}">${escapeHtml(sideLabel)}</span>
        </header>
        <div class="body action-body">
          <div class="action-detail">${escapeHtml(actionDetail)}</div>
          ${action.note ? `<div class="meta-line">${escapeHtml(action.note)}</div>` : "<div class=\"meta-line\">No note</div>"}
        </div>
        <div class="meta">${escapeHtml(fmtTime(action.time))}</div>
      </article>
    `;
  }).join("");
}

function renderPortfolios(snapshot) {
  const portfolios = snapshot.portfolios || snapshot.agents || [];
  if (!portfolios.length) {
    els.agentPortfolios.innerHTML = `<div class="empty-state">No portfolio data yet.</div>`;
    return;
  }

  const sorted = [...portfolios].sort((left, right) => right.equity - left.equity);
  els.agentPortfolios.innerHTML = sorted.map((portfolio) => {
    const pnl = (portfolio.realized ?? 0) + (portfolio.unrealized ?? 0);
    const pnlClass = pnl >= 0 ? "buy" : "sell";
    const statusLabel = portfolio.active ? "active" : (portfolio.status || "inactive");
    const deactivationLabel = portfolio.deactivatedReason ? `reason: ${portfolio.deactivatedReason}` : `${portfolio.openOrders ?? 0} open orders`;
    return `
      <article class="portfolio-card">
        <div class="portfolio-head">
          <div>
            <h4>${escapeHtml(portfolio.id)}</h4>
            <div class="portfolio-type">${escapeHtml(portfolio.type)}</div>
          </div>
          <div class="portfolio-score">
            <span class="side-chip ${portfolio.active ? "buy" : "sell"}">${escapeHtml(statusLabel)}</span>
            <strong class="portfolio-equity">${fmtNumber(portfolio.equity, 2)}</strong>
            <span class="portfolio-pnl ${pnlClass}">${fmtSigned(pnl, 2)}</span>
          </div>
        </div>
        <div class="portfolio-badges">
          <span class="portfolio-badge ${portfolio.active ? "buy" : "sell"}">${portfolio.active ? "trading" : "inactive"}</span>
          <span class="portfolio-badge">open orders ${portfolio.openOrders ?? 0}</span>
          ${portfolio.deactivatedReason ? `<span class="portfolio-badge">ruin: ${escapeHtml(portfolio.deactivatedReason)}</span>` : ""}
        </div>
        <div class="portfolio-note">${escapeHtml(deactivationLabel)}</div>
        <div class="portfolio-stats">
          <div class="portfolio-stat">
            <span>Equity</span>
            <strong>${fmtNumber(portfolio.equity, 2)}</strong>
          </div>
          <div class="portfolio-stat">
            <span>Free equity</span>
            <strong>${fmtNumber(portfolio.freeEquity ?? portfolio.equity, 2)}</strong>
          </div>
          <div class="portfolio-stat">
            <span>PnL</span>
            <strong class="${pnlClass}">${fmtSigned(pnl, 2)}</strong>
          </div>
          <div class="portfolio-stat">
            <span>Realized</span>
            <strong>${fmtSigned(portfolio.realized, 2)}</strong>
          </div>
          <div class="portfolio-stat">
            <span>Unrealized</span>
            <strong>${fmtSigned(portfolio.unrealized, 2)}</strong>
          </div>
          <div class="portfolio-stat">
            <span>Cash</span>
            <strong>${fmtNumber(portfolio.cash, 2)}</strong>
          </div>
          <div class="portfolio-stat">
            <span>Inventory</span>
            <strong>${fmtQty(portfolio.inventory)}</strong>
          </div>
          <div class="portfolio-stat">
            <span>Available cash</span>
            <strong>${fmtNumber(portfolio.availableCash ?? portfolio.cash, 2)}</strong>
          </div>
        </div>
        <div class="portfolio-status">
          <span class="meta">${portfolio.lastAction ? escapeHtml(portfolio.lastAction) : "idle"}</span>
          <span class="meta">${escapeHtml(deactivationLabel)}</span>
        </div>
      </article>
    `;
  }).join("");
}

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  renderMarketMeta(snapshot);
  renderRecentNews(snapshot);
  renderOrderBook(snapshot);
  renderTrades(snapshot);
  renderLatestOrders(snapshot);
  renderAgentSummary(snapshot);
  renderActions(snapshot);
  renderPortfolios(snapshot);
  updateModeButtons();
  drawChart(snapshot);
}

function backendAvailable() {
  return state.source === "backend";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: {
      Accept: "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

async function fetchJsonCandidates(urls, options = {}) {
  let lastError = null;
  for (const url of urls) {
    try {
      const data = await fetchJson(url, options);
      return { url, data };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("No endpoint responded");
}

function deriveControlUrl(stateUrl) {
  if (stateUrl.endsWith("/api/live/state")) return stateUrl.replace("/api/live/state", "/api/live/control");
  if (stateUrl.endsWith("/api/state")) return stateUrl.replace("/api/state", "/api/control");
  return ENDPOINTS.control[0] || null;
}

async function loadBackendSnapshot() {
  const { url, data } = await fetchJsonCandidates(ENDPOINTS.state);
  state.backendStateUrl = url;
  state.backendControlUrl = deriveControlUrl(url);
  return normalizeBackendState(data);
}

async function sendControl(action, extra = {}) {
  if (!backendAvailable()) return null;
  const controlUrl = state.backendControlUrl || ENDPOINTS.control[0];
  if (!controlUrl) return null;
  return fetchJson(controlUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action,
      speed: Number.parseFloat(els.speedSelect.value || String(CONFIG.defaultSpeed)),
      ...extra,
    }),
  });
}

function demoSnapshotAt(index) {
  return state.demoWorld.frameAt(index);
}

function stepDemo() {
  const nextIndex = clamp(state.demoIndex + 1, 0, state.demoWorld.timeline.length - 1);
  state.demoIndex = nextIndex;
  renderSnapshot(demoSnapshotAt(nextIndex));
  setModeStatus(`Demo frame ${nextIndex + 1}/${state.demoWorld.timeline.length}`);
}

function resetDemo() {
  state.demoIndex = 0;
  renderSnapshot(demoSnapshotAt(0));
  setModeStatus("Demo reset");
}

async function refreshBackend() {
  try {
    const snapshot = await loadBackendSnapshot();
    state.connected = true;
    state.lastError = null;
    setConnectionStatus("Connected", "live");
    setModeStatus("Backend live stream");
    renderSnapshot(snapshot);
  } catch (error) {
    state.connected = false;
    state.lastError = error;
    if (state.source === "backend") {
      setConnectionStatus("Connection lost", "error");
      setModeStatus(`Retrying backend - ${error.message}`);
      if (state.snapshot) {
        renderSnapshot(state.snapshot);
      }
      return;
    }
    setConnectionStatus("Demo mode", "demo");
    setModeStatus("Backend unavailable, using local demo");
    renderSnapshot(demoSnapshotAt(state.demoIndex));
  }
}

function restartTimer() {
  if (state.timer) {
    clearInterval(state.timer);
  }
  const speed = Number.parseFloat(els.speedSelect.value || String(CONFIG.defaultSpeed)) || 1;
  const interval = Math.max(250, Math.round(CONFIG.pollMs / Math.max(speed, 0.25)));
  state.timer = setInterval(async () => {
    if (state.source === "backend") {
      await refreshBackend();
      return;
    }
    if (state.autoplay) {
      stepDemo();
      return;
    }
    if (state.snapshot) {
      renderSnapshot(state.snapshot);
    }
  }, interval);
}

function setChartMode(mode) {
  state.chartMode = mode;
  updateModeButtons();
  if (state.snapshot) {
    renderChartPrice(state.snapshot);
    drawChart(state.snapshot);
  }
}

function bindEvents() {
  bindTradeContextTooltipEvents();
  els.chartModeCandles.addEventListener("click", () => setChartMode("candles"));
  els.chartModeLine.addEventListener("click", () => setChartMode("line"));
  els.priceChart.addEventListener("pointermove", (event) => {
    if (!state.snapshot) return;
    const rect = els.priceChart.getBoundingClientRect();
    state.chartHover = {
      x: event.clientX - rect.left,
    };
    renderChartPrice(state.snapshot);
    drawChart(state.snapshot);
  });
  els.priceChart.addEventListener("pointerleave", () => {
    if (!state.snapshot) return;
    state.chartHover = null;
    renderChartPrice(state.snapshot);
    drawChart(state.snapshot);
  });

  els.playButton.addEventListener("click", async () => {
    state.autoplay = true;
    setModeStatus("Autoplay on");
    if (backendAvailable()) {
      await sendControl("play");
      await refreshBackend();
      return;
    }
    restartTimer();
  });

  els.pauseButton.addEventListener("click", async () => {
    state.autoplay = false;
    setModeStatus("Paused");
    if (backendAvailable()) {
      await sendControl("pause");
      await refreshBackend();
      return;
    }
    restartTimer();
  });

  els.stepButton.addEventListener("click", async () => {
    if (backendAvailable()) {
      await sendControl("step", { steps: 1 });
      await refreshBackend();
      return;
    }
    stepDemo();
  });

  els.resetButton.addEventListener("click", async () => {
    state.autoplay = false;
    if (backendAvailable()) {
      await sendControl("reset");
      await refreshBackend();
      return;
    }
    resetDemo();
  });

  els.speedSelect.value = String(CONFIG.defaultSpeed || 1);
  els.speedSelect.addEventListener("change", async () => {
    if (backendAvailable()) {
      await sendControl("speed");
      await refreshBackend();
    }
    restartTimer();
    setModeStatus(`Speed ${els.speedSelect.value}x`);
  });

  window.addEventListener("resize", resizeCanvas);
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => resizeCanvas());
    observer.observe(els.priceChart.parentElement);
  }
}

async function init() {
  bindEvents();
  resizeCanvas();
  updateModeButtons();
  setConnectionStatus("Connecting", "pending");
  setModeStatus("Loading market view");

  if (CONFIG.mode === "demo") {
    state.source = "demo";
    setConnectionStatus("Demo mode", "demo");
    setModeStatus("Local demo timeline");
    renderSnapshot(demoSnapshotAt(state.demoIndex));
    restartTimer();
    return;
  }

  try {
    const snapshot = await loadBackendSnapshot();
    state.source = "backend";
    state.connected = true;
    setConnectionStatus("Connected", "live");
    setModeStatus("Backend live stream");
    renderSnapshot(snapshot);
  } catch (error) {
    state.source = "demo";
    state.connected = false;
    state.lastError = error;
    setConnectionStatus("Demo mode", "demo");
    setModeStatus("Backend unavailable, using local demo");
    renderSnapshot(demoSnapshotAt(state.demoIndex));
  }

  restartTimer();
}

init().catch((error) => {
  console.error(error);
  setConnectionStatus("Error", "error");
  setModeStatus("Initialization failed");
  els.newsFeed.textContent = `Failed to initialize market view: ${error.message}`;
});
