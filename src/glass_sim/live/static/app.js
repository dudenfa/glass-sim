"use strict";

// ============================================================== constants

const TYPE_COLORS = {
  market_maker: "#50d2c1",
  retail: "#8fa3b8",
  momentum: "#f59e0b",
  arbitrageur: "#3b82f6",
  informed: "#e879f9",
  whale: "#facc15",
  copy_trader: "#a3e635",
  flow_follower: "#fb7185",
  liquidation_hunter: "#f6465d",
  external: "#ffffff",
  rl_agent: "#ffffff",
};
const TYPE_LABELS = {
  market_maker: "Market maker",
  retail: "Retail",
  momentum: "Momentum",
  arbitrageur: "Basis arbitrageur",
  informed: "Informed",
  whale: "Whale",
  copy_trader: "Copy trader",
  flow_follower: "Flow follower",
  liquidation_hunter: "Liquidation hunter",
  external: "External / RL",
  rl_agent: "RL agent",
};
const TYPE_DESCRIPTIONS = {
  market_maker: "Posts multi-level buy and sell quotes around a fair price (book mid blended with the index). Widens with volatility and after news, and skews quotes to shed inventory. Earns the spread, loses to informed flow.",
  retail: "Opens random long/short positions with a mild bias toward the recent move, often at high leverage (3–40×). Closes on a random holding time or at a take-profit; with no stop-loss it can be liquidated.",
  momentum: "Trend follower: holds a position proportional to the gap between a fast and a slow moving average of the trade price.",
  arbitrageur: "Sells the perp when its bid is above the index and buys when its ask is below, with price-limited orders; closes when the gap closes. This keeps the perpetual anchored to the index.",
  informed: "Receives a noisy private signal about each news item 20–90 s before it becomes public and trades ahead of it. Its flow moves prices before the news.",
  whale: "A very large account running multi-hour accumulation and unwind campaigns, sliced into TWAP child orders. Highly visible when positions are public.",
  flow_follower: "Needs ATTRIBUTED or FULL transparency. Reads buyer and seller on the public trade tape, finds the account with the strongest one-directional flow over the last 10 minutes (e.g. a whale working an order) and trades in the same direction.",
  copy_trader: "Needs FULL transparency. Mirrors the direction of the most profitable visible account that is taking leveraged directional risk.",
  liquidation_hunter: "Needs FULL transparency. Scans public liquidation prices; when the positions that would be force-closed outweigh the cost of walking the book there, it pushes the price into the cluster and rests its exit where the forced orders arrive.",
  external: "Controlled from outside the simulator, e.g. by a reinforcement-learning policy.",
  rl_agent: "A trained reinforcement-learning policy (MaskablePPO). It sees a 32-number summary of the market and its own account, and picks one of 9 actions every ~5 s.",
};

const S = {
  bars: null,
  barCount: 0,
  sessionKey: null,
  state: null,
  tf: 15,
  visible: 140,
  offsetRight: 0,
  hover: null,
  show: { mark: true, index: true, liqs: true, news: true },
  sort: { key: "total_pnl", dir: -1 },
  filterType: "",
  filterOpen: false,
  tab: "positions",
  drawerId: null,
  drawerFetched: 0,
  controlsInit: false,
  finishedPrompted: null, // session key for which the form was already opened at the end
  aboutTransparency: null,
};

// ============================================================== helpers

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function num(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
}
function compact(v, d = 1) {
  if (v === null || v === undefined) return "—";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(d) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(d) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(d) + "k";
  return Number(v).toFixed(a < 10 && v % 1 ? 2 : 0);
}
function signed(v, d = 2) {
  if (v === null || v === undefined) return "—";
  return (v > 0 ? "+" : "") + num(v, d);
}
function cls(v) { return v > 0 ? "pos" : v < 0 ? "neg" : ""; }
function clock(sec) {
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return [h, m, s].map((x) => String(x).padStart(2, "0")).join(":");
}
function dot(type) { return `<span class="dot" style="background:${TYPE_COLORS[type] || "#888"}"></span>`; }
function typeChip(type) { return `<span class="type-chip">${dot(type)}${esc(TYPE_LABELS[type] || type)}</span>`; }
function bps(v, d = 1) { return v === null || v === undefined ? "—" : num(v, d) + " bps"; }

async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const res = await fetch(path, opts);
  return res.json();
}
async function control(action, extra = {}) {
  await api("/api/control", { action, ...extra });
  poll(true);
}

// ============================================================== polling

let polling = false;
let pollTimer = null;
async function poll(immediate) {
  if (polling) return;
  polling = true;
  clearTimeout(pollTimer);
  try {
    const st = await api(`/api/state?bars_since=${S.barCount}`);
    ingest(st);
    render();
  } catch (err) {
    console.error(err);
  } finally {
    polling = false;
    const playing = S.state && S.state.session.playing;
    pollTimer = setTimeout(() => poll(), playing ? 350 : 900);
  }
}

function resetBars() {
  S.bars = { time: [], open: [], high: [], low: [], close: [], volume: [], mark: [], index: [], open_interest: [], spread: [], predicted_funding: [] };
  S.barCount = 0;
}

function ingest(st) {
  const s = st.session;
  const key = `${s.scenario}|${s.seed}|${s.transparency}`;
  const lastTime = S.bars && S.bars.time.length ? S.bars.time[S.bars.time.length - 1] : -1;
  const restarted = key !== S.sessionKey || (st.bars.time.length && st.bars.time[0] < lastTime && st.bars_offset === 0) || s.time + 1e-9 < lastTime;
  if (!S.bars || restarted) {
    resetBars();
    S.sessionKey = key;
    S.offsetRight = 0;
    if (st.bars_offset !== 0) {
      S.state = st;
      setTimeout(() => poll(true), 0);
      return;
    }
  }
  if (st.bars_offset === S.barCount) {
    for (const k of Object.keys(S.bars)) S.bars[k].push(...(st.bars[k] || []));
    S.barCount = S.bars.time.length;
  } else if (st.bars_offset !== S.barCount) {
    resetBars();
    setTimeout(() => poll(true), 0);
  }
  S.state = st;
}

// ============================================================== rendering

function render() {
  const st = S.state;
  if (!st) return;
  renderHeader(st);
  renderControls(st);
  drawChart();
  renderBook(st);
  renderTrades(st);
  renderTab(st);
  if (S.drawerId && performance.now() - S.drawerFetched > 1500) openDrawer(S.drawerId, true);
}

function renderHeader(st) {
  const p = st.prices, x = st.stats, s = st.session;
  $("symbol").textContent = s.symbol;
  $("t-last").textContent = num(p.last, 2);
  const change = (p.last / p.open - 1) * 100;
  $("t-change").innerHTML = `<span class="${cls(change)}">${signed(change, 2)}% session</span>`;
  $("t-last").className = "value " + cls(change);
  $("t-mark").textContent = num(p.mark, 3);
  $("t-index").textContent = num(p.index, 3);
  const spreadBps = p.spread !== null && p.mid ? (p.spread / p.mid) * 1e4 : null;
  $("t-spread").textContent = p.spread === null ? "one-sided" : `${num(p.spread, 2)} (${num(spreadBps, 1)} bps)`;
  const countdown = Math.max(0, x.next_funding - s.time);
  $("t-funding").innerHTML = `<span class="${cls(-x.predicted_funding)}">${signed(x.predicted_funding * 100, 4)}%</span> · ${clock(countdown)}`;
  $("t-oi").textContent = `${compact(x.open_interest)} (${compact(x.open_interest * p.mark)} $)`;
  $("t-volume").textContent = `${compact(x.volume)} · ${compact(x.trades, 0)} trades`;
  $("t-liqs").textContent = x.liquidations;
  $("t-liqs").className = "value " + (x.liquidations ? "neg" : "");
  $("t-insurance").textContent = compact(x.insurance_fund, 2) + " $";
  $("t-active").textContent = `${x.active_agents} / ${x.agents}`;
  $("clock").textContent = s.clock;
  $("duration").textContent = "/ " + clock(s.duration);
  $("progress-bar").style.width = `${(100 * s.time) / s.duration}%`;
  document.title = `${num(p.last, 2)} · ${s.symbol} · ${s.clock}`;
}

function renderControls(st) {
  const s = st.session;
  const play = $("btn-play");
  play.textContent = s.finished ? "Finished" : s.playing ? "❚❚ Pause" : "▶ Play";
  play.classList.toggle("playing", s.playing);
  play.disabled = s.finished;
  if (!S.controlsInit) {
    const sel = $("scenario");
    sel.innerHTML = `<option value="" disabled selected>choose…</option>` + s.scenarios.map((n) => `<option value="${n}">${n}</option>`).join("");
    S.controlsInit = true;
  }
  const replay = s.mode === "replay";
  // The running session's settings are always shown, read-only. A new session is configured
  // in a form that starts empty: scenario, transparency and seed are chosen before it starts.
  $("session-active").textContent = `${s.scenario} · ${s.transparency} · seed ${s.seed}`;
  const key = `${s.scenario}|${s.transparency}|${s.seed}`;
  if (s.finished && !replay && S.finishedPrompted !== key) {
    S.finishedPrompted = key;
    openSessionForm();
  }
  $("btn-reset").classList.toggle("primary", s.finished);
  $("btn-reset").textContent = s.finished ? "▶ New session…" : "↺ New session…";
  $("session-dialog-title").textContent = s.finished ? "Session finished: start a new one" : "New session";
  $("session-dialog-note").textContent = s.finished
    ? "Choose the scenario, what the agents can see and the seed for the next session."
    : "Choose the scenario, what the agents can see and the seed. The current session is discarded when the new one starts.";
  $("btn-start").textContent = s.finished ? "▶ Start new session" : "▶ Start session";
  $("btn-start").disabled = !sessionFormValid();
  if (document.activeElement !== $("speed")) $("speed").value = String(s.speed);
  // Replay mode: the run is fixed, so the new-session controls give way to the replay's description and check.
  $("mode-badge").classList.toggle("hidden", !replay);
  $("btn-rewind").classList.toggle("hidden", replay); // replays have their own restart button
  $("session-group").classList.toggle("hidden", replay);
  $("replay-group").classList.toggle("hidden", !replay);
  if (replay) {
    $("replay-desc").textContent = s.replay.description;
    const v = s.replay.verification;
    if (!v) {
      $("replay-status").innerHTML = `<span class="muted">checked against the original at the end</span>`;
    } else if (v.identical) {
      $("replay-status").innerHTML = `<span class="pos">✓ identical to the original run</span>`;
    } else {
      const keys = Object.keys(v.differences).join(", ");
      $("replay-status").innerHTML = `<span class="neg" title="recorded with GLASS ${v.recorded_with}, replayed with ${v.replayed_with}">✗ differs from the original (${keys})</span>`;
    }
  }
  const note = {
    full: "agents see owners, positions and liquidation prices",
    attributed: "agents see order owners, not positions",
    anonymous: "agents see no owners (you do, as the researcher)",
  }[s.transparency];
  $("book-note").textContent = note;
}

// ------------------------------------------------------------ session form

function sessionFormValid() {
  const seed = $("seed").value.trim();
  return $("scenario").value !== "" && $("transparency").value !== "" && seed !== "" && Number.isInteger(Number(seed)) && Number(seed) >= 0;
}

function openSessionForm() {
  // Every field is cleared: the next session's settings are chosen explicitly, never inherited.
  $("scenario").value = "";
  $("transparency").value = "";
  $("seed").value = "";
  const dialog = $("session-dialog");
  if (!dialog.open) dialog.showModal();
  if (S.state) renderControls(S.state);
  $("scenario").focus();
}

function closeSessionForm() {
  const dialog = $("session-dialog");
  if (dialog.open) dialog.close();
}

function startSession() {
  if (!sessionFormValid()) return;
  const settings = { scenario: $("scenario").value, transparency: $("transparency").value, seed: Number($("seed").value) };
  closeSessionForm();
  S.finishedPrompted = null;
  control("reset", settings);
}

// -------------------------------------------------------------- chart

function buildCandles(tf) {
  const b = S.bars;
  const out = [];
  let cur = null;
  for (let i = 0; i < b.time.length; i++) {
    const bucket = Math.floor(b.time[i] / tf);
    if (!cur || cur.bucket !== bucket) {
      cur = { bucket, t: bucket * tf, o: b.open[i], h: b.high[i], l: b.low[i], c: b.close[i], v: b.volume[i], mark: b.mark[i], index: b.index[i], oi: b.open_interest[i] };
      out.push(cur);
    } else {
      cur.h = Math.max(cur.h, b.high[i]);
      cur.l = Math.min(cur.l, b.low[i]);
      cur.c = b.close[i];
      cur.v += b.volume[i];
      cur.mark = b.mark[i];
      cur.index = b.index[i];
      cur.oi = b.open_interest[i];
    }
  }
  return out;
}

function niceTicks(lo, hi, count) {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  return ticks;
}

const chart = { candles: [], start: 0, end: 0, cw: 1, geom: null };

function drawChart() {
  const canvas = $("chart");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  if (!S.bars || !S.bars.time.length) return;

  const candles = buildCandles(S.tf);
  const n = candles.length;
  S.offsetRight = Math.max(0, Math.min(S.offsetRight, Math.max(0, n - 10)));
  const end = n - Math.round(S.offsetRight);
  const start = Math.max(0, end - S.visible);
  const view = candles.slice(start, end);
  const axisW = 70, timeH = 18, left = 8;
  const plotW = W - axisW - left;
  const priceH = (H - timeH) * 0.68, volTop = priceH + 8, volH = (H - timeH) * 0.14, oiTop = volTop + volH + 8, oiH = H - timeH - oiTop - 4;
  const cw = plotW / S.visible;
  Object.assign(chart, { candles, start, end, cw, geom: { left, plotW, priceH, volTop, volH, oiTop, oiH, W, H, axisW, timeH } });

  let lo = Infinity, hi = -Infinity;
  for (const c of view) {
    lo = Math.min(lo, c.l); hi = Math.max(hi, c.h);
    if (S.show.mark) { lo = Math.min(lo, c.mark); hi = Math.max(hi, c.mark); }
    if (S.show.index) { lo = Math.min(lo, c.index); hi = Math.max(hi, c.index); }
  }
  if (!isFinite(lo)) return;
  const pad = Math.max((hi - lo) * 0.08, 0.02);
  lo -= pad; hi += pad;
  const y = (v) => 6 + (priceH - 12) * (1 - (v - lo) / (hi - lo));
  const xOf = (i) => left + (i - start + (S.visible - (end - start))) * cw + cw / 2;

  // grid + price axis
  ctx.font = "11px SF Mono, Menlo, monospace";
  ctx.textBaseline = "middle";
  for (const t of niceTicks(lo, hi, 6)) {
    const yy = Math.round(y(t)) + 0.5;
    ctx.strokeStyle = "#18202a"; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(left + plotW, yy); ctx.stroke();
    ctx.fillStyle = "#6f7d8c"; ctx.fillText(t.toFixed(2), left + plotW + 8, yy);
  }

  // news markers
  const liqs = (S.state && S.state.liquidations) || [];
  const news = (S.state && S.state.news) || [];
  if (S.show.news) {
    for (const item of news) {
      const idx = Math.floor(item.time / S.tf) - candles[0].bucket;
      if (idx < start || idx >= end) continue;
      const xx = Math.round(xOf(idx)) + 0.5;
      ctx.strokeStyle = "rgba(96,165,250,0.45)"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(xx, 4); ctx.lineTo(xx, priceH); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#60a5fa"; ctx.fillText(item.log_jump >= 0 ? "N▲" : "N▼", xx + 3, 12);
    }
  }

  // candles
  const bodyW = Math.max(1, Math.min(14, cw * 0.7));
  view.forEach((c, k) => {
    const i = start + k;
    const xx = xOf(i);
    const up = c.c >= c.o;
    ctx.strokeStyle = ctx.fillStyle = up ? "#2ebd85" : "#f6465d";
    ctx.beginPath(); ctx.moveTo(Math.round(xx) + 0.5, y(c.h)); ctx.lineTo(Math.round(xx) + 0.5, y(c.l)); ctx.stroke();
    const top = y(Math.max(c.o, c.c)), bot = y(Math.min(c.o, c.c));
    ctx.fillRect(xx - bodyW / 2, top, bodyW, Math.max(1, bot - top));
  });

  // mark / index lines
  const line = (key, color, dash) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.setLineDash(dash || []);
    ctx.beginPath();
    view.forEach((c, k) => { const xx = xOf(start + k), yy = y(c[key]); k ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); });
    ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  };
  if (S.show.index) line("index", "#a78bfa", [4, 3]);
  if (S.show.mark) line("mark", "rgba(240,185,11,0.9)");

  // liquidation markers
  if (S.show.liqs) {
    for (const l of liqs) {
      const idx = Math.floor(l.time / S.tf) - candles[0].bucket;
      if (idx < start || idx >= end) continue;
      const c = candles[idx];
      const xx = xOf(idx);
      const r = Math.min(9, 3 + Math.log10(1 + l.quantity));
      ctx.fillStyle = l.side === "long" ? "#f6465d" : "#2ebd85";
      ctx.beginPath();
      if (l.side === "long") { const yy = y(c.l) + 6; ctx.moveTo(xx, yy); ctx.lineTo(xx - r, yy + r * 1.6); ctx.lineTo(xx + r, yy + r * 1.6); }
      else { const yy = y(c.h) - 6; ctx.moveTo(xx, yy); ctx.lineTo(xx - r, yy - r * 1.6); ctx.lineTo(xx + r, yy - r * 1.6); }
      ctx.fill();
    }
  }

  // last / mark price tags
  const lastC = candles[n - 1];
  const tag = (value, color, textColor) => {
    const yy = y(value);
    if (yy < 0 || yy > priceH) return;
    ctx.strokeStyle = color; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(left + plotW, yy); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.fillRect(left + plotW + 2, yy - 8, axisW - 4, 16);
    ctx.fillStyle = textColor; ctx.fillText(value.toFixed(2), left + plotW + 8, yy);
  };
  if (S.show.mark) tag(lastC.mark, "#f0b90b", "#1a1400");
  tag(lastC.c, lastC.c >= lastC.o ? "#2ebd85" : "#f6465d", "#04120c");

  // volume pane
  let vmax = 1;
  for (const c of view) vmax = Math.max(vmax, c.v);
  view.forEach((c, k) => {
    const xx = xOf(start + k);
    const h = (c.v / vmax) * volH;
    ctx.fillStyle = c.c >= c.o ? "rgba(46,189,133,0.45)" : "rgba(246,70,93,0.45)";
    ctx.fillRect(xx - bodyW / 2, volTop + volH - h, bodyW, h);
  });
  ctx.fillStyle = "#6f7d8c"; ctx.fillText("Volume " + compact(vmax), left + plotW + 8, volTop + 6);

  // open interest pane
  let omin = Infinity, omax = -Infinity;
  for (const c of view) { omin = Math.min(omin, c.oi); omax = Math.max(omax, c.oi); }
  if (omax === omin) { omax += 1; omin -= 1; }
  ctx.strokeStyle = "#50d2c1"; ctx.beginPath();
  view.forEach((c, k) => { const xx = xOf(start + k), yy = oiTop + oiH * (1 - (c.oi - omin) / (omax - omin)); k ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); });
  ctx.stroke();
  ctx.fillStyle = "#6f7d8c"; ctx.fillText("OI " + compact(view.length ? view[view.length - 1].oi : 0), left + plotW + 8, oiTop + 6);
  ctx.strokeStyle = "#18202a";
  for (const yy of [volTop - 4, oiTop - 4]) { ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(left + plotW, yy); ctx.stroke(); }

  // time axis
  ctx.fillStyle = "#6f7d8c"; ctx.textBaseline = "alphabetic";
  const every = Math.max(1, Math.round(90 / cw));
  for (let i = start; i < end; i++) {
    if (candles[i].bucket % every !== 0) continue;
    ctx.fillText(clock(candles[i].t), xOf(i) - 26, H - 5);
  }

  // crosshair + tooltip
  const tip = $("chart-tip");
  if (S.hover && S.hover.x < left + plotW) {
    const idx = Math.floor((S.hover.x - left) / cw) + start - (S.visible - (end - start));
    ctx.strokeStyle = "rgba(200,210,220,0.25)"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(S.hover.x, 0); ctx.lineTo(S.hover.x, H - timeH); ctx.stroke();
    if (S.hover.y < priceH) {
      ctx.beginPath(); ctx.moveTo(left, S.hover.y); ctx.lineTo(left + plotW, S.hover.y); ctx.stroke();
      const pv = lo + (1 - (S.hover.y - 6) / (priceH - 12)) * (hi - lo);
      ctx.setLineDash([]); ctx.fillStyle = "#2a3644"; ctx.fillRect(left + plotW + 2, S.hover.y - 8, axisW - 4, 16);
      ctx.fillStyle = "#d5dde7"; ctx.textBaseline = "middle"; ctx.fillText(pv.toFixed(2), left + plotW + 8, S.hover.y);
    }
    ctx.setLineDash([]);
    const c = candles[idx];
    if (c) {
      tip.textContent = `${clock(c.t)}  O ${c.o.toFixed(2)}  H ${c.h.toFixed(2)}  L ${c.l.toFixed(2)}  C ${c.c.toFixed(2)}  Vol ${compact(c.v)}\nMark ${c.mark.toFixed(3)}  Index ${c.index.toFixed(3)}  Basis ${(((c.mark - c.index) / c.index) * 1e4).toFixed(1)} bps  OI ${compact(c.oi)}`;
    }
  } else {
    const c = lastC;
    tip.textContent = `${S.tf}s candles · ${clock(c.t)}  C ${c.c.toFixed(2)}  Mark ${c.mark.toFixed(3)}  Index ${c.index.toFixed(3)}\nscroll to zoom · drag to pan · double-click for live`;
  }
}

function setupChartInteraction() {
  const canvas = $("chart");
  let drag = null;
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    S.visible = Math.max(20, Math.min(1500, Math.round(S.visible * (e.deltaY > 0 ? 1.12 : 0.89))));
    drawChart();
  }, { passive: false });
  canvas.addEventListener("mousedown", (e) => { drag = { x: e.clientX, off: S.offsetRight }; });
  window.addEventListener("mouseup", () => { drag = null; });
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    S.hover = { x: e.clientX - r.left, y: e.clientY - r.top };
    if (drag) S.offsetRight = Math.max(0, drag.off + (e.clientX - drag.x) / chart.cw);
    drawChart();
  });
  canvas.addEventListener("mouseleave", () => { S.hover = null; drawChart(); });
  canvas.addEventListener("dblclick", () => { S.offsetRight = 0; drawChart(); });
  $("btn-live").addEventListener("click", () => { S.offsetRight = 0; drawChart(); });
  document.querySelectorAll("#timeframes button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#timeframes button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    S.tf = Number(b.dataset.tf);
    S.offsetRight = 0;
    drawChart();
  }));
  for (const k of ["mark", "index", "liqs", "news"]) {
    $("show-" + k).addEventListener("change", (e) => { S.show[k] = e.target.checked; drawChart(); });
  }
  window.addEventListener("resize", () => drawChart());
}

// -------------------------------------------------------------- book + trades

function renderBook(st) {
  const depth = 16;
  const asks = st.book.asks.slice(0, depth);
  const bids = st.book.bids.slice(0, depth);
  let cum = 0;
  const askRows = asks.map((l) => ({ ...l, cum: (cum += l.quantity) }));
  cum = 0;
  const bidRows = bids.map((l) => ({ ...l, cum: (cum += l.quantity) }));
  const maxCum = Math.max(1, askRows.length ? askRows[askRows.length - 1].cum : 0, bidRows.length ? bidRows[bidRows.length - 1].cum : 0);
  const row = (l, side) => {
    const owners = l.owners.map((o) => dot(o.type)).join("");
    const more = l.orders > l.owners.length ? `<span class="muted small">+${l.orders - l.owners.length}</span>` : "";
    return `<div class="book-row ${side}" data-owners='${esc(JSON.stringify(l.owners))}' data-price="${l.price}">
      <div class="bar" style="width:${(100 * l.cum) / maxCum}%"></div>
      <span class="price">${num(l.price, 2)}</span><span>${compact(l.quantity)}</span><span class="muted">${compact(l.cum)}</span><span class="owners">${owners}${more}</span></div>`;
  };
  $("book-asks").innerHTML = askRows.slice().reverse().map((l) => row(l, "ask")).join("");
  $("book-bids").innerHTML = bidRows.map((l) => row(l, "bid")).join("");
  const p = st.prices;
  const spreadBps = p.spread !== null && p.mid ? (p.spread / p.mid) * 1e4 : null;
  $("book-mid").innerHTML = `<span>Spread ${p.spread === null ? "—" : num(p.spread, 2)} <span class="muted">${spreadBps === null ? "" : num(spreadBps, 1) + " bps"}</span></span><span class="muted">Mark <span style="color:var(--mark)">${num(p.mark, 3)}</span></span>`;
}

function shortId(id) { return id.length > 12 ? id.slice(0, 11) + "…" : id; }

function renderTrades(st) {
  $("trades").innerHTML = st.trades.slice().reverse().map((t) => {
    const color = t.aggressor === "buy" ? "pos" : "neg";
    const tag = t.liquidation ? `<span class="liq-tag">${t.adl ? "ADL" : "LIQ"}</span>` : "";
    return `<div class="trade-row ${t.liquidation ? "liq" : ""}"><span class="muted">${clock(t.time)}</span><span class="${color}">${num(t.price, 2)}</span><span>${compact(t.quantity)}</span>
      <span class="who">${tag}${dot(t.buyer_type)}${esc(shortId(t.buyer))} ← ${dot(t.seller_type)}${esc(shortId(t.seller))}</span></div>`;
  }).join("");
}

function setupBookTooltip() {
  const tip = $("tooltip");
  for (const id of ["book-asks", "book-bids"]) {
    $(id).addEventListener("mousemove", (e) => {
      const rowEl = e.target.closest(".book-row");
      if (!rowEl) { tip.classList.add("hidden"); return; }
      const owners = JSON.parse(rowEl.dataset.owners || "[]");
      tip.innerHTML = `<div class="muted" style="margin-bottom:4px">Resting at ${num(Number(rowEl.dataset.price), 2)}</div>` +
        owners.map((o) => `<div class="row"><span>${dot(o.type)} ${esc(o.id)}</span><span>${compact(o.quantity)}</span></div>`).join("");
      tip.style.left = e.clientX + 14 + "px";
      tip.style.top = e.clientY + 10 + "px";
      tip.classList.remove("hidden");
    });
    $(id).addEventListener("mouseleave", () => tip.classList.add("hidden"));
  }
}

// -------------------------------------------------------------- bottom tabs

function renderTab(st) {
  $("badge-liqs").textContent = st.stats.liquidations;
  const fns = { positions: renderPositions, groups: renderGroups, liquidations: renderLiquidations, funding: renderFunding, news: renderNews, orders: renderOrders, quality: renderQuality, about: renderAbout };
  fns[S.tab](st);
}

function table(el, columns, rows, opts = {}) {
  const head = "<tr>" + columns.map((c) => `<th class="${c.left ? "left" : ""}" data-key="${c.key || ""}">${c.label}${c.key && opts.sortKey === c.key ? (opts.sortDir > 0 ? " ▲" : " ▼") : ""}</th>`).join("") + "</tr>";
  const body = rows.map((r) => `<tr class="${opts.rowClass ? opts.rowClass(r) : ""}" ${opts.rowAttr ? opts.rowAttr(r) : ""}>` + columns.map((c) => `<td class="${c.left ? "left" : ""}">${c.render(r)}</td>`).join("") + "</tr>").join("");
  el.innerHTML = head + (body || `<tr><td class="left muted" colspan="${columns.length}">nothing yet</td></tr>`);
}

function renderPositions(st) {
  const typeSel = $("filter-type");
  const types = [...new Set(st.accounts.map((a) => a.agent_type))];
  if (typeSel.options.length - 1 !== types.length) {
    typeSel.innerHTML = `<option value="">all</option>` + types.map((t) => `<option value="${t}">${TYPE_LABELS[t] || t}</option>`).join("");
    typeSel.value = S.filterType;
  }
  $("positions-visibility").textContent = {
    full: "Every agent in this market can see this table (FULL transparency).",
    attributed: "Agents cannot see this table; they only see who owns each order and trade.",
    anonymous: "Agents cannot see this table: the market is anonymous. You see it as the researcher.",
  }[st.session.transparency];
  let rows = st.accounts.filter((a) => (!S.filterType || a.agent_type === S.filterType) && (!S.filterOpen || a.position !== 0));
  const k = S.sort.key;
  rows.sort((a, b) => {
    const va = a[k], vb = b[k];
    if (typeof va === "string") return S.sort.dir * va.localeCompare(vb);
    return S.sort.dir * ((va ?? -Infinity) - (vb ?? -Infinity));
  });
  const cols = [
    { key: "agent_id", label: "Account", left: true, render: (a) => `${dot(a.agent_type)} ${esc(a.agent_id)}${a.public_id !== a.agent_id ? ` <span class="muted">${esc(a.public_id)}</span>` : ""}` },
    { key: "agent_type", label: "Type", left: true, render: (a) => esc(TYPE_LABELS[a.agent_type] || a.agent_type) },
    { key: "position", label: "Position", render: (a) => a.position === 0 ? `<span class="muted">flat</span>` : `<span class="${cls(a.position)}">${a.position > 0 ? "LONG" : "SHORT"} ${compact(Math.abs(a.position))}</span>` },
    { key: "notional", label: "Notional", render: (a) => compact(a.notional) },
    { key: "entry_price", label: "Entry", render: (a) => a.position ? num(a.entry_price, 2) : "—" },
    { key: "liquidation_price", label: "Liq. price", render: (a) => a.liquidation_price ? num(a.liquidation_price, 2) : "—" },
    { key: "leverage", label: "Leverage", render: (a) => a.position ? num(a.leverage, 1) + "×" : "—" },
    { key: "unrealized_pnl", label: "uPnL", render: (a) => `<span class="${cls(a.unrealized_pnl)}">${signed(a.unrealized_pnl, 0)}</span>` },
    { key: "equity", label: "Equity", render: (a) => compact(a.equity, 2) },
    { key: "total_pnl", label: "Total PnL", render: (a) => `<span class="${cls(a.total_pnl)}">${signed(a.total_pnl, 0)}</span>` },
    { key: "return_pct", label: "Return", render: (a) => `<span class="${cls(a.return_pct)}">${signed(a.return_pct, 2)}%</span>` },
    { key: "max_drawdown_pct", label: "Max DD", render: (a) => num(a.max_drawdown_pct, 1) + "%" },
    { key: "trade_count", label: "Fills", render: (a) => compact(a.trade_count, 0) },
    { key: "liquidations", label: "Liqs", render: (a) => a.liquidations ? `<span class="neg">${a.liquidations}</span>` : "0" },
    { key: "active", label: "Status", render: (a) => a.active ? `<span class="pos">active</span>` : `<span class="neg">out</span>` },
  ];
  table($("positions-table"), cols, rows, {
    sortKey: S.sort.key,
    sortDir: S.sort.dir,
    rowClass: (a) => "clickable" + (a.active ? "" : " inactive"),
    rowAttr: (a) => `data-agent="${esc(a.agent_id)}"`,
  });
}

function renderGroups(st) {
  const maxAbs = Math.max(0.01, ...st.groups.map((g) => Math.abs(g.return_pct)));
  const cols = [
    { label: "Participant type", left: true, render: (g) => typeChip(g.agent_type) },
    { label: "Agents", render: (g) => `${g.active} / ${g.agents}` },
    { label: "Long", render: (g) => compact(g.long) },
    { label: "Short", render: (g) => compact(g.short) },
    { label: "Net", render: (g) => `<span class="${cls(g.long - g.short)}">${signed(g.long - g.short, 0)}</span>` },
    { label: "Volume ($)", render: (g) => compact(g.volume) },
    { label: "Total PnL", render: (g) => `<span class="${cls(g.pnl)}">${signed(g.pnl, 0)}</span>` },
    {
      label: "Return on capital",
      render: (g) => {
        const w = (50 * Math.abs(g.return_pct)) / maxAbs;
        const style = g.return_pct >= 0 ? `left:50%;width:${w}%;background:var(--green)` : `right:50%;width:${w}%;background:var(--red)`;
        return `<div class="bar-cell"><div class="fill" style="${style}"></div><span class="${cls(g.return_pct)}">${signed(g.return_pct, 2)}%</span></div>`;
      },
    },
    { label: "Liquidations", render: (g) => g.liquidations || "0" },
  ];
  table($("groups-table"), cols, st.groups);
}

function renderLiquidations(st) {
  const cols = [
    { label: "Time", left: true, render: (l) => clock(l.time) },
    { label: "Account", left: true, render: (l) => `${dot(l.agent_type)} ${esc(l.agent_id)}` },
    { label: "Position", render: (l) => `<span class="${l.side === "long" ? "pos" : "neg"}">${l.side.toUpperCase()} ${compact(l.quantity)}</span>` },
    { label: "Mark", render: (l) => num(l.mark_price, 3) },
    { label: "Avg fill", render: (l) => num(l.avg_fill_price, 3) },
    { label: "ADL", render: (l) => l.adl_quantity || "0" },
    { label: "Penalty → fund", render: (l) => num(l.penalty, 2) },
    { label: "Bad debt", render: (l) => l.bad_debt ? `<span class="neg">${num(l.bad_debt, 2)}</span>` : "0" },
    { label: "Outcome", render: (l) => l.deactivated ? `<span class="neg">wiped out</span>` : "continues" },
  ];
  table($("liquidations-table"), cols, st.liquidations.slice().reverse(), { rowClass: () => "clickable", rowAttr: (l) => `data-agent="${esc(l.agent_id)}"` });
}

function renderFunding(st) {
  const x = st.stats, s = st.session;
  $("funding-now").innerHTML = `<div><span class="muted">Predicted rate</span><br><span class="${cls(-x.predicted_funding)}">${signed(x.predicted_funding * 100, 5)}%</span></div>
    <div><span class="muted">Next settlement</span><br>${clock(x.next_funding)} (in ${clock(x.next_funding - s.time)})</div>
    <div><span class="muted">Rule</span><br>rate = clip(mean premium + interest); longs pay shorts when positive</div>`;
  const cols = [
    { label: "Time", left: true, render: (f) => clock(f.time) },
    { label: "Rate", render: (f) => `<span class="${cls(-f.rate)}">${signed(f.rate * 100, 5)}%</span>` },
    { label: "Mean premium", render: (f) => bps(f.premium * 1e4, 2) },
    { label: "Mark", render: (f) => num(f.mark, 3) },
    { label: "Index", render: (f) => num(f.index, 3) },
    { label: "Paid by longs ($)", render: (f) => signed(f.paid_by_longs, 2) },
  ];
  table($("funding-table"), cols, st.funding.slice().reverse());
}

function renderNews(st) {
  const cols = [
    { label: "Time", left: true, render: (n) => clock(n.time) },
    { label: "Headline", left: true, render: (n) => esc(n.headline) },
    { label: "Index jump", render: (n) => `<span class="${cls(n.log_jump)}">${signed(n.log_jump * 1e4, 1)} bps</span>` },
  ];
  table($("news-table"), cols, st.news.slice().reverse());
}

function renderOrders(st) {
  const cols = [
    { label: "Time", left: true, render: (o) => clock(o.time) },
    { label: "Event", left: true, render: (o) => `<span class="${o.kind === "rejected" ? "neg" : o.kind === "accepted" ? "pos" : "muted"}">${o.kind}</span>` },
    { label: "Account", left: true, render: (o) => `${dot(o.agent_type)} ${esc(o.agent_id)}` },
    { label: "Side", render: (o) => `<span class="${o.side === "buy" ? "pos" : "neg"}">${o.side}</span>` },
    { label: "Type", render: (o) => o.type },
    { label: "Qty", render: (o) => compact(o.quantity) },
    { label: "Price", render: (o) => (o.price === null ? "market" : num(o.price, 2)) },
    { label: "Reason", left: true, render: (o) => `<span class="muted">${esc(o.reason || "")}</span>` },
  ];
  table($("orders-table"), cols, st.orders.slice().reverse(), { rowClass: () => "clickable", rowAttr: (o) => `data-agent="${esc(o.agent_id)}"` });
}

function kv(pairs) { return `<div class="kv">${pairs.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("")}</div>`; }

function renderQuality(st) {
  const m = st.summary.market || {};
  const facts = st.summary.stylized_facts || {};
  const ifacts = st.summary.index_stylized_facts || {};
  const horizons = Object.keys(facts);
  const factRows = horizons.map((h) => {
    const f = facts[h], g = ifacts[h] || {};
    // Statistics from fewer than 60 returns are too noisy to read.
    const enough = (x) => x && x.n_returns >= 60;
    const val = (x, key) => (enough(x) && x[key] !== null && x[key] !== undefined ? num(x[key], 3) : "—");
    const cell = (key) => `${val(f, key)} <span class="muted">/ ${val(g, key)}</span>`;
    return `<tr><td class="left">${h} <span class="muted small">(${f.n_returns} returns)</span></td><td>${cell("excess_kurtosis")}</td><td>${cell("acf_return_lag1")}</td><td>${cell("acf_abs_return_lag1")}</td><td>${cell("acf_abs_return_lag5")}</td><td>${num(f.std_bps, 1)}</td></tr>`;
  }).join("");
  $("quality").innerHTML = `
    <div class="card"><h3>Liquidity</h3>${kv([
      ["Two-sided book (fraction of time)", num(m.two_sided_book_fraction, 3)],
      ["Median spread", bps(m.median_spread_bps, 2)],
      ["Mean bid depth (≤50 ticks)", compact(m.mean_bid_depth)],
      ["Mean ask depth (≤50 ticks)", compact(m.mean_ask_depth)],
    ])}</div>
    <div class="card"><h3>Price discovery</h3>${kv([
      ["|mark − index| mean", bps(m.mark_index_mean_abs_bps, 2)],
      ["|last − index| mean", bps(m.last_index_mean_abs_bps, 2)],
      ["|last − index| max", bps(m.last_index_max_abs_bps, 1)],
      ["1-minute return volatility", bps(m.return_vol_1m_bps, 1)],
    ])}</div>
    <div class="card"><h3>Activity</h3>${kv([
      ["Trades per minute", num(m.trades_per_minute, 1)],
      ["Volume (contracts)", compact(m.volume_contracts)],
      ["Mean open interest", compact(m.mean_open_interest)],
      ["Max open interest", compact(m.max_open_interest)],
    ])}</div>
    <div class="card" style="grid-column: 1 / -1"><h3>Stylized facts: perp trade price <span class="muted">/ index</span></h3>
      ${horizons.length ? `<table><tr><th class="left">Horizon</th><th>Excess kurtosis</th><th>ACF return lag 1</th><th>ACF |return| lag 1</th><th>ACF |return| lag 5</th><th>Std (bps)</th></tr>${factRows}</table>` : `<p class="muted">Needs a few minutes of simulated data.</p>`}
      <p class="muted small">Real markets show heavy tails (kurtosis &gt; 0, shrinking at longer horizons), almost no autocorrelation of returns, and positive, slowly decaying autocorrelation of absolute returns (volatility clustering). The index column shows what comes from the exogenous reference price; the perp column adds what the agents' trading creates.</p>
    </div>`;
}

function renderAbout(st) {
  const t = st.session.transparency;
  if (S.aboutTransparency === t && $("about").innerHTML) return;
  S.aboutTransparency = t;
  const matrix = [
    ["Aggregated order book, trade prices and sizes", 1, 1, 1],
    ["Mark, index, funding rate, open interest", 1, 1, 1],
    ["Owner of every resting order", 0, 1, 1],
    ["Buyer and seller of every trade", 0, 1, 1],
    ["Every account's position, entry, leverage, PnL", 0, 0, 1],
    ["Every account's liquidation price", 0, 0, 1],
  ];
  const yn = (v) => (v ? `<span class="yes">✓</span>` : `<span class="no">✗</span>`);
  const colIdx = { anonymous: 1, attributed: 2, full: 3 }[t];
  const participants = Object.keys(TYPE_LABELS).filter((k) => k !== "external")
    .map((k) => `<tr><td>${typeChip(k)}</td><td style="white-space:normal;text-align:left">${TYPE_DESCRIPTIONS[k]}</td></tr>`).join("");
  $("about").innerHTML = `
    <div class="card"><h3>What you are looking at</h3>
      <p>A simulated <b>perpetual futures</b> exchange. Every participant is a program trading through a single public limit order book. Prices, liquidity and liquidations are not replayed from data: they emerge from what the agents do.</p>
      <p>Agents can go <b>long or short with leverage</b> (up to ${st.session.max_leverage}×). Each account posts collateral; positions must stay above a maintenance margin or they are <b>liquidated</b>: orders canceled, position closed at market, penalty paid to the insurance fund. If the book cannot absorb the position it is <b>auto-deleveraged</b> against the most profitable opposite positions.</p>
      <p>The <b>index</b> (purple) is the external spot reference, moved by a random process and by news. The <b>mark</b> (yellow) is a robust median used for margin. Every hour, <b>funding</b> is exchanged between longs and shorts to pull the perp toward the index.</p>
      <p>Fees: maker ${(st.session.maker_fee * 1e4).toFixed(1)} bps, taker ${(st.session.taker_fee * 1e4).toFixed(1)} bps. Orders reach the exchange after a small random latency.</p>
    </div>
    <div class="card"><h3>What agents can see</h3>
      <p>Transparency is an experimental setting. You, the researcher, always see everything; the table shows what the <b>agents</b> see. Current setting highlighted.</p>
      <table><tr><th class="left">Information</th><th>Anonymous</th><th>Attributed</th><th>Full</th></tr>
      ${matrix.map((r) => `<tr><td class="left" style="white-space:normal">${r[0]}</td>${[1, 2, 3].map((i) => `<td style="text-align:center;${i === colIdx ? "background:#102b28" : ""}">${yn(r[i])}</td>`).join("")}</tr>`).join("")}
      </table>
      <p class="muted small">Full transparency is how on-chain perpetual exchanges such as Hyperliquid work: every wallet's positions and liquidation levels are public. Flow followers need at least ATTRIBUTED; copy traders and liquidation hunters need FULL.</p>
    </div>
    <div class="card" style="grid-column: 1 / -1"><h3>Participants</h3><table>${participants}</table></div>`;
}

// -------------------------------------------------------------- drawer

async function openDrawer(agentId, refresh) {
  S.drawerId = agentId;
  S.drawerFetched = performance.now();
  let d;
  try { d = await api(`/api/account?id=${encodeURIComponent(agentId)}`); } catch (e) { return; }
  if (d.error || S.drawerId !== agentId) return;
  const a = d.account;
  $("drawer").classList.remove("hidden");
  $("drawer-title").innerHTML = `${dot(a.agent_type)} ${esc(a.agent_id)}`;
  $("drawer-sub").textContent = `${TYPE_LABELS[a.agent_type] || a.agent_type} · public id ${a.public_id}${d.private_signals ? " · receives private signals" : ""}${a.active ? "" : " · OUT OF THE MARKET"}`;
  const stat = (label, value, c = "") => `<div class="stat"><div class="label">${label}</div><div class="value ${c}">${value}</div></div>`;
  $("drawer-stats").innerHTML = [
    stat("Equity", compact(a.equity, 2)),
    stat("Total PnL", signed(a.total_pnl, 0), cls(a.total_pnl)),
    stat("Return", signed(a.return_pct, 2) + "%", cls(a.return_pct)),
    stat("Position", a.position ? `${a.position > 0 ? "LONG" : "SHORT"} ${compact(Math.abs(a.position))}` : "flat", cls(a.position)),
    stat("Entry", a.position ? num(a.entry_price, 2) : "—"),
    stat("Liq. price", a.liquidation_price ? num(a.liquidation_price, 2) : "—", a.liquidation_price ? "neg" : ""),
    stat("Leverage", a.position ? num(a.leverage, 1) + "×" : "—"),
    stat("Realized PnL", signed(a.realized_pnl, 0), cls(a.realized_pnl)),
    stat("Unrealized PnL", signed(a.unrealized_pnl, 0), cls(a.unrealized_pnl)),
    stat("Fees paid", num(a.fees_paid, 0)),
    stat("Funding paid", signed(a.funding_paid, 0)),
    stat("Fills / volume", `${compact(a.trade_count, 0)} / ${compact(a.volume)}`),
    stat("Max drawdown", `${num(a.max_drawdown_pct, 1)}% (${compact(a.max_drawdown, 1)})`, a.max_drawdown_pct > 20 ? "neg" : ""),
    stat("Peak / worst equity", `${compact(a.peak_equity, 1)} / ${compact(a.min_equity, 1)}`),
    stat("Max position / leverage", `${compact(a.max_abs_position)} / ${num(a.max_leverage, 1)}×`),
    stat("Maker share", `${num(100 * a.maker_share, 0)}%`),
    stat("Orders acc / rej / canc", `${compact(a.orders_accepted, 0)} / ${compact(a.orders_rejected, 0)} / ${compact(a.orders_canceled, 0)}`),
    stat("First liquidation", a.first_liquidation_time === null ? "never" : clock(a.first_liquidation_time), a.first_liquidation_time === null ? "" : "neg"),
  ].join("");
  sparkline($("spark-equity"), d.equity_series.equity, "#50d2c1", a.initial_collateral);
  sparkline($("spark-position"), d.equity_series.position, "#f0b90b", 0);
  $("drawer-params").innerHTML = Object.entries(d.params).filter(([k]) => !["agent_id", "public_id", "agent_type", "default_prefix"].includes(k))
    .map(([k, v]) => `<span>${esc(k)}</span><span>${esc(typeof v === "number" ? +v.toFixed(4) : v)}</span>`).join("");
  table($("drawer-orders"), [
    { label: "Side", left: true, render: (o) => `<span class="${o.side === "buy" ? "pos" : "neg"}">${o.side}</span>` },
    { label: "Price", render: (o) => num(o.price, 2) },
    { label: "Remaining", render: (o) => `${o.remaining} / ${o.quantity}` },
    { label: "Placed", render: (o) => clock(o.time) },
    { label: "Tag", render: (o) => `<span class="muted">${esc(o.tag)}</span>` },
  ], d.orders);
  table($("drawer-fills"), [
    { label: "Time", left: true, render: (f) => clock(f.time) },
    { label: "Side", render: (f) => `<span class="${f.side === "buy" ? "pos" : "neg"}">${f.side}</span>` },
    { label: "Price", render: (f) => num(f.price, 2) },
    { label: "Qty", render: (f) => compact(f.quantity) },
    { label: "Role", render: (f) => (f.liquidation ? `<span class="liq-tag">LIQ</span>` : f.maker ? "maker" : "taker") },
    { label: "Counterparty", render: (f) => esc(shortId(f.counterparty)) },
  ], d.trades);
}

function sparkline(canvas, values, color, baseline) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!values || values.length < 2) return;
  let lo = Math.min(...values, baseline), hi = Math.max(...values, baseline);
  if (hi === lo) { hi += 1; lo -= 1; }
  const y = (v) => 6 + (h - 12) * (1 - (v - lo) / (hi - lo));
  ctx.strokeStyle = "#2a3644"; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, y(baseline)); ctx.lineTo(w, y(baseline)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.beginPath();
  values.forEach((v, i) => { const xx = (i / (values.length - 1)) * w; i ? ctx.lineTo(xx, y(v)) : ctx.moveTo(xx, y(v)); });
  ctx.stroke();
  ctx.fillStyle = "#7a8898"; ctx.font = "10px SF Mono, Menlo, monospace";
  ctx.fillText(compact(hi, 2), 4, 12); ctx.fillText(compact(lo, 2), 4, h - 4);
}

// -------------------------------------------------------------- wiring

function setup() {
  $("btn-play").addEventListener("click", () => control(S.state && S.state.session.playing ? "pause" : "play"));
  $("btn-step").addEventListener("click", () => control("step", { seconds: 10 }));
  $("btn-step60").addEventListener("click", () => control("step", { seconds: 60 }));
  $("btn-end").addEventListener("click", () => control("finish"));
  $("speed").addEventListener("change", (e) => control("speed", { speed: Number(e.target.value) }));
  $("btn-reset").addEventListener("click", openSessionForm);
  $("btn-start").addEventListener("click", startSession);
  $("btn-cancel").addEventListener("click", closeSessionForm);
  // Closing (also with Escape) returns focus to the button that opened the dialog; drop it so the space-bar play shortcut cannot press it.
  $("session-dialog").addEventListener("close", () => $("btn-reset").blur());
  $("btn-restart").addEventListener("click", () => control("reset"));
  // Same settings, back to time 0: runs are deterministic, so this is an exact re-watch. The
  // end-of-session prompt is armed again so it shows when the rerun finishes.
  $("btn-rewind").addEventListener("click", () => { closeSessionForm(); S.finishedPrompted = null; control("reset"); });
  $("progress").addEventListener("click", (e) => {
    if (!S.state) return;
    const box = e.currentTarget.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
    $("clock").textContent = "seeking…";
    control("seek", { time: fraction * S.state.session.duration });
  });
  for (const id of ["scenario", "transparency", "seed"]) {
    $(id).addEventListener("input", () => { $("btn-start").disabled = !sessionFormValid(); });
    $(id).addEventListener("change", () => { $("btn-start").disabled = !sessionFormValid(); });
  }
  $("seed").addEventListener("keydown", (e) => { if (e.key === "Enter") startSession(); });
  document.querySelectorAll("#bottom-tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#bottom-tabs button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    S.tab = b.dataset.tab;
    document.querySelectorAll(".tab-body").forEach((el) => el.classList.toggle("hidden", el.dataset.body !== S.tab));
    if (S.state) renderTab(S.state);
  }));
  $("filter-type").addEventListener("change", (e) => { S.filterType = e.target.value; renderTab(S.state); });
  $("filter-open").addEventListener("change", (e) => { S.filterOpen = e.target.checked; renderTab(S.state); });
  $("positions-table").addEventListener("click", (e) => {
    const th = e.target.closest("th");
    if (th && th.dataset.key) {
      S.sort = { key: th.dataset.key, dir: S.sort.key === th.dataset.key ? -S.sort.dir : -1 };
      renderTab(S.state);
      return;
    }
    const tr = e.target.closest("tr[data-agent]");
    if (tr) openDrawer(tr.dataset.agent);
  });
  for (const id of ["liquidations-table", "orders-table"]) {
    $(id).addEventListener("click", (e) => { const tr = e.target.closest("tr[data-agent]"); if (tr) openDrawer(tr.dataset.agent); });
  }
  $("drawer-close").addEventListener("click", () => { S.drawerId = null; $("drawer").classList.add("hidden"); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { S.drawerId = null; $("drawer").classList.add("hidden"); }
    if (e.key === " " && !$("session-dialog").open && !["INPUT", "SELECT"].includes(document.activeElement.tagName)) { e.preventDefault(); $("btn-play").click(); }
  });
  setupChartInteraction();
  setupBookTooltip();
  poll(true);
}

document.addEventListener("DOMContentLoaded", setup);
