/* ==========================================================================
   AutoML Workspace — core: utilities, state, overlays, charts, websocket, router
   Pure UI layer. Talks to the existing API exactly as it already does.
   ========================================================================== */
"use strict";

/* ------------------------------------------------------------------ utils */

const $ = (id) => document.getElementById(id);
const el = (sel, root) => (root || document).querySelector(sel);
const els = (sel, root) => Array.from((root || document).querySelectorAll(sel));

const esc = (s) => String(s === null || s === undefined ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

const nf = (v) => (v === null || v === undefined || v === "" ? "—" : Number(v).toLocaleString());

function num(v, d) {
  if (v === null || v === undefined || v === "" || !isFinite(Number(v))) return "—";
  const n = Number(v);
  if (Number.isInteger(n) && Math.abs(n) < 1e6) return String(n);
  return n.toFixed(d === undefined ? 4 : d);
}
const pct = (v) => (isFinite(Number(v)) ? (Number(v) * 100).toFixed(1) + "%" : "—");

function dur(sec) {
  if (sec === null || sec === undefined || !isFinite(Number(sec))) return "—";
  const s = Number(sec);
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60);
  if (m < 60) return m + "m " + Math.round(s % 60) + "s";
  return Math.floor(m / 60) + "h " + (m % 60) + "m";
}
function ago(iso) {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (!isFinite(t)) return String(iso).slice(0, 16).replace("T", " ");
  const d = (Date.now() - t) / 1000;
  if (d < 60) return "just now";
  if (d < 3600) return Math.floor(d / 60) + "m ago";
  if (d < 86400) return Math.floor(d / 3600) + "h ago";
  if (d < 604800) return Math.floor(d / 86400) + "d ago";
  return new Date(t).toISOString().slice(0, 10);
}
const titleCase = (s) => String(s || "").replace(/[_-]+/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}));
  const txt = await res.text();
  let data = {};
  try { data = txt ? JSON.parse(txt) : {}; } catch (e) { data = { detail: txt }; }
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === "string" ? d : (d && d.msg) || "HTTP " + res.status);
  }
  return data;
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });
const del = (path) => api(path, { method: "DELETE" });

function downloadText(name, text, mime) {
  const blob = new Blob([text], { type: mime || "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1500);
}

/* ------------------------------------------------------------------ state */

const sel = {
  projectId: null, project: null, page: null, runId: null,
  sessionId: null, datasetId: null, mode: "ai",
};
let run = null;
let lastState = {};
const cache = {};
const CFG_KEY = "studio.config.v1";

let runCfg = {
  goal: "", metric: "", dataset_source: "", dataset_query: "", dataset_path: "",
  max_rows: "", target_column: "", max_iterations: "", feature_engineering_every_n: "",
  model_hint: "", mode: "ai",
};
let cfgDirty = false;

/* real, backend-supported value sets (never invented) */
const SOURCES = [
  ["synthetic", "Synthetic demo"],
  ["upload", "Uploaded file"],
  ["openml", "OpenML search"],
  ["kaggle", "Kaggle (needs keys)"],
];
const METRICS = [
  ["accuracy", "Accuracy"], ["f1", "F1"], ["roc_auc", "ROC-AUC"], ["precision", "Precision"],
  ["recall", "Recall"], ["pr_auc", "PR-AUC"], ["logloss", "Log loss"],
  ["rmse", "RMSE"], ["mae", "MAE"], ["mse", "MSE"], ["r2", "R²"],
];
const MINIMIZE = new Set(["rmse", "mae", "mse", "logloss"]);
const MODELS = [
  ["logistic_regression", "Logistic Regression", "Linear baseline for classification"],
  ["linear_regression", "Linear Regression", "Linear baseline for regression"],
  ["random_forest", "Random Forest", "Bagged trees, strong tabular default"],
  ["lightgbm", "LightGBM", "Gradient boosting, leaf-wise"],
  ["xgboost", "XGBoost", "Gradient boosting, level-wise"],
  ["gradient_boosting", "Gradient Boosting", "scikit-learn boosting"],
  ["svm", "SVM", "Support vector classifier / regressor"],
  ["neural_network", "Neural Network", "Small multilayer perceptron"],
  ["decision_tree", "Decision Tree", "Single tree, interpretable"],
  ["ridge", "Ridge", "Regularised linear model"],
];
const BUSY = ["TRAINING", "RESEARCH", "DATA_FETCH", "STARTING"];
const modelName = (m) => {
  const hit = MODELS.find((x) => x[0] === m);
  return hit ? hit[1] : titleCase(m);
};

/* Display names for the metric keys the backend actually stores. */
const METRIC_LABEL = {
  accuracy: "Accuracy", f1: "F1", f1_macro: "F1 macro", f1_weighted: "F1 weighted",
  precision: "Precision", recall: "Recall", roc_auc: "ROC-AUC", pr_auc: "PR-AUC",
  logloss: "Log loss", mae: "MAE", mse: "MSE", rmse: "RMSE", r2: "R²",
  explained_variance: "Explained variance", target_metric: "Target metric",
};
const metricLabel = (m) => METRIC_LABEL[String(m || "").toLowerCase()] || titleCase(m);

const PAGES = {};

/* ------------------------------------------------------------------ config */

function loadCfg() {
  try {
    const raw = localStorage.getItem(CFG_KEY);
    if (raw) Object.assign(runCfg, JSON.parse(raw));
  } catch (e) { /* private mode */ }
}
function saveCfg() {
  try { localStorage.setItem(CFG_KEY, JSON.stringify(runCfg)); } catch (e) { /* ignore */ }
}
function setCfg(k, v) {
  runCfg[k] = v;
  cfgDirty = true;
  saveCfg();
  paintSaveState();
}
function paintSaveState() {
  const b = $("saveChip");
  if (b) b.textContent = cfgDirty ? "Unsaved" : "Saved";
}

/* ------------------------------------------------------------------ toasts */

function toast(msg, kind, sub) {
  const host = $("toasts");
  const t = document.createElement("div");
  t.className = "toast " + (kind || "info");
  const body = document.createElement("div");
  body.className = "ti";
  body.textContent = msg;
  if (sub) {
    const s = document.createElement("span");
    s.className = "sub";
    s.textContent = sub;
    body.appendChild(s);
  }
  const x = document.createElement("button");
  x.className = "xbtn";
  x.type = "button";
  x.setAttribute("aria-label", "Dismiss notification");
  x.textContent = "✕";
  x.onclick = () => kill();
  t.appendChild(body);
  t.appendChild(x);
  host.appendChild(t);
  let done = false;
  function kill() {
    if (done) return;
    done = true;
    t.classList.add("out");
    setTimeout(() => { if (t.parentNode) t.remove(); }, 220);
  }
  setTimeout(kill, kind === "err" ? 9000 : 4200);
  return kill;
}

/* ------------------------------------------------------- overlay machinery */
/* One stack so Esc always closes the top-most thing. Every overlay has an X,
   a Close, and Esc — the user is never trapped. */

const overlays = [];
function pushOverlay(id) {
  if (!overlays.some((o) => o.id === id)) overlays.push({ id });
  $("scrim").classList.add("on");
}
function popOverlay(id) {
  const i = overlays.findIndex((o) => o.id === id);
  if (i >= 0) overlays.splice(i, 1);
  if (!overlays.length) $("scrim").classList.remove("on");
}
function closeTop() {
  const top = overlays[overlays.length - 1];
  if (!top) return false;
  if (top.id === "drawer") closeDrawer();
  else if (top.id === "palette") closePalette();
  else if (top.id === "modal") closeModal();
  return true;
}

function openDrawer(title, sub, bodyHTML, footHTML) {
  $("drawerTitle").textContent = title;
  $("drawerSub").textContent = sub || "";
  $("drawerBody").innerHTML = bodyHTML || "";
  $("drawerFoot").innerHTML = footHTML || "";
  $("drawerFoot").style.display = (footHTML || "").trim() ? "" : "none";
  $("drawer").classList.add("on");
  $("drawer").setAttribute("aria-hidden", "false");
  pushOverlay("drawer");
  return $("drawerBody");
}
function closeDrawer() {
  $("drawer").classList.remove("on");
  $("drawer").setAttribute("aria-hidden", "true");
  popOverlay("drawer");
}

function modal(title, bodyHTML, actions) {
  return new Promise((resolve) => {
    $("modalTitle").textContent = title;
    $("modalBody").innerHTML = bodyHTML;
    const foot = $("modalFoot");
    foot.innerHTML = "";
    let settled = false;
    const done = (v) => {
      if (settled) return;
      settled = true;
      $("modal").classList.remove("on");
      $("modal").setAttribute("aria-hidden", "true");
      popOverlay("modal");
      resolve(v);
    };
    (actions || [{ label: "Close", kind: "ghost", value: null }]).forEach((a) => {
      const b = document.createElement("button");
      b.className = "btn " + (a.kind || "ghost");
      b.type = "button";
      b.textContent = a.label;
      b.onclick = () => done(a.value);
      foot.appendChild(b);
    });
    const x = document.createElement("button");
    x.className = "xbtn";
    x.type = "button";
    x.setAttribute("aria-label", "Close dialog");
    x.title = "Close (Esc)";
    x.textContent = "✕";
    x.onclick = () => done(null);
    foot.appendChild(x);
    $("modal").classList.add("on");
    $("modal").setAttribute("aria-hidden", "false");
    pushOverlay("modal");
    const first = foot.querySelector("button");
    if (first) first.focus();
  });
}
function closeModal() {
  const x = $("modalFoot").querySelector(".xbtn");
  if (x) x.click();
}

const confirmModal = (title, body, okLabel, danger) =>
  modal(title, body, [
    { label: "Cancel", kind: "ghost", value: false },
    { label: okLabel || "Confirm", kind: danger ? "danger" : "pri", value: true },
  ]);

/* Unsaved-changes guard: never silently discard work. */
async function guardUnsaved() {
  if (!cfgDirty) return true;
  const r = await modal("Unsaved changes", "<p>Your changes have not been saved.</p>", [
    { label: "Keep editing", kind: "ghost", value: "keep" },
    { label: "Discard", kind: "danger", value: "discard" },
    { label: "Save", kind: "pri", value: "save" },
  ]);
  if (r === "keep") return false;
  cfgDirty = false;
  paintSaveState();
  saveCfg();
  return true;
}

/* ------------------------------------------------------------- empty/error */

const emptyState = (icon, title, text, actionHTML) => `
  <div class="empty">
    <div class="ei" aria-hidden="true">${icon}</div>
    <h3>${esc(title)}</h3>
    <p>${esc(text)}</p>
    ${actionHTML || ""}
  </div>`;

const errorState = (title, msg, err) => `
  <div class="errbox">
    <div class="eh"><i aria-hidden="true">!</i>${esc(title)}</div>
    <p>${esc(msg || "The request could not be completed.")}</p>
    <div class="btnrow">
      <button class="btn" data-act="retry" type="button"><i aria-hidden="true">↻</i> Retry</button>
      <button class="btn ghost" data-act="dismiss" type="button">Close</button>
    </div>
    ${err ? `<details class="tech"><summary>Technical details</summary><pre>${esc(String(err.stack || err.message || err))}</pre></details>` : ""}
  </div>`;

const skeleton = (rows, note) => `
  <div class="loading-note"><span class="spin" aria-hidden="true"></span>${esc(note || "Loading…")}</div>
  <div class="skelrows" aria-hidden="true">
    ${Array.from({ length: rows || 4 }, (_, i) => `<div class="skel" style="opacity:${Math.max(0.2, 1 - i * 0.16)}"></div>`).join("")}
  </div>`;

function wireErrorBox(root, retryFn) {
  const r = el('[data-act="retry"]', root);
  if (r && retryFn) r.onclick = retryFn;
  const d = el('[data-act="dismiss"]', root);
  if (d) d.onclick = () => { const b = d.closest(".errbox"); if (b) b.remove(); };
}

function statusBadge(s) {
  const v = String(s || "idle").toLowerCase();
  const map = {
    done: "ok", completed: "ok", ready: "ok", healthy: "ok", success: "ok", live: "ok",
    running: "warn", training: "warn", processing: "warn", pending: "warn",
    paused: "warn", uploading: "warn", starting: "warn", research: "warn", data_fetch: "warn",
    failed: "bad", error: "bad", stopped: "bad", cancelled: "bad",
    compatible: "ok", incompatible: "bad", partially_compatible: "warn",
  };
  const cls = map[v] || "";
  const live = ["running", "training"].indexOf(v) >= 0 ? " live" : "";
  return `<span class="badge ${cls}${live}"><span class="dot" aria-hidden="true"></span>${esc(titleCase(s || "idle"))}</span>`;
}

const kpi = (l, v, i, s, spark) => `
  <div class="kpi">
    <div class="kpi-l"><span aria-hidden="true">${i || "◈"}</span>${esc(l)}</div>
    <div class="kpi-v">${esc(v)}</div>
    ${spark ? sparkline(spark) : ""}
    ${s ? `<div class="kpi-s">${esc(s)}</div>` : ""}
  </div>`;

/* Inline trend line for a KPI. Real values only — never invented. */
let _spk = 0;
function sparkline(vals, opts) {
  const o = opts || {};
  const v = (vals || []).map(Number).filter((x) => isFinite(x));
  if (v.length < 2) return "";
  const W = 132, H = 30, pad = 4;
  const mn = Math.min.apply(null, v), mx = Math.max.apply(null, v);
  const rng = (mx - mn) || 1;
  const sx = (i) => (i / (v.length - 1)) * (W - 2) + 1;
  const sy = (x) => H - pad - ((x - mn) / rng) * (H - pad * 2);
  const d = v.map((x, i) => (i ? "L" : "M") + sx(i).toFixed(1) + " " + sy(x).toFixed(1)).join(" ");
  const area = d + " L" + sx(v.length - 1).toFixed(1) + " " + (H - pad) + " L" + sx(0).toFixed(1) + " " + (H - pad) + " Z";
  const up = v[v.length - 1] >= v[0];
  const col = o.color || (up ? "var(--ok)" : "var(--bad)");
  const id = "spk" + (++_spk);
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${col}" stop-opacity=".30"/><stop offset="100%" stop-color="${col}" stop-opacity="0"/>
    </linearGradient></defs>
    <path d="${area}" fill="url(#${id})"/>
    <path class="spk-ln" d="${d}" fill="none" stroke="${col}" stroke-width="1.7"
          stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
    <circle cx="${sx(v.length - 1).toFixed(1)}" cy="${sy(v[v.length - 1]).toFixed(1)}" r="2.4" fill="${col}"/>
  </svg>`;
}
const kv = (k, v) => `<div><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`;
const pre = (s) => `<pre class="code wrap">${esc(s)}</pre>`;
const sectHead = (title, right) => `<div class="sect-h"><h2>${esc(title)}</h2><span class="rule"></span>${right || ""}</div>`;

/** Per-project memo. Cleared on every project switch so no data leaks across. */
function clearProjectCache() {
  Object.keys(cache).forEach((k) => { delete cache[k]; });
}

/**
 * Best real dataset metadata available. The orchestrator's snapshot carries
 * only a preview-shaped table (rows/columns/dtypes), while the persisted
 * project context carries the descriptive fields (source, target, task), so
 * prefer whichever candidate actually has metadata.
 * Never invents values; returns {} when the backend has nothing.
 */
function currentDataset() {
  const cands = [];
  if (run && run.dataset && run.dataset.rows) cands.push(run.dataset);
  if (lastState && lastState.dataset && lastState.dataset.rows) cands.push(lastState.dataset);
  if (cache.ctx && cache.ctx.dataset && cache.ctx.dataset.rows) cands.push(cache.ctx.dataset);
  for (let i = 0; i < cands.length; i++) {
    const d = cands[i];
    if (d.target_column || d.source || d.task_type) return d;
  }
  return cands[0] || {};
}

/** Fetch and cache the project's recorded context (real, persisted). */
async function loadProjectContext() {
  if (cache.ctx && cache.ctx.dataset) return cache.ctx;
  try { cache.ctx = await api("/projects/" + sel.projectId + "/context"); } catch (e) { cache.ctx = {}; }
  return cache.ctx;
}

/* ------------------------------------------------------------------ charts */
/* Real backend numbers only. No data → no chart is drawn. */

function lineChart(points, opts) {
  const o = opts || {};
  const raw = (points || []).filter((p) => isFinite(p[1]));
  if (raw.length < 2) {
    return `<div class="empty" style="padding:28px"><p>${esc(o.emptyNote || "Not enough data points yet.")}</p></div>`;
  }
  /* categorical x (e.g. model names) is plotted by index and labelled by value */
  const categorical = !raw.every((p) => isFinite(p[0]));
  const pts = raw.map((p, i) => (categorical ? [i, p[1]] : [p[0], p[1]]));
  const label = categorical ? (p) => String(raw[p[0]][0]) : (p) => (o.xLabel ? o.xLabel(p) : String(p));

  const W = 720, H = o.height || 220, P = { l: 48, r: 14, t: 14, b: 28 };
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
  let y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
  if (y1 - y0 < 1e-9) y1 = y0 + 1;
  const pad = (y1 - y0) * 0.12;
  y0 -= pad; y1 += pad;
  if (o.fixed0) y0 = 0;
  const sx = (v) => P.l + ((v - x0) / (x1 - x0 || 1)) * (W - P.l - P.r);
  const sy = (v) => H - P.b - ((v - y0) / (y1 - y0 || 1)) * (H - P.t - P.b);
  const path = pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
  const area = path + " L" + sx(pts[pts.length - 1][0]).toFixed(1) + " " + sy(y0).toFixed(1) +
               " L" + sx(pts[0][0]).toFixed(1) + " " + sy(y0).toFixed(1) + " Z";
  let g = "";
  /* tick precision follows the value range, so a 0.90→0.91 spread does not
     print four identical "0.91" labels */
  const span = Math.abs(y1 - y0) || 1;
  const tickDp = span >= 100 ? 0 : span >= 1 ? 1 : span >= 0.1 ? 2 : span >= 0.01 ? 3 : 4;
  for (let i = 0; i <= 4; i++) {
    const v = y0 + ((y1 - y0) * i) / 4, y = sy(v);
    g += `<line class="grid" x1="${P.l}" y1="${y.toFixed(1)}" x2="${W - P.r}" y2="${y.toFixed(1)}"/>` +
         `<text x="${P.l - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end">${esc(v.toFixed(tickDp))}</text>`;
  }
  const step = Math.max(1, Math.ceil(pts.length / 7));
  const ticks = pts.filter((_, i) => i % step === 0 || i === pts.length - 1);
  const xl = ticks.map((p) =>
    `<text x="${sx(p[0]).toFixed(1)}" y="${H - 8}" text-anchor="middle">${esc(label(p))}</text>`).join("");
  const dots = pts.map((p, i) =>
    `<circle class="pt" cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="${pts.length > 30 ? 1.6 : 2.8}"
      data-cx="${sx(p[0]).toFixed(1)}" data-cy="${sy(p[1]).toFixed(1)}"
      data-lx="${esc(label(p))}" data-ly="${esc(num(p[1]))}"><title>${esc(label(p))}: ${esc(num(p[1]))}</title></circle>`).join("");
  /* hover layer: crosshair + value pill, driven by the delegated handler below */
  const hover = `<g class="hv" aria-hidden="true" style="display:none">
      <line class="hv-l" x1="0" y1="${P.t}" x2="0" y2="${H - P.b}"/>
      <circle class="hv-c" cx="0" cy="0" r="4.5"/>
      <g class="hv-t"><rect rx="5"/><text></text></g>
    </g>`;
  return `<svg class="chart hov" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img"
      aria-label="${esc(o.aria || "Line chart")}">
    <defs><linearGradient id="gAccent" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="rgba(124,108,255,.26)"/><stop offset="100%" stop-color="rgba(124,108,255,0)"/>
    </linearGradient></defs>
    ${g}
    ${o.ref !== undefined && isFinite(o.ref)
      ? `<line class="ref" x1="${P.l}" y1="${sy(o.ref).toFixed(1)}" x2="${W - P.r}" y2="${sy(o.ref).toFixed(1)}"/>` : ""}
    <path class="area" d="${area}"/><path class="ln" pathLength="1" d="${path}"/>
    ${dots}
    ${hover}
    ${xl}
  </svg>`;
}

/* One delegated listener powers every line chart: hover shows the exact value
   at the nearest point. Falls back silently on browsers without getScreenCTM. */
function initChartHover() {
  if (initChartHover.done) return;
  initChartHover.done = true;
  document.addEventListener("mousemove", (e) => {
    const svg = e.target.closest ? e.target.closest("svg.chart.hov") : null;
    if (!svg) return;
    const hv = svg.querySelector(".hv");
    if (!hv || !svg.getScreenCTM) return;
    let loc;
    try { loc = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse()); }
    catch (err) { return; }
    const pts = svg.querySelectorAll(".pt");
    if (!pts.length) return;
    let best = pts[0], bestD = Infinity;
    for (const p of pts) {
      const d = Math.abs(Number(p.dataset.cx) - loc.x);
      if (d < bestD) { bestD = d; best = p; }
    }
    const cx = Number(best.dataset.cx), cy = Number(best.dataset.cy);
    hv.style.display = "";
    const line = hv.querySelector(".hv-l"), dot = hv.querySelector(".hv-c");
    const tip = hv.querySelector(".hv-t"), txt = hv.querySelector(".hv-t text");
    line.setAttribute("x1", cx); line.setAttribute("x2", cx);
    dot.setAttribute("cx", cx); dot.setAttribute("cy", cy);
    txt.textContent = best.dataset.ly + "  ·  " + best.dataset.lx;
    const bb = txt.getBBox();
    const vb = svg.viewBox.baseVal;
    let tx = cx + 12, ty = cy - 12;
    const tw = bb.width + 14, th = bb.height + 10;
    if (tx + tw > vb.width - 6) tx = cx - 12 - tw;
    if (ty < 4) ty = cy + 14;
    tip.setAttribute("transform", "translate(" + tx.toFixed(1) + "," + ty.toFixed(1) + ")");
    tip.querySelector("rect").setAttribute("width", tw.toFixed(1));
    tip.querySelector("rect").setAttribute("height", th.toFixed(1));
    tip.querySelector("rect").setAttribute("y", (-bb.height - 4).toFixed(1));
    tip.querySelector("text").setAttribute("y", 0);
  }, { passive: true });
  document.addEventListener("mouseleave", (e) => {
    const svg = e.target.closest ? e.target.closest("svg.chart.hov") : null;
    if (svg) { const hv = svg.querySelector(".hv"); if (hv) hv.style.display = "none"; }
  }, true);
}

function curveChart(xs, ys, opts) {
  const o = opts || {};
  if (!xs || !ys || xs.length < 2) return "";
  const W = 540, H = o.height || 290, P = { l: 46, r: 16, t: 16, b: 34 };
  const sx = (v) => P.l + Math.max(0, Math.min(1, v)) * (W - P.l - P.r);
  const sy = (v) => H - P.b - Math.max(0, Math.min(1, v)) * (H - P.t - P.b);
  const path = xs.map((x, i) => (i ? "L" : "M") + sx(x).toFixed(1) + " " + sy(ys[i]).toFixed(1)).join(" ");
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const y = (H - P.t - P.b) * i / 4 + P.t;
    g += `<line class="grid" x1="${P.l}" y1="${y.toFixed(1)}" x2="${W - P.r}" y2="${y.toFixed(1)}"/>` +
         `<text x="${P.l - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end">${(1 - i / 4).toFixed(1)}</text>`;
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || "Curve")}">
    <line class="ref" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}"/>
    ${g}
    <line class="axis" x1="${P.l}" y1="${P.t}" x2="${P.l}" y2="${H - P.b}"/>
    <line class="axis" x1="${P.l}" y1="${H - P.b}" x2="${W - P.r}" y2="${H - P.b}"/>
    <path class="${o.dash ? "ln2" : "ln"}" d="${path}"/>
    <text x="${(P.l + W - P.r) / 2}" y="${H - 8}" text-anchor="middle" class="lbl">${esc(o.xLabel || "False positive rate")}</text>
    <text x="13" y="${(P.t + H - P.b) / 2}" text-anchor="middle" class="lbl" transform="rotate(-90 13 ${(P.t + H - P.b) / 2})">${esc(o.yLabel || "True positive rate")}</text>
  </svg>`;
}

function hbars(obj, opts) {
  const o = opts || {};
  const rows = Object.entries(obj || {}).filter(([, v]) => isFinite(Number(v)));
  if (!rows.length) return emptyState("⌗", o.emptyTitle || "Nothing to show", o.emptyNote || "No values were recorded by the backend.");
  const sorted = rows.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, o.limit || 14);
  const max = Math.max.apply(null, sorted.map((r) => Math.abs(r[1]))) || 1;
  return `<div class="hbars">${sorted.map(([k, v]) => `
    <div class="hbar">
      <span class="hn mono" title="${esc(k)}">${esc(String(k).replace(/^(numeric|categorical)__/, ""))}</span>
      <span class="ht"><i style="width:${((Math.abs(v) / max) * 100).toFixed(1)}%"></i></span>
      <span class="hv">${esc(num(v, 3))}</span>
    </div>`).join("")}</div>`;
}

function confusionMatrix(cm) {
  const labels = cm && cm.labels;
  const m = cm && (cm.matrix || cm.values);
  if (!labels || !m || !m.length) return "";
  const flat = m.flat().filter((v) => isFinite(Number(v)));
  if (!flat.length) return "";
  const max = Math.max.apply(null, flat);
  return `<div style="overflow-x:auto"><div class="cm" style="grid-template-columns:repeat(${labels.length + 1},auto)">
    <div class="c h"></div>${labels.map((l) => `<div class="c h">pred ${esc(l)}</div>`).join("")}
    ${m.map((row, i) => `<div class="c h">true ${esc(labels[i])}</div>` +
      row.map((v, j) => `<div class="c${i === j ? " diag" : ""}" style="${v === max && max > 0 ? "background:var(--accent-soft)" : ""}">${esc(nf(v))}</div>`).join("")
    ).join("")}
  </div></div>
  <div class="legend"><span><i style="background:var(--accent)"></i>diagonal = correct</span>
    <span>rows = actual · columns = predicted</span></div>`;
}

/* --------------------------------------------------------------- websocket */

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const scope = sel.projectId ? "?project_id=" + encodeURIComponent(sel.projectId) : "?project_id=none";
  try { if (window.ws) { window.ws.onclose = null; window.ws.close(); } } catch (e) {}
  const ws = new WebSocket(proto + "://" + location.host + "/ws/automl" + scope);
  ws.onmessage = (e) => { try { onEvent(JSON.parse(e.data)); } catch (err) {} };
  ws.onclose = () => {
    setTimeout(() => { if (window.ws === ws) connectWS(); }, 2200);
  };
  window.ws = ws;
  window.wsScope = sel.projectId;
  paintLink();
}
function paintLink() {
  const p = $("runPill");
  if (!p) return;
  const on = window.ws && window.ws.readyState === 1;
  p.className = "pill" + (on ? "" : " warn");
  p.title = on ? "Connected to the agent" : "Reconnecting to the agent…";
}

const mine = (ev) => !!(sel.projectId && ev && ev.project_id === sel.projectId);

function onEvent(m) {
  if (m.type === "snapshot") { applySnapshot(m.data); return; }
  if (m.type === "ack") {
    if (m.status === "busy") toast("A run is already in progress.", "err");
    if (m.status === "error") toast("Rejected: " + (m.detail || "invalid context"), "err");
    if (m.snapshot) applySnapshot(m.snapshot);
    return;
  }
  if (m.type === "chat_response") {
    const card = (m.dataset && m.dataset.columns && m.dataset.columns.length)
      ? datasetFactsCard(m.dataset) : "";
    const node = addMsg("agent", m.text, "interviewer", card);
    if (node) {
      // Label how the reply was produced so a deterministic fallback is never
      // mistaken for an LLM answer, and surface provider failures honestly.
      const source = m.source || "assistant";
      const badge = source === "llm"
        ? '<span class="src-badge llm" title="Composed by the configured LLM provider">LLM</span>'
        : '<span class="src-badge" title="Built-in assistant (no LLM provider configured, or it was unavailable)">built-in</span>';
      const who = node.querySelector(".who");
      if (who) who.insertAdjacentHTML("beforeend", badge);
      if (m.provider_error) {
        const sys = document.createElement("div");
        sys.className = "sysline";
        sys.textContent = "Provider unavailable — using the built-in assistant. " + m.provider_error;
        node.appendChild(sys);
      }
    }
    drawerNote(m.text, "interviewer");
    suggestFor(m.text);
    return;
  }
  if (m.type === "agent_message") {
    if (mine(m)) {
      noteEvent(m);
      addMsg("agent", m.text, m.agent);
      drawerNote(m.text, m.agent);
      suggestFor(m.text);
    }
    return;
  }
  if (m.type === "status") {
    if (mine(m)) { noteEvent(m); setStage(m.state, m.message); }
    return;
  }
  if (m.type === "iteration" && mine(m)) { noteEvent(m); setStage("TRAINING", m.message); return; }
  if (mine(m) && ["research_card", "dataset_preview", "leaderboard", "feature_importance",
                   "best_model_ready", "profiling"].indexOf(m.type) >= 0) {
    noteEvent(m);
  }
}

function refreshIfVisible(pages) {
  if (sel.page && pages.indexOf(sel.page) >= 0) {
    const v = $("v-" + sel.page);
    if (v && v.classList.contains("on")) {
      const fn = PAGES[sel.page];
      if (fn) fn();
    }
  }
}

function applySnapshot(d) {
  if (!d) return;
  lastState = d;
  const ctx = d.run_context || {};
  if (ctx.project_id && ctx.project_id === sel.projectId) {
    run = {
      runId: ctx.run_id, stage: d.state, iters: (d.iterations || []).length,
      iterations: (d.iterations || []).slice(), log: [], last: "",
      best: d.best, dataset: d.dataset, startedAt: d.started_at, finishedAt: d.finished_at,
    };
    (d.iterations || []).forEach((it) => {
      run.log.push({
        t: Date.parse(d.started_at) || Date.now(), type: "iteration", state: "TRAINING",
        msg: "iter " + it.iteration + ": " + modelName(it.model) + " → " + num(it.target_metric),
      });
    });
    setStage(d.state, "");
  } else {
    run = null;
  }
  paintHeader();
  refreshIfVisible(["home", "monitor"]);
}

function noteEvent(m) {
  if (!run) run = { runId: m.run_id, stage: m.state || "STARTING", iters: 0, iterations: [], log: [], last: "" };
  if (m.run_id) run.runId = m.run_id;
  if (m.type === "iteration") {
    run.iters = Math.max(run.iters, m.iteration || 0);
    if (m.model) {
      run.iterations = (run.iterations || []).filter((i) => i.iteration !== m.iteration);
      run.iterations.push({ iteration: m.iteration, model: m.model, metrics: m.metrics || {}, target_metric: m.target_metric });
      run.iterations.sort((a, b) => a.iteration - b.iteration);
    }
  }
  if (m.state) run.stage = m.state;
  if (typeof m.progress === "number") run.progress = m.progress;
  if (m.text || m.message) run.last = String(m.message || m.text).slice(0, 160);
  run.log.push({
    t: m.ts ? Date.parse(m.ts) || Date.now() : Date.now(),
    type: m.type, state: m.state || m.stage || "", msg: m.message || m.text || m.type,
  });
  if (run.log.length > 400) run.log = run.log.slice(-400);
  paintHeader();
}

function setStage(stage, message) {
  if (!stage) return;
  if (run) { run.stage = stage; if (message) run.last = String(message).slice(0, 160); }
  lastState.stage = stage;
  const t = $("chatTyping");
  if (t) t.classList.toggle("on", BUSY.indexOf(stage) >= 0);
  paintHeader();
  refreshIfVisible(["monitor"]);
}

function paintHeader() {
  const stage = (run && run.stage) || lastState.stage || "IDLE";
  const p = $("runPill"), t = $("runPillT");
  if (!p || !t) return;
  t.textContent = titleCase(stage);
  p.className = "pill" + (BUSY.indexOf(stage) >= 0 ? " live" : stage === "DONE" ? " ok"
    : stage === "FAILED" ? " bad" : stage === "PAUSED" ? " warn" : "");
  p.title = run && run.runId ? "Run " + run.runId : "No active run";
  if (sel.project) {
    $("sideProjMeta").textContent = run && run.runId ? run.runId : (sel.project.status || "active");
  }
}

/* ------------------------------------------------------------------ router */

const NAV = [
  ["Workspace", [
    ["home", "◈", "Overview"],
    ["chat", "✦", "Chat"],
    ["data", "▤", "Data"],
    ["code", "⌸", "Code"],
    ["experiments", "◈", "Experiments"],
  ]],
  ["Models", [
    ["lab", "⚙", "Model Lab"],
    ["compare", "⇄", "Compare"],
  ]],
  ["Monitoring", [
    ["monitor", "▶", "Live Runs"],
    ["reports", "✎", "Reports"],
  ]],
  ["Settings", [
    ["settings", "⌗", "Project Settings"],
  ]],
];

const WHERE = {
  projects: ["◫", "All projects"], home: ["◈", "Overview"], chat: ["✦", "Chat"],
  data: ["▤", "Data"], dataset: ["▤", "Dataset inspector"], lab: ["⚙", "Model Lab"],
  experiments: ["◈", "Experiments"], run: ["◈", "Experiment"], monitor: ["▶", "Live Runs"],
  reports: ["✎", "Reports"], report: ["✎", "Model report"], compare: ["⇄", "Compare"],
  code: ["⌸", "Code & Artifacts"], settings: ["⌗", "Project settings"],
};

function setWhere(icon, label) {
  $("hdrWhere").querySelector("i").textContent = icon;
  $("hdrWhereT").textContent = label;
}

function buildNav() {
  const nav = $("sideNav");
  nav.innerHTML = "";
  if (!sel.projectId) {
    const box = document.createElement("div");
    box.className = "navgroup";
    box.innerHTML = '<h5>Workspace</h5>' +
      '<a class="navitem on" href="#/projects" aria-current="page"><i aria-hidden="true">◫</i><span class="nlab">All projects</span></a>';
    nav.appendChild(box);
    return;
  }
  NAV.forEach((pair) => {
    const title = pair[0], items = pair[1];
    const box = document.createElement("div");
    box.className = "navgroup";
    box.innerHTML = "<h5>" + esc(title) + "</h5>";
    items.forEach((it) => {
      const key = it[0], icon = it[1], label = it[2];
      const a = document.createElement("a");
      a.className = "navitem";
      a.href = "#/projects/" + sel.projectId + "/" + key;
      a.dataset.page = key;
      a.innerHTML = '<i aria-hidden="true">' + icon + '</i><span class="nlab">' + esc(label) + "</span>";
      box.appendChild(a);
    });
    nav.appendChild(box);
  });
}

function markNav() {
  els("#sideNav .navitem").forEach((a) => {
    const on = a.dataset.page === sel.page;
    a.classList.toggle("on", on);
    a.setAttribute("aria-current", on ? "page" : "false");
  });
}

function parseHash() {
  const parts = (location.hash || "#/projects").replace(/^#/, "").split("/").filter(Boolean)
    .map((x) => { try { return decodeURIComponent(x); } catch (e) { return x; } });
  if (parts[0] !== "projects" || !parts[1]) return { view: "projects" };
  const pid = parts[1];
  if (parts[2] === "runs" && parts[3]) return { view: "run", pid: pid, runId: parts[3] };
  if (parts[2] === "report" && parts[3]) return { view: "report", pid: pid, runId: parts[3] };
  if (parts[2] === "datasets" && parts[3]) return { view: "dataset", pid: pid, datasetId: parts[3] };
  const page = parts[2] || "home";
  if (!PAGES[page]) return { view: "page", pid: pid, page: "home" };
  return { view: "page", pid: pid, page: page };
}

async function render() {
  const r = parseHash();
  els(".page").forEach((v) => v.classList.remove("on"));
  $("app").classList.remove("sidemenu");

  if (r.view === "projects") {
    sel.projectId = null; sel.project = null; sel.page = null;
    sel.sessionId = null; sel.runId = null; sel.datasetId = null;
    $("v-projects").classList.add("on");
    $("hdrProject").textContent = "No project";
    $("sideProjName").textContent = "All projects";
    $("sideProjMeta").textContent = "Choose a project to begin";
    $("sideBackBtn").hidden = true;
    $("chatFab").hidden = true;
    run = null;
    const w = WHERE.projects;
    setWhere(w[0], w[1]);
    buildNav();
    connectWS();
    paintHeader();
    PAGES.projects();
    return;
  }

  if (sel.projectId !== r.pid) await selectProject(r.pid);
  if (!sel.project) return;

  const page = r.view === "page" ? r.page : r.view;
  sel.page = page;
  sel.runId = r.runId || null;
  sel.datasetId = r.datasetId || null;

  const view = $("v-" + page);
  if (view) view.classList.add("on");
  $("hdrProject").textContent = sel.project.name;
  const w = WHERE[page] || ["◈", titleCase(page)];
  setWhere(w[0], w[1]);
  buildNav();
  markNav();
  $("sideBackBtn").hidden = false;
  $("chatFab").hidden = page !== "chat";
  $("main").scrollTop = 0;

  const fn = PAGES[page];
  if (fn) fn();
}

async function selectProject(id) {
  try {
    const p = await api("/projects/" + id);
    if (p.status === "deleted") throw new Error("This project was deleted.");
    if (sel.projectId !== p.id) clearProjectCache();
    sel.projectId = p.id;
    sel.project = p;
    $("sideProjName").textContent = p.name;
    $("sideProjMeta").textContent = p.status || "active";
    if (p.description && !runCfg.goal) runCfg.goal = p.description;
    connectWS();
    await loadSessions();
    try { lastState = await api("/automl/state"); } catch (e) { lastState = {}; }
    applySnapshot(lastState);
  } catch (e) {
    sel.projectId = null;
    sel.project = null;
    toast("Project not found: " + e.message, "err");
    location.hash = "#/projects";
  }
}

window.addEventListener("hashchange", () => { render(); });

/* --------------------------------------------------------- projects page */

PAGES.projects = async function () {
  const host = $("projectsBody");
  host.innerHTML = skeleton(4, "Loading projects…");
  let items = [];
  try {
    items = ((await api("/projects?page=1&page_size=100")).items || [])
      .filter((p) => p.status !== "deleted");
  } catch (e) {
    host.innerHTML = errorState("Could not load projects", e.message, e);
    wireErrorBox(host, PAGES.projects);
    return;
  }
  if (!items.length) {
    host.innerHTML = emptyState("◫", "No projects yet",
      "A project holds its own datasets, agents, runs and reports. Create one to begin.",
      '<button class="btn pri" data-act="newproj" type="button"><i aria-hidden="true">＋</i> Create project</button>');
    const b = el('[data-act="newproj"]', host);
    if (b) b.onclick = newProjectDialog;
    return;
  }
  host.innerHTML = `<div class="card flush"><div class="tbl-scroll"><table class="tbl">
    <thead><tr><th>Project</th><th>Status</th><th>Last modified</th><th></th></tr></thead>
    <tbody>${items.map((p) => `
      <tr class="clickable" data-pid="${esc(p.id)}">
        <td><b>${esc(p.name)}</b>${p.description ? `<div class="hint" style="margin-top:2px">${esc(p.description)}</div>` : ""}</td>
        <td>${statusBadge(p.status)}</td>
        <td class="muted">${esc(ago(p.updated_at))}</td>
        <td><div class="tbl-act"><button class="btn ghost sm" data-open="${esc(p.id)}" type="button">Open</button></div></td>
      </tr>`).join("")}</tbody></table></div></div>`;
  els("tr[data-pid]", host).forEach((tr) => {
    tr.onclick = () => { location.hash = "#/projects/" + tr.dataset.pid + "/home"; };
  });
  els("[data-open]", host).forEach((b) => {
    b.onclick = (e) => { e.stopPropagation(); location.hash = "#/projects/" + b.dataset.open + "/home"; };
  });
};

async function newProjectDialog() {
  const r = await modal("New project", `
    <div class="field">
      <label class="lbl" for="npName">Project name</label>
      <input class="inp" id="npName" placeholder="Customer churn" />
    </div>
    <div class="field">
      <label class="lbl" for="npDesc">What are you building? <span class="opt">optional</span></label>
      <textarea class="inp" id="npDesc" placeholder="Predict which customers will churn this month."></textarea>
      <div class="help">The agent uses this as its starting goal.</div>
    </div>`, [
    { label: "Cancel", kind: "ghost", value: null },
    { label: "Create project", kind: "pri", value: "go" },
  ]);
  if (r !== "go") return;
  const name = $("npName").value.trim();
  if (!name) { toast("Give the project a name.", "err"); return; }
  try {
    const p = await post("/projects", { name: name, description: $("npDesc").value.trim() || null });
    toast("Project created.", "ok", p.name);
    location.hash = "#/projects/" + p.id + "/chat";
  } catch (e) {
    toast("Create failed: " + e.message, "err");
  }
}

/* ------------------------------------------------- project command centre */

PAGES.home = function () {
  const host = $("homeBody");
  $("homeName").textContent = sel.project ? sel.project.name : "—";
  $("homeDesc").textContent = (sel.project && sel.project.description) ||
    "A calm starting point. Ask the agent, or open a workspace section.";
  $("homeEyebrow").textContent = (sel.project && sel.project.status) || "Project";
  host.innerHTML = skeleton(3, "Preparing project overview…");

  (async () => {
    if (sel.projectId !== sel.project.id && !sel.projectId) return;
    let ctx = null, runs = [];
    try { ctx = cache.ctx = await api("/projects/" + sel.projectId + "/context"); } catch (e) {}
    try { runs = (await api("/runs?project_id=" + sel.projectId + "&limit=50")).runs || []; } catch (e) {}
    const ds = (ctx && ctx.dataset) || {};
    const best = ctx && ctx.best_model;
    const live = currentDataset();
    const cards = [
      /* rows live on the live table, not on ctx.dataset — show the truth */
      ["Dataset", live.rows ? nf(live.rows) + " rows" : "none yet",
        live.source ? titleCase(live.source) : (ds.source ? titleCase(ds.source) : "no source yet"), "▤"],
      ["Target", live.target_column || ds.target_column || ctx.target || "—",
        live.target_column || ds.target_column ? "used by the last run" : "not chosen yet", "◎"],
      ["Experiments", nf(runs.length), runs.length ? runs[0].run_id : "none yet", "◈"],
      ["Best model", best ? modelName(best.model) : "—", best ? num(best.target_metric) + " metric" : "no run yet", "⚙"],
      ["Features", best ? nf(best.n_features) : "0", best ? "in the trained table" : "no run yet", "⌗"],
    ];
    /* A trend of the recorded target metric, oldest → newest. Only real runs. */
    const trend = runs.filter((r) => r.best && isFinite(Number(r.best.target_metric)))
      .slice().reverse().map((r) => Number(r.best.target_metric));
    const cardHtml = cards.map((c) => kpi(c[0], c[1], c[3], c[2])).join("") +
      (trend.length > 1
        ? kpi("Metric trend", num(trend[trend.length - 1]),
            "◆", trend.length + " recorded runs, oldest → newest", trend)
        : "");
    const recent = runs.slice(0, 5);
    const decisions = (ctx && ctx.decisions) || [];
    host.innerHTML = `
      <div class="sect"><div class="kpis">${cardHtml}</div></div>
      <div class="sect">
        ${sectHead("Recent experiments",
          '<a class="btn ghost sm" href="#/projects/' + esc(sel.projectId) + '/experiments">All experiments</a>')}
        ${recent.length ? `<div class="card flush"><div class="tbl-scroll"><table class="tbl">
          <thead><tr><th>Run</th><th>Model</th><th>Metric</th><th>Status</th><th></th></tr></thead>
          <tbody>${recent.map((r) => `
            <tr class="clickable" data-run="${esc(r.run_id)}">
              <td class="mono">${esc(r.run_id)}</td>
              <td>${esc(modelName((r.best && r.best.model) || "—"))}</td>
              <td class="num">${esc(num(r.best && r.best.target_metric))}</td>
              <td>${statusBadge(r.status)}</td>
              <td><div class="tbl-act"><button class="btn ghost sm" type="button">Open</button></div></td>
            </tr>`).join("")}</tbody></table></div></div>`
          : emptyState("◈", "No experiments yet",
              "Run your first experiment to compare models and metrics.",
              '<button class="btn pri" data-go="lab" type="button"><i aria-hidden="true">▶</i> Configure a run</button>')}
      </div>
      <div class="sect">
        ${sectHead("Dataset in use")}
        <div class="kv">
          ${kv("Source", titleCase(live.source || ds.source || "—"))}
          ${kv("Rows", live.rows || ds.rows ? nf(live.rows || ds.rows) : "—")}
          ${kv("Target", live.target_column || ds.target_column || ctx.target || "—")}
          ${kv("Task", titleCase(live.task_type || ds.task_type || ctx.task || "—"))}
        </div>
      </div>
      <div class="sect">
        ${sectHead("What the agent has decided")}
        ${decisions.length
          ? '<div class="card"><ul class="speclist">' + decisions.slice(0, 6).map((a) => {
              const what = (a && (a.what || a.key)) || (typeof a === "string" ? a : "decision");
              const why = (a && (a.why || a.reason)) || "";
              return '<li><span>' + esc(what) + '</span><span class="sp"></span><span class="muted">' + esc(why) + "</span></li>";
            }).join("") + "</ul></div>"
          : '<div class="card"><p class="muted" style="margin:0">Nothing recorded yet. Describe a task in chat and the agent will write its decisions here.</p></div>'}
      </div>`;
    els("tr[data-run]", host).forEach((tr) => {
      tr.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/runs/" + tr.dataset.run; };
    });
    const g = el('[data-go="lab"]', host);
    if (g) g.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/lab"; };
  })();
};
