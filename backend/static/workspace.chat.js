/* ==========================================================================
   AutoML Workspace — chat workspace, sessions, AI task cards, agent dialogs
   ========================================================================== */

const AGENTS = {
  interviewer: "Interviewer", "data_fetcher": "Data agent", "data-fetcher": "Data agent",
  researcher: "Researcher", research: "Researcher",
  trainer: "Trainer", orchestrator: "Orchestrator",
};

function scrollChat() {
  const w = el(".chatwrap");
  if (w && w.scrollHeight > w.clientHeight) w.scrollTop = w.scrollHeight;
}

/** Light, safe formatting for agent prose. Never renders raw JSON. */
function richText(text) {
  const inline = (s) => esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  let out = "", list = false;
  String(text || "").split("\n").forEach((raw) => {
    const m = raw.match(/^\s*[-*•]\s+(.*)$/);
    if (m) {
      if (!list) { out += "<ul>"; list = true; }
      out += "<li>" + inline(m[1]) + "</li>";
      return;
    }
    if (list) { out += "</ul>"; list = false; }
    if (raw.trim()) out += "<p>" + inline(raw) + "</p>";
  });
  if (list) out += "</ul>";
  return out || "<p></p>";
}

function addMsg(who, text, agent, cardHTML) {
  const thread = $("chatThread");
  if (!thread) return null;
  $("chatHero").style.display = "none";
  const wrap = document.createElement("div");
  if (who === "user") {
    wrap.className = "msg u";
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    wrap.appendChild(b);
  } else if (who === "sys") {
    wrap.className = "msg sys";
    wrap.innerHTML = '<div class="sysline">' + esc(text) + "</div>";
  } else {
    wrap.className = "msg a";
    const b = document.createElement("div");
    b.className = "bubble";
    b.innerHTML = '<div class="who"><i aria-hidden="true">◈</i>' + esc(AGENTS[agent] || "Agent") + "</div>" +
      '<div class="txt">' + richText(text) + "</div>" + (cardHTML || "");
    wrap.appendChild(b);
  }
  thread.appendChild(wrap);
  scrollChat();
  return wrap;
}

/** A compact dataset fact-sheet appended to agent messages when the user asks
 *  about the data / rows / features — built from the backend's real summary. */
function datasetFactsCard(d) {
  if (!d) return "";
  const cols = (d.stats ? Object.keys(d.stats) : d.columns || []);
  const target = d.target || null;
  const prov = d.provenance || {};
  const rows = d.preview && d.preview.rows;
  return '<div class="msgcard">' +
    '<div class="msgcard-h"><i aria-hidden="true">▤</i>' + esc(nf(rows || 0)) + " rows · " +
    esc(nf(cols.length)) + " columns</div><dl>" +
    cols.slice(0, 30).map((c) => {
      const st = d.stats && d.stats[c];
      const dt = (st && st.type) || (d.dtypes && d.dtypes[c]) || "—";
      let detail = "";
      if (st && ("min" in st)) detail = "min " + esc(sv(st.min)) + " · max " + esc(sv(st.max)) +
        " · mean " + esc(sv(st.mean)) + (st.nulls ? " · " + esc(nf(st.nulls)) + " nulls" : "");
      else if (st && st.top && st.top.length) detail = "top: " + st.top.map((v) => esc(String(v))).join(", ");
      if (st && st.nulls) detail += (detail ? " · " : "") + esc(nf(st.nulls)) + " null";
      return "<div><dt>" + esc(c) + (c === target ? " Ⓣ" : "") + "</dt><dd>" +
        '<span class="hint">' + esc(dt) + "</span> " + (detail || "—") + "</dd></div>";
    }).join("") +
    (prov.source
      ? '<div class="msgcard-ft"><span class="hint">source ' + esc(prov.source) +
        (prov.path ? " · " + esc(prov.path) : "") +
        (prov.imported_snapshot_of ? " · imported from " + esc(prov.imported_snapshot_of) : "") + "</span></div>"
      : "") +
    '</dl><div class="btnrow">' +
    '<button class="btn sm" data-goto="data" type="button"><i aria-hidden="true">▤</i> Inspect dataset</button>' +
    "</div></div>";
}

function sv(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (Math.abs(v) >= 1e5 || (Math.abs(v) > 0 && Math.abs(v) < 1e-4)) return v.toExponential(3);
    return Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : String(Math.round(v * 1e4) / 1e4);
  }
  return String(v);
}

/** Fallback "declared types only" bars for a column list + dtype map. */
function statsFallback(list, dtypes) {
  const names = (list || []).slice(0, 24);
  if (!names.length) return emptyState("⌗", "No columns", "The backend reported no columns.");
  const total = names.length;
  return '<div class="hbars">' + names.map((c) => {
    const t = String((dtypes || {})[c] || "");
    const numish = /INT|DOUBLE|FLOAT|DECIMAL|REAL/.test(t);
    let wpc = 0;
    names.forEach((x) => {
      if (/INT|DOUBLE|FLOAT|DECIMAL|REAL/.test(String((dtypes || {})[x] || "")) === numish) wpc++;
    });
    return '<div class="hbar"><span class="hn mono" title="' + esc(c) + '">' + esc(c) + "</span>" +
      '<span class="ht"><i style="width:' + ((wpc / total) * 100).toFixed(1) + '%"></i></span>' +
      '<span class="hv">' + esc(t || "—") + "</span></div>";
  }).join("") + "</div>";
}

function datasetCard(d) {
  if (!d) return "";
  const cols = (d.columns || []).length;
  return '<div class="msgcard">' +
    '<div class="msgcard-h"><i aria-hidden="true">▤</i>Dataset analysed</div><dl>' +
    "<div><dt>Rows</dt><dd>" + esc(nf(d.rows)) + "</dd></div>" +
    "<div><dt>Columns</dt><dd>" + esc(nf(cols)) + "</dd></div>" +
    "<div><dt>Source</dt><dd>" + esc(d.source || "—") + "</dd></div>" +
    (d.target_column ? "<div><dt>Target</dt><dd>" + esc(d.target_column) + "</dd></div>" : "") +
    (d.task_type ? "<div><dt>Task</dt><dd>" + esc(titleCase(d.task_type)) + "</dd></div>" : "") +
    '</dl><div class="btnrow">' +
    '<button class="btn sm" data-goto="data" type="button"><i aria-hidden="true">▤</i> Inspect dataset</button>' +
    "</div></div>";
}

/** The AI task card: the user reviews, edits, starts, or cancels. Never forced. */
function taskCard(cfg) {
  const c = cfg || {};
  const hit = SOURCES.find((s) => s[0] === c.dataset_source);
  const src = hit ? hit[1] : "auto-selected";
  return '<div class="taskcard" data-taskcard>' +
    '<div class="taskcard-h"><i aria-hidden="true">◈</i>AI proposes</div>' +
    '<div class="taskcard-b"><h4>Train ' + esc(modelName(c.model_hint || "logistic_regression")) + " on " + esc(src.toLowerCase()) + " data</h4>" +
    '<div class="msgcard" style="border:none;padding:0;background:none;margin:0">' +
    "<dl>" +
    "<div><dt>Target</dt><dd>" + esc(c.target_column || "auto-detect") + "</dd></div>" +
    "<div><dt>Metric</dt><dd>" + esc(c.metric || "agent will choose") + "</dd></div>" +
    "<div><dt>Row limit</dt><dd>" + esc(c.max_rows ? nf(c.max_rows) : "agent will choose") + "</dd></div>" +
    "<div><dt>Iterations</dt><dd>" + esc(c.max_iterations ? nf(c.max_iterations) : "agent will choose") + "</dd></div>" +
    "</dl></div></div>" +
    '<div class="taskcard-f">' +
    '<button class="btn sm" data-task="edit" type="button"><i aria-hidden="true">✎</i> Edit</button>' +
    '<button class="btn pri sm" data-task="start" type="button"><i aria-hidden="true">▶</i> Start</button>' +
    '<span class="sp" style="flex:1"></span>' +
    '<button class="xbtn" data-task="cancel" type="button" aria-label="Cancel this proposed run" title="Cancel">✕</button>' +
    "</div></div>";
}

function wireTaskCard(root) {
  els("[data-taskcard]", root).forEach((c) => {
    const b = (k) => el('[data-task="' + k + '"]', c);
    const cancel = b("cancel");
    if (cancel) cancel.onclick = () => {
      c.remove();
      addMsg("sys", "Proposed run cancelled. Nothing was started.");
    };
    const edit = b("edit");
    if (edit) edit.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/lab"; };
    const start = b("start");
    if (start) start.onclick = () => startRun();
  });
  els("[data-goto]", root).forEach((b) => {
    b.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/" + b.dataset.goto; };
  });
}

function renderSugs(list) {
  ["chatSugs", "chatSugs2"].forEach((id) => {
    const box = $(id);
    if (!box) return;
    box.innerHTML = "";
    (list || []).forEach((s) => {
      const b = document.createElement("button");
      b.className = "sug" + (s[2] ? " go" : "");
      b.type = "button";
      b.textContent = s[0];
      b.onclick = () => chatSend(s[1]);
      box.appendChild(b);
    });
  });
}

function suggestFor(agentText) {
  const t = String(agentText || "").toLowerCase();
  let out = [];
  if (/shall i proceed|reply.*start|proceed|ready to start/.test(t)) out = [["Start the run", "yes", 1]];
  else if (/resum|paused|stopped/.test(t)) out = [["Resume the run", "resume the run"], ["Start over", "start over"]];
  else if (/metric|train towards/.test(t)) out = [["accuracy 0.90", "accuracy >= 0.90"], ["f1 0.85", "f1 >= 0.85"], ["RMSE 1.2", "rmse <= 1.2"], ["auto", "auto"]];
  else if (/how many rows|row cap|cap/.test(t)) out = [["5k rows", "5000 rows"], ["50k rows", "50000 rows"], ["auto", "auto"]];
  else if (/which column|predict\?|target column|detect/.test(t)) out = [["auto-detect", "auto"]];
  else if (/data source|where.*data|dataset/.test(t)) out = [["Synthetic demo", "use the synthetic demo dataset"], ["I'll upload a file", "I will upload a file"], ["OpenML", "search openml"]];
  else if (/what.*(task|build|goal)|describe your/.test(t)) out = [
    ["Classify demo data", "Classify the demo synthetic dataset"],
    ["Predict house prices", "Predict house prices from property data"]];
  renderSugs(out);
}

PAGES.chat = async function () {
  renderSugs([
    ["Classify the demo dataset", "Classify the demo synthetic dataset"],
    ["Predict churn at 90% accuracy", "Predict churn at accuracy >= 0.90"],
  ]);
  await loadSessions();
  const thread = $("chatThread");
  thread.innerHTML = "";
  if (!sel.sessionId) await newSession();
  if (!sel.sessionId) { $("chatHero").style.display = ""; return; }
  try {
    const msgs = (await api("/sessions/" + sel.sessionId + "/messages?limit=200")).messages || [];
    if (!msgs.length) {
      /* fresh session: the centred hero IS the empty state — no bubble needed */
      $("chatHero").style.display = "";
    } else {
      $("chatHero").style.display = "none";
      msgs.forEach((m) => {
        if (m.role === "user") addMsg("user", m.text);
        else if (m.role === "agent") addMsg("agent", m.text, m.agent || "interviewer");
        else addMsg("sys", m.text);
      });
    }
  } catch (e) {
    $("chatHero").style.display = "";
    addMsg("sys", "Could not load history: " + e.message);
  }
  scrollChat();
};

async function loadSessions() {
  const picker = $("sessionSel");
  if (!sel.projectId) return;
  let items = [];
  try { items = (await api("/sessions?project_id=" + sel.projectId + "&page=1&page_size=100")).items || []; }
  catch (e) { return; }
  if (sel.sessionId && !items.some((s) => s.id === sel.sessionId)) sel.sessionId = null;
  if (!sel.sessionId && items.length) sel.sessionId = items[0].id;
  picker.innerHTML = items.length
    ? items.map((s) => '<option value="' + esc(s.id) + '"' + (s.id === sel.sessionId ? " selected" : "") + ">" + esc(s.name || "Session") + "</option>").join("")
    : '<option value="">No sessions</option>';
}

async function newSession() {
  try {
    /* project_id is a required query parameter here; the body only takes name */
    const s = await api("/sessions?project_id=" + encodeURIComponent(sel.projectId), {
      method: "POST",
      body: JSON.stringify({
        name: "Session " + new Date().toISOString().slice(5, 16).replace("T", " "),
      }),
    });
    sel.sessionId = s.id;
    await loadSessions();
    return s;
  } catch (e) {
    toast("Could not create a session: " + e.message, "err");
  }
}

function sendWS(obj) {
  if (window.ws && window.ws.readyState === 1) { window.ws.send(JSON.stringify(obj)); return true; }
  return false;
}

function chatSend(text) {
  const t = String(text || "").trim();
  if (!t) return;
  if (!sel.projectId) { toast("Select a project first.", "err"); return; }
  addMsg("user", t);
  if (!sendWS({ event: "chat", message: t, project_id: sel.projectId, session_id: sel.sessionId })) {
    addMsg("sys", "Not connected yet — retrying…");
  }
  $("chatInput").value = "";
  autoGrow();
}

function autoGrow() {
  const t = $("chatInput");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 190) + "px";
}

/** Start a campaign over the WebSocket, using only keys the backend accepts. */
async function startRun() {
  if (!sel.projectId) { toast("Select a project first.", "err"); return; }
  const cfg = {};
  if (runCfg.goal) cfg.goal = runCfg.goal;
  if (runCfg.metric) cfg.metric = runCfg.metric;
  if (runCfg.dataset_source) cfg.dataset_source = runCfg.dataset_source;
  if (runCfg.dataset_query) cfg.dataset_query = runCfg.dataset_query;
  if (runCfg.dataset_path) cfg.dataset_path = runCfg.dataset_path;
  if (runCfg.max_rows) cfg.max_rows = Number(runCfg.max_rows);
  if (runCfg.target_column) cfg.target_column = runCfg.target_column;
  if (runCfg.max_iterations) cfg.max_iterations = Number(runCfg.max_iterations);
  if (runCfg.feature_engineering_every_n) cfg.feature_engineering_every_n = Number(runCfg.feature_engineering_every_n);
  if (runCfg.model_hint) cfg.model_hint = runCfg.model_hint;
  if (!cfg.goal) cfg.chat_mode = true;
  if (!sendWS({ event: "start", project_id: sel.projectId, session_id: sel.sessionId, config: cfg })) {
    toast("Not connected to the agent yet — retrying…", "err");
    return;
  }
  toast("Run started.", "ok", "You can navigate away — it continues in the background.");
  cfgDirty = false;
  paintSaveState();
  location.hash = "#/projects/" + sel.projectId + "/monitor";
}

/* ------------------------------------------------- small agent-facing dialogs */

function metricDialog() {
  const cur = String(runCfg.metric || "").split(/\s*(>=|<=|>|<|=)\s*/);
  modal("Choose metric", `
    <div class="field">
      <label class="lbl" for="mdMet">Metric</label>
      <select class="inp" id="mdMet">
        ${METRICS.map((m) => '<option value="' + m[0] + '"' + (cur[0] === m[0] ? " selected" : "") + ">" + esc(m[1]) + "</option>").join("")}
        <option value=""${cur[0] ? "" : " selected"}>Agent decides</option>
      </select>
    </div>
    <div class="field">
      <label class="lbl" for="mdVal">Target value</label>
      <input class="inp" id="mdVal" type="number" step="0.01" value="${esc(cur[1] || "")}" placeholder="0.90" />
      <div class="help">Score metrics use “at least”; error metrics use “at most”.</div>
    </div>`, [
    { label: "Cancel", kind: "ghost", value: null },
    { label: "Save", kind: "pri", value: "go" },
  ]).then((r) => {
    if (r !== "go") return;
    const m = $("mdMet").value, v = $("mdVal").value;
    setCfg("metric", m ? m + (MINIMIZE.has(m) ? " <= " : " >= ") + (v || "0.85") : "");
    toast("Metric preference saved.", "ok");
  });
}

function constraintDialog() {
  modal("Add a constraint", `
    <p>Constraints are passed to the agent as part of your goal.</p>
    <div class="field">
      <label class="lbl" for="cdTxt">Constraint</label>
      <textarea class="inp" id="cdTxt" placeholder="Prefer interpretable models. Keep each run under two minutes."></textarea>
    </div>`, [
    { label: "Cancel", kind: "ghost", value: null },
    { label: "Save", kind: "pri", value: "go" },
  ]).then((r) => {
    if (r !== "go") return;
    const v = $("cdTxt").value.trim();
    if (!v) return;
    setCfg("goal", ((runCfg.goal || "") + " " + v).trim());
    toast("Constraint saved.", "ok");
  });
}

async function uploadDialog() {
  const r = await modal("Add dataset", `
    <p>Upload a CSV or similar file. It is registered to this project and staged for the next run.</p>
    <div class="field">
      <label class="lbl" for="upFile">File</label>
      <input class="inp" id="upFile" type="file" accept=".csv,.tsv,.txt,.parquet,.json" />
      <div class="help">Staging a file tells the data agent to use it instead of generating data.</div>
    </div>`, [
    { label: "Cancel", kind: "ghost", value: null },
    { label: "Upload", kind: "pri", value: "go" },
  ]);
  if (r !== "go") return;
  const f = $("upFile").files && $("upFile").files[0];
  if (!f) { toast("Choose a file first.", "err"); return; }
  const fd = new FormData();
  fd.append("file", f);
  try {
    await api("/automl/upload", { method: "POST", body: fd, headers: {} });
    setCfg("dataset_source", "upload");
    toast("File staged for the data agent.", "ok", f.name);
    if (sel.page === "data") PAGES.data();
  } catch (e) {
    toast("Upload failed: " + e.message, "err");
  }
}

/* ------------------------------------------------------------- data pages */

let DS_ROWS = [];

PAGES.data = async function () {
  const host = $("dataBody");
  host.innerHTML = skeleton(3, "Preparing dataset registry…");

  const rows = [];
  let registered = [];
  try { registered = (await api("/datasets?project_id=" + sel.projectId + "&page=1&page_size=100")).items || []; }
  catch (e) { /* none registered */ }
  registered.forEach((d) => rows.push({
    kind: "registered", id: d.id, name: d.name,
    version: d.version || "—",
    format: String(d.filename || d.name || "").split(".").pop().toUpperCase(),
    rows: d.row_count !== undefined ? d.row_count : d.rows,
    cols: d.column_count !== undefined ? d.column_count : (d.columns || []).length,
    source: "upload", updated: d.created_at, status: d.status || "ready",
  }));

  let live = null;
  try { live = await api("/automl/dataset-preview?limit=5"); } catch (e) {}
  if (live && (live.columns || []).length) {
    const ds = currentDataset();
    rows.push({
      kind: "live", id: "live", name: "Working table", version: "live",
      format: "duckdb", rows: live.rows, cols: (live.columns || []).length,
      source: ds.source || "working table", updated: null, status: "ready",
    });
  }
  DS_ROWS = rows;

  if (!rows.length) {
    host.innerHTML = emptyState("▤", "No datasets",
      "Add a dataset to begin — a CSV you upload, or let the agent generate one.",
      '<div class="btnrow">' +
      '<button class="btn pri" data-act="add" type="button"><i aria-hidden="true">＋</i> Add dataset</button>' +
      '<button class="btn" data-act="synth" type="button"><i aria-hidden="true">◈</i> Use synthetic demo</button>' +
      "</div>");
    el('[data-act="add"]', host).onclick = uploadDialog;
    el('[data-act="synth"]', host).onclick = () => {
      setCfg("dataset_source", "synthetic");
      toast("Synthetic demo selected.", "ok", "Start a run to generate it.");
    };
    return;
  }

  const draw = () => {
    const q = $("dataSearch").value.trim().toLowerCase();
    const src = $("dataFilter").value;
    const list = rows.filter((r) => (!q || String(r.name).toLowerCase().indexOf(q) >= 0) && (!src || r.source === src));
    const srcs = [];
    rows.forEach((r) => { if (srcs.indexOf(r.source) < 0) srcs.push(r.source); });
    $("dataFilter").innerHTML = '<option value="">All sources</option>' +
      srcs.map((s) => '<option value="' + esc(s) + '"' + (src === s ? " selected" : "") + ">" + esc(titleCase(s)) + "</option>").join("");
    if (!list.length) {
      host.innerHTML = emptyState("⌕", "No matches", "No dataset matches that search.");
      return;
    }
    host.innerHTML = '<div class="card flush"><div class="tbl-scroll"><table class="tbl">' +
      "<thead><tr><th>Dataset</th><th>Version</th><th>Rows</th><th>Columns</th><th>Format</th><th>Source</th><th>Updated</th><th>Status</th><th></th></tr></thead><tbody>" +
      list.map((r) => '<tr class="clickable" data-ds="' + esc(r.id) + '">' +
        "<td><b>" + esc(r.name) + "</b></td>" +
        '<td class="muted">' + esc(r.version) + "</td>" +
        '<td class="num">' + esc(nf(r.rows)) + "</td>" +
        '<td class="num">' + esc(nf(r.cols)) + "</td>" +
        '<td class="muted">' + esc(r.format || "—") + "</td>" +
        '<td class="muted">' + esc(titleCase(r.source)) + "</td>" +
        '<td class="muted">' + esc(r.updated ? ago(r.updated) : "—") + "</td>" +
        "<td>" + statusBadge(r.status) + "</td>" +
        '<td><div class="tbl-act">' +
        '<button class="btn ghost sm" data-open="' + esc(r.id) + '" type="button">Open</button>' +
        '<button class="btn ghost sm" data-use="' + esc(r.id) + '" type="button">Use</button>' +
        "</div></td></tr>").join("") +
      "</tbody></table></div></div>";
    els("tr[data-ds]", host).forEach((tr) => {
      tr.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/datasets/" + encodeURIComponent(tr.dataset.ds); };
    });
    els("[data-open]", host).forEach((b) => {
      b.onclick = (e) => {
        e.stopPropagation();
        location.hash = "#/projects/" + sel.projectId + "/datasets/" + encodeURIComponent(b.dataset.open);
      };
    });
    els("[data-use]", host).forEach((b) => {
      b.onclick = (e) => {
        e.stopPropagation();
        const r = rows.find((x) => x.id === b.dataset.use);
        setCfg("dataset_source", "upload");
        toast("Staged for the next run.", "ok", r ? r.name : "");
      };
    });
  };
  $("dataSearch").oninput = draw;
  $("dataFilter").onchange = draw;
  draw();
};

PAGES.dataset = async function () {
  const id = sel.datasetId;
  const host = $("dsBody");
  $("dsName").textContent = "Dataset";
  $("dsMeta").textContent = "";
  $("dsActions").innerHTML = "";
  $("dsTabs").innerHTML = "";
  host.innerHTML = skeleton(4, "Preparing dataset…");

  let preview = null, record = null, runs = [], details = null;
  try { preview = await api("/automl/dataset-preview?limit=50"); } catch (e) {}
  try { details = await api("/automl/dataset-details?project_id=" + encodeURIComponent(sel.projectId)); } catch (e) {}
  try { runs = (await api("/runs?project_id=" + sel.projectId + "&limit=50")).runs || []; } catch (e) {}
  if (id && id !== "live") { try { record = await api("/datasets/" + id); } catch (e) { record = null; } }
  await loadProjectContext();

  const colStats = (details && details.stats) || {};
  const prov = (details && details.provenance) || {};
  const isLive = id === "live" || !record;
  const meta = isLive ? (preview || {}) : (record || {});
  const cols = meta.columns || [];
  const liveDs = currentDataset();
  const target = liveDs.target_column || meta.target_column || (cache.ctx && cache.ctx.target) || null;

  $("dsName").textContent = isLive ? "Working table" : (record ? record.name : "Dataset");
  $("dsMeta").textContent = [
    nf(meta.rows !== undefined ? meta.rows : record && record.row_count) + " rows",
    nf(cols.length || (record && record.column_count)) + " columns",
    record && record.filename ? record.filename : (isLive ? "live table in the working store" : ""),
  ].filter(Boolean).join(" · ");

  $("dsActions").innerHTML =
    '<button class="btn pri" id="dsUse" type="button"><i aria-hidden="true">▶</i> Use in experiment</button>' +
    '<button class="btn" id="dsExport" type="button"><i aria-hidden="true">↓</i> Export</button>' +
    '<button class="iconbtn" id="dsMore" type="button" aria-label="More dataset actions" title="More">⋯</button>';
  $("dsUse").onclick = () => {
    if (!isLive) setCfg("dataset_source", "upload");
    location.hash = "#/projects/" + sel.projectId + "/lab";
  };
  $("dsExport").onclick = () => {
    downloadText((isLive ? "working-table" : (record && record.name) || "dataset") + ".json",
      JSON.stringify(meta, null, 2), "application/json");
  };
  $("dsMore").onclick = async () => {
    const actions = [
      { label: "Close", kind: "ghost", value: null },
      { label: "View raw response", kind: "ghost", value: "json" },
    ];
    if (record) actions.push({ label: "Delete dataset", kind: "danger", value: "del" });
    const v = await modal("Dataset actions", "<p>Choose an action for this dataset.</p>", actions);
    if (v === "json") openDrawer("Dataset", "Raw backend response", pre(JSON.stringify(meta, null, 2)));
    else if (v === "del" && record) {
      const ok = await confirmModal("Delete dataset",
        "<p>Permanently delete <b>" + esc(record.name) + "</b>? This cannot be undone.</p>", "Delete", true);
      if (!ok) return;
      try { await del("/datasets/" + record.id); toast("Deleted.", "ok"); location.hash = "#/projects/" + sel.projectId + "/data"; }
      catch (e) { toast("Delete failed: " + e.message, "err"); }
    }
  };

  const TABS = [
    ["overview", "Overview"], ["preview", "Preview"], ["columns", "Columns"],
    ["quality", "Quality"], ["stats", "Statistics"], ["versions", "Versions"],
  ];
  $("dsTabs").innerHTML = TABS.map((t, i) =>
    '<button class="tab' + (i === 0 ? " on" : "") + '" data-tab="' + t[0] + '" type="button" role="tab">' + esc(t[1]) + "</button>").join("");

  const dtypes = meta.dtypes || {};
  const rowCount = meta.rows !== undefined ? meta.rows : record && record.row_count;
  const colCount = cols.length || (record && record.column_count);

  const panes = {
    overview:
      '<div class="kpis">' +
      kpi("Rows", nf(rowCount), "▤") +
      kpi("Columns", nf(colCount), "⌗") +
      kpi("Target", target || "not set", "◎") +
      kpi("Task", titleCase(liveDs.task_type || meta.task_type || "—"), "◈") +
      "</div>" +
      '<div class="sect" style="margin-top:var(--s6)">' + sectHead("Source") +
      '<div class="kv">' +
      kv("Source", titleCase(liveDs.source || prov.source || meta.source || (record && record.source) || "—")) +
      kv("Query", meta.query || (record && record.query) || "—") +
      kv("Path", prov.path || meta.path || (record && record.path) || "—") +
      (prov.imported_snapshot_of
        ? kv("Imported from run", prov.imported_snapshot_of) : "") +
      kv("Registered", record ? ago(record.created_at) : "not a registered file") +
      kv("Storage", isLive ? "single-source working table (DuckDB)" : "project storage") +
      "</div></div>" +
      '<div class="sect">' + sectHead("Column statistics", '<span class="hint">computed from the live table</span>') +
      (Object.keys(colStats).length
        ? '<div class="grid2">' + cols.slice(0, 32).map((c) => {
            const st = colStats[c] || {};
            let stats = "";
            if ("min" in st) {
              stats = "range <b>" + esc(sv(st.min)) + " – " + esc(sv(st.max)) + "</b>" +
                " · mean <b>" + esc(sv(st.mean)) + "</b>" +
                (st.std !== undefined ? " · σ " + esc(sv(st.std)) : "");
            } else if (st.top && st.top.length) {
              stats = "top <b>" + st.top.map((v) => esc(String(v))).join(", ") + "</b>";
            } else {
              stats = "—";
            }
            return '<div class="card mini"><div class="card-t"><h3 class="mono">' + esc(c) +
              (c === target ? ' <span class="badge acc">target</span>' : "") +
              '</h3><span class="sp"></span><span class="hint">' + esc(st.type || "—") + "</span></div>" +
              '<p class="muted" style="margin:var(--s2) 0 0">' + stats + "</p>" +
              '<p class="hint" style="margin:var(--s2) 0 0">' +
              esc(nf(st.nulls || 0)) + " null" + (st.nulls ? " (" + esc(st.null_pct) + "%)" : "") +
              " · " + esc(nf(st.uniques || 0)) + " unique</p></div>";
          }).join("") + "</div>"
        : emptyState("◔", "No live statistics", "Start a run (or import a dataset snapshot) so the backend records a working table.")) +
      "</div>" +
      '<details class="disc"><summary><span class="chev">▸</span> What the backend does not measure' +
      '<span class="dspacer"></span><span class="hint">transparency</span></summary>' +
      '<div class="dbody"><p class="muted" style="margin:0">Missing-value counts, duplicate counts, constant-column ' +
      "detection and leakage analysis are not computed by the current backend, so they are not shown here. " +
      "Nothing is estimated.</p></div></details>",

    preview: (preview && preview.rows && preview.rows.length)
      ? '<div class="card flush"><div class="tbl-scroll"><table class="tbl"><thead><tr>' +
        (preview.columns || []).map((c) => "<th>" + esc(c) + "</th>").join("") + "</tr></thead><tbody>" +
        preview.rows.map((r) => "<tr>" + (preview.columns || []).map((c) => {
          const v = r[c];
          const isNum = typeof v === "number";
          return '<td class="' + (isNum ? "num" : "") + '">' +
            esc(v === null || v === undefined ? "—" : String(v).slice(0, 60)) + "</td>";
        }).join("") + "</tr>").join("") +
        "</tbody></table></div></div>" +
        '<p class="hint" style="margin-top:var(--s3)">Showing ' + esc(nf(preview.rows.length)) + " of " +
        esc(nf(preview.rows)) + " rows. The backend caps previews at 100.</p>"
      : emptyState("▤", "No preview available", "The backend returned no rows for this dataset."),

    columns: cols.length
      ? '<div class="card flush"><div class="tbl-scroll"><table class="tbl">' +
        "<thead><tr><th>Column</th><th>Type</th><th>Role</th><th>Details (from live table)</th><th></th></tr></thead><tbody>" +
        cols.map((c) => {
          const st = colStats[c] || {};
          let det = "—";
          if ("min" in st) det = "min " + sv(st.min) + " · max " + sv(st.max) +
            " · mean " + sv(st.mean) + (st.nulls ? " · " + esc(nf(st.nulls)) + " nulls" : "");
          else if (st.top && st.top.length) det = "top " + st.top.map((v) => esc(String(v))).join(", ") +
            (st.nulls ? " · " + esc(nf(st.nulls)) + " nulls" : "");
          return "<tr><td class=\"mono\">" + esc(c) + '</td><td class="muted">' + esc(dtypes[c] || "—") +
            "</td><td>" + (c === target ? '<span class="badge acc">target</span>' : '<span class="hint">feature</span>') +
            '</td><td class="hint">' + det + '</td><td><div class="tbl-act">' +
            (c === target ? "" : '<button class="btn ghost sm" data-tgt="' + esc(c) + '" type="button">Set as target</button>') +
            "</div></td></tr>";
        }).join("") + "</tbody></table></div></div>" +
        '<p class="hint" style="margin-top:var(--s3)">Statistics are computed from the live working table. ' +
        'Setting a target here updates the next run and is remembered in project memory.</p>'
      : emptyState("⌗", "No columns", "The backend reported no column list for this dataset."),

    quality: (function () {
      const warn = (lastState.warnings || []).map((w) => (w && w.message) || String(w));
      const rep = cache.report && cache.report.data_quality;
      const dw = rep && rep.warnings ? rep.warnings : [];
      const all = warn.concat(dw);
      return sectHead("Data quality") +
        (all.length
          ? '<ul class="speclist">' + all.map((x) => "<li><span>" + esc(x) + "</span></li>").join("") + "</ul>"
          : '<div class="card"><p class="muted" style="margin:0">No quality warnings were reported for this dataset.</p></div>') +
        (rep && rep.note ? '<p class="hint" style="margin-top:var(--s3)">' + esc(rep.note) + "</p>" : "");
    })(),

    stats: cols.length
      ? (Object.keys(colStats).length
          ? sectHead("Per-column statistics", '<span class="hint">real, from the live table</span>') +
            '<div class="card flush"><div class="tbl-scroll"><table class="tbl">' +
            "<thead><tr><th>Column</th><th>Type</th><th>Range / top values</th><th>Nulls</th><th>Unique</th></tr></thead><tbody>" +
            cols.slice(0, 40).map((c) => {
              const st = colStats[c] || {};
              const t = String(st.type || dtypes[c] || "");
              const range = ("min" in st)
                ? '<span class="hint">min ' + esc(sv(st.min)) + " · max " + esc(sv(st.max)) +
                  " · mean " + esc(sv(st.mean)) + (st.std !== undefined ? " · σ " + esc(sv(st.std)) : "") + "</span>"
                : '<span class="hint">' + (st.top && st.top.length
                    ? "top " + st.top.map((v) => esc(String(v))).join(", ")
                    : "—") + "</span>";
              return "<tr><td class=\"mono\">" + esc(c) +
                (c === target ? ' <span class="badge acc">target</span>' : "") +
                '</td><td class="muted">' + esc(t) + "</td><td>" + range +
                '</td><td class="num">' + esc(nf(st.nulls || 0)) + " (" + esc(st.null_pct || 0) + "%)</td><td class=\"num\">" +
                esc(nf(st.uniques || 0)) + "</td></tr>";
            }).join("") + "</tbody></table></div></div>" +
            '<p class="hint" style="margin-top:var(--s3)">Every value is computed from the actual rows — no estimates.</p>'
          : sectHead("Declared types") + statsFallback(cols, dtypes) +
            '<p class="hint" style="margin-top:var(--s3)">Declared types only — the backend has not computed live statistics yet.</p>')
      : emptyState("◑", "No statistics", "Run a campaign so the backend records a working table."),

    versions: (function () {
      const snaps = runs.map((r) => ({
        run: r.run_id, when: r.completed_at || r.started_at,
        rows: r.dataset && r.dataset.rows, target: r.dataset && r.dataset.target_column,
        model: r.best && r.best.model,
      })).filter((s) => s.rows);
      if (!snaps.length) return emptyState("⑂", "No versions", "This project has no run history yet.");
      return '<p class="hint" style="margin-bottom:var(--s4)">Each run records the exact table it trained on, ' +
        "so every experiment is reproducible against a specific dataset state.</p>" +
        '<div class="lineage">' + snaps.map((s, i) =>
          (i ? '<div class="larrow">↓</div>' : "") +
          '<div class="lnode"><span class="li" aria-hidden="true">▤</span>' +
          "<span>" + esc(nf(s.rows)) + ' rows · target <b>' + esc(s.target || "—") + "</b></span>" +
          '<span class="sp"></span><span class="lv">' + esc(s.run) + "</span>" +
          '<button class="btn ghost sm" data-ver="' + esc(s.run) + '" type="button">Inspect</button></div>').join("") +
        "</div>";
    })(),
  };

  host.innerHTML = TABS.map((t) =>
    '<div class="tabpane' + (t[0] === "overview" ? " on" : "") + '" data-pane="' + t[0] + '">' + panes[t[0]] + "</div>").join("");
  els("[data-tab]", $("dsTabs")).forEach((b) => {
    b.onclick = () => {
      els("[data-tab]", $("dsTabs")).forEach((x) => x.classList.toggle("on", x === b));
      els("[data-pane]", host).forEach((p) => p.classList.toggle("on", p.dataset.pane === b.dataset.tab));
    };
  });
  els("[data-ver]", host).forEach((b) => {
    b.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/runs/" + b.dataset.ver; };
  });
  els("[data-tgt]", host).forEach((b) => {
    b.onclick = async () => {
      try {
        await post("/automl/target", { column: b.dataset.tgt, project_id: sel.projectId });
        setCfg("target_column", b.dataset.tgt);
        toast("Target updated.", "ok", b.dataset.tgt);
        PAGES.dataset();
      } catch (e) { toast("Could not set target: " + e.message, "err"); }
    };
  });
};
