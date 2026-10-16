/* ==========================================================================
   AutoML Workspace — Model Lab, Experiments, Run inspector, Live Run monitor
   ========================================================================== */

/* ------------------------------------------------------------- model lab */

PAGES.lab = async function () {
  const host = $("labBody");
  host.innerHTML = skeleton(4, "Preparing Model Lab…");

  els("#modeSeg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === sel.mode));
  runCfg.mode = sel.mode;

  let preview = null, ctx = null, runs = [], board = null;
  try { preview = await api("/automl/dataset-preview?limit=1"); } catch (e) {}
  try { ctx = await api("/projects/" + sel.projectId + "/context"); cache.ctx = ctx; } catch (e) {}
  try { runs = (await api("/runs?project_id=" + sel.projectId + "&limit=50")).runs || []; } catch (e) {}
  try { board = await api("/automl/leaderboard"); } catch (e) {}

  const used = [];
  runs.forEach((r) => { if (r.best && r.best.model && used.indexOf(r.best.model) < 0) used.push(r.best.model); });
  const cols = (preview && preview.columns) || [];
  const liveDs = currentDataset();
  const target = liveDs.target_column || (ctx && ctx.target) || "";
  const req = (lastState && lastState.requirements) || {};
  const mparts = String(runCfg.metric || req.metric_expr || "").split(/\s*(>=|<=|>|<|=)\s*/);
  const mName = mparts[0] || req.metric || "accuracy";
  const mVal = mparts[1] || (req.target_value !== undefined ? req.target_value : "");

  const HINT = {
    ai: "The agent manages the workflow and fills in anything you leave blank.",
    manual: "You control the configuration. Leave a field blank to use the backend default.",
    hybrid: "You set the major choices — model, metric, data — and the agent tunes the rest.",
  }[sel.mode];

  /* The trainer previously stored some metrics as 0.0 (roc_auc for binary, and
     therefore target_metric when that was the target). The run page reconciles
     saved predictions; here, if the stored target metric is 0 but a real value
     exists for the campaign's metric, surface it with an honest note. */
  const lbMain = () => {
    const b = board.best;
    const stored = Number(b.target_metric);
    if (isFinite(stored) && stored !== 0) return { v: num(b.target_metric), s: "" };
    const key = String((mName || "accuracy") === "target_metric" ? "accuracy" : (mName || "accuracy")).toLowerCase();
    const real = b.metrics && Number(b.metrics[key]);
    if (isFinite(real) && real !== 0)
      return { v: num(real), s: "stored " + num(stored) + " · shows " + metricLabel(key) };
    return { v: num(b.target_metric), s: "" };
  };

  host.innerHTML =
    (board && board.best
      ? '<div class="sect">' + sectHead("Current campaign leaderboard",
          '<span class="hint">best model of the live campaign</span>') +
        '<div class="kpis">' +
        kpi("Model", modelName(board.best.model || "—"), "⚙") +
        kpi("Iteration", nf(board.best.iteration), "◷") +
        kpi("Main metric", lbMain().v, "◎", lbMain().s) +
        kpi("Train time", dur(board.best.training_time_seconds), "◷") +
        "</div>" +
        '<div class="btnrow" style="margin-top:var(--s4)">' +
        (board.run_context && board.run_context.run_id
          ? '<button class="btn ghost sm" id="lbOpen" type="button"><i aria-hidden="true">◈</i> Open experiment</button>' +
            '<button class="btn ghost sm" id="lbRep" type="button"><i aria-hidden="true">✎</i> Model report</button>'
          : "") +
        "</div></div>"
      : "") +

    '<div class="sect">' + sectHead("Run configuration", '<span class="pill" id="saveChip">' + (cfgDirty ? "Unsaved" : "Saved") + "</span>") +
    '<div class="grid2">' +
    '<div class="card">' +
    '<div class="field"><label class="lbl" for="labSrc">Dataset source</label><select class="inp" id="labSrc">' +
    SOURCES.map((s) => '<option value="' + s[0] + '"' + (runCfg.dataset_source === s[0] ? " selected" : "") + ">" + esc(s[1]) + "</option>").join("") +
    '<option value=""' + (runCfg.dataset_source ? "" : " selected") + ">Agent decides</option></select></div>" +
    '<div class="field"><label class="lbl" for="labTarget">Target column</label><select class="inp" id="labTarget">' +
    '<option value="">Auto-detect</option>' +
    cols.map((c) => '<option value="' + esc(c) + '"' + (runCfg.target_column === c || target === c ? " selected" : "") + ">" + esc(c) + "</option>").join("") +
    '</select><div class="help">Options come from the working table the backend reported.</div></div>' +
    '<div class="field"><label class="lbl" for="labRows">Row limit <span class="opt">optional</span></label>' +
    '<input class="inp" id="labRows" type="number" min="100" step="1000" value="' + esc(runCfg.max_rows || "") +
    '" placeholder="' + esc(req.max_rows || 50000) + '" /></div>' +
    "</div>" +
    '<div class="card">' +
    '<div class="field"><label class="lbl" for="labMetric">Metric</label><select class="inp" id="labMetric">' +
    METRICS.map((m) => '<option value="' + m[0] + '"' + (mName === m[0] ? " selected" : "") + ">" + esc(m[1]) + "</option>").join("") +
    "</select></div>" +
    '<div class="field"><label class="lbl" for="labTarget2">Target value <span class="opt">optional</span></label>' +
    '<input class="inp" id="labTarget2" type="number" step="0.01" value="' + esc(mVal) + '" placeholder="0.85" />' +
    '<div class="help" id="labDir"></div></div>' +
    '<div class="field"><label class="lbl" for="labIters">Max iterations <span class="opt">optional</span></label>' +
    '<input class="inp" id="labIters" type="number" min="1" step="1" value="' + esc(runCfg.max_iterations || "") +
    '" placeholder="' + esc(req.max_iterations || "agent default") + '" /></div>' +
    "</div></div></div>" +

    '<div class="sect">' + sectHead("Available models", '<span class="hint">' + esc(HINT) + "</span>") +
    '<div class="grid-auto">' + MODELS.map((m) => {
      const on = runCfg.model_hint === m[0];
      return '<label class="check' + (on ? " on" : "") + '" data-model="' + esc(m[0]) + '">' +
        '<input type="checkbox"' + (on ? " checked" : "") + ' aria-label="Use ' + esc(m[1]) + '" />' +
        '<span><span class="cn">' + esc(m[1]) + '</span><br /><span class="cd">' + esc(m[2]) + "</span></span>" +
        '<span class="sp"></span>' +
        (used.indexOf(m[0]) >= 0 ? '<span class="badge ok">used</span>' : "") +
        "</label>";
    }).join("") + "</div>" +
    '<p class="hint" style="margin-top:var(--s3)">Selecting a model makes it train first; the campaign then ' +
    "continues through the backend's own candidate ladder. “used” marks models this project has actually trained.</p></div>" +

    '<details class="disc" id="labAdv"><summary><span class="chev">▸</span> Advanced' +
    '<span class="dspacer"></span><span class="hint">feature engineering, query, path, feature pipeline, research</span></summary>' +
    '<div class="dbody"><div class="grid2">' +
    '<div class="field"><label class="lbl" for="labFe">Feature engineering every N iterations <span class="opt">optional</span></label>' +
    '<input class="inp" id="labFe" type="number" min="1" step="1" value="' + esc(runCfg.feature_engineering_every_n || "") +
    '" placeholder="' + esc(req.feature_engineering_every_n || "agent default") + '" />' +
    '<div class="help">How often the researcher proposes new features.</div></div>' +
    '<div class="field"><label class="lbl" for="labQuery">Dataset query <span class="opt">optional</span></label>' +
    '<input class="inp" id="labQuery" value="' + esc(runCfg.dataset_query || "") + '" placeholder="e.g. churn in the last 90 days" /></div>' +
    "</div>" +
    '<div class="field"><label class="lbl" for="labPath">Dataset path <span class="opt">optional</span></label>' +
    '<input class="inp mono" id="labPath" value="' + esc(runCfg.dataset_path || "") + '" placeholder="/path/to/data.csv" /></div>' +
    "</div></details>" +

    '<div class="sect">' + sectHead("Pipeline control",
        '<span class="hint">inspect or steer what the agents are doing</span>') +
    '<div class="grid2">' +
    '<button class="card" id="labFeat" type="button" style="text-align:left;cursor:pointer;display:block;width:100%">' +
    '<div class="card-t"><h3><span aria-hidden="true">⌗</span> Feature pipeline</h3></div>' +
    '<p class="muted" style="margin:0;font-size:13.5px">See every engineered feature, its source paper, and enable, disable or remove it.</p></button>' +
    '<button class="card" id="labRes" type="button" style="text-align:left;cursor:pointer;display:block;width:100%">' +
    '<div class="card-t"><h3><span aria-hidden="true">⌕</span> Research</h3></div>' +
    '<p class="muted" style="margin:0;font-size:13.5px">The papers the researcher actually used, and the features each one produced.</p></button>' +
    "</div></div>" +

    '<div class="sect"><div class="btnrow">' +
    '<button class="btn pri lg" id="labStart" type="button"><i aria-hidden="true">▶</i> Run experiment</button>' +
    '<button class="btn" id="labReview" type="button"><i aria-hidden="true">◈</i> Review configuration</button>' +
    '<button class="btn ghost" id="labReset" type="button">Reset</button>' +
    '<button class="btn ghost" id="labSave" type="button">Save configuration</button>' +
    '<button class="btn ghost" id="labClose" type="button"><i aria-hidden="true">✕</i> Close</button>' +
    "</div>" +
    '<p class="hint" style="margin-top:var(--s3)">Closing returns to the overview without starting anything.</p></div>';

  const updDir = () => {
    const d = $("labDir");
    if (!d) return;
    d.textContent = MINIMIZE.has($("labMetric").value)
      ? "Error metric — the backend minimises it, so the target becomes “at most”."
      : "Score metric — the backend maximises it, so the target becomes “at least”.";
  };
  updDir();
  $("labMetric").onchange = updDir;

  const lbOpen = $("lbOpen");
  if (lbOpen) lbOpen.onclick = () => {
    location.hash = "#/projects/" + sel.projectId + "/runs/" + board.run_context.run_id;
  };
  const lbRep = $("lbRep");
  if (lbRep) lbRep.onclick = () => {
    location.hash = "#/projects/" + sel.projectId + "/report/" + board.run_context.run_id;
  };

  const readCfg = () => {
    setCfg("dataset_source", $("labSrc").value);
    setCfg("target_column", $("labTarget").value);
    setCfg("max_rows", $("labRows").value);
    setCfg("max_iterations", $("labIters").value);
    setCfg("feature_engineering_every_n", $("labFe").value);
    setCfg("dataset_query", $("labQuery").value);
    setCfg("dataset_path", $("labPath").value);
    const m = $("labMetric").value, v = $("labTarget2").value;
    setCfg("metric", m + (MINIMIZE.has(m) ? " <= " : " >= ") + (v || (MINIMIZE.has(m) ? "0" : "0.85")));
  };

  els(".check[data-model]", host).forEach((c) => {
    const box = c.querySelector("input");
    box.onchange = () => {
      if (!box.checked) return;
      els(".check[data-model]", host).forEach((o) => {
        o.classList.remove("on");
        o.querySelector("input").checked = false;
      });
      c.classList.add("on");
      setCfg("model_hint", c.dataset.model);
    };
  });

  $("labReset").onclick = () => {
    const goal = runCfg.goal;
    runCfg = {
      goal: goal, metric: "", dataset_source: "", dataset_query: "", dataset_path: "",
      max_rows: "", target_column: "", max_iterations: "", feature_engineering_every_n: "",
      model_hint: "", mode: sel.mode,
    };
    cfgDirty = true;
    saveCfg();
    paintSaveState();
    PAGES.lab();
    toast("Configuration reset.", "ok");
  };
  $("labSave").onclick = () => { readCfg(); cfgDirty = false; paintSaveState(); toast("Configuration saved.", "ok"); };
  $("labClose").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/home"; };
  $("labFeat").onclick = featuresDrawer;
  $("labRes").onclick = researchDrawer;

  $("labReview").onclick = () => {
    readCfg();
    location.hash = "#/projects/" + sel.projectId + "/chat";
    const w = addMsg("agent", "Here is the run I would start with. Review it, edit it, or cancel it.", "interviewer", taskCard(runCfg));
    if (w) wireTaskCard(w);
  };

  $("labStart").onclick = async () => {
    readCfg();
    const body = taskCard(runCfg).replace(/<div class="taskcard-f">[\s\S]*?<\/div><\/div>$/, "</div>");
    const ok = await modal("Start this run?", body +
      '<p class="hint" style="margin-top:var(--s4)">You can leave the page while it runs — it continues in the background.</p>', [
      { label: "Cancel", kind: "ghost", value: null },
      { label: "Start run", kind: "pri", value: "go" },
    ]);
    if (ok === "go") startRun();
  };
};

els("#modeSeg button").forEach((b) => {
  b.onclick = () => {
    sel.mode = b.dataset.mode;
    runCfg.mode = sel.mode;
    saveCfg();
    PAGES.lab();
  };
});

async function featuresDrawer() {
  const body = openDrawer("Feature pipeline", "Real columns in the working table",
    '<div class="loading-note"><span class="spin" aria-hidden="true"></span>Loading feature state…</div>');
  let f = null;
  try { f = await api("/automl/features"); } catch (e) {
    body.innerHTML = errorState("Could not load features", e.message, e);
    wireErrorBox(body, featuresDrawer);
    return;
  }
  cache.features = f;
  const added = f.added || [], hist = f.run_history || [], excl = f.excluded || [];
  const isExcl = (n) => excl.indexOf(n) >= 0;
  const call = async (path, payload) => {
    try {
      const r = await post(path, Object.assign({ project_id: sel.projectId }, payload));
      toast(r.note || "Done.", "ok");
      featuresDrawer();
    } catch (e) { toast(e.message, "err"); }
  };
  body.innerHTML =
    '<div class="sect">' + sectHead("Engineered features") +
    (added.length
      ? '<div class="tbl-wrap"><div class="tbl-scroll"><table class="tbl">' +
        "<thead><tr><th>Feature</th><th>Operation</th><th>Paper</th><th>State</th><th></th></tr></thead><tbody>" +
        added.map((x) => "<tr><td class=\"mono\">" + esc(x.name) + '</td><td class="muted">' + esc(x.kind) +
          (x.a ? " · " + esc(x.a) : "") + '</td><td class="hint">' + esc(x.paper_title || x.paper_id || "—") + "</td><td>" +
          (isExcl(x.name) ? '<span class="badge warn">disabled</span>' : '<span class="badge ok">active</span>') +
          '</td><td><div class="btnrow tight end">' +
          (isExcl(x.name)
            ? '<button class="btn ghost sm" data-en="' + esc(x.name) + '" type="button">Enable</button>'
            : '<button class="btn ghost sm" data-dis="' + esc(x.name) + '" type="button">Disable</button>') +
          '<button class="btn ghost sm" data-rm="' + esc(x.name) + '" type="button">Remove</button>' +
          "</div></td></tr>").join("") + "</tbody></table></div></div>"
      : emptyState("⌗", "No engineered features yet", "The researcher proposes features during a campaign.")) +
    "</div>" +
    '<div class="sect">' + sectHead("Feature pipeline by run") +
    (hist.length
      ? '<div class="tbl-wrap"><div class="tbl-scroll"><table class="tbl">' +
        "<thead><tr><th>Run</th><th>Status</th><th>Count</th></tr></thead><tbody>" +
        hist.map((h) => '<tr><td class="mono">' + esc(h.run_id) + "</td><td>" + statusBadge(h.status) +
          '</td><td class="num">' + esc(nf(h.n_features)) + "</td></tr>").join("") +
        "</tbody></table></div></div>"
      : '<p class="muted">No run history yet.</p>') + "</div>";

  els("[data-dis]", body).forEach((b) => { b.onclick = () => call("/automl/features/disable", { column: b.dataset.dis }); });
  els("[data-en]", body).forEach((b) => { b.onclick = () => call("/automl/features/enable", { column: b.dataset.en }); });
  els("[data-rm]", body).forEach((b) => {
    b.onclick = async () => {
      const ok = await confirmModal("Remove feature",
        "<p>Remove <b>" + esc(b.dataset.rm) + "</b> from the working table?</p>", "Remove", true);
      if (ok) call("/automl/features/remove", { column: b.dataset.rm });
    };
  });
  $("drawerFoot").innerHTML =
    '<button class="btn" id="featUndo" type="button" style="flex:1">Undo last change</button>' +
    '<button class="btn pri" id="featClose" type="button">Close</button>';
  $("featUndo").onclick = () => call("/automl/features/undo", {});
  $("featClose").onclick = closeDrawer;
}

async function researchDrawer() {
  const body = openDrawer("Research", "Papers the agent actually used",
    '<div class="loading-note"><span class="spin" aria-hidden="true"></span>Loading research…</div>');
  let list = [];
  try {
    const runs = (await api("/runs?project_id=" + sel.projectId + "&limit=50")).runs || [];
    if (runs.length) list = (await api("/runs/" + runs[0].run_id + "/research")).research || [];
  } catch (e) { /* shown as empty below */ }
  if (!list.length) {
    body.innerHTML = emptyState("⌕", "No research yet",
      "Papers are recorded during a campaign, when the researcher proposes features.");
    $("drawerFoot").innerHTML = '<button class="btn pri" id="resClose" type="button" style="flex:1">Close</button>';
    $("resClose").onclick = closeDrawer;
    return;
  }
  body.innerHTML = list.map((p) =>
    '<div class="card" style="margin-bottom:var(--s3)"><div class="card-t"><h3>' + esc(p.title || p.paper_id) +
    '</h3><span class="sp"></span><span class="badge">' + esc(p.paper_id) + "</span></div>" +
    '<div class="kv">' + kv("Technique", p.technique || "—") + kv("Feature", p.feature || "—") + "</div>" +
    '<div class="btnrow" style="margin-top:var(--s3)"><button class="btn ghost sm" data-paper="' +
    esc(p.paper_id) + '" type="button">Open paper</button></div></div>').join("");
  els("[data-paper]", body).forEach((b) => {
    b.onclick = async () => {
      try {
        const p = await api("/automl/papers/" + encodeURIComponent(b.dataset.paper));
        openDrawer("Paper " + b.dataset.paper, p.title || "", pre(p.text || p.content || JSON.stringify(p, null, 2)));
      } catch (e) { toast("Could not open paper: " + e.message, "err"); }
    };
  });
  $("drawerFoot").innerHTML = '<button class="btn pri" id="resClose2" type="button" style="flex:1">Close</button>';
  $("resClose2").onclick = closeDrawer;
}

/* ------------------------------------------------------------- experiments */

PAGES.experiments = async function () {
  const host = $("expBody");
  host.innerHTML = skeleton(5, "Loading experiments…");
  let runs = [];
  try { runs = (await api("/runs?project_id=" + sel.projectId + "&limit=100")).runs || []; }
  catch (e) {
    host.innerHTML = errorState("Could not load experiments", e.message, e);
    wireErrorBox(host, PAGES.experiments);
    return;
  }
  cache.runs = runs;
  const models = [];
  runs.forEach((r) => {
    const m = (r.best && r.best.model) || "";
    if (m && models.indexOf(m) < 0) models.push(m);
  });
  $("expModel").innerHTML = '<option value="">All models</option>' +
    models.map((m) => '<option value="' + esc(m) + '">' + esc(modelName(m)) + "</option>").join("");

  const durOf = (r) => {
    if (!r.started_at) return 0;
    const a = Date.parse(r.started_at), b = Date.parse(r.completed_at || r.started_at);
    return isFinite(a) && isFinite(b) ? (b - a) / 1000 : 0;
  };

  const draw = () => {
    const q = $("expSearch").value.trim().toLowerCase();
    const mf = $("expModel").value, sf = $("expStatus").value, sort = $("expSort").value;
    let list = runs.filter((r) => {
      const m = (r.best && r.best.model) || "";
      if (q && (r.run_id + " " + m).toLowerCase().indexOf(q) < 0) return false;
      if (mf && m !== mf) return false;
      if (sf && String(r.status || "").toLowerCase() !== sf) return false;
      return true;
    });
    list = list.slice().sort((a, b) => {
      if (sort === "metric") return ((b.best && b.best.target_metric) || -1) - ((a.best && a.best.target_metric) || -1);
      if (sort === "dur") return durOf(b) - durOf(a);
      if (sort === "model") return String((a.best && a.best.model) || "").localeCompare(String((b.best && b.best.model) || ""));
      return String(b.started_at || "").localeCompare(String(a.started_at || ""));
    });

    if (!list.length) {
      host.innerHTML = emptyState("◈", runs.length ? "No matches" : "No experiments yet",
        runs.length ? "No experiment matches these filters."
          : "Run your first experiment to compare models and metrics.",
        runs.length ? "" : '<button class="btn pri" data-go="lab" type="button"><i aria-hidden="true">▶</i> Create experiment</button>');
      const g = el('[data-go="lab"]', host);
      if (g) g.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/lab"; };
      return;
    }

    host.innerHTML = '<div class="card flush"><div class="tbl-scroll"><table class="tbl">' +
      "<thead><tr><th>Experiment</th><th>Model</th><th>Dataset</th><th>Metric</th><th>Status</th><th>Duration</th><th>Created</th><th></th></tr></thead><tbody>" +
      list.map((r, i) => {
        const ds = r.dataset || {};
        return '<tr class="clickable" data-run="' + esc(r.run_id) + '">' +
          '<td class="mono">EXP-' + String(list.length - i).padStart(4, "0") +
          '<div class="hint mono" style="font-size:11px">' + esc(r.run_id) + "</div></td>" +
          "<td>" + esc(modelName((r.best && r.best.model) || "—")) + "</td>" +
          '<td class="muted">' + esc(ds.target_column || "—") + ' <span class="hint">· ' + esc(nf(ds.rows)) + " rows</span></td>" +
          '<td class="num">' + esc(num(r.best && r.best.target_metric)) + "</td>" +
          "<td>" + statusBadge(r.status) + "</td>" +
          '<td class="muted">' + esc(dur(durOf(r))) + "</td>" +
          '<td class="muted">' + esc(ago(r.started_at)) + "</td>" +
          '<td><div class="tbl-act">' +
          '<button class="btn ghost sm" data-open="' + esc(r.run_id) + '" type="button">Open</button>' +
          '<button class="btn ghost sm" data-cmp="' + esc(r.run_id) + '" type="button">Compare</button>' +
          '<button class="xbtn" data-more="' + esc(r.run_id) + '" type="button" aria-label="More actions for ' + esc(r.run_id) + '" title="More">⋯</button>' +
          "</div></td></tr>";
      }).join("") + "</tbody></table></div></div>";

    els("tr[data-run]", host).forEach((tr) => {
      tr.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/runs/" + tr.dataset.run; };
    });
    els("[data-open]", host).forEach((b) => {
      b.onclick = (e) => { e.stopPropagation(); location.hash = "#/projects/" + sel.projectId + "/runs/" + b.dataset.open; };
    });
    els("[data-cmp]", host).forEach((b) => {
      b.onclick = (e) => {
        e.stopPropagation();
        const m = location.hash.match(/compare\/([^/]+)/);
        const other = m ? m[1].split(",")[0] : "";
        location.hash = "#/projects/" + sel.projectId + "/compare/" + [other, b.dataset.cmp].filter(Boolean).join(",");
      };
    });
    els("[data-more]", host).forEach((b) => {
      b.onclick = (e) => { e.stopPropagation(); runMenu(b.dataset.more); };
    });
  };
  ["expSearch", "expModel", "expStatus", "expSort"].forEach((id) => {
    $(id).oninput = draw;
    $(id).onchange = draw;
  });
  draw();
};

async function runMenu(rid) {
  const v = await modal("Run actions", '<p>Actions for <span class="mono">' + esc(rid) + "</span>.</p>", [
    { label: "Close", kind: "ghost", value: null },
    { label: "Open", kind: "ghost", value: "open" },
    { label: "Reproduce", kind: "ghost", value: "repro" },
    { label: "Diagnostics", kind: "ghost", value: "diag" },
    { label: "Model report", kind: "pri", value: "rep" },
  ]);
  if (v === "open") location.hash = "#/projects/" + sel.projectId + "/runs/" + rid;
  else if (v === "rep") location.hash = "#/projects/" + sel.projectId + "/report/" + rid;
  else if (v === "diag") openDiagnostics(rid);
  else if (v === "repro") reproduceRun(rid);
}

async function reproduceRun(rid) {
  try {
    const d = await api("/runs/" + rid);
    const cfg = Object.assign({}, (d.run && d.run.config) || {}, { reproduces_run_id: rid });
    if (!sendWS({ event: "start", project_id: sel.projectId, session_id: sel.sessionId, config: cfg })) {
      toast("Not connected yet — retrying…", "err");
      return;
    }
    toast("Reproducing " + rid + "…", "ok", "Lineage is recorded on the new run.");
    location.hash = "#/projects/" + sel.projectId + "/monitor";
  } catch (e) { toast("Could not reproduce: " + e.message, "err"); }
}

async function openDiagnostics(rid) {
  const body = openDrawer("Diagnostics", rid,
    '<div class="loading-note"><span class="spin" aria-hidden="true"></span>Recomputing metrics…</div>');
  try {
    const d = await api("/runs/" + rid + "/diagnostics");
    const rec = Object.entries(d.metrics_recomputed || {}).map((e) => kv(metricLabel(e[0]), num(e[1])));
    const sto = Object.entries(d.stored_metrics || {}).map((e) => kv(metricLabel(e[0]), num(e[1])));
    body.innerHTML =
      '<div class="sect">' + sectHead("Recomputed") +
      '<div class="kv">' + (rec.length ? rec.join("") : kv("—", "none")) + "</div></div>" +
      '<div class="sect">' + sectHead("Stored") +
      '<div class="kv">' + (sto.length ? sto.join("") : kv("—", "none")) + "</div></div>" +
      (confusionMatrix(d.confusion_matrix)
        ? '<div class="sect">' + sectHead("Confusion matrix") + confusionMatrix(d.confusion_matrix) + "</div>" : "") +
      '<details class="disc"><summary><span class="chev">▸</span> Metric glossary</summary><div class="dbody">' +
      '<div class="kv">' + Object.entries(d.glossary || {}).map((e) => kv(titleCase(e[0]), e[1])).join("") + "</div></div></details>";
  } catch (e) {
    body.innerHTML = errorState("Diagnostics unavailable", e.message, e);
    wireErrorBox(body, () => openDiagnostics(rid));
  }
  $("drawerFoot").innerHTML = '<button class="btn pri" id="diagClose" type="button" style="flex:1">Close</button>';
  $("diagClose").onclick = closeDrawer;
}

/* ------------------------------------------------------------ run inspector */

PAGES.run = async function () {
  const rid = sel.runId;
  const host = $("runBody");
  $("runName").textContent = rid || "—";
  $("runActions").innerHTML = "";
  $("runTabs").innerHTML = "";
  host.innerHTML = skeleton(4, "Loading run " + (rid || "") + "…");

  let d = null, diag = null;
  try { d = await api("/runs/" + rid); } catch (e) {
    host.innerHTML = errorState("Could not load this run", e.message, e);
    wireErrorBox(host, PAGES.run);
    return;
  }
  const r = d.run || {};
  const best = r.best || {};
  const ds = r.dataset || {};
  cache.run = d;

  $("runName").textContent = best.model ? modelName(best.model) : (r.stage ? titleCase(r.stage) : "Run");
  $("runMeta").textContent = [
    rid,
    ds.target_column ? "target " + ds.target_column : null,
    ds.rows ? nf(ds.rows) + " rows" : null,
    ago(r.started_at),
  ].filter(Boolean).join(" · ");

  $("runActions").innerHTML =
    '<button class="btn pri" id="runRep" type="button"><i aria-hidden="true">✎</i> Model report</button>' +
    '<button class="btn" id="runRepro" type="button"><i aria-hidden="true">⧉</i> Reproduce</button>' +
    '<button class="btn" id="runCmp" type="button"><i aria-hidden="true">⇄</i> Compare</button>' +
    '<button class="xbtn" id="runX" type="button" aria-label="Close and return to experiments" title="Close (Esc)">✕</button>';
  $("runRep").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/report/" + rid; };
  $("runRepro").onclick = () => reproduceRun(rid);
  $("runCmp").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/compare/" + rid; };
  $("runX").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/experiments"; };

  try { diag = await api("/runs/" + rid + "/diagnostics"); cache.diag = diag; } catch (e) { diag = null; }

  const TABS = [
    ["overview", "Overview"], ["metrics", "Metrics"], ["params", "Parameters"],
    ["features", "Features"], ["preds", "Predictions"], ["explain", "Explainability"],
    ["code", "Code"], ["artifacts", "Artifacts"], ["lineage", "Lineage"],
  ];
  $("runTabs").innerHTML = TABS.map((t, i) =>
    '<button class="tab' + (i === 0 ? " on" : "") + '" data-tab="' + t[0] + '" type="button" role="tab">' + esc(t[1]) + "</button>").join("");

  const fi = best.feature_importance || {};
  const imp = (diag && diag.permutation_importance) || {};
  const impv = imp.importance || {};
  const pd = (diag && diag.prediction_distribution) || null;
  const conf = diag && diag.confusion_matrix;
  const roc = diag && diag.roc;
  const pr = diag && diag.pr;

  const panes = {
    overview:
      '<div class="kpis">' +
      kpi("Status", titleCase(r.status || "—"), "◈") +
      kpi("Model", best.model ? modelName(best.model) : "—", "⚙") +
      kpi("Main metric", num(best.target_metric), "◎", (r.config && r.config.metric) || "") +
      kpi("Features", nf(best.n_features), "⌗") +
      kpi("Train time", dur(best.training_time_seconds), "◷") +
      kpi("Target met", best.target_met === undefined || best.target_met === null ? "—" : (best.target_met ? "yes" : "no"), "◈") +
      "</div>" +
      '<div class="sect" style="margin-top:var(--s6)">' + sectHead("Dataset") +
      '<div class="kv">' + kv("Source", titleCase(ds.source || "—")) + kv("Rows", nf(ds.rows)) +
      kv("Columns", nf((ds.columns || []).length)) + kv("Target", ds.target_column || "—") +
      kv("Task", titleCase(ds.task_type || "—")) + "</div></div>" +
      (r.error ? '<div class="sect"><div class="errbox"><div class="eh"><i aria-hidden="true">!</i>Run failed</div>' +
        "<p>" + esc(String(r.error)) + "</p></div></div>" : ""),

    metrics: (function () {
      const m = best.metrics || {};
      const keys = Object.keys(m);
      if (!keys.length) return emptyState("◎", "No metrics", "This run recorded no metrics.");
      return '<div class="kpis">' + keys.map((k) => kpi(metricLabel(k), num(m[k]), "◎")).join("") + "</div>" +
        (diag && diag.glossary
          ? '<details class="disc" style="margin-top:var(--s5)"><summary><span class="chev">▸</span> What each metric means</summary>' +
            '<div class="dbody"><div class="kv">' +
            Object.entries(diag.glossary).map((e) => kv(titleCase(e[0]), e[1])).join("") + "</div></div></details>"
          : "");
    })(),

    params: (function () {
      const hp = best.hyperparameters || {};
      const keys = Object.keys(hp);
      return '<div class="sect">' + sectHead("Model parameters") +
        (keys.length
          ? '<div class="kv">' + keys.map((k) => kv(k, hp[k])).join("") + "</div>"
          : '<p class="muted">This model ran with its library defaults — the backend recorded no overrides.</p>') +
        "</div>" +
        '<div class="sect">' + sectHead("Run configuration") +
        '<div class="kv">' + Object.entries(r.config || {}).map((e) => kv(titleCase(e[0]),
          e[1] === null || e[1] === undefined || e[1] === "" ? "—"
            : typeof e[1] === "object" ? JSON.stringify(e[1]) : String(e[1]))).join("") + "</div></div>";
    })(),

    features: Object.keys(fi).length
      ? hbars(fi, { limit: 20 })
      : emptyState("⌗", "No feature importance", "The backend recorded none for this model."),

    preds: pd && pd.classes
      ? '<div class="sect">' + sectHead("Predicted class distribution") + '<div class="hbars">' +
        pd.classes.map((c, i) => {
          const v = (pd.counts || [])[i] || 0;
          const tot = (pd.counts || []).reduce((a, b) => a + b, 0) || 1;
          return '<div class="hbar"><span class="hn">' + esc(c) + '</span><span class="ht"><i style="width:' +
            ((v / tot) * 100).toFixed(1) + '%"></i></span><span class="hv">' + esc(nf(v)) + "</span></div>";
        }).join("") + "</div></div>"
      : emptyState("◌", "No prediction samples",
          "The backend does not store individual predictions for this run."),

    explain:
      '<div class="sect">' + sectHead("Feature importance") +
      (Object.keys(fi).length ? hbars(fi, { limit: 16 }) : emptyState("⌗", "None recorded", "")) + "</div>" +
      '<div class="sect">' + sectHead("Permutation importance",
        '<span class="hint">' + esc(imp.scoring ? "scoring: " + imp.scoring : "") + "</span>") +
      (Object.keys(impv).length
        ? hbars(impv, { limit: 16 })
        : emptyState("◌", "Not computed", imp.note || "The backend did not compute permutation importance for this run.")) +
      "</div>" +
      (roc
        ? '<div class="sect">' + sectHead("ROC curve", '<span class="hint">AUC ' + esc(num(roc.auc)) + "</span>") +
          curveChart(roc.fpr, roc.tpr, { aria: "ROC curve", yLabel: "True positive rate" }) + "</div>" : "") +
      (pr
        ? '<div class="sect">' + sectHead("Precision–recall curve", '<span class="hint">AUC ' + esc(num(pr.auc)) + "</span>") +
          curveChart(pr.recall, pr.precision, { dash: true, aria: "Precision recall curve", xLabel: "Recall", yLabel: "Precision" }) + "</div>" : ""),

    code: '<p class="hint" style="margin-bottom:var(--s4)">This is the exact configuration the run executed with. ' +
      "The backend exposes no editable source files for this project.</p>" +
      pre(JSON.stringify({
        config: r.config || {},
        dataset: ds,
        best_model: { model: best.model, hyperparameters: best.hyperparameters },
      }, null, 2)),

    artifacts: (d.artifacts || []).length
      ? '<div class="card flush"><div class="tbl-scroll"><table class="tbl">' +
        "<thead><tr><th>Kind</th><th>Path</th></tr></thead><tbody>" +
        (d.artifacts || []).map((a) => "<tr><td>" + statusBadge(a.kind) + '</td><td class="mono hint">' +
          esc(a.path) + "</td></tr>").join("") + "</tbody></table></div></div>"
      : emptyState("◫", "No artifacts", "This run produced no files."),

    lineage: (function () {
      const nodes = [
        ["▤", "Dataset", (ds.source || "—") + " · " + nf(ds.rows) + " rows", ds.target_column ? "target " + ds.target_column : ""],
        ["⌗", "Preprocessing", titleCase(ds.task_type || "—"), ""],
        ["⊞", "Features", nf(best.n_features) + " features", (r.feature_history || []).length + " engineered"],
        ["◈", "Experiment", rid, titleCase(r.status || "")],
        ["⚙", "Model", modelName(best.model || "—"), num(best.target_metric) + " metric"],
        ["✎", "Report", "Model report", best.target_met === false ? "target not met" : "available"],
      ];
      return '<div class="lineage">' + nodes.map((n, k) =>
        (k ? '<div class="larrow">↓</div>' : "") +
        '<div class="lnode"><span class="li" aria-hidden="true">' + n[0] + "</span><span>" + esc(n[1]) +
        '</span><span class="sp"></span><span class="muted">' + esc(n[2]) + "</span>" +
        (n[3] ? '<span class="lv">' + esc(n[3]) + "</span>" : "") + "</div>").join("") + "</div>" +
        '<div class="btnrow" style="margin-top:var(--s5)">' +
        '<button class="btn" id="lnRep" type="button"><i aria-hidden="true">✎</i> Open report</button>' +
        '<button class="btn ghost" id="lnCode" type="button"><i aria-hidden="true">⌸</i> Open code</button></div>';
    })(),
  };

  host.innerHTML = TABS.map((t) =>
    '<div class="tabpane' + (t[0] === "overview" ? " on" : "") + '" data-pane="' + t[0] + '">' + panes[t[0]] + "</div>").join("");
  els("[data-tab]", $("runTabs")).forEach((b) => {
    b.onclick = () => {
      els("[data-tab]", $("runTabs")).forEach((x) => x.classList.toggle("on", x === b));
      els("[data-pane]", host).forEach((p) => p.classList.toggle("on", p.dataset.pane === b.dataset.tab));
    };
  });
  const lr = el("#lnRep", host), lc = el("#lnCode", host);
  if (lr) lr.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/report/" + rid; };
  if (lc) lc.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/code"; };
};

/* --------------------------------------------------------- live run monitor */

function logHTML(lines) {
  return lines.map((l) =>
    '<div class="logline"><span class="lt">' + esc(new Date(l.t).toISOString().slice(11, 19)) +
    '</span><span class="le">' + esc(l.type || "") + '</span><span class="ls">' + esc(titleCase(l.state || "")) +
    '</span><span class="lm">' + esc(String(l.msg || "")) + "</span></div>").join("");
}

function wireLogButtons(host) {
  const lines = (run && run.log) || [];
  const asText = lines.map((l) => [new Date(l.t).toISOString(), l.type, l.state, l.msg].join("  ")).join("\n");
  const c = el("#logCopy", host);
  if (c) c.onclick = () => {
    if (!navigator.clipboard) { toast("Copy is not available in this browser.", "err"); return; }
    navigator.clipboard.writeText(asText).then(() => toast("Logs copied.", "ok"), () => toast("Copy failed.", "err"));
  };
  const dl = el("#logDl", host);
  if (dl) dl.onclick = () => downloadText("run-" + ((run && run.runId) || "events") + ".log", asText);
  const cl = el("#logClear", host);
  if (cl) cl.onclick = () => {
    const d = el("#monLogs", host);
    if (d) d.open = false;
    toast("Log view collapsed.", "info");
  };
}

PAGES.monitor = function () {
  const host = $("monBody");
  const stage = (run && run.stage) || lastState.stage || "IDLE";
  const busy = BUSY.indexOf(stage) >= 0;
  const canResume = stage === "PAUSED" || stage === "FAILED";

  $("monActions").innerHTML =
    '<button class="btn" id="monLog" type="button"><i aria-hidden="true">▤</i> Developer logs</button>' +
    (canResume ? '<button class="btn pri" id="monResume" type="button"><i aria-hidden="true">▶</i> Resume</button>' : "") +
    '<button class="xbtn" id="monX" type="button" aria-label="Leave monitoring — the run continues in the background" title="Leave monitoring">✕</button>';
  $("monX").onclick = leaveMonitor;
  const mr = $("monResume");
  if (mr) mr.onclick = () => resumeRun();
  $("monLog").onclick = () => {
    const d = $("monLogs");
    if (d) { d.open = true; d.scrollIntoView({ block: "nearest" }); }
  };

  const iters = (run && run.iterations) || [];
  const req = (lastState && lastState.requirements) || {};
  const metricName = req.metric || "target_metric";
  const series = iters.map((i) => [i.iteration, i.target_metric]).filter((p) => isFinite(p[1]));
  const bestSoFar = series.length ? Math.max.apply(null, series.map((p) => p[1])) : null;
  const cur = series.length ? series[series.length - 1][1] : null;
  const elapsed = run && run.startedAt
    ? ((Date.parse(run.finishedAt) || Date.now()) - Date.parse(run.startedAt)) / 1000 : null;
  const maxIter = req.max_iterations || 0;
  const progress = (run && typeof run.progress === "number")
    ? run.progress
    : (maxIter ? Math.min(100, (iters.length / maxIter) * 100) : 0);

  const order = ["STARTING", "DATA_FETCH", "RESEARCH", "TRAINING", "DONE"];
  const curIdx = order.indexOf(run && run.stage);
  const stages = order.map((s, i) => {
    const done = (curIdx >= 0 && i < curIdx) || (stage === "DONE" && i <= 4);
    const now = s === stage;
    return '<div class="stage ' + (now ? "now" : done ? "done" : "") + '">' +
      '<span class="si" aria-hidden="true">' + (now ? "●" : done ? "✓" : "○") + "</span><span>" +
      esc(titleCase(s)) + "</span></div>";
  }).join("");

  let chartHTML = "";
  if (series.length) {
    const bestLine = [];
    let b = -Infinity;
    series.forEach((p) => { b = Math.max(b, p[1]); bestLine.push([p[0], b]); });
    chartHTML =
      '<div class="chartbox"><div class="cbt"><span aria-hidden="true">◈</span>' + esc(metricLabel(metricName)) + " over iterations</div>" +
      lineChart(series, { aria: titleCase(metricName) + " per iteration", ref: req.target_value }) +
      '<div class="legend"><span><i style="background:var(--accent)"></i>metric</span>' +
      '<span><i style="background:var(--ok)"></i>best so far</span>' +
      "<span>target " + esc(num(req.target_value)) + "</span></div></div>" +
      '<div class="chartbox" style="margin-top:var(--s4)"><div class="cbt"><span aria-hidden="true">◈</span>Best so far</div>' +
      lineChart(bestLine, { aria: "Best metric so far" }) + "</div>";
  }

  host.innerHTML =
    '<div class="sect"><div class="card"><div class="card-t"><h3>' +
    esc(run && run.runId ? run.runId : "No active run") + '</h3><span class="sp"></span>' + statusBadge(stage) + "</div>" +
    '<div class="btnrow" style="margin-bottom:var(--s4)">' +
    (busy ? '<button class="btn" id="mp2" type="button"><i aria-hidden="true">⏸</i> Pause</button>' : "") +
    (busy ? '<button class="btn danger" id="ms2" type="button"><i aria-hidden="true">■</i> Stop</button>' : "") +
    (canResume ? '<button class="btn pri" id="mr2" type="button"><i aria-hidden="true">▶</i> Resume</button>' : "") +
    '<button class="btn ghost" id="mh2" type="button">Hide logs</button></div>' +
    '<div class="bar' + (stage === "DONE" ? " ok" : stage === "FAILED" ? " warn" : "") + '"><i style="width:' +
    Math.max(2, Math.min(100, progress)).toFixed(1) + '%"></i></div>' +
    '<div class="grid-auto" style="margin-top:var(--s5)">' +
    kpi("Progress", (progress || 0).toFixed(0) + "%", "◈") +
    kpi("Iteration", iters.length ? iters[iters.length - 1].iteration + (maxIter ? " / " + maxIter : "") : "—", "◷") +
    kpi("Best " + metricLabel(metricName), num(bestSoFar), "✓") +
    kpi("Current", num(cur), "◎") +
    kpi("Elapsed", dur(elapsed), "◷") + "</div>" +
    (run && run.last ? '<p class="hint" style="margin-top:var(--s4)">' + esc(run.last) + "</p>" : "") +
    "</div></div>" +

    (busy
      ? '<div class="sect"><div class="card tight" style="border-color:var(--accent-line);background:var(--accent-soft)">' +
        "<p style=\"margin:0;font-size:13.5px\">The run continues in the background. You can leave this page at any " +
        "time — use the <b>✕</b> above, or pick another section in the sidebar.</p></div></div>"
      : "") +

    '<div class="sect">' + sectHead("Current stage") + '<div class="stages">' + stages + "</div></div>" +

    (series.length
      ? '<div class="sect">' + sectHead("Progress") + chartHTML + "</div>"
      : emptyState("◈", "Nothing to chart yet",
          "Charts appear as soon as the trainer records real metrics for this run.")) +

    '<div class="sect"><details class="disc" id="monLogs"><summary><span class="chev">▸</span> Developer logs' +
    '<span class="dspacer"></span><span class="hint">' + esc(nf((run && run.log || []).length)) + " events</span></summary>" +
    '<div class="dbody">' +
    ((run && run.log && run.log.length) ? logHTML(run.log.slice(-200).reverse()) : '<p class="muted" style="margin:0">No events recorded yet.</p>') +
    '<div class="btnrow" style="margin-top:var(--s4)">' +
    '<button class="btn ghost sm" id="logCopy" type="button"><i aria-hidden="true">⧉</i> Copy</button>' +
    '<button class="btn ghost sm" id="logDl" type="button"><i aria-hidden="true">↓</i> Download</button>' +
    '<button class="btn ghost sm" id="logClear" type="button">Clear view</button>' +
    "</div></div></details></div>";

  const mp2 = el("#mp2", host), ms2 = el("#ms2", host), mr2 = el("#mr2", host), mh2 = el("#mh2", host);
  if (mp2) mp2.onclick = () => requestStop(false);
  if (ms2) ms2.onclick = () => requestStop(true);
  if (mr2) mr2.onclick = () => resumeRun();
  if (mh2) mh2.onclick = () => {
    const d = $("monLogs");
    if (d) d.open = false;
    toast("Logs hidden.", "info", "The run continues.");
  };
  wireLogButtons(host);
};

async function requestStop(hard) {
  const ok = await confirmModal(hard ? "Stop this run" : "Pause this run",
    hard
      ? "<p>Stopping ends the current campaign. Everything already recorded is kept, and you can start a new run at any time.</p>"
      : "<p>Pausing asks the trainer to stop after the current iteration. You can resume from the last checkpoint.</p>",
    hard ? "Stop run" : "Pause run", hard);
  if (!ok) return;
  try {
    await post("/automl/stop", {});
    toast(hard ? "Stop requested." : "Pause requested.", "ok", "The trainer finishes its current step first.");
  } catch (e) { toast(e.message, "err"); }
}

async function resumeRun() {
  try {
    const r = await post("/automl/resume", {
      project_id: sel.projectId,
      run_id: run && run.runId,
      session_id: sel.sessionId,
    });
    if (r.status === "not_found") { toast("Nothing to resume: " + (r.detail || ""), "err"); return; }
    if (r.status === "wrong_project") { toast("That run belongs to another project.", "err"); return; }
    toast("Resumed " + (r.run_id || "") + ".", "ok");
    location.hash = "#/projects/" + sel.projectId + "/monitor";
  } catch (e) {
    toast("Resume failed: " + e.message, "err");
  }
}

/** Leaving the monitor never stops the run — the user is told the truth. */
async function leaveMonitor() {
  const stage = (run && run.stage) || lastState.stage || "IDLE";
  const busy = BUSY.indexOf(stage) >= 0;
  const v = await modal("Leave monitoring?", busy
    ? "<p>This page only watches the run. Closing it will <b>not</b> stop anything — the campaign continues in the background and you will still be notified when it finishes.</p>"
    : "<p>This run is not active, so there is nothing to keep running. You can return to it at any time from Experiments.</p>", [
    { label: "Stay", kind: "ghost", value: "stay" },
    { label: "Leave", kind: "pri", value: "leave" },
  ]);
  if (v !== "leave") return;
  location.hash = "#/projects/" + sel.projectId + "/home";
}
