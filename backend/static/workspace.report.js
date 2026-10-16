/* ==========================================================================
   AutoML Workspace — Reports, Compare, Code & Artifacts, Settings, palette, shell
   ========================================================================== */

/* --------------------------------------------------------- reports index */

PAGES.reports = async function () {
  const host = $("reportsBody");
  host.innerHTML = skeleton(4, "Loading reports…");
  let runs = [];
  try { runs = (await api("/runs?project_id=" + sel.projectId + "&limit=100")).runs || []; }
  catch (e) {
    host.innerHTML = errorState("Could not load reports", e.message, e);
    wireErrorBox(host, PAGES.reports);
    return;
  }
  const done = runs.filter((r) => r.status === "done" || r.best);
  if (!done.length) {
    host.innerHTML = emptyState("✎", "No reports",
      "Completed model reports will appear here once a run finishes evaluating.",
      '<button class="btn pri" data-go="lab" type="button"><i aria-hidden="true">▶</i> Run an experiment</button>');
    const g = el('[data-go="lab"]', host);
    if (g) g.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/lab"; };
    return;
  }
  host.innerHTML = '<div class="card flush"><div class="tbl-scroll"><table class="tbl">' +
    "<thead><tr><th>Model</th><th>Experiment</th><th>Dataset</th><th>Metric</th><th>Target met</th><th>Finished</th><th></th></tr></thead><tbody>" +
    done.map((r) => {
      const b = r.best || {}, ds = r.dataset || {};
      return '<tr class="clickable" data-rep="' + esc(r.run_id) + '">' +
        "<td><b>" + esc(modelName(b.model || "—")) + "</b></td>" +
        '<td class="mono hint">' + esc(r.run_id) + "</td>" +
        '<td class="muted">' + esc(ds.target_column || "—") + ' <span class="hint">· ' + esc(nf(ds.rows)) + " rows</span></td>" +
        '<td class="num">' + esc(num(b.target_metric)) + "</td>" +
        "<td>" + (b.target_met === undefined ? '<span class="hint">—</span>'
          : b.target_met ? '<span class="badge ok">met</span>' : '<span class="badge warn">not met</span>') + "</td>" +
        '<td class="muted">' + esc(ago(r.completed_at || r.started_at)) + "</td>" +
        '<td><div class="tbl-act"><button class="btn ghost sm" data-open="' + esc(r.run_id) + '" type="button">Open report</button></div></td></tr>';
    }).join("") + "</tbody></table></div></div>";
  els("[data-rep]", host).forEach((tr) => {
    tr.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/report/" + tr.dataset.rep; };
  });
  els("[data-open]", host).forEach((b) => {
    b.onclick = (e) => { e.stopPropagation(); location.hash = "#/projects/" + sel.projectId + "/report/" + b.dataset.open; };
  });
};

/* ------------------------------------------------------------ model report */

PAGES.report = async function () {
  const rid = sel.runId;
  const host = $("repBody");
  $("repName").textContent = rid || "—";
  $("repMeta").textContent = "";
  $("repActions").innerHTML = "";
  host.innerHTML = skeleton(5, "Building the model report…");

  let d = null, diag = null, rep = null;
  try { d = await api("/runs/" + rid); } catch (e) {
    host.innerHTML = errorState("Could not load this report", e.message, e);
    wireErrorBox(host, PAGES.report);
    return;
  }
  const r = d.run || {}, best = r.best || {}, ds = r.dataset || {};
  cache.run = d;
  try { diag = await api("/runs/" + rid + "/diagnostics"); cache.diag = diag; } catch (e) {}
  try { rep = cache.report = await api("/projects/" + sel.projectId + "/report?format=json"); } catch (e) {}

  $("repName").textContent = best.model ? modelName(best.model) : "Model report";
  $("repMeta").textContent = [ds.target_column ? ds.target_column + " dataset" : null, nf(ds.rows) + " rows",
    "experiment " + rid, ago(r.completed_at || r.started_at)].filter(Boolean).join(" · ");

  $("repActions").innerHTML =
    '<button class="btn" id="repCmp" type="button"><i aria-hidden="true">⇄</i> Compare</button>' +
    '<button class="btn" id="repRun" type="button"><i aria-hidden="true">◈</i> View experiment</button>' +
    '<button class="btn" id="repCode" type="button"><i aria-hidden="true">⌸</i> View code</button>' +
    '<button class="btn" id="repExp" type="button"><i aria-hidden="true">↓</i> Export report</button>' +
    '<button class="btn pri" id="repMdl" type="button"><i aria-hidden="true">↓</i> Export model</button>' +
    '<button class="xbtn" id="repX" type="button" aria-label="Close report and return to reports" title="Close (Esc)">✕</button>';
  $("repCmp").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/compare/" + rid; };
  $("repRun").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/runs/" + rid; };
  $("repCode").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/code"; };
  $("repExp").onclick = () => {
    const payload = rep || { run: d.run, artifacts: d.artifacts, diagnostics: diag };
    downloadText("model-report-" + rid + ".json", JSON.stringify(payload, null, 2), "application/json");
  };
  $("repMdl").onclick = () => {
    if (!best.model_path) { toast("No trained model file was recorded for this run.", "err"); return; }
    toast("Exporting the best model…", "ok", best.model_path);
    window.open("/automl/export-best-model", "_blank", "noopener");
  };
  $("repX").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/reports"; };

  const metrics = best.metrics || {};
  const keys = Object.keys(metrics);
  const conf = diag && diag.confusion_matrix;
  const roc = diag && diag.roc;
  const pr = diag && diag.pr;
  const fi = best.feature_importance || {};
  const repro = (rep && rep.reproducibility) || {};

  /* The trainer stores some metrics as 0 (roc_auc here) while diagnostics
     recomputes them from the saved predictions. Show the real value and say
     where it came from instead of printing a bare 0 next to AUC 0.9752. */
  const metricNote = (k) => {
    const stored = Number(metrics[k]);
    if (!isFinite(stored) || stored !== 0) return null;
    const alt = k === "roc_auc" && roc && isFinite(roc.auc) ? roc.auc
      : diag && diag.metrics_recomputed ? diag.metrics_recomputed[k] : null;
    if (!isFinite(alt) || !alt) return null;
    return { value: num(alt), note: "recomputed from saved predictions — the trainer stored " + num(stored) };
  };

  host.innerHTML =
    '<div class="sect">' + sectHead("Key metrics") +
    (keys.length
      ? '<div class="kpis">' + keys.map((k) => {
          const alt = metricNote(k);
          return kpi(metricLabel(k), alt ? alt.value : num(metrics[k]), "◎", alt ? alt.note : "", null);
        }).join("") + "</div>"
      : emptyState("◎", "No metrics recorded", "This run produced no stored metrics.")) +
    (best.target_met !== undefined && best.target_met !== null
      ? '<p class="hint" style="margin-top:var(--s4)">Target ' +
        (best.target_met ? '<span class="badge ok">met</span>' : '<span class="badge warn">not met</span>') +
        " — goal " + esc(String((r.config && r.config.metric) || "—")) + "</p>"
      : "") + "</div>" +

    (conf
      ? '<div class="sect">' + sectHead("Confusion matrix") +
        '<div class="chartbox">' + confusionMatrix(conf) + "</div></div>" : "") +

    (roc
      ? '<div class="sect">' + sectHead("ROC curve", '<span class="hint">AUC ' + esc(num(roc.auc)) + "</span>") +
        '<div class="chartbox">' + curveChart(roc.fpr, roc.tpr, { aria: "ROC curve", height: 340 }) + "</div></div>" : "") +

    (pr
      ? '<div class="sect">' + sectHead("Precision–recall", '<span class="hint">AUC ' + esc(num(pr.auc)) + "</span>") +
        '<div class="chartbox">' +
        curveChart(pr.recall, pr.precision, { dash: true, aria: "Precision recall curve", xLabel: "Recall", yLabel: "Precision", height: 340 }) +
        "</div></div>" : "") +

    (Object.keys(fi).length
      ? '<div class="sect">' + sectHead("Feature importance") +
        '<div class="chartbox">' + hbars(fi, { limit: 16 }) + "</div></div>" : "") +

    (diag && diag.classification_report
      ? '<div class="sect"><details class="disc"><summary><span class="chev">▸</span> Classification report' +
        '<span class="dspacer"></span><span class="hint">precision · recall · f1 by class</span></summary>' +
        '<div class="dbody"><div class="tbl-wrap"><div class="tbl-scroll"><table class="tbl">' +
        "<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>" +
        Object.entries(diag.classification_report).filter((e) => e[0] !== "accuracy")
          .map((e) => "<tr><td>" + esc(titleCase(e[0])) + "</td>" +
            ["precision", "recall", "f1-score", "support"].map((k) => '<td class="num">' + esc(num(e[1][k])) + "</td>").join("") +
            "</tr>").join("") +
        "</tbody></table></div></div></div></details></div>" : "") +

    (diag && diag.prediction_distribution
      ? '<div class="sect">' + sectHead("Error analysis", '<span class="hint">from the recorded prediction distribution</span>') +
        '<div class="chartbox"><div class="hbars">' +
        diag.prediction_distribution.classes.map((c, i) => {
          const v = (diag.prediction_distribution.counts || [])[i] || 0;
          const tot = (diag.prediction_distribution.counts || []).reduce((a, b) => a + b, 0) || 1;
          return '<div class="hbar"><span class="hn">' + esc(c) + '</span><span class="ht"><i style="width:' +
            ((v / tot) * 100).toFixed(1) + '%"></i></span><span class="hv">' + esc(nf(v)) + "</span></div>";
        }).join("") + "</div></div></div>" : "") +

    '<div class="sect">' + sectHead("Model details") +
    '<div class="kv">' +
    kv("Model", modelName(best.model || "—")) +
    kv("Parameters", Object.keys(best.hyperparameters || {}).length
      ? Object.entries(best.hyperparameters).map((e) => e[0] + "=" + e[1]).join(", ")
      : "library defaults") +
    kv("Dataset", (ds.source || "—") + " · " + nf(ds.rows) + " rows · " + nf((ds.columns || []).length) + " columns") +
    kv("Target", ds.target_column || "—") +
    kv("Features", nf(best.n_features)) +
    kv("Training time", dur(best.training_time_seconds)) +
    kv("Random seed", repro.trainer_seed !== undefined ? repro.trainer_seed : "—") +
    kv("Run id", rid) +
    kv("Started", r.started_at || "—") +
    kv("Completed", r.completed_at || "—") +
    (repro.package_versions && Object.keys(repro.package_versions).length
      ? kv("Packages", Object.entries(repro.package_versions).map((e) => e[0] + " " + e[1]).join(", ")) : "") +
    "</div></div>" +

    (rep && (rep.limitations || []).length
      ? '<div class="sect"><details class="disc"><summary><span class="chev">▸</span> Limitations recorded by the backend</summary>' +
        '<div class="dbody"><ul class="speclist">' +
        rep.limitations.map((x) => "<li><span>" + esc(typeof x === "string" ? x : JSON.stringify(x)) + "</span></li>").join("") +
        "</ul></div></details></div>" : "") +

    (rep && (rep.recommendations || []).length
      ? '<div class="sect"><details class="disc"><summary><span class="chev">▸</span> Recommendations recorded by the backend</summary>' +
        '<div class="dbody"><ul class="speclist">' +
        rep.recommendations.map((x) => "<li><span>" + esc(typeof x === "string" ? x : JSON.stringify(x)) + "</span></li>").join("") +
        "</ul></div></details></div>" : "") +

    '<div class="sect"><div class="btnrow">' +
    '<button class="btn" id="raCmp" type="button"><i aria-hidden="true">⇄</i> Compare</button>' +
    '<button class="btn" id="raRun" type="button"><i aria-hidden="true">◈</i> View experiment</button>' +
    '<button class="btn" id="raCode" type="button"><i aria-hidden="true">⌸</i> View code</button>' +
    '<button class="btn ghost" id="raX" type="button"><i aria-hidden="true">✕</i> Close</button>' +
    "</div></div>";

  const pair = [["#raCmp", "compare/" + rid], ["#raRun", "runs/" + rid], ["#raCode", "code"]];
  pair.forEach((p) => {
    const b = el(p[0], host);
    if (b) b.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/" + p[1]; };
  });
  const ax = el("#raX", host);
  if (ax) ax.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/reports"; };
};

/* ---------------------------------------------------------------- compare */

PAGES.compare = function () {
  const host = $("compareBody");
  const fromHash = (location.hash.match(/compare\/([^/]+)/) || [])[1] || "";
  const preselect = fromHash.split(",").filter(Boolean);
  host.innerHTML = skeleton(4, "Loading runs to compare…");

  (async () => {
    let runs = [];
    try { runs = cache.runs || (await api("/runs?project_id=" + sel.projectId + "&limit=100")).runs || []; }
    catch (e) {
      host.innerHTML = errorState("Could not load runs", e.message, e);
      wireErrorBox(host, PAGES.compare);
      return;
    }
    const usable = runs.filter((r) => r.best && r.best.metrics);
    if (!usable.length) {
      host.innerHTML = emptyState("⇄", "Nothing to compare",
        "Comparison needs at least one completed run with recorded metrics.");
      return;
    }
    if (cache.runs) { /* keep */ } else { cache.runs = runs; }

    /* named `picked`, not `sel`: a `sel` here would shadow the global state
       object and put line 208's sel.projectId into the temporal dead zone. */
    const picked = {};
    usable.forEach((r) => { picked[r.run_id] = preselect.indexOf(r.run_id) >= 0; });
    if (!preselect.length) {
      picked[usable[0].run_id] = true;
      if (usable[1]) picked[usable[1].run_id] = true;
    } else if (preselect.length === 1 && usable.length > 1) {
      /* deep-linked to one run: add the newest other run so the comparison and
         its charts are actually usable. The user can untick it at any time. */
      const other = usable.find((r) => !picked[r.run_id]);
      if (other) picked[other.run_id] = true;
    }

    const metricKeys = [];
    usable.forEach((r) => Object.keys(r.best.metrics || {}).forEach((k) => {
      if (metricKeys.indexOf(k) < 0) metricKeys.push(k);
    }));

    const draw = () => {
      const chosen = usable.filter((r) => picked[r.run_id]);
      const rows = chosen.map((r) => {
        const b = r.best || {};
        const m = b.metrics || {};
        const cells = metricKeys.map((k) => '<td class="num">' + esc(num(m[k])) + "</td>").join("");
        return "<tr><td><b>" + esc(modelName(b.model || "—")) + '</b><div class="hint mono" style="font-size:11px">' +
          esc(r.run_id) + "</div></td>" + cells +
          '<td class="muted">' + esc(dur(b.training_time_seconds)) + "</td>" +
          '<td class="muted hint">' + esc(nf(b.n_features)) + "</td></tr>";
      }).join("");

      const charts = metricKeys.map((k) => {
        const pts = chosen.map((r) => [r.best.model || r.run_id, (r.best.metrics || {})[k]])
          .filter((p) => isFinite(p[1]));
        if (pts.length < 2) return "";
        return '<div class="chartbox"><div class="cbt"><span aria-hidden="true">◈</span>' +
          esc(metricLabel(k)) + "</div>" + lineChart(pts, { aria: metricLabel(k) + " by model", height: 180 }) + "</div>";
      }).join("");

      host.innerHTML =
        '<div class="sect">' + sectHead("Select runs to compare",
          '<span class="hint">' + esc(nf(chosen.length)) + " selected</span>") +
        '<div class="grid-auto">' + usable.map((r) => {
          const b = r.best || {};
          return '<label class="check' + (picked[r.run_id] ? " on" : "") + '" data-cmp="' + esc(r.run_id) + '">' +
            '<input type="checkbox"' + (picked[r.run_id] ? " checked" : "") + ' aria-label="Compare ' + esc(modelName(b.model || r.run_id)) + '" />' +
            "<span><span class=\"cn\">" + esc(modelName(b.model || "—")) + '</span><br /><span class="cd">' +
            esc((r.best.metrics && Object.keys(r.best.metrics)[0]) || "no metrics") + " " +
            esc(num(b.target_metric)) + " · " + esc(r.run_id) + "</span></span></label>";
        }).join("") + "</div></div>" +

        (chosen.length
          ? '<div class="sect">' + sectHead("Metric comparison",
              '<button class="btn ghost sm" id="cmpClear" type="button">Clear selection</button>') +
            '<div class="tbl-wrap"><div class="tbl-scroll"><table class="tbl"><thead><tr><th>Model</th>' +
            metricKeys.map((k) => "<th>" + esc(metricLabel(k)) + "</th>").join("") +
            "<th>Train time</th><th>Features</th></tr></thead><tbody>" + rows + "</tbody></table></div></div>" +
            (charts ? '<div class="grid2" style="margin-top:var(--s5)">' + charts + "</div>" : "") +
            '<p class="hint" style="margin-top:var(--s4)">Differences are shown as measured. No model is marked as a winner — ' +
            "the choice is yours.</p></div>"
          : emptyState("⇄", "Nothing selected", "Tick at least one run above to compare.")) +

        (chosen.length >= 2
          ? '<div class="sect"><div class="btnrow">' +
            '<button class="btn" id="cmpRep" type="button"><i aria-hidden="true">✎</i> Open first report</button>' +
            '<button class="btn ghost" id="cmpX" type="button"><i aria-hidden="true">✕</i> Close</button>' +
            "</div></div>" : "");

      els("[data-cmp]", host).forEach((c) => {
        c.querySelector("input").onchange = () => { picked[c.dataset.cmp] = c.querySelector("input").checked; draw(); };
      });
      const cl = el("#cmpClear", host);
      if (cl) cl.onclick = () => { usable.forEach((r) => { picked[r.run_id] = false; }); draw(); };
      const rp = el("#cmpRep", host);
      if (rp) rp.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/report/" + chosen[0].run_id; };
      const cx = el("#cmpX", host);
      if (cx) cx.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/experiments"; };
    };

    draw();
  })();
};

/* ------------------------------------------------- code & artifacts viewer */

let CODE_FILES = [];
let CODE_ACTIVE = null;

PAGES.code = async function () {
  const tree = $("codeTree");
  tree.innerHTML = '<div class="loading-note" style="padding:var(--s4)"><span class="spin" aria-hidden="true"></span>Collecting files…</div>';

  let runs = [], datasets = [];
  try { runs = (await api("/runs?project_id=" + sel.projectId + "&limit=50")).runs || []; } catch (e) {}
  try { datasets = (await api("/datasets?project_id=" + sel.projectId + "&page=1&page_size=100")).items || []; } catch (e) {}

  const groups = [];

  runs.forEach((r) => {
    const d = cache.run && cache.run.run && cache.run.run.run_id === r.run_id ? cache.run : null;
    const arts = (d && d.artifacts) || [];
    arts.forEach((a) => {
      groups.push({
        group: "Run " + r.run_id,
        items: [{
          name: String(a.path).split("/").pop(),
          icon: a.kind === "model" ? "◈" : "▤",
          size: a.kind,
          load: async () => {
            const detail = await api("/runs/" + r.run_id);
            const found = (detail.artifacts || []).find((x) => x.path === a.path);
            return {
              name: found ? found.path : a.path,
              body: JSON.stringify({
                artifact_kind: found ? found.kind : a.kind,
                path: found ? found.path : a.path,
                run_id: r.run_id,
                note: "The backend records artifact paths, not file contents. Download the file from the server filesystem at this path.",
              }, null, 2),
            };
          },
        }],
      });
    });
  });

  runs.slice(0, 10).forEach((r) => {
    groups.push({
      group: "Experiments",
      items: [{
        name: r.run_id,
        icon: "◈",
        size: r.status,
        load: async () => {
          const detail = await api("/runs/" + r.run_id);
          return { name: "runs/" + r.run_id + "/run.json", body: JSON.stringify(detail.run, null, 2) };
        },
      }],
    });
  });

  if (runs.length) {
    groups.push({
      group: "Reports",
      items: [{
        name: "project-report.json",
        icon: "✎",
        size: "json",
        load: async () => {
          const rep = await api("/projects/" + sel.projectId + "/report?format=json");
          cache.report = rep;
          return { name: "reports/project-report.json", body: JSON.stringify(rep, null, 2) };
        },
      }],
    });
  }

  if (runs.length) {
    groups.push({
      group: "Data",
      items: [{
        name: "dataset-preview.json",
        icon: "▤",
        size: "json",
        load: async () => {
          const pv = await api("/automl/dataset-preview?limit=25");
          return { name: "data/dataset-preview.json", body: JSON.stringify(pv, null, 2) };
        },
      }],
    });
  }

  runs.forEach((r) => {
    groups.push({
      group: "Configuration",
      items: [{
        name: r.run_id + "/config.json",
        icon: "⌗",
        size: "json",
        load: async () => {
          const detail = await api("/runs/" + r.run_id);
          return { name: "experiments/" + r.run_id + "/config.json", body: JSON.stringify(detail.run.config, null, 2) };
        },
      }],
    });
  });

  datasets.forEach((d) => {
    groups.push({
      group: "Registered datasets",
      items: [{
        name: d.name,
        icon: "▤",
        size: d.status,
        load: async () => {
          const rec = await api("/datasets/" + d.id);
          return { name: "data/" + d.name, body: JSON.stringify(rec, null, 2) };
        },
      }],
    });
  });

  if (runs.length) {
    groups.push({
      group: "Model",
      items: [{
        name: "export-best-model",
        icon: "↓",
        size: "endpoint",
        load: async () => ({
          name: "models/export-best-model",
          body: "GET /automl/export-best-model\n\n" +
            "Returns the best model of the current campaign as a downloadable file.\n" +
            "Use the “Export model” button on a model report to download it.",
        }),
      }],
    });
  }

  CODE_FILES = groups;

  if (!groups.length) {
    tree.innerHTML = '<div style="padding:var(--s5)">' +
      emptyState("⌸", "No files yet", "Files appear here as soon as this project produces runs, datasets or reports.") + "</div>";
    $("codeView").textContent = "Nothing to show yet.";
    renderAssistant();
    renderCodeLogs();
    return;
  }

  const html = groups.map((g, gi) =>
    '<div class="fgroup">' + esc(g.group) + "</div>" +
    g.items.map((it) =>
      '<button class="fitem" data-g="' + gi + '" data-i="' + groups[gi].items.indexOf(it) + '" type="button">' +
      '<i aria-hidden="true">' + it.icon + "</i><span>" + esc(it.name) + "</span><span class=\"sp\"></span>" +
      '<span class="fs">' + esc(it.size || "") + "</span></button>").join("")).join("");
  tree.innerHTML = '<div class="ftree" style="padding:var(--s2)">' + html + "</div>";

  els("[data-g]", tree).forEach((b) => {
    b.onclick = () => openCodeFile(Number(b.dataset.g), Number(b.dataset.i));
  });
  await openCodeFile(0, 0);
  renderAssistant();
  renderCodeLogs();
};

async function openCodeFile(gi, ii) {
  const g = CODE_FILES[gi];
  if (!g) return;
  const f = g.items[ii];
  if (!f) return;
  CODE_ACTIVE = f;
  els("#codeTree .fitem").forEach((b) => b.classList.toggle("on",
    Number(b.dataset.g) === gi && Number(b.dataset.i) === ii));
  $("codeView").textContent = "Opening " + f.name + "…";
  $("codeName").textContent = f.name;
  $("codeSize").textContent = "";
  try {
    const out = await f.load();
    $("codeName").textContent = out.name;
    $("codeView").textContent = out.body;
    $("codeSize").textContent = nf(out.body.length) + " chars";
  } catch (e) {
    $("codeView").textContent = "Could not load this file: " + e.message;
    $("codeSize").textContent = "";
  }
}

function renderAssistant() {
  $("codeAsst").innerHTML =
    '<div class="card-t"><h3>Assistant</h3><span class="sp"></span>' +
    '<span class="badge acc">project</span></div>' +
    '<p class="muted" style="font-size:13.5px;margin:0 0 var(--s4)">Ask the agent about this project while you read. ' +
    "It has the same context as the chat workspace.</p>" +
    '<div class="composer"><textarea id="codeAsk" rows="2" placeholder="Ask about this run…" aria-label="Ask the agent"></textarea>' +
    '<div class="composer-f"><span class="sp"></span>' +
    '<button class="btn pri sm" id="codeAskSend" type="button">Send <i aria-hidden="true">↑</i></button></div></div>' +
    '<div class="thread" id="codeThread" style="max-height:280px;overflow-y:auto;margin-top:var(--s4)"></div>' +
    '<div class="typing" id="codeTyping"><span class="ticks"><i></i><i></i><i></i></span> agents working…</div>';
  $("codeAskSend").onclick = () => {
    const v = $("codeAsk").value.trim();
    if (!v) return;
    $("codeThread").insertAdjacentHTML("beforeend", '<div class="msg u"><div class="bubble"></div></div>');
    const last = $("codeThread").lastElementChild.firstElementChild;
    last.textContent = v;
    $("codeAsk").value = "";
    $("codeTyping").classList.add("on");
    chatSend(v);
  };
}

function renderCodeLogs() {
  const box = $("codeLogBody");
  if (!box) return;
  const lines = (run && run.log) || [];
  $("codeLogN").textContent = nf(lines.length) + " events";
  box.innerHTML = lines.length
    ? logHTML(lines.slice(-100).reverse())
    : '<p class="muted" style="margin:0">No events recorded yet.</p>';
  wireLogButtons(box);
}

$("codeCopy").onclick = () => {
  if (!navigator.clipboard) { toast("Copy is not available in this browser.", "err"); return; }
  navigator.clipboard.writeText($("codeView").textContent)
    .then(() => toast("Copied to clipboard.", "ok"), () => toast("Copy failed.", "err"));
};
$("codeDl").onclick = () => {
  if (!CODE_ACTIVE) { toast("No file selected.", "err"); return; }
  downloadText(String($("codeName").textContent).split("/").pop() || "file.txt", $("codeView").textContent);
};

/* Real resource types the backend import endpoint accepts. */
const IMPORT_TYPES = [
  ["feature_pipeline", "Feature pipeline", "Engineered features and their provenance"],
  ["dataset", "Dataset", "The registered dataset record"],
  ["dataset_snapshot", "Dataset snapshot", "The exact table a run trained on"],
  ["model_artifact", "Model artifact", "Trained model files"],
  ["experiment_config", "Experiment config", "A run's configuration"],
  ["research", "Research", "Papers the researcher used"],
];

/* -------------------------------------------------------- project settings */
let settingsTab = "general";  // survives PAGES.settings() re-renders


PAGES.settings = async function () {
  const host = $("setBody");
  $("setTabs").innerHTML = "";
  host.innerHTML = skeleton(3, "Loading project settings…");

  let ctx = null, res = null, rep = null, imports = [];
  try { ctx = cache.ctx = await api("/projects/" + sel.projectId + "/context"); } catch (e) {}
  try { res = await api("/projects/" + sel.projectId + "/resources"); } catch (e) {}
  try { rep = await api("/projects/" + sel.projectId + "/report?format=json"); cache.report = rep; } catch (e) {}
  try { imports = (await api("/projects/" + sel.projectId + "/imports")).imports || []; } catch (e) {}

  const TABS = [["general", "General"], ["context", "Agent context"], ["import", "Import"], ["danger", "Project"]];
  /* Remember the open tab across re-renders (e.g. after an import applies), so
     the user is not bounced back to General and can see the result they caused. */
  if (!TABS.some((t) => t[0] === settingsTab)) settingsTab = "general";
  $("setTabs").innerHTML = TABS.map((t) =>
    '<button class="tab' + (t[0] === settingsTab ? " on" : "") + '" data-tab="' + t[0] + '" type="button" role="tab">' + esc(t[1]) + "</button>").join("");

  const mem = (res && res.memory) || {};
  const runs = (res && res.runs) || [];
  const dsList = (res && res.datasets) || [];

  const panes = {
    general: '<div class="card"><div class="grid2">' +
      '<div class="field"><label class="lbl" for="stName">Project name</label>' +
      '<input class="inp" id="stName" value="' + esc(sel.project ? sel.project.name : "") + '" /></div>' +
      '<div class="field"><label class="lbl">Status</label>' +
      '<input class="inp" value="' + esc(titleCase((sel.project && sel.project.status) || "active")) + '" disabled />' +
      '<div class="help">The API exposes project status as read-only, so it is shown rather than edited here.</div></div>' +
      "</div>" +
      '<div class="field"><label class="lbl" for="stDesc">Description</label>' +
      '<textarea class="inp" id="stDesc">' + esc(sel.project && sel.project.description ? sel.project.description : "") + "</textarea>" +
      '<div class="help">The agent reads this as the project goal when a new interview starts.</div></div>' +
      '<div class="btnrow"><button class="btn pri" id="stSave" type="button">Save changes</button>' +
      '<button class="btn ghost" id="stCancel" type="button">Cancel</button></div></div>',

    context: sectHead("Agent context", '<span class="hint">recorded by the backend</span>') +
      '<div class="kv">' +
      kv("Goal", (ctx && ctx.goal) || mem.goal || "—") +
      kv("Task", titleCase((ctx && ctx.task) || mem.task_type || "—")) +
      kv("Target", (ctx && ctx.target) || mem.target || "—") +
      kv("Dataset", (ctx && ctx.dataset && ctx.dataset.source) || (mem.dataset && mem.dataset.source) || "—") +
      kv("Rows", nf(ctx && ctx.dataset && ctx.dataset.rows ? ctx.dataset.rows : mem.dataset && mem.dataset.rows)) +
      "</div>" +
      '<div class="sect" style="margin-top:var(--s6)">' + sectHead("Decisions") +
      (((ctx && ctx.decisions) || mem.decisions || []).length
        ? '<div class="card"><ul class="speclist">' +
          ((ctx && ctx.decisions) || mem.decisions || []).slice(0, 12).map((a) => {
            const what = (a && (a.what || a.key)) || (typeof a === "string" ? a : "decision");
            const why = (a && (a.why || a.reason)) || "";
            return "<li><span>" + esc(what) + '</span><span class="sp"></span><span class="muted">' + esc(why) + "</span></li>";
          }).join("") + "</ul></div>"
        : '<p class="muted">Nothing recorded yet.</p>') + "</div>" +
      '<div class="sect">' + sectHead("Resources in this project") +
      '<div class="kpis">' + kpi("Runs", nf(runs.length), "◈") + kpi("Datasets", nf(dsList.length), "▤") +
      kpi("Research papers", nf((mem.research || []).length), "⌕") +
      kpi("Excluded features", nf((mem.excluded_features || []).length), "⌗") + "</div></div>",

    import: (function () {
      (async () => {
        try {
          const others = ((await api("/projects?page=1&page_size=100")).items || [])
            .filter((p) => p.id !== sel.projectId && p.status !== "deleted");
          const sel2 = $("impSrc");
          if (sel2) {
            sel2.innerHTML = others.length
              ? others.map((p) => '<option value="' + esc(p.id) + '">' + esc(p.name) + "</option>").join("")
              : '<option value="">No other projects</option>';
          }
        } catch (e) { /* reported when the user presses the button */ }
      })();
      return '<div class="card">' +
        "<p>Bring resources from another project. <b>Review</b> is a dry run — the backend reports exactly " +
        "what it would change before anything is written.</p>" +
        '<div class="grid2">' +
        '<div class="field"><label class="lbl" for="impSrc">From project</label>' +
        '<select class="inp" id="impSrc"><option value="">Loading…</option></select></div>' +
        '<div class="field"><label class="lbl" for="impMode">Mode</label><select class="inp" id="impMode">' +
        '<option value="review">Review — dry run, nothing is written</option>' +
        '<option value="quick">Quick — import fully compatible resources</option>' +
        '<option value="smart">Smart — adapt partially compatible resources</option>' +
        "</select></div></div>" +
        '<div class="field"><span class="lbl">Resources</span><div class="grid-auto" id="impTypes">' +
        IMPORT_TYPES.map((t) => '<label class="check on" data-imp="' + t[0] + '">' +
          '<input type="checkbox" checked aria-label="Import ' + esc(t[1]) + '" />' +
          '<span><span class="cn">' + esc(t[1]) + '</span><br /><span class="cd">' + esc(t[2]) + "</span></span></label>").join("") +
        "</div></div>" +
        '<div id="impErr" aria-live="polite"></div>' +
        '<div class="btnrow"><button class="btn pri" id="impGo" type="button"><i aria-hidden="true">→</i> Inspect source</button>' +
        '<button class="btn ghost" id="impClose" type="button">Close</button></div></div>' +
        (imports.length
          ? '<div class="sect">' + sectHead("Import history") +
            '<div class="tbl-wrap"><div class="tbl-scroll"><table class="tbl">' +
            "<thead><tr><th>When</th><th>Resource</th><th>Mode</th><th>Verdict</th><th>Adapted</th></tr></thead><tbody>" +
            imports.map((i) => '<tr><td class="muted">' + esc(ago(i.created_at)) + "</td>" +
              '<td class="hint">' + esc(titleCase(i.resource_type || "—")) +
              (i.source_run_id ? '<div class="hint mono" style="font-size:11px">' + esc(i.source_run_id) + "</div>" : "") + "</td>" +
              '<td class="muted">' + esc(titleCase(i.mode || "—")) + "</td>" +
              "<td>" + statusBadge(i.verdict) + "</td>" +
              '<td class="muted">' + (i.adapted ? "yes" : "no") + "</td></tr>").join("") +
            "</tbody></table></div></div></div>"
          : "");
    })(),

    danger: '<div class="card"><div class="card-t"><h3>Destructive actions</h3></div>' +
      "<p class=\"muted\">This cannot be undone. A confirmation is always required.</p>" +
      '<div class="btnrow">' +
      '<button class="btn danger" id="stDelete" type="button"><i aria-hidden="true">■</i> Delete project</button>' +
      "</div>" +
      '<p class="hint" style="margin-top:var(--s4)">Deleting removes the project record, its runs, datasets and ' +
      "chat history. Exported artifacts already downloaded are not affected.</p></div>",
  };

  host.innerHTML = TABS.map((t) =>
    '<div class="tabpane' + (t[0] === settingsTab ? " on" : "") + '" data-pane="' + t[0] + '">' + panes[t[0]] + "</div>").join("");

  els("[data-tab]", $("setTabs")).forEach((b) => {
    b.onclick = () => {
      settingsTab = b.dataset.tab;
      els("[data-tab]", $("setTabs")).forEach((x) => x.classList.toggle("on", x === b));
      els("[data-pane]", host).forEach((p) => p.classList.toggle("on", p.dataset.pane === b.dataset.tab));
    };
  });

  const save = $("stSave");
  if (save) save.onclick = async () => {
    const name = $("stName").value.trim();
    if (!name) { toast("The project needs a name.", "err"); return; }
    try {
      const p = await api("/projects/" + sel.projectId, {
        method: "PATCH",
        body: JSON.stringify({ name: name, description: $("stDesc").value.trim() || null }),
      });
      sel.project = p;
      $("hdrProject").textContent = p.name;
      $("sideProjName").textContent = p.name;
      toast("Project updated.", "ok");
    } catch (e) { toast("Could not save: " + e.message, "err"); }
  };
  const cancel = $("stCancel");
  if (cancel) cancel.onclick = () => PAGES.settings();

  const impClose = $("impClose");
  if (impClose) impClose.onclick = () => { location.hash = "#/projects/" + sel.projectId + "/home"; };
  els("[data-imp]", host).forEach((c) => {
    const box = c.querySelector("input");
    box.onchange = () => c.classList.toggle("on", box.checked);
  });
  const impGo = $("impGo");
  if (impGo) impGo.onclick = runImportFlow;

  const de = $("stDelete");
  if (de) de.onclick = async () => {
    const ok = await confirmModal("Delete project",
      "<p>Permanently delete <b>" + esc(sel.project ? sel.project.name : "") + "</b> and all of its runs, " +
      "datasets and chat history? This cannot be undone.</p>", "Delete project", true);
    if (!ok) return;
    try { await del("/projects/" + sel.projectId); toast("Project deleted.", "ok"); location.hash = "#/projects"; }
    catch (e) { toast("Could not delete: " + e.message, "err"); }
  };
};

/**
 * Import is a two-step contract with the backend:
 *   1. mode="review"  → dry run, returns verdicts + would_apply, writes nothing
 *   2. mode="quick"|"smart" → the same call actually applies
 * There is no separate apply endpoint, so we re-issue the request with the
 * mode the user chooses after reading the real diff.
 */
async function runImport(sourceId, types, mode) {
  return post("/projects/" + sel.projectId + "/imports", {
    source_project_id: sourceId,
    mode: mode,
    resources: types.map((t) => ({ type: t })),
  });
}

function importResultCard(r) {
  const wa = r.would_apply || {};
  const items = wa.items || [];
  const reasons = r.reasons || [];
  return '<div class="card" style="margin-bottom:var(--s3)">' +
    '<div class="card-t"><h3>' + esc(titleCase(r.type || "resource")) + "</h3><span class=\"sp\"></span>" +
    statusBadge(r.verdict) + (r.adapted ? ' <span class="badge acc">adapted</span>' : "") + "</div>" +
    '<div class="kv">' +
    kv("Applied", r.applied ? "yes" : "no") +
    (r.detail ? kv("Note", r.detail) : "") +
    kv("Would change", items.length ? nf(items.length) + " item(s)" : (wa.note || "nothing")) +
    "</div>" +
    (items.length
      ? '<div class="tbl-wrap" style="margin-top:var(--s4)"><div class="tbl-scroll"><table class="tbl">' +
        "<thead><tr><th>Change</th><th>Operation</th><th>Inputs</th></tr></thead><tbody>" +
        items.slice(0, 20).map((it) => "<tr><td class=\"mono hint\">" + esc(it.add || it.remove || it.set || JSON.stringify(it)) + "</td>" +
          '<td class="muted">' + esc(it.operation || "—") + "</td>" +
          '<td class="muted hint">' + esc((it.inputs || []).join(", ") || "—") + "</td></tr>").join("") +
        "</tbody></table></div></div>"
      : "") +
    (reasons.length
      ? '<details class="disc" style="margin-top:var(--s4)"><summary><span class="chev">▸</span> Why</summary>' +
        "<div class=\"dbody\"><ul class=\"speclist\">" +
        reasons.map((x) => "<li><span>" + esc(x) + "</span></li>").join("") + "</ul></div></details>"
      : "") +
    "</div>";
}

/**
 * Failures are shown inline in the import pane (not a modal) so the form
 * stays usable, the message stays in context, and Retry re-runs the dry run.
 */
function showImportError(e) {
  const box = $("impErr");
  if (!box) { toast("Import failed: " + e.message, "err"); return; }
  const err = e instanceof Error ? e : new Error(String(e));
  box.innerHTML = errorState("The backend could not inspect this project",
    "The import service returned an error. Nothing was written to either project.", err);
  wireErrorBox(box, runImportFlow);
}

/** Read the form and run the dry-run review; reusable as an error Retry. */
async function runImportFlow() {
  const src = $("impSrc") ? $("impSrc").value : "";
  if (!src) { toast("Pick a source project first.", "err"); return; }
  const types = els("[data-imp]")
    .filter((c) => c.querySelector("input").checked)
    .map((c) => c.dataset.imp);
  if (!types.length) { toast("Tick at least one resource to import.", "err"); return; }
  const errBox = $("impErr");
  if (errBox) errBox.innerHTML = "";
  const go = $("impGo");
  if (go) { go.disabled = true; go.textContent = "Inspecting…"; }
  try { await showImportReview(src, types); }
  finally {
    if (go) { go.disabled = false; go.innerHTML = '<i aria-hidden="true">→</i> Inspect source'; }
  }
}

async function showImportReview(sourceId, types) {
  let r;
  try {
    r = await runImport(sourceId, types, "review");
  } catch (e) {
    /* A 500/404 here is a backend fault, not something the user can fix.
       Say so plainly, keep the detail behind a disclosure, offer Retry. */
    showImportError(e);
    toast("Import failed: " + e.message, "err");
    return;
  }
  const results = r.results || [];
  if (!results.length) {
    await modal("Import review", "<p>The backend returned no results for these resource types.</p>", [
      { label: "Close", kind: "ghost", value: null }]);
    return;
  }
  const v = await modal("Import review — nothing has been written",
    '<p class="hint">Source run <span class="mono">' + esc(r.source_run_id || "—") + "</span>. " +
    "Choose how to apply, or close to leave everything untouched.</p>" +
    results.map(importResultCard).join(""), [
    { label: "Close without importing", kind: "ghost", value: null },
    { label: "Import (quick)", kind: "ghost", value: "quick" },
    { label: "Import (smart)", kind: "pri", value: "smart" },
  ]);
  if (!v) return;
  try {
    const applied = await runImport(sourceId, types, v);
    const done = (applied.results || []).filter((x) => x.applied).length;
    const skipped = (applied.results || []).length - done;
    toast("Imported " + done + " resource(s).", done ? "ok" : "info",
      skipped ? skipped + " skipped as incompatible." : "");
    PAGES.settings();
  } catch (e) {
    showImportError(e);
    toast("Import failed: " + e.message, "err");
  }
}

/* -------------------------------------------------------- command palette */

let PAL_INDEX = -1;

async function paletteItems() {
  const out = [];
  if (!sel.projectId) return out;
  let runs = [], datasets = [];
  try { runs = cache.runs || (await api("/runs?project_id=" + sel.projectId + "&limit=100")).runs || []; } catch (e) {}
  try { datasets = (await api("/datasets?project_id=" + sel.projectId + "&page=1&page_size=100")).items || []; } catch (e) {}
  cache.runs = runs;

  NAV.forEach((g) => g[1].forEach((it) => {
    out.push({ ic: it[1], label: it[2], type: "Page", href: "#/projects/" + sel.projectId + "/" + it[0] });
  }));
  runs.forEach((r) => {
    const m = r.best ? modelName(r.best.model) : (r.stage || "run");
    out.push({ ic: "◈", label: r.run_id, sub: m, type: "Experiment",
      href: "#/projects/" + sel.projectId + "/runs/" + r.run_id });
    if (r.best) {
      out.push({ ic: "✎", label: "Report — " + m, sub: r.run_id, type: "Report",
        href: "#/projects/" + sel.projectId + "/report/" + r.run_id });
    }
  });
  datasets.forEach((d) => {
    out.push({ ic: "▤", label: d.name, sub: d.status, type: "Dataset",
      href: "#/projects/" + sel.projectId + "/datasets/" + d.id });
  });
  out.push({ ic: "⌕", label: "Developer logs and artifacts", type: "File", href: "#/projects/" + sel.projectId + "/code" });
  return out;
}

async function drawPalette(q) {
  const all = await paletteItems();
  const needle = String(q || "").trim().toLowerCase();
  const list = (needle
    ? all.filter((i) => (i.label + " " + (i.sub || "") + " " + i.type).toLowerCase().indexOf(needle) >= 0)
    : all).slice(0, 40);
  PAL_INDEX = list.length ? 0 : -1;
  const box = $("palResults");
  if (!list.length) {
    box.innerHTML = '<div style="padding:var(--s5)"><p class="muted" style="margin:0">Nothing matches “' +
      esc(needle) + "”.</p></div>";
    return;
  }
  box.innerHTML = list.map((i, n) =>
    '<button class="pal-item' + (n === 0 ? " on" : "") + '" data-p="' + n + '" type="button">' +
    '<i aria-hidden="true">' + i.ic + "</i><span>" + esc(i.label) + "</span>" +
    (i.sub ? '<span class="pt">' + esc(i.sub) + "</span>" : "") +
    '<span class="sp"></span><span class="pt">' + esc(i.type) + "</span></button>").join("");
  els("[data-p]", box).forEach((b) => {
    b.onclick = () => { closePalette(); location.hash = list[Number(b.dataset.p)].href.slice(1); };
  });
}

function openPalette() {
  $("palette").classList.add("on");
  $("palette").setAttribute("aria-hidden", "false");
  $("palInput").value = "";
  pushOverlay("palette");
  drawPalette("");
  $("palInput").focus();
}
function closePalette() {
  $("palette").classList.remove("on");
  $("palette").setAttribute("aria-hidden", "true");
  popOverlay("palette");
}

$("palInput").addEventListener("input", (e) => drawPalette(e.target.value));
$("palInput").addEventListener("keydown", (e) => {
  const items = els("#palResults .pal-item");
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (!items.length) return;
    PAL_INDEX = Math.max(0, Math.min(items.length - 1, PAL_INDEX + (e.key === "ArrowDown" ? 1 : -1)));
    items.forEach((b, n) => b.classList.toggle("on", n === PAL_INDEX));
    items[PAL_INDEX].scrollIntoView({ block: "nearest" });
  } else if (e.key === "Enter") {
    e.preventDefault();
    const b = items[PAL_INDEX];
    if (b) b.click();
  }
});

/* ------------------------------------------------------------ shell wiring */

$("menuBtn").onclick = () => $("app").classList.toggle("sidemenu");
$("collapseBtn").onclick = () => {
  const app = $("app");
  const now = app.classList.toggle("collapsed");
  $("collapseBtn").textContent = now ? "⇥" : "⇤";
  $("collapseBtn").setAttribute("aria-label", now ? "Expand sidebar" : "Collapse sidebar");
  $("collapseBtn").title = now ? "Expand sidebar" : "Collapse sidebar";
};
$("paletteBtn").onclick = openPalette;
$("sideBackBtn").onclick = () => { location.hash = "#/projects"; };
$("scrim").onclick = () => closeTop();
$("drawerClose").onclick = closeDrawer;

$("newProjBtn").onclick = newProjectDialog;
$("newProjBtn2").onclick = newProjectDialog;
$("homeAsk").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/chat"; };
$("homeRun").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/lab"; };
$("sendBtn").onclick = () => chatSend($("chatInput").value);
$("sessionSel").onchange = (e) => { sel.sessionId = e.target.value; PAGES.chat(); };
$("newSessionBtn").onclick = async () => { await newSession(); PAGES.chat(); };
$("clearChatBtn").onclick = async () => {
  if (!sel.sessionId) { toast("No session selected.", "err"); return; }
  const r = await confirmModal("Clear chat",
    "<p>This removes every message in this session. It cannot be undone.</p>", "Clear chat", true);
  if (!r) return;
  try { await del("/sessions/" + sel.sessionId + "/messages"); toast("Chat cleared.", "ok"); PAGES.chat(); }
  catch (e) { toast("Could not clear chat: " + e.message, "err"); }
};
$("chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); chatSend($("chatInput").value); }
});
$("chatInput").addEventListener("input", autoGrow);
$("chatConfigBtn").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/lab"; };
$("chipAttach").onclick = uploadDialog;
$("chipMetric").onclick = metricDialog;
$("chipConstraint").onclick = constraintDialog;
$("chatFab").onclick = () => {
  const w = $("chatDrawer");
  if (w && w.classList.contains("on")) closeChatDrawer();
  else openChatDrawer();
};
$("drawerChatClose").onclick = closeChatDrawer;
$("drawerChatSend").onclick = drawerSend;
$("drawerChatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); drawerSend(); }
});
$("drawerChatMetrics").onclick = async () => { closeChatDrawer(); metricDialog(); };
$("drawerChatLab").onclick = () => { closeChatDrawer(); location.hash = "#/projects/" + sel.projectId + "/lab"; };
$("dsBack").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/data"; };
$("runBack").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/experiments"; };
$("repBack").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/reports"; };
$("newExpBtn").onclick = () => { location.hash = "#/projects/" + sel.projectId + "/lab"; };
$("addDataBtn").onclick = uploadDialog;
$("dataSearch").addEventListener("input", () => { if (sel.page === "data") PAGES.data(); });
$("dataFilter").addEventListener("change", () => { if (sel.page === "data") PAGES.data(); });

/* keyboard: Esc closes the top overlay, Ctrl/Cmd+K opens search */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    const w = $("chatDrawer");
    if (w && w.classList.contains("on")) { closeChatDrawer(); return; }
    if (closeTop()) return;
    /* nothing open — leave the run/detail view the way the user came from */
    if (sel.page === "run") location.hash = "#/projects/" + sel.projectId + "/experiments";
    else if (sel.page === "report") location.hash = "#/projects/" + sel.projectId + "/reports";
    else if (sel.page === "dataset") location.hash = "#/projects/" + sel.projectId + "/data";
    return;
  }
  if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
    e.preventDefault();
    if ($("palette").classList.contains("on")) closePalette();
    else openPalette();
  }
});

/* ------------------------------------------------------------------- init */

loadCfg();
paintSaveState();
connectWS();
initChartHover();
setInterval(paintLink, 3000);
autoGrow();
render();
