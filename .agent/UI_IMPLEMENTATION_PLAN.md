# UI Implementation Plan — AI-First AutoML Workspace

Constraint: no-build static frontend (keep shipping `.html` + one shared JS core
unless a phase explicitly justifies otherwise). Preserve all working behavior;
`GET /ui`, `WS /ws/automl`, and the 11-event contract stay compatible. Rule per
phase: implement → run app → test → fix → update report → continue. Test suite
must stay green (currently 32 passed).

## PHASE A — Audit (DONE 2026-09-22)

Outputs: `.agent/UI_AUDIT.md`, `.agent/PROJECT_MODEL.md`, this file. Exit criteria:
reports reviewed, Phase B scope confirmed.

## PHASE B — Project/workspace routing + app shell

Goal: persistent shell (top bar / project sidebar / main chat / activity rail) and
real routes. Since there is no frontend router, routes = static shells
(`projects.html`, `project.html` + query/hash sub-views, or one `app.html` with
hash routing `#/projects/:id/chat`) sharing `static/js/automl-core.js` (extracted
from current inline script: WS client, snapshot rendering, chart, toast).
- Backend (minimal): `GET /projects/{id}/workspace` aggregator (project +
  sessions + datasets + experiments + latest automl summary) to avoid N+1 fetches.
  MISSING BACKEND CAPABILITY otherwise: none (CRUD exists).
- Bridge start: `POST /automl/start` accepts optional `project_id`/`session_id`,
  stored on orchestrator, echoed in snapshot. No DB writes yet.
- Acceptance: sidebar lists projects (`GET /projects`), `+ New` creates and opens
  workspace, chat works, activity rail shows live run. Old `/ui` untouched.

## PHASE C — Persistent chat (per-project, survives refresh)

Goal: chat history per project/session, agent keeps project context automatically.
- Backend: persist chat turns. Smallest change: append `{ts, role, agent?, text,
  cards?}` to `Session.metadata_['automl']['chat']` (bounded, e.g. last 200) on
  every `_emit` of `chat_response/agent_message/status`; `GET
  /sessions/{id}` returns it; WS `snapshot` includes it for late join (extends
  existing replay). MISSING BACKEND CAPABILITY: none if metadata_ approach taken
  (no migration); a `chat_messages` table is the cleaner follow-up.
- Frontend: chat pane loads history on open; suggestion chips stay; add
  project-scoped starter prompts.
- Acceptance: reload mid-run → history intact; new message gets context-aware reply
  (requirements prefilled from stored session state, no repeated questions).

## PHASE D — Live activity/progress (recording-style, no chain-of-thought)

Goal: secondary rail showing WHAT/WHY/COMPLETED/NEXT/NEEDS-DECISION/FAILED, plus
full run view `/projects/:id/runs/:runId` (timeline with durations, artifacts,
metrics, errors; expandable events).
- Backend: structured run events. Orchestrator already emits all needed signals;
  add `{run_id, t_start/t_end, status, detail_refs}` envelope and persist the
  append-only log (JSONL per run under `automl_outputs/runs/{runId}/events.jsonl`
  first; `run_events` table later). Add progress = completed/total steps + elapsed.
  Introduce `run_id = {started_at}-{counter}` (no orchestrator rewrite).
  MISSING BACKEND CAPABILITY: event persistence + `GET /automl/runs/{run_id}`.
- Frontend: activity rail component (shared core), run page rendering the timeline.
- Acceptance: start run → rail animates through steps with elapsed time; refresh →
  run page replays from persisted log; errors show friendly card + [Retry]
  [Reduce data] [Smaller model] actions mapping to real config changes.

## PHASE E — Project pages (Data/Analysis/Research/Features/Experiments/Evaluation/Report)

Goal: one page per stage backed by REAL data; chat drawer available on every page.
- Backend: read-only aggregators first (all data exists): dataset preview/profile
  (`Dataset.profile` + central store preview), analysis (`profile_dataset`
  output), research (feature_history + papers proxy), features (feature_history +
  store columns), experiments (iterations + MLflow runs + `Model` rows once Phase B
  writes them), evaluation (`/ml/.../results`-style ranking over iterations),
  report (assemble stored dicts → new `GET /projects/{id}/report` returning
  markdown-ready JSON; file export later). MISSING BACKEND CAPABILITY: report
  assembler endpoint; write-through of iterations→`Experiment`/`Model` rows.
- Frontend: per-stage shells sharing core + a persistent chat drawer (fixed right
  drawer reusing the same WS client; badge on agent question/decision-needed).
- Acceptance: each page loads with real content after a run; chat drawer works
  from every page; stage nav shows ✓/⟳/○/! from actual state.

## PHASE F — Project memory + token efficiency

Goal: structured memory, compact context, no full-history resends.
- Backend: `Session.metadata_['automl']['memory']` = `{goal, dataset, target,
  stage, best_model, findings[], known_issues[], decisions[], imports[]}` updated
  at stage boundaries (interview confirm, data ready, research done, each
  iteration best-change, finalize). Expose `GET /projects/{id}/context` returning
  the compact `PROJECT_CONTEXT.json` shape from the spec. Retrieval = this doc +
  current task + referenced rows only. (Already cheap: agents are rule-based; this
  future-proofs LLM providers and multi-project chats.)
- Acceptance: context endpoint returns <2KB summary; agent answers use stored
  dataset/target without re-asking; memory visible in a UI section.

## PHASE G — Import/reuse system (quick / smart / review + provenance)

Goal: "Import from project" with resource picker, compatibility analysis, and
provenance on everything imported.
- Backend: `GET /projects/{id}/resources` (datasets, pipelines/configs, features,
  models, research, report templates — assembled from existing rows + artifacts);
  `POST /projects/{id}/imports {source_project_id, resources[], mode:
  quick|smart|review}` performing copy + compatibility check (schema/feature overlap
  via stored profiles; verdict COMPATIBLE/PARTIAL/INCOMPATIBLE + reasons) and
  writing `project_imports` provenance rows. MISSING BACKEND CAPABILITY: resources
  aggregator, import job, `project_imports` table, diff generator for review mode.
- Frontend: import dialog (picker → verdict → diff → confirm), provenance badges
  on imported items (`Source: X · EXP-nn · date · modified by`).
- Acceptance: import dataset+pipeline across projects; PARTIAL verdict blocks
  auto-apply in CONFIRM/MANUAL modes; provenance visible and correct.

## PHASE H — Experiment comparison + evaluation visuals

Goal: comparison table (EXP/Model/Features/F1/Acc/Runtime + Compare/Open/Reproduce/
Clone/Import/Export), model detail (confusion matrix, ROC/PR where applicable,
error analysis, explainability via existing feature importance + residual plots).
- Backend: mostly present (`comparison` endpoint, ranking helpers); add clone/
  reproduce endpoints reusing stored configs. Confusion-matrix/ROC data: compute
  from stored predictions — MISSING BACKEND CAPABILITY: persist per-model
  predictions (or recompute endpoint `GET /models/{id}/diagnostics` rerunning the
  stored `.pkl` on the stored dataset snapshot — artifacts already saved per
  iteration, so this is feasible without new storage).
- Acceptance: 3-run comparison renders; clone reproduces metrics within tolerance.

## PHASE I — UI polish (spec §24)

Dark theme kept; typography/spacing/hierarchy pass; status colors reserved for
semantics; empty/loading/error states completed (incl. export-disabled state,
upload validation messages, chart single-point + NaN handling, avatar key fix,
`llmPill` live value); `esc()` hardened; `prefers-reduced-motion` respected.

## PHASE J — Responsive behavior (spec §25)

Sidebar → collapsible; activity rail → drawer; chat primary on all sizes; header
wraps; tables scroll; `acquire_ui.html` gets the breakpoint treatment or is folded
into the workspace.

## PHASE K — Testing (spec §28 scenarios)

Extend suite per phase (project CRUD+workspace open, chat round-trip, activity
updates, run page, restart persistence, import + provenance, conflict verdict,
autonomy modes, full nav with chat accessible). E2E script driving WS + REST
against synthetic runs. Update `.agent/UI_FINAL_REPORT.md` (spec §29, all 20
items) only after K is green.

## Cross-phase backend capability checklist (minimum changes)

1. `project_id/session_id` on automl start + snapshot echo (B).
2. Chat persistence (metadata_ or table) + history in snapshot (C).
3. Run events JSONL + `GET /automl/runs/{run_id}` + progress fields (D).
4. Iterations→`Experiment`/`Model` write-through; report assembler (E).
5. `GET /projects/{id}/context` compact memory doc (F).
6. Resources aggregator + import job + `project_imports` + diff (G).
7. `GET /models/{id}/diagnostics` (recompute from stored artifacts) (H).
8. AuthN/Z (recommended before any multi-user exposure; currently none).

## Explicit non-goals for this plan

No frontend framework migration, no SSE (WS suffices), no Postgres requirement
(SQLite stays default), no chain-of-thought display (operational events only),
no blind file copying in imports (compatibility gate is mandatory).
