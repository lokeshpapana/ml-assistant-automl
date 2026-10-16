# Phase B Implementation Report

Date: 2026-09-22. Result: **complete, all acceptance criteria met.** Stopped here
per instruction — no Phase C work started.

## Files changed (3, all additive/minimal)

1. `backend/orchestrator.py` — added `project_id`/`session_id`/`run_id` fields
   (`__init__`), set from start-config + generated `run-{12 hex}` in `start()`,
   `run_context()` helper, `run_context` block in `snapshot()` and
   `leaderboard()`, context stamps in `_emit()`, top-level→config merge in
   `handle_message(start)`. Thread/state-machine logic untouched.
2. `backend/api/automl.py` — `POST /automl/start` validates `project_id`/
   `session_id` (404/400, starts nothing on failure); WS handler validates
   `event=="start"` the same way (error `ack`, socket survives); new
   `GET /workspace`; `_validate_ws_start` helper; docstring updated. `GET /ui`
   and every other endpoint byte-identical in behavior.
3. `tests/test_phase_b.py` — new (8 tests). No existing test modified.

## Files added (4)

1. `backend/project_context.py` — `extract_run_context()` (both payload shapes,
   old flat-config bodies unaffected) + `validate_run_context()` (UUID parse →
   Project exists → optional Session exists + belongs to project).
2. `backend/static/workspace.html` — self-contained workspace shell (no shared
   modules, so `automl_ui.html` is provably untouched): topbar, project sidebar
   (real `GET /projects`, inline create via `POST /projects`), hash routing,
   Overview (real project + dataset/experiment counts), Chat (WS, suggestions,
   project-scoped start), 7 honest placeholders, live activity rail, live run
   view, responsive drawers.
3. `.agent/PHASE_B_AUDIT.md`, `.agent/PHASE_B_BLOCKERS.md`,
   `.agent/SECURITY_LIMITATIONS.md` (Step 2), this file,
   `.agent/PROJECT_ARCHITECTURE.md`.

## APIs changed

- `POST /automl/start`: same body shape accepted; optional `project_id`/
  `session_id` (top-level or config-nested) now validated. Invalid → 404/400.
- `WS /ws/automl`: `start` messages validated identically; failures return
  `{"type":"ack","status":"error","detail","code"}` instead of starting.
- `snapshot()`/`leaderboard()`: new `run_context{project_id,session_id,run_id,
  status,started_at}` block. All WS events carry the three tags (`setdefault`,
  so classic ad-hoc runs emit `null`s — backward compatible).
- New: `GET /workspace` (static shell). Nothing else added, removed, or renamed.

## project_id flow (verified end-to-end)

UI selection → hash `#/projects/:id/chat` (+sessionStorage) → WS
`{event:"start", project_id, config}` → `_validate_ws_start` → `db.get(Project)`
→ `handle_message` merges into config → `start()` stores context →
every `_emit` + `snapshot`/`leaderboard` carries it → workspace filters panels by
selected project. REST path mirrors via `extract/validate_run_context`.

## run_id flow

Generated in `start()` as `run-{uuid4hex:12}` (per campaign, survives on the
finished run for display). Stamped on all events + `run_context`. Run view route
`#/projects/:id/runs/:runId` renders the in-memory log for the live run only.

## WebSocket changes

Validation gate for `start` only; pump/replay/subscribe/disconnect logic
identical. Error acks never raise (socket survives — verified live).

## Frontend state changes

`sel{projectId,page,runId,project}` derived from `location.hash` (URL-identifiable,
survives refresh of *selection*, not of run data); per-project live `run{runId,
stage,iters,maxIters,lastTask,error,log≤500}` rebuilt from tagged events;
switching projects resets the run view (honest: no persistence in Phase B).

## Routing

Hash routes (no framework): `#/projects`, `#/projects/:id/chat` (default),
`#/projects/:id/overview|data|analysis|research|features|experiments|evaluation|
report` (7 placeholders), `#/projects/:id/runs/:runId` (live view). Invalid id →
toast + back to list.

## Known limitations (all labeled in-UI, none faked)

- One global run (singleton); second start → `busy` naming the state.
- Runs not persisted/resumable; refresh clears activity; run view = live only.
- Classic-`/ui` runs (`project_id null`) hidden from project views (isolation).
- `POST /sessions` has a pre-existing response-serialization bug (found while
  testing; NOT fixed per minimal-change rule — test creates the row directly).
- Upload endpoint reused as-is (validation gaps documented, unchanged).
- 8 audit bugs triaged in `PHASE_B_BLOCKERS.md`; only avatar mapping handled
  (both spellings mapped in the new shell).

## Tests performed → passed

- `tests/test_phase_b.py` (8): shell route + `/ui` regression, context shapes,
  emit tagging, unknown→404 + no state change, malformed→400, WS helper
  error-ack + no state change, session/project mismatch→400, two-project
  isolation incl. busy-guard + stop + per-run tagging. **8/8.**
- Full suite: **40/40** (32 pre-existing + 8 new), 15.39s.
- Live drill vs `:8000` (`/tmp/opencode/live_iso.py`): invalid→error 404; A via
  WS started with 9 tagged events + run_id; B while busy→busy; classic
  start→started with null tags. **OK.**
- Content checks: `/workspace` 200 `text/html`, markers present, zero banned
  claims (`resumable/checkpoint/my projects`).

## Tests not possible / not done

- No ASGI socket-level WS test harness (httpx has no WS; TestClient portal
  mixing judged riskier than value) — WS path covered by helper-level test +
  live drill above instead.
- No frontend unit tests (no harness exists in repo); verified via live serving
  + content assertions + manual drill.
- Multi-process concurrency beyond singleton semantics: N/A by design.
