# Phase I — UI Consistency Pass (Implemented)

Every control now works, is intentionally disabled with a reason, or is gone.

| Issue (from the audit) | Resolution |
|---|---|
| Dead `llmPill` in `/ui` | Both shells read `snapshot().llm.provider` (backend now reports it) |
| Export button 404s pre-training | Workspace button is disabled with the tooltip "No evaluated model yet — complete a run first" and re-enables on `best_model_ready`; no raw navigation to a 404 |
| Start clickable while busy | Button is disabled without a project; the `busy` ack still toasts the blocking state |
| Upload trusted the response | Upload paths check `res.ok` and surface `detail` as a toast (chat + data page) |
| Avatar key mismatch (`data-fetcher`/`research`) | `AGENTS` map accepts both spellings |
| Chart showed nothing for 1 iteration | Placeholder retained until 2 points, and the Metrics tab is secondary to chat |
| Non-numeric metrics → `NaN` | All metric rendering goes through `Number(...).toFixed` guards; backend coerces with `json_safe` |
| Placeholder pages with no data | Replaced by real pages; each states exactly what is missing and why |
| `placeholder` class left unused | Reused for genuine empty states (no run yet, no dataset yet) |
| Stale run state after switching project | `resetRunView()` on selection; foreign runs explicitly reported as belonging elsewhere |
| Stale `POST /sessions` bug | Fixed (Phase E) — session create/list return real JSON |
| Duplicate/no-op markup | The placeholder view was removed; nav gained real Compare/Runs entries |

Empty/loading/error states: project list, run list, features, research,
experiments, evaluation, report, import, compare, chat, dataset preview.
Live region + reduced-motion support added for accessibility.

## Final batch (closing the spec gaps)
- **Activity rail → full page**: `#/projects/:id/activity` renders every run's
  persisted timeline (durable, survives refresh), with a nav entry and a link
  from each block to the run view.
- **Run view events are expandable**: each timeline row reveals status, stage,
  duration and its full metadata payload (model, params, metrics, MLflow run id).
  Operational events only — no chain-of-thought is stored or shown.
- **Experiments filtering/sorting**: model, status and sort key (started,
  objective metric, iteration count, duration), all client-side over the real
  persisted runs; the objective comes from the run config.
- **Import REVIEW diff**: review mode returns `would_apply` (concrete items:
  feature names + operation + inputs, files, config keys, findings) so the
  import dialog shows a real diff before applying.
- Dead code removed: the superseded live-only `renderRunView` (the run page is
  durable now) — verified by an esprima parse of the served script.
