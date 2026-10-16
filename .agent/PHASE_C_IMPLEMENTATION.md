# Phase C — Persistent Project Chat (Implemented)

Goal: chat survives refresh, is bound to `project_id` + `session_id`, restores on
open, understands project state, and never resends whole history.

## Schema
- `chat_messages(id, project_id?, session_id?, role, agent?, text, created_at)` —
  same SQLite DB, dedicated table (NOT a text blob in `Session.metadata_`: that
  would grow unbounded and contend on concurrent writes). Indexes on
  `(session_id, created_at)` and `(project_id, created_at)`.

## Files
- Added `backend/runlog.py::save_chat / list_chat / clear_chat / count_chat / compact_session`.
- `backend/orchestrator.py::handle_message(chat)` persists the user turn, routes
  it (pending confirmation → structured action → interview), and persists the
  agent turn through `_agent_reply`.
- `backend/api/sessions.py`: `GET /sessions/{id}/messages`, `DELETE /sessions/{id}/messages`.

## Behaviour
- Session management: list, create, select, per-session clear, session title.
- `chatMsg` now sends `project_id` + `session_id`, so history is attributable.
- Rule-based compaction: >300 turns digests the oldest into memory
  (`chat_digest`) and keeps the newest 100 verbatim. No LLM involved.
- Context resolution uses project memory, not re-asking: "use the same dataset",
  "try the previous model", "undo that", "use the features from experiment 3".
- Structured, validated actions (never free-form DB/SQL):
  `select_model` (registry-validated), `remove_feature`, `undo_last_feature`,
  `set_target_column`, `use_dataset`. Destructive ones require an explicit
  confirmation turn; a confirmation can never simultaneously start a run.

## Tests
`tests/test_phases_c_h.py`: history endpoint + clear, per-project isolation,
dedupe/scalar memory sections, confirm-gate regression, action parsing,
`/tmp/opencode/e2e_drill.py` (persistence across re-read + confirmation gating).

## Notes
- Chat is intentionally NOT compacted by an LLM (would be a hidden cost and a
  non-deterministic record); the digest is factual and inspectable.
