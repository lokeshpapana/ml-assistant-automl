# Phase B Blockers — Known-Issue Triage

Rule: fix ONLY what blocks Phase B. Everything else is documented and deferred.

| # | Issue (from UI_AUDIT) | Impact on Phase B | Blocks? | Action |
|---|---|---|---|---|
| 1 | Agent avatar key mismatch (`data-fetcher`/`research` vs `data_fetcher`/`researcher`) | Workspace chat would show grey fallback avatars for data/research agents | NO (cosmetic) | Map **both** spellings in new `workspace.html` avatar map; leave `automl_ui.html` untouched |
| 2 | `leaderboard()` always-None `best` | Workspace "best model" widget would be empty if it used this endpoint | NO (avoidable) | Workspace reads `best` from `snapshot()` (correct source); no backend fix in Phase B |
| 3 | DB `ModelType` ⇄ registry enum mismatch | None in Phase B — workspace creates no `Model` rows | NO | Deferred to Phase E/H |
| 4 | Researcher `source` vs `a/b` spec gap | None in Phase B — research runs unchanged, cards render as today | NO | Deferred (Phase E research page) |
| 5 | Celery `project_id` gap | None — Phase B uses no Celery paths | NO | Deferred (separate track) |
| 6 | `ml_tasks.py:53` `db.get` bug | Celery-only path, unused by workspace | NO | Deferred |
| 7 | Upload validation gaps | Workspace reuses `POST /automl/upload` as-is; same exposure as `/ui`, no new risk | NO | Deferred; document in implementation report |
| 8 | Dead `llmPill` | Workspace will NOT show an LLM-provider pill (nothing truthful to show without new backend data) | NO | Show static "local mode" deployment badge instead; no backend change |

**Phase-B-specific blockers found during audit: none.** Required capabilities all
exist: project CRUD (`api/projects.py`), session read (`api/sessions.py`),
config-dict start (`api/automl.py:46-48`), `ingest_form` ignoring unknown keys
(`interviewer.py:132-158`), async test client pattern (`tests/test_main.py`,
`ASGITransport`, `asyncio_mode=auto`), UUID models with `get_async_session`.
