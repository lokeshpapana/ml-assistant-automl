# Security Limitations (current, Phase B scope)

> **Project access is currently not user-isolated.**

1. **No authentication.** `backend/auth/` is an empty directory; no login, token,
   API-key, or session-cookie mechanism exists anywhere in the codebase. The
   dormant `secret_key` in `backend/config/settings.py:32` is unused.
2. **No authorization.** Every endpoint (all 47 HTTP + WS `/ws/automl`) is open to
   anyone who can reach the host. Any client can list/create/delete projects,
   upload files, start/stop runs, and read all state and artifacts.
3. **No project ownership.** Projects have no owner field (`database/models.py:
   Project` has `id/name/description/status/timestamps` only). The workspace MUST
   NOT use "My Projects" or any ownership language — it lists *all* projects.
4. **No tenant isolation.** `project_id` in Phase B is an association label for
   correct event routing, NOT a security boundary. A client can pass any
   project's id.
5. **Uploads are unvalidated** (any suffix/size server-side) and CORS allows the
   configured origins with credentials (`backend/main.py:36-42`).

Do not implement auth in Phase B (explicitly out of scope). Revisit before any
multi-user or network-exposed deployment.
