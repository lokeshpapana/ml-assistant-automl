"""Single Source-of-Truth dataset store for the Interactive AutoML engine.

CRITICAL RULE (from the system specification): no duplicated/versioned CSV files
are kept on disk during feature engineering. The active dataset lives in exactly
one table (DuckDB preferred, SQLite fallback). Every cleaning / sampling /
feature-engineering operation mutates that single table in place via
ALTER / UPDATE SQL (or a vectorized write-back that mutates the same table), so
all agents always observe the exact same state.
"""

from __future__ import annotations

import re
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_TABLE = "central_dataset"

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _json_num(value: Any) -> Any:
    """Convert numpy/DuckDB scalars to JSON-safe Python primitives."""
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)) and not isinstance(value, bool):
        return value
    try:
        import numpy as _np
        from numpy import float64, integer

        if isinstance(value, (integer, float64, _np.floating, _np.integer)):
            return float(value) if isinstance(value, _np.floating) else int(value)
    except Exception:
        pass
    return value


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _column_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name)).strip("_")
    if not value or not _IDENTIFIER_RE.match(value):
        value = f"col_{len(value) if value else 0}"
    return value


class CentralStore:
    """Thread-safe single-instance dataset store backed by DuckDB (or SQLite)."""

    def __init__(self, db_path: str | None = None, memory: bool = False):
        """``db_path`` defaults to ``settings.central_store_path``.

        Reading the default from settings (instead of a hardcoded literal)
        keeps one source of truth for storage locations, so an overridden
        ``CENTRAL_STORE_PATH`` is honoured by every construction site.
        """
        if db_path is None:
            from backend.config import settings

            db_path = settings.central_store_path
        self._lock = threading.RLock()
        self._db_path = db_path
        self._memory = memory
        self._backend = "duckdb"
        self._conn = None
        self._name = "central_dataset"
        self._columns: list[str] = []
        self._dtypes: dict[str, str] = {}
        self.feature_log: list[dict[str, Any]] = []
        self.op_log: list[dict[str, Any]] = []  # every add/drop, for Features page + undo
        self._connect()

    # ------------------------------------------------------------------ lifecycle

    def _connect(self) -> None:
        try:
            import duckdb  # type: ignore

            if self._memory:
                self._conn = duckdb.connect(":memory:")
            else:
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
                self._conn = duckdb.connect(self._db_path)
            self._backend = "duckdb"
        except Exception:
            import sqlite3

            self._backend = "sqlite"
            self._conn = sqlite3.connect(
                ":memory:" if self._memory else self._db_path, check_same_thread=False
            )
        self._refresh_schema_if_exists()

    def reset(self) -> None:
        """Drop the active table and all derived features. Prepares a fresh run."""
        with self._lock:
            self._exec(f"DROP TABLE IF EXISTS {_quote(_TABLE)}")
            self._columns = []
            self._dtypes = {}
            self.feature_log = []
            self.op_log = []

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def table_name(self) -> str:
        return _TABLE

    # ------------------------------------------------------------------ helpers

    def _exec(self, sql: str) -> Any:
        if self._backend == "duckdb":
            return self._conn.execute(sql)
        return self._conn.execute(sql)

    def _fetch(self, sql: str) -> pd.DataFrame:
        if self._backend == "duckdb":
            return self._conn.execute(sql).df()
        return pd.read_sql_query(sql, self._conn)

    def _fetchall(self, sql: str) -> list[tuple]:
        if self._backend == "duckdb":
            return self._conn.execute(sql).fetchall()
        return self._conn.execute(sql).fetchall()

    def _refresh_schema(self) -> None:
        rows = (
            self._fetchall(f"PRAGMA table_info({_quote(_TABLE)})")
            if self._backend == "sqlite"
            else self._fetchall(f"DESCRIBE {_quote(_TABLE)}")
        )
        self._columns = []
        self._dtypes = {}
        for row in rows:
            if self._backend == "sqlite":
                # PRAGMA returns (cid, name, type, notnull, dflt_value, pk)
                name, dtype = row[1], row[2]
            else:
                name, dtype = row[0], row[1]
            self._columns.append(name)
            self._dtypes[name] = str(dtype)

    def _refresh_schema_if_exists(self) -> None:
        """Populate the in-memory column/dtype cache when the persisted table
        already exists (e.g. a fresh process reconnecting to a DuckDB file).

        Without this, ``preview()``/``columns()`` report zero columns for a live
        table and the UI can't list the dataset or its features.
        """
        with self._lock:
            try:
                if self._exec(
                    f"SELECT count(*) FROM sqlite_master WHERE type='table' AND name={_quote(_TABLE)}"
                ).fetchall():
                    self._refresh_schema()
            except Exception:
                try:
                    self._fetchall(f"SELECT 1 FROM {_quote(_TABLE)} LIMIT 1")
                    self._refresh_schema()
                except Exception:
                    pass

    # ------------------------------------------------------------------ loading

    def load_dataframe(
        self, df: pd.DataFrame, max_rows: int | None = None, sample_method: str = "random"
    ) -> dict[str, Any]:
        """Load a DataFrame into the single table, applying an optional row limit."""
        with self._lock:
            self.reset()
            df = df.copy()
            if max_rows and max_rows > 0 and len(df) > max_rows:
                if sample_method == "stratified" and "ny" not in df.columns:
                    # No target known at load time yet; random sampling is safe.
                    df = df.sample(n=max_rows, random_state=42)
                else:
                    df = df.sample(n=max_rows, random_state=42)
            df.columns = [_column_name(c) for c in df.columns]
            if hasattr(df, "index") and df.index.name is None:
                df = df.reset_index(drop=True)

            if self._backend == "duckdb":
                self._conn.begin()
                try:
                    self._conn.register("_df_in", df)
                    self._conn.execute(f"CREATE TABLE {_quote(_TABLE)} AS SELECT * FROM _df_in")
                    self._conn.unregister("_df_in")
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise
            else:
                df.to_sql(_TABLE, self._conn, if_exists="replace", index=False)
                self._conn.commit()

            self._refresh_schema()
            return {"rows": len(df), "columns": self._columns, "backend": self._backend}

    def load_path(self, path: str | Path, max_rows: int | None = None) -> dict[str, Any]:
        p = Path(path)
        if p.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        elif p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        elif p.suffix.lower() == ".json":
            df = pd.read_json(p)
        else:
            df = pd.read_csv(p)
        return self.load_dataframe(df, max_rows=max_rows)

    # ------------------------------------------------------------------ reads

    def shape(self) -> tuple[int, int]:
        with self._lock:
            row = self._fetchall(f"SELECT COUNT(*) FROM {_quote(_TABLE)}")[0][0]
            return int(row), len(self._columns)

    def columns(self) -> list[str]:
        with self._lock:
            return list(self._columns)

    def dtypes(self) -> dict[str, str]:
        with self._lock:
            return dict(self._dtypes)

    def preview(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        with self._lock:
            n_rows, n_cols = self.shape()
            sample = self._fetch(
                f"SELECT * FROM {_quote(_TABLE)} ORDER BY rowid LIMIT {int(limit)} OFFSET {int(offset)}"
            )
            records = sample.replace({np.nan: None}).to_dict(orient="records")
            return {
                "rows": n_rows,
                "columns": list(self._columns),
                "dtypes": dict(self._dtypes),
                "preview": records,
                "limit": int(limit),
            }

    def column_stats(self, limit: int = 120) -> dict[str, Any]:
        """Real per-column statistics computed from the live working table.

        Values are derived from the actual single-source table via SQL
        aggregates — never estimated. Each column reports null/unique counts,
        and either numeric min/mean/std/max or the top categorical values.
        """
        with self._lock:
            cols = list(self._columns)[: int(limit)]
            out: dict[str, Any] = {}
            for col in cols:
                name = _quote(col)
                dtype = str(self._dtypes.get(col, "")).upper()
                numeric = any(
                    k in dtype for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC")
                )
                base = {
                    "type": self._dtypes.get(col, ""),
                    "nulls": int(
                        self._fetchall(f"SELECT COUNT(*) - COUNT({name}) FROM {_quote(_TABLE)}")[0][
                            0
                        ]
                        or 0
                    ),
                }
                try:
                    base["uniques"] = int(
                        self._fetchall(f"SELECT COUNT(DISTINCT {name}) FROM {_quote(_TABLE)}")[0][0]
                        or 0
                    )
                except Exception:
                    base["uniques"] = None
                total = self._fetchall(f"SELECT COUNT(*) FROM {_quote(_TABLE)}")[0][0] or 1
                base["null_pct"] = round(base["nulls"] / total * 100, 2) if total else 0.0
                if numeric:
                    row = self._fetchall(
                        f"SELECT MIN({name}), AVG({name}), STDDEV({name}), MAX({name}) "
                        f"FROM {_quote(_TABLE)} WHERE {name} IS NOT NULL"
                    )[0]
                    base.update(
                        {
                            "min": _json_num(row[0]),
                            "mean": _json_num(row[1]),
                            "std": _json_num(row[2]),
                            "max": _json_num(row[3]),
                        }
                    )
                else:
                    try:
                        tops = self._fetchall(
                            f"SELECT {name}, COUNT(*) c FROM {_quote(_TABLE)} "
                            f"WHERE {name} IS NOT NULL GROUP BY {name} ORDER BY c DESC, {name} LIMIT 5"
                        )
                        base["top"] = [
                            _json_num(t[0]) if _is_number(t[0]) else str(t[0]) for t in tops
                        ]
                    except Exception:
                        base["top"] = []
                out[col] = base
            return {"columns": out, "count": len(out)}

    def to_dataframe(self) -> pd.DataFrame:
        with self._lock:
            return self._fetch(f"SELECT * FROM {_quote(_TABLE)}")

    def export_csv(self, path: str | Path) -> str:
        df = self.to_dataframe()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)

    # ------------------------------------------------------------------ in-place mutation

    def add_feature(self, feature: dict[str, Any]) -> dict[str, Any]:
        """Apply one engineered feature directly into the single table.

        Canonical spec: ``{name, kind, a?, b?, ...}`` where ``a`` is the primary
        operand column and ``b`` the second operand for binary ops. ``source``
        is accepted as a legacy alias of ``a``. Supported kinds: ratio,
        product, sum, difference, log1p, sqrt, square, abs, interaction,
        quantile_bin, freq_encode, year, month, day, dayofweek.
        """
        with self._lock:
            kind = feature.get("kind", "custom")
            name = _column_name(feature.get("name") or f"feat_{len(self.feature_log) + 1}")
            if name in self._columns:
                raise ValueError(f"Feature column '{name}' already exists in the central store")

            try:
                if kind == "freq_encode":
                    col = self._require_column(feature.get("source") or feature.get("a", ""))
                    self._add_column(name, "DOUBLE")
                    self._exec(
                        f"UPDATE {_quote(_TABLE)} SET {_quote(name)} = "
                        f"(SELECT cnt FROM (SELECT {_quote(col)} AS key_, COUNT(*) AS cnt "
                        f"FROM {_quote(_TABLE)} GROUP BY {_quote(col)}) AS g "
                        f"WHERE g.key_ = {_quote(_TABLE)}.{_quote(col)})"
                    )
                elif kind in ("quantile_bin",):
                    col = self._require_column(feature.get("source") or feature.get("a", ""))
                    nbins = max(2, int(feature.get("bins", 4)))
                    self._add_column(name, "BIGINT")
                    # A window function cannot appear in UPDATE ... SET, and a
                    # scalar subquery returning one row per row is rejected by
                    # DuckDB. UPDATE ... FROM a windowed projection (matched on
                    # rowid) is the portable form (DuckDB + SQLite >= 3.33).
                    try:
                        self._exec(
                            f"UPDATE {_quote(_TABLE)} SET {_quote(name)} = s.bin FROM "
                            f"(SELECT rowid AS rid, NTILE({nbins}) OVER (ORDER BY {_quote(col)}) AS bin "
                            f"FROM {_quote(_TABLE)}) AS s WHERE {_quote(_TABLE)}.rowid = s.rid"
                        )
                    except Exception:
                        # Engines without rowid support: compute in pandas and
                        # write the column back through the same single table.
                        self._assign_from_pandas(name, self._compute_bins(col, nbins))
                elif kind in ("year", "month", "day", "dayofweek"):
                    col = self._require_column(feature.get("source") or feature.get("a", ""))
                    part = {
                        "year": "YEAR",
                        "month": "MONTH",
                        "day": "DAY",
                        "dayofweek": "DAYOFWEEK",
                    }[kind]
                    self._add_column(name, "DOUBLE")
                    self._exec(
                        f"UPDATE {_quote(_TABLE)} SET {_quote(name)} = CAST(EXTRACT({part} FROM CAST({_quote(col)} AS TIMESTAMP)) AS DOUBLE)"
                    )
                else:
                    expr = feature.get("expression") or self._expression(kind, feature)
                    self._add_column(name, "DOUBLE")
                    self._exec(f"UPDATE {_quote(_TABLE)} SET {_quote(name)} = {expr}")
            except Exception as exc:
                raise ValueError(
                    f"Feature engineering failed for '{name}' ({kind}): {exc}"
                ) from exc

            self._refresh_schema()
            log_entry = dict(feature)
            log_entry["name"] = name
            log_entry["applied_ts"] = time.time()
            self.feature_log.append(log_entry)
            self.op_log.append(
                {"op": "add", "name": name, "kind": kind, "ts": log_entry["applied_ts"]}
            )
            return log_entry

    def drop_column(self, name: str) -> None:
        """Remove a column in place (feature disable/remove, undo). Logged in op_log."""
        with self._lock:
            col = self._require_column(name)
            self._exec(f"ALTER TABLE {_quote(_TABLE)} DROP COLUMN {_quote(col)}")
            self._refresh_schema()
            self.op_log.append({"op": "drop", "name": col, "ts": time.time()})

    def _compute_bins(self, col: str, nbins: int) -> list[int]:
        """NTILE(n) bin labels aligned with the table's current row order."""
        order = self._fetch(f"SELECT {_quote(col)} AS v FROM {_quote(_TABLE)}")
        values = order["v"].tolist()
        n = len(values)
        bins = [0] * n
        if n:
            indexed = sorted(range(n), key=lambda i: (values[i] is None, values[i]))
            per = n / float(nbins)
            for rank, i in enumerate(indexed):
                bins[i] = min(nbins, int(rank / per) + 1)
        return bins

    def _assign_from_pandas(self, name: str, bins: list[int]) -> None:
        """Write a computed column back into the single table (fallback path)."""
        if self._backend == "duckdb":
            frame = self._fetch(f"SELECT rowid AS __rid FROM {_quote(_TABLE)}")
            for rid, value in zip(frame["__rid"].tolist(), bins):
                self._exec(
                    f"UPDATE {_quote(_TABLE)} SET {_quote(name)} = {int(value)} WHERE rowid = {int(rid)}"
                )
        else:
            frame = self._fetch(f"SELECT rowid AS __rid FROM {_quote(_TABLE)}")
            for rid, value in zip(frame["__rid"].tolist(), bins):
                self._exec(
                    f"UPDATE {_quote(_TABLE)} SET {_quote(name)} = {int(value)} WHERE rowid = {int(rid)}"
                )

    def _add_column(self, name: str, dtype: str) -> None:
        self._exec(f"ALTER TABLE {_quote(_TABLE)} ADD COLUMN {_quote(name)} {dtype}")

    def _require_column(self, col: str) -> str:
        if col not in self._columns:
            raise KeyError(f"Column '{col}' not in central store {self._columns}")
        return col

    def _expression(self, kind: str, feature: dict[str, Any]) -> str:
        """Return an engine-portable SQL expression for the given feature kind.

        DuckDB and SQLite both support: CASE WHEN, NULLIF, ABS, SQRT, POWER,
        LN (SQLite >= 3.35 with math functions; fall back to Python if missing).
        """
        cols = {
            k: self._require_column(v)
            for k, v in feature.items()
            if k in ("a", "b") and isinstance(v, str)
        }
        a = _quote(cols.get("a", feature.get("a", "")))
        b = _quote(cols.get("b", feature.get("b", "")))

        if not a:
            raise ValueError(f"Feature kind '{kind}' requires at least one operand column")

        case_guard = lambda expr: f"CASE WHEN {expr} THEN {expr} ELSE 0 END"  # noqa: E731

        return {
            "ratio": f"CASE WHEN {b} = 0 OR {b} IS NULL THEN {a} ELSE {a} / {b} END",
            "product": case_guard(f"{a} * {b}"),
            "sum": case_guard(f"{a} + {b}"),
            "difference": case_guard(f"{a} - {b}"),
            "log1p": f"CASE WHEN {a} + 1 > 0 THEN LN({a} + 1) ELSE 0 END",
            "sqrt": f"CASE WHEN {a} >= 0 THEN SQRT({a}) ELSE 0 END",
            "square": f"{a} * {a}",
            "abs": f"ABS({a})",
            "interaction": case_guard(f"{a} * {b}"),
        }.get(kind)

    # ------------------------------------------------------------------ SQL escaping

    def safe_identifier(self, name: str) -> str:
        return _column_name(name)


class CentralDataStore(CentralStore):
    """Alias keeping the spec's naming; single instance per run."""


_store_lock = threading.Lock()
_store_instance: CentralStore | None = None


def get_central_store() -> CentralStore:
    """Process-wide singleton central store."""
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            from backend.config import settings

            _store_instance = CentralStore(
                db_path=settings.central_store_path,
                memory=settings.central_store_in_memory,
            )
        return _store_instance


def reset_central_store() -> CentralStore:
    global _store_instance
    with _store_lock:
        _store_instance = None
        from backend.config import settings

        _store_instance = CentralStore(
            db_path=settings.central_store_path,
            memory=settings.central_store_in_memory,
        )
        return _store_instance
