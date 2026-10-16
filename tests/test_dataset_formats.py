from __future__ import annotations

import pandas as pd
from pathlib import Path

from ml.dataset_acquisition import acquire_dataset
from ml.dataset_acquisition.agent import _write_sql


def test_output_formats_create_files(tmp_path):
    df = pd.DataFrame(
        {
            "id": range(10),
            "age": [20, 30, 40, 50, 60, 70, 80, 25, 35, 45],
            "income": [1000.5, 2000.0, 3000.5, 4000.0, 5000.5, 6000.0, 7000.5, 800.0, 900.0, 1500.0],
            "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            "fraud": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        }
    )
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)

    for fmt in ["csv", "xlsx", "parquet", "json", "sql"]:
        out = tmp_path / fmt
        report = acquire_dataset(
            request="predict fraud with 4 features",
            local_file_path=str(path),
            output_format=fmt,
            output_dir=str(out),
        )
        assert report["status"] == "success", fmt
        op = report["output_path"]
        assert op and Path(op).exists(), f"{fmt} file not created"
        suffix = {
            "csv": ".csv", "xlsx": ".xlsx", "excel": ".xlsx",
            "parquet": ".parquet", "json": ".json", "sql": ".sql",
        }[fmt]
        assert Path(op).suffix == suffix, f"{fmt} wrong suffix: {Path(op).suffix}"


def test_sql_dump_is_loadable(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [1000.5, 2000.0, 3000.5],
            "active": [True, False, True],
            "city": ["A", "B", "C"],
            "fraud": [0, 1, 0],
        }
    )
    sql_path = tmp_path / "out.sql"
    _write_sql(df, sql_path, table_name="optimized")
    content = sql_path.read_text()
    assert "CREATE TABLE" in content
    assert "INSERT INTO" in content
    assert '"active" BOOLEAN' in content
    assert "TRUE" in content or "FALSE" in content
