"""Tests for the CentralStore single-source-of-truth dataset table."""

import pytest

from backend.central_store import CentralStore, reset_central_store


@pytest.fixture(autouse=True)
def clean_store():
    reset_central_store()
    yield
    reset_central_store()


def test_store_load_and_preview():
    store = CentralStore()
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    store.load_dataframe(df)

    preview = store.preview(limit=2)
    assert preview["rows"] == 3
    assert "a" in preview["columns"]
    assert "target" in preview["columns"]
    assert len(preview["preview"]) == 2


def test_store_add_feature_ratio():
    store = CentralStore()
    import pandas as pd

    df = pd.DataFrame({"a": [10, 20, 30], "b": [2, 4, 5], "target": [0, 1, 0]})
    store.load_dataframe(df)

    new_col = store.add_feature({"kind": "ratio", "a": "a", "b": "b", "name": "ratio_a_b"})
    assert new_col["name"] == "ratio_a_b"
    assert new_col["name"] in store.columns()
    assert len(store.columns()) == 4


def test_store_add_feature_multiple():
    store = CentralStore()
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "target": [0, 1, 0]})
    store.load_dataframe(df)

    store.add_feature({"kind": "product", "a": "x", "b": "y", "name": "product_x_y"})
    store.add_feature({"kind": "sum", "a": "x", "b": "y", "name": "sum_x_y"})
    store.add_feature({"kind": "difference", "a": "x", "b": "y", "name": "diff_x_y"})
    store.add_feature({"kind": "log1p", "a": "x", "name": "log1p_x"})
    store.add_feature({"kind": "sqrt", "a": "y", "name": "sqrt_y"})

    cols = store.columns()
    assert "product_x_y" in cols
    assert "sum_x_y" in cols
    assert "diff_x_y" in cols
    assert "log1p_x" in cols
    assert "sqrt_y" in cols


def test_store_to_dataframe():
    store = CentralStore()
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    store.load_dataframe(df)

    out = store.to_dataframe()
    assert len(out) == 2
    assert list(out.columns) == ["a", "b", "target"]


def test_store_load_path_parquet(tmp_path):
    store = CentralStore()
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    path = tmp_path / "test.parquet"
    df.to_parquet(path, index=False)

    store.load_path(str(path))
    assert store.shape()[0] == 3
    assert "a" in store.columns()


def test_store_singleton():
    from backend.central_store import get_central_store, reset_central_store

    reset_central_store()
    s1 = get_central_store()
    s2 = get_central_store()
    assert s1 is s2


def test_reopen_persisted_file_reports_schema(tmp_path):
    """A fresh process reconnecting to a persisted table must see its columns
    and dtypes — otherwise preview()/columns() are empty and the UI can't list
    the dataset or its features."""
    import pandas as pd

    path = tmp_path / "persist.duckdb"
    first = CentralStore(db_path=str(path))
    first.load_dataframe(pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]}))

    reopened = CentralStore(db_path=str(path))
    preview = reopened.preview(limit=2)
    assert preview["rows"] == 3
    assert preview["columns"] == ["a", "target"]
    assert preview["dtypes"]["a"]
    assert "a" in reopened.columns()
    assert len(reopened.dtypes()) == 2