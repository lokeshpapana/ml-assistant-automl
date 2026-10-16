from __future__ import annotations

from typing import Any

import pandas as pd


def rank_models(
    results: list[dict[str, Any]],
    primary_metric: str = "f1",
    secondary_metrics: list[str] | None = None,
    optimization_direction: str = "maximize",
) -> list[dict[str, Any]]:
    if not results:
        return []

    secondary_metrics = secondary_metrics or ["accuracy", "roc_auc", "training_time_seconds"]

    def sort_key(result: dict[str, Any]) -> tuple:
        metrics = result.get("metrics", {})
        primary = metrics.get(primary_metric, 0)

        if optimization_direction == "minimize":
            primary = -primary

        secondary = tuple(metrics.get(m, 0) for m in secondary_metrics)
        return (primary,) + secondary

    ranked = sorted(results, key=sort_key, reverse=(optimization_direction == "maximize"))

    for i, result in enumerate(ranked):
        result["rank"] = i + 1

    return ranked


def create_comparison_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        row = {
            "rank": r.get("rank", 0),
            "model_name": r.get("name", "Unknown"),
            "model_type": r.get("model_type", "Unknown"),
            "training_time_sec": r.get("training_time_seconds", 0),
        }
        metrics = r.get("metrics", {})
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    metric_cols = [c for c in df.columns if c not in ["rank", "model_name", "model_type", "training_time_sec"]]
    df = df[["rank", "model_name", "model_type", "training_time_sec"] + metric_cols]

    return df


def select_best_model(
    results: list[dict[str, Any]],
    primary_metric: str = "f1",
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not results:
        return None

    constraints = constraints or {}

    feasible = []
    for r in results:
        metrics = r.get("metrics", {})

        if "max_training_time" in constraints:
            if r.get("training_time_seconds", float("inf")) > constraints["max_training_time"]:
                continue

        if "min_metric" in constraints:
            if metrics.get(primary_metric, 0) < constraints["min_metric"]:
                continue

        feasible.append(r)

    if not feasible:
        return None

    feasible.sort(key=lambda x: x.get("metrics", {}).get(primary_metric, 0), reverse=True)
    return feasible[0]


def get_pareto_frontier(
    results: list[dict[str, Any]],
    metrics: list[str] = ("f1", "training_time_seconds"),
    directions: tuple[str, str] = ("maximize", "minimize"),
) -> list[dict[str, Any]]:
    if len(results) < 2 or len(metrics) != 2:
        return results

    m1, m2 = metrics
    d1, d2 = directions

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        ma1, ma2 = a.get("metrics", {}).get(m1, 0), a.get("metrics", {}).get(m2, float("inf"))
        mb1, mb2 = b.get("metrics", {}).get(m1, 0), b.get("metrics", {}).get(m2, float("inf"))

        if d1 == "minimize":
            ma1, mb1 = -ma1, -mb1
        if d2 == "minimize":
            ma2, mb2 = -ma2, -mb2

        return ma1 >= mb1 and ma2 >= mb2 and (ma1 > mb1 or ma2 > mb2)

    pareto = []
    for i, a in enumerate(results):
        dominated = False
        for j, b in enumerate(results):
            if i != j and dominates(b, a):
                dominated = True
                break
        if not dominated:
            pareto.append(a)

    return pareto


def generate_model_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"summary": "No models evaluated", "models": []}

    ranked = rank_models(results)
    df = create_comparison_dataframe(ranked)

    best = ranked[0]
    worst = ranked[-1]

    return {
        "summary": {
            "total_models": len(results),
            "best_model": best.get("name"),
            "best_model_type": best.get("model_type"),
            "best_metric_value": best.get("metrics", {}).get("f1", 0),
            "worst_model": worst.get("name"),
            "metric_range": {
                "min": worst.get("metrics", {}).get("f1", 0),
                "max": best.get("metrics", {}).get("f1", 0),
            },
        },
        "ranking": ranked,
        "comparison_table": df.to_dict("records"),
        "pareto_frontier": get_pareto_frontier(ranked),
    }