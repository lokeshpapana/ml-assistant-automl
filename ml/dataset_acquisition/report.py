from __future__ import annotations

from typing import Any


def build_report(
    candidate_summary: dict[str, Any],
    spec: Any,
    inspection: dict[str, Any],
    target_info: dict[str, Any],
    assessments: list[Any],
    score: dict[str, Any],
    optimization: Any,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Assemble the final report (spec section 26)."""
    selected_features = [
        {
            "feature": a.name,
            "type": a.dtype,
            "reason": a.reason,
            "importance_score": round(a.importance_score, 4),
            "classification": a.classification.value,
        }
        for a in assessments
        if a.name in optimization.selected_features
    ]
    final_feature_columns = [
        c for c in optimization.df.columns if c != optimization.target
    ]
    removed_features = optimization.removed_features
    return {
        "dataset_selected": {
            "name": candidate_summary.get("name"),
            "source": candidate_summary.get("source"),
            "url": candidate_summary.get("url"),
            "license": candidate_summary.get("license"),
        },
        "problem": {
            "task": spec.task.value if hasattr(spec.task, "value") else spec.task,
            "target": target_info.get("best_target", {}).get("column"),
            "target_type": target_info.get("best_target", {}).get("type"),
        },
        "dimensions": {
            "requested_rows": spec.requested_rows,
            "final_rows": inspection["n_rows"],
            "requested_features": spec.requested_features,
            "final_input_features": optimization.final_input_feature_count,
            "target_included_in_feature_count": spec.features_include_target,
        },
        "dataset_quality": {
            "overall_score": score["overall_score"],
            **score["component_scores"],
        },
        "selected_features": selected_features,
        "final_feature_columns": final_feature_columns,
        "removed_features": removed_features,
        "engineered_features": optimization.engineered_features,
        "data_cleaning": {
            "transformations": optimization.transformation_log,
            "duplicate_rows_removed": inspection.get("duplicate_rows", 0),
        },
        "warnings": optimization.warnings,
        "final_recommendation": (
            f"Selected '{candidate_summary.get('name')}' as the best available dataset "
            f"for the stated {spec.task} problem (overall quality score "
            f"{score['overall_score']}). Leakage/identifier/constant features were removed "
            f"and the final dataset has {optimization.final_input_feature_count} input features."
        ),
        "output_path": output_path,
    }
