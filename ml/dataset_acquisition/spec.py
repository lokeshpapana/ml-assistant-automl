from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MLTaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series_forecasting"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    OTHER = "other"


class DataType(StrEnum):
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TIMESERIES = "timeseries"
    GRAPH = "graph"
    OTHER = "other"


class LicenseRequirement(StrEnum):
    ANY = "any"
    OPEN = "open"
    CC = "cc"
    COMMERCIAL_OK = "commercial_ok"
    NONE = "none"


class DatasetSpecification(BaseModel):
    """Structured specification derived from a user's request.

    Every field that the user did not explicitly state is marked via
    `inferred_fields` so downstream steps can surface assumptions.
    """

    problem_description: str = Field(..., description="Free-text problem statement")
    task: MLTaskType = MLTaskType.CLASSIFICATION
    target: str | None = Field(None, description="Requested target variable name or description")
    target_type: str | None = Field(None, description="binary/multiclass/continuous/etc.")
    requested_rows: int | None = None
    requested_features: int | None = Field(None, description="Requested NUMBER of input features")
    features_include_target: bool = False
    required_data_types: list[DataType] = Field(default_factory=list)
    domain: str | None = None
    geographic_requirements: str | None = None
    time_period_requirements: str | None = None
    class_requirements: str | None = None
    min_acceptable_rows: int | None = None
    max_acceptable_rows: int | None = None
    performance_goals: str | None = None
    computational_constraints: str | None = None
    license_requirement: LicenseRequirement = LicenseRequirement.ANY
    privacy_restrictions: str | None = None
    other_constraints: list[str] = Field(default_factory=list)
    inferred_fields: list[str] = Field(
        default_factory=list, description="Field names whose value was inferred, not user-stated"
    )

    @classmethod
    def from_user_request(
        cls, request: str, target: str | None = None, **overrides: Any
    ) -> "DatasetSpecification":
        """Build a spec from a natural-language request with safe defaults.

        This is a rule-based parser (no LLM dependency) intended for laptop use.
        It extracts the obvious signals and infers the rest, recording inferences.
        """
        inferred: set[str] = set()
        text = (request or "").lower()
        data: dict[str, Any] = {"problem_description": request}

        # Task inference
        if any(k in text for k in ["classif", "predict category", "spam", "fraud detect", "churn", "diagnosis"]):
            data["task"] = MLTaskType.CLASSIFICATION
        elif any(k in text for k in ["regress", "predict price", "forecast", "estimate"]):
            data["task"] = MLTaskType.REGRESSION
        else:
            data["task"] = MLTaskType.CLASSIFICATION
            inferred.add("task")

        # Target
        if target:
            data["target"] = target
        else:
            inferred.add("target")

        # Domain extraction (lightweight keyword map)
        domain_map = {
            "health": "healthcare", "medical": "healthcare", "diabet": "healthcare",
            "cancer": "healthcare", "finance": "finance", "bank": "finance",
            "credit": "finance", "fraud": "finance", "housing": "real_estate",
            "price": "real_estate", "sales": "retail", "customer": "marketing",
            "image": "computer_vision", "text": "nlp", "review": "nlp",
        }
        domain = None
        for kw, dom in domain_map.items():
            if kw in text:
                domain = dom
                break
        data["domain"] = domain
        if domain is None:
            inferred.add("domain")

        # Row / feature hints
        import re

        m_rows = re.search(r"(\d[\d,]*)[\s]*rows|(\d[\d,]*)[\s]*samples", text)
        if m_rows:
            num = m_rows.group(1) or m_rows.group(2)
            data["requested_rows"] = int(num.replace(",", ""))
        else:
            inferred.add("requested_rows")

        m_feats = re.search(r"(\d[\d,]*)[\s]*features|(\d[\d,]*)[\s]*columns", text)
        if m_feats:
            num = m_feats.group(1) or m_feats.group(2)
            data["requested_features"] = int(num.replace(",", ""))
        else:
            inferred.add("requested_features")

        data["inferred_fields"] = sorted(inferred)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def resolve_final_feature_count(self) -> int | None:
        """Compute intended input-feature count given include_target flag."""
        if self.requested_features is None:
            return None
        if self.features_include_target:
            return max(0, self.requested_features - 1)
        return self.requested_features
