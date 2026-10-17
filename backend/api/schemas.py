from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class ProjectResponse(ProjectBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    created_at: datetime
    updated_at: datetime


class SessionBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class SessionCreate(SessionBase):
    pass


class SessionUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class SessionResponse(SessionBase):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    project_id: UUID
    # ORM attribute is `metadata_` (trailing underscore avoids Base.metadata clash);
    # read from it, serialize as "metadata".
    metadata: dict = Field(
        default_factory=dict, validation_alias="metadata_", serialization_alias="metadata"
    )
    created_at: datetime
    updated_at: datetime


class DatasetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class DatasetCreate(DatasetBase):
    pass


class DatasetProfileRequest(BaseModel):
    target_column: str | None = None
    sample_size: int | None = None


class DatasetProfileResponse(BaseModel):
    dataset_id: UUID
    profile: dict
    profile_generated_at: datetime


class DatasetResponse(DatasetBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    session_id: UUID | None
    original_filename: str
    file_path: str
    file_size_bytes: int
    mime_type: str
    status: str
    rows: int | None
    columns: int | None
    column_info: dict | None
    profile: dict | None
    created_at: datetime
    updated_at: datetime


class ExperimentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    config: dict = Field(default_factory=dict)


class ExperimentCreate(ExperimentBase):
    dataset_id: UUID


class ExperimentRunRequest(BaseModel):
    target_column: str
    task_type: str = "classification"
    models: list[str] | None = None
    test_size: float = 0.2
    cv_folds: int = 5
    max_training_time: int = 300


class DatasetAcquisitionRequest(BaseModel):
    request: str | None = Field(None, description="Natural-language description of the ML task")
    target: str | None = Field(None, description="Explicit target column name if known")
    local_file_path: str | None = Field(None, description="Path to a user-provided dataset file")
    source_dataset_id: UUID | None = Field(
        None, description="Use an already-uploaded dataset as the candidate"
    )
    query: str | None = Field(None, description="Search query when acquiring from external sources")
    project_id: UUID | None = Field(
        None, description="Project to register the acquired dataset under"
    )
    output_format: str = Field("csv", description="csv | xlsx | excel | parquet | json | sql")
    output_dir: str | None = Field(None, description="Directory to save the optimized dataset")
    save_csv: bool = True
    max_candidates: int = 5
    requested_input_features: int | None = Field(
        None, description="Desired number of input features"
    )
    register_as_dataset: bool = True


class ExperimentUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    config: dict | None = None


class ExperimentResponse(ExperimentBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    session_id: UUID | None
    dataset_id: UUID
    status: str
    metrics: dict | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class ModelBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    model_type: str
    hyperparameters: dict = Field(default_factory=dict)


class ModelCreate(ModelBase):
    pass


class ModelResponse(ModelBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_id: UUID
    metrics: dict
    training_time_seconds: float | None
    model_path: str | None
    feature_importance: dict | None
    created_at: datetime


class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int
