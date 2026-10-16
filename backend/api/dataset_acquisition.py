from __future__ import annotations

import uuid
from pathlib import Path

from backend.api.schemas import DatasetAcquisitionRequest
from backend.database import get_async_session
from backend.database.models import Dataset, DatasetStatus, Project
from backend.tasks.ml_tasks import acquire_dataset_task
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/ml/dataset-acquisition", tags=["dataset-acquisition"])


@router.post("/acquire")
async def acquire(
    request: DatasetAcquisitionRequest,
    db: AsyncSession = Depends(get_async_session),
    async_mode: bool = Query(False, description="Run acquisition as a Celery task"),
) -> dict:
    """Find or optimize the best dataset for the given ML requirement.

    - If `local_file_path` or `source_dataset_id` is provided, the user-supplied
      dataset is evaluated/optimized.
    - Otherwise the agent searches external sources (OpenML, HuggingFace) and
      selects the highest-quality candidate.
    """
    local_path = request.local_file_path
    if request.source_dataset_id:
        ds = await db.get(Dataset, request.source_dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Source dataset not found")
        local_path = ds.file_path

    project = None
    if request.project_id:
        project = await db.get(Project, request.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    kwargs = {
        "request": request.request,
        "local_file_path": local_path,
        "query": request.query,
        "save_csv": request.save_csv,
        "output_format": request.output_format,
        "output_dir": request.output_dir,
        "max_candidates": request.max_candidates,
        "requested_input_features": request.requested_input_features,
        # Threaded through so the async worker can register under the project.
        "project_id": str(request.project_id) if request.project_id else None,
    }

    if async_mode:
        task = acquire_dataset_task.delay(kwargs, request.project_id is not None)
        return {"status": "pending", "task_id": task.id}

    from ml.dataset_acquisition import acquire_dataset

    report = acquire_dataset(**kwargs)

    if report.get("status") == "success" and request.register_as_dataset and project:
        await _register_dataset(db, project, report, request.source_dataset_id)

    return report


async def _register_dataset(
    db: AsyncSession, project: Project, report: dict, source_dataset_id: uuid.UUID | None
) -> None:
    output_path = report.get("output_path")
    output_format = report.get("output_format", "csv")
    mime_map = {
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "parquet": "application/parquet",
        "json": "application/json",
        "sql": "application/sql",
    }
    name = report["dataset_selected"]["name"] or "acquired_dataset"
    ds = Dataset(
        project_id=project.id,
        name=f"acquired_{name}",
        original_filename=Path(output_path).name if output_path else name,
        file_path=output_path or "",
        file_size_bytes=Path(output_path).stat().st_size if output_path else 0,
        mime_type=mime_map.get(output_format, "text/csv"),
        status=DatasetStatus.READY,
        rows=report["dimensions"]["final_rows"],
        columns=report["dimensions"]["final_input_features"] + 1,
        profile=report,
    )
    db.add(ds)
    await db.commit()


@router.get("/ui")
async def acquisition_ui() -> FileResponse:
    """Basic browser interface for interacting with and testing the agent."""
    html_path = Path(__file__).parent.parent / "static" / "acquire_ui.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI template not found")
    return FileResponse(html_path)
