import shutil
import uuid
from pathlib import Path
from uuid import UUID

from backend.api.schemas import (
    DatasetResponse,
    PaginatedResponse,
)
from backend.config import settings
from backend.database import get_async_session
from backend.database.models import Dataset, DatasetStatus, Project, Session
from backend.filecheck import check_filename, check_size, dataset_info, read_tabular
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/datasets", tags=["datasets"])

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    project_id: UUID = Query(...),
    session_id: UUID | None = Query(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_session),
) -> DatasetResponse:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if session_id:
        session = await db.get(Session, session_id)
        if not session or session.project_id != project_id:
            raise HTTPException(status_code=404, detail="Session not found in this project")

    if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB",
        )
    check_filename(file.filename)

    dataset_id = uuid.uuid4()
    file_ext = Path(file.filename or "").suffix.lower() or ".csv"
    file_path = UPLOAD_DIR / f"{project_id}" / f"{dataset_id}{file_ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = file_path.stat().st_size
    try:
        check_size(file_size, context=file.filename or "upload")
        df = read_tabular(file_path)
        info = dataset_info(df)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    dataset = Dataset(
        id=dataset_id,
        project_id=project_id,
        session_id=session_id,
        name=file.filename or "unnamed_dataset",
        original_filename=file.filename or "unnamed_dataset",
        file_path=str(file_path),
        file_size_bytes=file_size,
        mime_type=file.content_type or "application/octet-stream",
        status=DatasetStatus.READY,
        rows=info["rows"],
        columns=info["columns"],
        column_info={"columns": info["column_names"]},
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


@router.get("", response_model=PaginatedResponse)
async def list_datasets(
    project_id: UUID = Query(...),
    session_id: UUID | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: DatasetStatus | None = None,
    db: AsyncSession = Depends(get_async_session),
) -> PaginatedResponse:
    query = select(Dataset).where(Dataset.project_id == project_id)
    if session_id:
        query = query.where(Dataset.session_id == session_id)
    if status:
        query = query.where(Dataset.status == status)

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = (
        query.order_by(Dataset.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )
    result = await db.execute(query)
    datasets = result.scalars().all()

    return PaginatedResponse(
        items=[DatasetResponse.model_validate(d) for d in datasets],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> DatasetResponse:
    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> None:
    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if file_path.exists():
        file_path.unlink()

    await db.delete(dataset)
    await db.commit()


@router.post("/{dataset_id}/stage")
async def stage_dataset(
    dataset_id: UUID,
    project_id: UUID = Query(...),
    db: AsyncSession = Depends(get_async_session),
) -> dict:
    """Stage a project dataset as the AutoML working input (sets the fetcher's
    upload path + project memory). The next run uses it; nothing trains yet."""
    from backend import runlog
    from backend.orchestrator import get_orchestrator

    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.project_id != project_id:
        raise HTTPException(status_code=400, detail="Dataset does not belong to this project")
    if not dataset.file_path or not Path(dataset.file_path).exists():
        raise HTTPException(status_code=400, detail="Dataset file is missing on disk.")
    orch = get_orchestrator()
    orch.upload_path = dataset.file_path
    runlog.remember(
        str(project_id),
        "dataset",
        {
            "source": "upload",
            "path": dataset.file_path,
            "rows": dataset.rows,
            "columns": dataset.columns,
            "name": dataset.name,
        },
    )
    return {"staged": dataset.file_path, "rows": dataset.rows, "columns": dataset.columns}
