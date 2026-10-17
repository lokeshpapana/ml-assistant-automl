from uuid import UUID

from backend.api.schemas import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentUpdate,
    PaginatedResponse,
)
from backend.database import get_async_session
from backend.database.models import Dataset, Experiment, ExperimentStatus, Project
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment_in: ExperimentCreate,
    project_id: UUID = Query(...),
    db: AsyncSession = Depends(get_async_session),
) -> ExperimentResponse:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = await db.get(Dataset, experiment_in.dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(status_code=404, detail="Dataset not found in this project")

    experiment = Experiment(
        project_id=project_id,
        dataset_id=experiment_in.dataset_id,
        name=experiment_in.name,
        description=experiment_in.description,
        config=experiment_in.config,
        status=ExperimentStatus.PENDING,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment


@router.get("", response_model=PaginatedResponse)
async def list_experiments(
    project_id: UUID = Query(...),
    dataset_id: UUID | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: ExperimentStatus | None = None,
    db: AsyncSession = Depends(get_async_session),
) -> PaginatedResponse:
    query = select(Experiment).where(Experiment.project_id == project_id)
    if dataset_id:
        query = query.where(Experiment.dataset_id == dataset_id)
    if status:
        query = query.where(Experiment.status == status)

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = (
        query.order_by(Experiment.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )
    result = await db.execute(query)
    experiments = result.scalars().all()

    return PaginatedResponse(
        items=[ExperimentResponse.model_validate(e) for e in experiments],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> ExperimentResponse:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: UUID,
    experiment_in: ExperimentUpdate,
    db: AsyncSession = Depends(get_async_session),
) -> ExperimentResponse:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    update_data = experiment_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(experiment, field, value)

    await db.commit()
    await db.refresh(experiment)
    return experiment


@router.post("/{experiment_id}/run", response_model=ExperimentResponse)
async def run_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> ExperimentResponse:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.status not in (ExperimentStatus.PENDING, ExperimentStatus.FAILED):
        raise HTTPException(status_code=400, detail="Experiment cannot be run in current state")

    experiment.status = ExperimentStatus.RUNNING
    await db.commit()
    await db.refresh(experiment)

    # TODO: Trigger background task to run experiment
    # For now, just return the experiment with RUNNING status
    return experiment


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> None:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    await db.delete(experiment)
    await db.commit()
