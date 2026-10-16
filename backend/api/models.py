from uuid import UUID

from backend.api.schemas import (
    ModelCreate,
    ModelResponse,
    PaginatedResponse,
)
from backend.database import get_async_session
from backend.database.models import Experiment, Model, ModelType
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    model_in: ModelCreate,
    experiment_id: UUID = Query(...),
    db: AsyncSession = Depends(get_async_session),
) -> ModelResponse:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    try:
        model_type = ModelType(model_in.model_type)
    except ValueError as err:
        raise HTTPException(
            status_code=400, detail=f"Invalid model type: {model_in.model_type}"
        ) from err

    model = Model(
        experiment_id=experiment_id,
        name=model_in.name,
        model_type=model_type,
        hyperparameters=model_in.hyperparameters,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


@router.get("", response_model=PaginatedResponse)
async def list_models(
    experiment_id: UUID = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    model_type: ModelType | None = None,
    db: AsyncSession = Depends(get_async_session),
) -> PaginatedResponse:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    query = select(Model).where(Model.experiment_id == experiment_id)
    if model_type:
        query = query.where(Model.model_type == model_type)

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = query.order_by(Model.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    models = result.scalars().all()

    return PaginatedResponse(
        items=[ModelResponse.model_validate(m) for m in models],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> ModelResponse:
    model = await db.get(Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.get("/experiment/{experiment_id}/comparison")
async def compare_models(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> dict:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    query = select(Model).where(Model.experiment_id == experiment_id)
    result = await db.execute(query)
    models = result.scalars().all()

    if not models:
        return {"models": [], "ranking": []}

    comparison = []
    for m in models:
        comparison.append(
            {
                "id": str(m.id),
                "name": m.name,
                "model_type": m.model_type.value,
                "hyperparameters": m.hyperparameters,
                "metrics": m.metrics,
                "training_time_seconds": m.training_time_seconds,
                "feature_importance": m.feature_importance,
            }
        )

    def sort_key(model):
        metrics = model["metrics"]
        return metrics.get("f1", metrics.get("accuracy", metrics.get("roc_auc", 0)))

    ranked = sorted(comparison, key=sort_key, reverse=True)

    return {
        "experiment_id": str(experiment_id),
        "models": comparison,
        "ranking": ranked,
    }


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> None:
    model = await db.get(Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    await db.delete(model)
    await db.commit()
