from uuid import UUID

from backend.api.schemas import (
    DatasetProfileRequest,
    DatasetProfileResponse,
    ExperimentCreate,
    ExperimentResponse,
    ExperimentRunRequest,
)
from backend.database import get_async_session
from backend.database.models import Dataset, DatasetStatus, Experiment, ExperimentStatus, Project
from backend.tasks.ml_tasks import profile_dataset_task, run_experiment_task
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post("/datasets/{dataset_id}/profile", response_model=DatasetProfileResponse)
async def profile_dataset(
    dataset_id: UUID,
    request: DatasetProfileRequest,
    db: AsyncSession = Depends(get_async_session),
    async_mode: bool = Query(True, description="Run profiling asynchronously"),
) -> DatasetProfileResponse:
    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if async_mode:
        task = profile_dataset_task.delay(
            str(dataset_id),
            target_column=request.target_column,
            sample_size=request.sample_size,
        )
        return DatasetProfileResponse(
            dataset_id=dataset.id,
            profile={"task_id": task.id, "status": "pending"},
            profile_generated_at=dataset.updated_at,
        )

    try:
        from ml.preprocessing.profiler import profile_dataset as ml_profile

        profile = ml_profile(dataset.file_path, target_column=request.target_column, sample_size=request.sample_size)

        dataset.status = DatasetStatus.READY
        dataset.rows = profile["overview"]["n_rows"]
        dataset.columns = profile["overview"]["n_columns"]
        dataset.column_info = profile["columns"]
        dataset.profile = profile

        await db.commit()
        await db.refresh(dataset)

        return DatasetProfileResponse(
            dataset_id=dataset.id, profile=profile, profile_generated_at=dataset.updated_at
        )
    except Exception as e:
        dataset.status = DatasetStatus.FAILED
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}") from e


@router.post("/experiments", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
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

    if dataset.status != DatasetStatus.READY:
        raise HTTPException(status_code=400, detail="Dataset must be profiled before running experiments")

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


@router.post("/experiments/{experiment_id}/run")
async def run_experiment(
    experiment_id: UUID,
    run_request: ExperimentRunRequest,
    db: AsyncSession = Depends(get_async_session),
    async_mode: bool = Query(True, description="Run experiment asynchronously"),
) -> dict:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.status not in (ExperimentStatus.PENDING, ExperimentStatus.FAILED):
        raise HTTPException(status_code=400, detail="Experiment cannot be run in current state")

    dataset = await db.get(Dataset, experiment.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    experiment.config = run_request.model_dump()
    await db.commit()

    if async_mode:
        task = run_experiment_task.delay(
            str(experiment_id),
            target_column=run_request.target_column,
            task_type=run_request.task_type,
            models=run_request.models,
            test_size=run_request.test_size,
            cv_folds=run_request.cv_folds,
            max_training_time=run_request.max_training_time,
        )
        return {
            "experiment_id": str(experiment_id),
            "status": "pending",
            "task_id": task.id,
        }

    try:
        import pandas as pd

        if dataset.file_path.endswith(".csv"):
            df = pd.read_csv(dataset.file_path)
        elif dataset.file_path.endswith(".parquet"):
            df = pd.read_parquet(dataset.file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if run_request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{run_request.target_column}' not found")

        from ml.experiments.runner import ExperimentRunner

        runner = ExperimentRunner(
            target_column=run_request.target_column,
            task_type=run_request.task_type,
            models_to_run=run_request.models,
            test_size=run_request.test_size,
            cv_folds=run_request.cv_folds,
            max_training_time=run_request.max_training_time,
        )

        report = runner.run(df)

        experiment.status = ExperimentStatus.COMPLETED
        experiment.metrics = report.get("summary", {})
        experiment.completed_at = func.now()

        from backend.database.models import Model, ModelType

        for model_result in report.get("ranking", []):
            if model_result.get("status") == "success":
                model = Model(
                    experiment_id=experiment.id,
                    name=model_result["name"],
                    model_type=ModelType(model_result["model_type"]),
                    hyperparameters=model_result.get("hyperparameters", {}),
                    metrics=model_result.get("metrics", {}),
                    training_time_seconds=model_result.get("training_time_seconds"),
                    feature_importance=model_result.get("feature_importance"),
                )
                db.add(model)

        await db.commit()
        await db.refresh(experiment)

        return {
            "experiment_id": str(experiment.id),
            "status": "completed",
            "report": report,
        }

    except Exception as e:
        experiment.status = ExperimentStatus.FAILED
        experiment.error_message = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Experiment failed: {str(e)}") from e


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> dict:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    from backend.database.models import Model

    query = select(Model).where(Model.experiment_id == experiment_id)
    result = await db.execute(query)
    models = result.scalars().all()

    model_results = []
    for m in models:
        model_results.append(
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

    from ml.evaluation.ranking import rank_models

    ranked = rank_models(
        [r for r in model_results if r.get("metrics")],
        primary_metric="f1" if experiment.config.get("task_type") == "classification" else "r2",
    )

    return {
        "experiment_id": str(experiment_id),
        "status": experiment.status,
        "config": experiment.config,
        "models": model_results,
        "ranking": ranked,
        "summary": experiment.metrics,
    }


@router.get("/experiments/{experiment_id}/best-model")
async def get_best_model(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> dict:
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    from backend.database.models import Model

    query = select(Model).where(Model.experiment_id == experiment_id)
    result = await db.execute(query)
    models = result.scalars().all()

    if not models:
        raise HTTPException(status_code=404, detail="No models found for this experiment")

    task_type = experiment.config.get("task_type", "classification")
    primary_metric = "f1" if task_type == "classification" else "r2"

    best = None
    best_score = -1
    for m in models:
        score = m.metrics.get(primary_metric, 0)
        if score > best_score:
            best_score = score
            best = m

    if not best:
        raise HTTPException(status_code=404, detail="No valid models found")

    return {
        "experiment_id": str(experiment_id),
        "best_model": {
            "id": str(best.id),
            "name": best.name,
            "model_type": best.model_type.value,
            "hyperparameters": best.hyperparameters,
            "metrics": best.metrics,
            "training_time_seconds": best.training_time_seconds,
            "feature_importance": best.feature_importance,
        },
    }


@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str) -> dict:
    from backend.tasks import celery_app

    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
        "traceback": result.traceback if result.failed() else None,
    }


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> dict:
    from backend.tasks import celery_app

    result = celery_app.AsyncResult(task_id)
    if not result.ready():
        raise HTTPException(status_code=400, detail="Task not yet completed")

    return {"task_id": task_id, "status": result.status, "result": result.result}
