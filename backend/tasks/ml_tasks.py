from __future__ import annotations

from pathlib import Path
from uuid import UUID

from backend.database import async_session_maker
from backend.database.models import Dataset, Experiment, ExperimentStatus, Model, ModelType
from backend.tasks import celery_app


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_experiment_task(
    self,
    experiment_id: str,
    target_column: str,
    task_type: str,
    models: list[str] | None,
    test_size: float,
    cv_folds: int,
    max_training_time: int,
) -> dict:
    import asyncio

    return asyncio.run(
        _run_experiment_async(
            experiment_id=experiment_id,
            target_column=target_column,
            task_type=task_type,
            models=models,
            test_size=test_size,
            cv_folds=cv_folds,
            max_training_time=max_training_time,
            task_instance=self,
        )
    )


async def _run_experiment_async(
    experiment_id: str,
    target_column: str,
    task_type: str,
    models: list[str] | None,
    test_size: float,
    cv_folds: int,
    max_training_time: int,
    task_instance,
) -> dict:
    async with async_session_maker() as db:
        experiment = await db.get(Experiment, UUID(experiment_id))
        if not experiment:
            return {"status": "failed", "error": "Experiment not found"}

        dataset = await db.get(Dataset, experiment.dataset_id)
        if not dataset:
            return {"status": "failed", "error": "Dataset not found"}

        try:
            experiment.status = ExperimentStatus.RUNNING
            await db.commit()

            import pandas as pd
            from ml.experiments.runner import ExperimentRunner

            if dataset.file_path.endswith(".csv"):
                df = pd.read_csv(dataset.file_path)
            elif dataset.file_path.endswith(".parquet"):
                df = pd.read_parquet(dataset.file_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset.file_path}")

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")

            runner = ExperimentRunner(
                target_column=target_column,
                task_type=task_type,
                models_to_run=models,
                test_size=test_size,
                cv_folds=cv_folds,
                max_training_time=max_training_time,
            )

            report = runner.run(df)

            experiment.status = ExperimentStatus.COMPLETED
            experiment.metrics = report.get("summary", {})
            await db.commit()

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

            return {"status": "completed", "experiment_id": experiment_id, "report": report}

        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = str(e)
            await db.commit()
            raise


@celery_app.task(bind=True)
def profile_dataset_task(
    self, dataset_id: str, target_column: str | None = None, sample_size: int | None = None
) -> dict:
    import asyncio

    return asyncio.run(_profile_dataset_async(dataset_id, target_column, sample_size))


async def _profile_dataset_async(
    dataset_id: str, target_column: str | None, sample_size: int | None
) -> dict:
    async with async_session_maker() as db:
        from backend.database.models import Dataset, DatasetStatus

        dataset = await db.get(Dataset, UUID(dataset_id))
        if not dataset:
            return {"status": "failed", "error": "Dataset not found"}

        try:
            from ml.preprocessing.profiler import profile_dataset as ml_profile

            profile = ml_profile(
                dataset.file_path, target_column=target_column, sample_size=sample_size
            )

            dataset.status = DatasetStatus.READY
            dataset.rows = profile["overview"]["n_rows"]
            dataset.columns = profile["overview"]["n_columns"]
            dataset.column_info = profile["columns"]
            dataset.profile = profile

            await db.commit()

            return {"status": "completed", "dataset_id": dataset_id, "profile": profile}

        except Exception as e:
            dataset.status = DatasetStatus.FAILED
            await db.commit()
            return {"status": "failed", "error": str(e)}


@celery_app.task(bind=True)
def acquire_dataset_task(self, kwargs: dict, register_as_dataset: bool = False) -> dict:
    import asyncio

    return asyncio.run(_acquire_dataset_async(kwargs, register_as_dataset))


async def _acquire_dataset_async(kwargs: dict, register_as_dataset: bool) -> dict:
    from ml.dataset_acquisition import acquire_dataset

    report = acquire_dataset(**kwargs)

    if report.get("status") == "success" and register_as_dataset:
        from backend.database.models import Dataset as DBDataset
        from backend.database.models import DatasetStatus, Project

        async with async_session_maker() as db:
            raw_pid = kwargs.get("project_id")
            try:
                project_id = UUID(str(raw_pid)) if raw_pid else None
            except (ValueError, AttributeError, TypeError):
                project_id = None
            project = await db.get(Project, project_id) if project_id else None
            if project:
                output_path = report.get("output_path")
                name = report["dataset_selected"]["name"] or "acquired_dataset"
                ds = DBDataset(
                    project_id=project.id,
                    name=f"acquired_{name}",
                    original_filename=Path(output_path).name if output_path else name,
                    file_path=output_path or "",
                    file_size_bytes=Path(output_path).stat().st_size if output_path else 0,
                    mime_type="text/csv",
                    status=DatasetStatus.READY,
                    rows=report["dimensions"]["final_rows"],
                    columns=report["dimensions"]["final_input_features"] + 1,
                    profile=report,
                )
                db.add(ds)
                await db.commit()

    return report
