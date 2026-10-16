from __future__ import annotations

from backend.config import settings
from celery import Celery
from celery.signals import worker_process_init

celery_app = Celery(
    "ml_assistant",
    broker=f"redis://{settings.redis_host}:{settings.redis_port}/0",
    backend=f"redis://{settings.redis_host}:{settings.redis_port}/1",
    include=["backend.tasks.ml_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3000,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
)


@worker_process_init.connect
def init_worker(**kwargs):
    pass


def get_celery_app() -> Celery:
    return celery_app
