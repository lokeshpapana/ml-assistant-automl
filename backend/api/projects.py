from uuid import UUID

from backend.api.schemas import (
    PaginatedResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
)
from backend.database import get_async_session
from backend.database.models import Project, ProjectStatus
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_in: ProjectCreate,
    session: AsyncSession = Depends(get_async_session),
) -> ProjectResponse:
    project = Project(**project_in.model_dump())
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


@router.get("", response_model=PaginatedResponse)
async def list_projects(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: ProjectStatus | None = None,
    session: AsyncSession = Depends(get_async_session),
) -> PaginatedResponse:
    query = select(Project)
    if status:
        query = query.where(Project.status == status)

    count_query = select(func.count()).select_from(query.subquery())
    total = await session.scalar(count_query)

    query = (
        query.order_by(Project.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )
    result = await session.execute(query)
    projects = result.scalars().all()

    return PaginatedResponse(
        items=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> ProjectResponse:
    project = await session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    project_in: ProjectUpdate,
    session: AsyncSession = Depends(get_async_session),
) -> ProjectResponse:
    project = await session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = project_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)

    await session.commit()
    await session.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> None:
    project = await session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.status = ProjectStatus.DELETED
    await session.commit()


@router.get("/{project_id}/context")
async def project_context(
    project_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    """Compact PROJECT_CONTEXT.json: structured memory + latest run pointer.

    This (plus the current task and the user's message) is the entire context
    budget — history, logs, and full project data are never resent wholesale.
    """
    from backend import runlog

    project = await session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    memory = runlog.get_memory(str(project_id))
    latest = runlog.list_runs(project_id=str(project_id), limit=1)
    latest_run = latest[0] if latest else None
    return {
        "project_id": str(project_id),
        "goal": memory.get("goal") or project.description or project.name,
        "task": memory.get("task_type"),
        "dataset": memory.get("dataset"),
        "target": memory.get("target"),
        "constraints": memory.get("constraints"),
        "decisions": (memory.get("decisions") or [])[-10:],
        "experiments": [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "best_model": (r.get("best") or {}).get("model"),
            }
            for r in runlog.list_runs(project_id=str(project_id), limit=10)
        ],
        "best_model": memory.get("best"),
        "research": (memory.get("research") or [])[-10:],
        "warnings": memory.get("warnings") or [],
        "pending_config": memory.get("pending_config"),
        "current_run": (
            {k: latest_run.get(k) for k in ("run_id", "status", "stage", "progress")}
            if latest_run
            else None
        ),
        "pending_features": memory.get("pending_features"),
        "excluded_features": memory.get("excluded_features"),
        "pending_model": memory.get("pending_model"),
    }
