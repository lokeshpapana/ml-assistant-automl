from uuid import UUID

from backend.api.schemas import (
    PaginatedResponse,
    SessionCreate,
    SessionResponse,
    SessionUpdate,
)
from backend.database import get_async_session
from backend.database.models import Session
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_in: SessionCreate,
    project_id: UUID = Query(...),
    db: AsyncSession = Depends(get_async_session),
) -> SessionResponse:
    from backend.database.models import Project

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    session = Session(project_id=project_id, **session_in.model_dump())
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("", response_model=PaginatedResponse)
async def list_sessions(
    project_id: UUID = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_async_session),
) -> PaginatedResponse:
    query = select(Session).where(Session.project_id == project_id)

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = (
        query.order_by(Session.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )
    result = await db.execute(query)
    sessions = result.scalars().all()

    return PaginatedResponse(
        items=[SessionResponse.model_validate(s) for s in sessions],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> SessionResponse:
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: UUID,
    session_in: SessionUpdate,
    db: AsyncSession = Depends(get_async_session),
) -> SessionResponse:
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    update_data = session_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(session, field, value)

    await db.commit()
    await db.refresh(session)
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> None:
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()


@router.get("/{session_id}/messages")
async def session_messages(
    session_id: UUID,
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_async_session),
) -> dict:
    """Persistent chat history for one session (oldest first)."""
    if not await db.get(Session, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    from backend import runlog

    return {
        "session_id": str(session_id),
        "messages": runlog.list_chat(session_id=str(session_id), limit=limit),
    }


@router.delete("/{session_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session_messages(
    session_id: UUID,
    db: AsyncSession = Depends(get_async_session),
) -> None:
    """Clear one session's chat history. Session-scoped only — never global."""
    if not await db.get(Session, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    from backend import runlog

    runlog.clear_chat(str(session_id))
