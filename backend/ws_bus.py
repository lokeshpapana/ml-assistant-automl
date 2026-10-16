"""Thread-safe bridge between the background orchestrator thread and the
FastAPI/WebSocket asyncio loop.

Agent worker threads call :meth:`EventBus.publish` which schedules the event onto
every subscribed client's asyncio queue via ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from typing import Any

import anyio


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._counter = itertools.count(1)
        self._history: list[dict[str, Any]] = []
        self.max_history = 2000

    # ------------------------------------------------------------- subscribe

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> tuple[str, asyncio.Queue]:
        client_id = f"client-{next(self._counter)}"
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers[client_id] = (loop, queue)
        return client_id, queue

    def unsubscribe(self, client_id: str) -> None:
        with self._lock:
            self._subscribers.pop(client_id, None)

    def replay(
        self,
        client_id: str,
        project_id: str | None = None,
        all_projects: bool = False,
    ) -> list[dict[str, Any]]:
        """History for a fresh connection.

        With ``all_projects=True`` (classic single-run /ui) the whole buffer is
        returned. Otherwise only events belonging to ``project_id`` are sent, so
        one project's activity can never appear in another project's UI
        (regression: the buffer was previously broadcast to every client).
        ``project_id=None, all_projects=False`` means "unassigned events only".
        """
        with self._lock:
            history = list(self._history)
        if all_projects:
            return history
        return [e for e in history if e.get("project_id") == project_id]

    # ------------------------------------------------------------- publish

    def publish(self, event: dict[str, Any]) -> None:
        """Called from worker threads; schedules delivery on each asyncio loop."""
        with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            subs = list(self._subscribers.items())
        for _, (loop, queue) in subs:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except (RuntimeError, ValueError):
                pass

    def publish_sync(self, event: dict[str, Any]) -> None:
        """Called from within the asyncio loop; puts directly without threadsafe hop."""
        with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            subs = list(self._subscribers.items())
        for _, (loop, queue) in subs:
            try:
                queue.put_nowait(event)
            except Exception:
                pass

    async def send(self, event: dict[str, Any]) -> None:
        """Awaitable variant safe from async contexts (anyio compatibility)."""
        with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            subs = list(self._subscribers.items())
        for _, (loop, queue) in subs:
            try:
                await anyio.to_thread.run_sync(lambda q=queue: q.put_nowait(dict(event)))
            except Exception:
                pass


_bus_lock = threading.Lock()
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = EventBus()
        return _bus