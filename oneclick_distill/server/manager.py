"""Job manager: runs pipelines in background threads and streams progress to
WebSocket subscribers. This is the single source of truth for job state."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from ..runner import run_pipeline
from ..schema import JobSpec, JobState, Status, make_progress_cb


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    # ---- accessors -------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    # ---- submission -------------------------------------------------------
    def submit(self, spec: JobSpec) -> JobState:
        state = JobState(
            id=spec.id,
            source=spec.source,
            spec=spec.to_dict(),
        )
        state.set_status(Status.RUNNING, "已提交，正在初始化")
        self._jobs[spec.id] = state

        thread = threading.Thread(target=self._run, args=(spec, state), daemon=True)
        thread.start()
        return state

    def _run(self, spec: JobSpec, state: JobState) -> None:
        base_cb = make_progress_cb(state)

        def cb(stage, progress, message, metrics=None):
            base_cb(stage, progress, message, metrics or {})
            self._publish(state)

        try:
            result = run_pipeline(spec, cb)
            state.result = result
            state.set_status(Status.DONE, "蒸馏完成 ✓")
        except Exception as e:  # noqa: BLE001
            state.set_status(Status.FAILED, str(e), str(e))
        finally:
            self._publish(state)

    # ---- subscription ------------------------------------------------------
    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id, [])
        if q in subs:
            subs.remove(q)

    def _publish(self, state: JobState) -> None:
        for q in self._subscribers.get(state.id, []):
            if self._loop:
                self._loop.call_soon_threadsafe(q.put_nowait, state.to_dict())


manager = JobManager()
