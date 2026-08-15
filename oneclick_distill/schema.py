"""Job contract shared by the GUI, the CLI and the MCP agent interface.

Every interface submits a JobSpec and receives progress through a JobState.
Keeping these as plain dataclasses + dict serialization means the contract is
stable, JSON-friendly and easy to expose over WebSocket / MCP / REST.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stage(str, Enum):
    PREPARE = "prepare"
    DATA = "data"
    SYNTHETIC = "synthetic"
    TRAIN = "train"
    QUANTIZE = "quantize"
    DONE = "done"


ProgressCallback = Callable[[Stage, float, str, dict[str, Any]], None]

STAGES = [Stage.PREPARE, Stage.DATA, Stage.SYNTHETIC, Stage.TRAIN, Stage.QUANTIZE, Stage.DONE]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class TeacherConfig:
    name: str = "none"          # deepseek | openai | local | none
    model: str = ""
    base_url: str = ""
    api_key: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else "",
        }


@dataclass
class JobSpec:
    """Everything needed to launch a distillation pipeline run."""

    id: str = field(default_factory=new_id)
    source: str = "cli"         # "ui" | "cli" | "mcp"
    data_paths: list[str] = field(default_factory=list)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    model: str = ""             # resolved HF model id
    size: str = "ultra"         # ultra | balanced | smoke
    max_steps: Optional[int] = None   # override for smoke/dry runs
    smoke: bool = False         # tiny model, tiny run
    dry_run: bool = False       # probe VRAM with 1 fake step, no real train
    quantize: bool = True       # export to GGUF after training
    out_dir: str = ""           # default: runs/<id>

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "source": self.source,
            "data_paths": list(self.data_paths),
            "teacher": self.teacher.to_dict(),
            "model": self.model,
            "size": self.size,
            "max_steps": self.max_steps,
            "smoke": self.smoke,
            "dry_run": self.dry_run,
            "quantize": self.quantize,
            "out_dir": self.out_dir,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobSpec":
        t = d.get("teacher", {}) or {}
        return cls(
            id=d.get("id") or new_id(),
            source=d.get("source", "cli"),
            data_paths=list(d.get("data_paths", [])),
            teacher=TeacherConfig(
                name=t.get("name", "none"),
                model=t.get("model", ""),
                base_url=t.get("base_url", ""),
                api_key=t.get("api_key", ""),
            ),
            model=d.get("model", ""),
            size=d.get("size", "ultra"),
            max_steps=d.get("max_steps"),
            smoke=bool(d.get("smoke", False)),
            dry_run=bool(d.get("dry_run", False)),
            quantize=bool(d.get("quantize", True)),
            out_dir=d.get("out_dir", ""),
        )


@dataclass
class JobState:
    """Mutable live state of a job, streamed to subscribers."""

    id: str
    status: Status = Status.QUEUED
    stage: Stage = Stage.PREPARE
    progress: float = 0.0
    message: str = "queued"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source: str = "cli"
    spec: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def update(self, stage: Stage, progress: float, message: str, metrics: dict[str, Any] | None = None):
        self.stage = stage
        self.progress = round(max(0.0, min(progress, 1.0)), 4)
        self.message = message
        if metrics:
            self.metrics.update(metrics)
        self.updated_at = time.time()

    def set_status(self, status: Status, message: str, error: str | None = None):
        self.status = status
        self.message = message
        self.error = error
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "stage": self.stage.value,
            "progress": self.progress,
            "message": self.message,
            "metrics": self.metrics,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "spec": self.spec,
            "result": self.result,
        }


def make_progress_cb(state: JobState) -> ProgressCallback:
    """Build a callback that mutates a JobState."""

    def cb(stage: Stage, progress: float, message: str, metrics: dict[str, Any] | None = None):
        state.update(stage, progress, message, metrics or {})

    return cb
