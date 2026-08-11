"""Typed task and frame-reference protocol for process-isolated GPU workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class TaskKind(str, Enum):
    BVS = "bvs"
    RIFE = "rife"
    SR = "sr"


class WorkerRole(str, Enum):
    TEMPORAL = "temporal"
    SR = "sr"


class FrameStorage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class FrameHandle:
    worker_id: int
    slot: int
    generation: int


@dataclass(frozen=True)
class FrameInput:
    storage: FrameStorage
    slot: int


@dataclass(frozen=True)
class BVSGroup:
    count: int
    emit_start: int
    emit_end: int


@dataclass(frozen=True)
class BVSTask:
    task_id: int
    groups: tuple[BVSGroup, ...]
    output_slots: tuple[int, ...]
    kind: TaskKind = TaskKind.BVS


@dataclass(frozen=True)
class RIFETask:
    task_id: int
    frame0: FrameInput
    frame1: FrameInput
    timesteps: tuple[float, ...]
    output_slots: tuple[int, ...]
    kind: TaskKind = TaskKind.RIFE


@dataclass(frozen=True)
class SRTask:
    task_id: int
    frame_ids: tuple[int, ...]
    frames: tuple[FrameInput, ...]
    output_slots: tuple[int, ...]
    kind: TaskKind = TaskKind.SR


WorkerTask: TypeAlias = BVSTask | RIFETask | SRTask


@dataclass(frozen=True)
class WorkerReady:
    worker_id: int
    gpu_id: int
    role: WorkerRole


@dataclass(frozen=True)
class TaskStarted:
    worker_id: int
    task_id: int
    kind: TaskKind


@dataclass(frozen=True)
class TaskComputeDone:
    """The task finished CUDA model compute and is draining transport/CPU work."""

    worker_id: int
    task_id: int
    kind: TaskKind


@dataclass(frozen=True)
class TaskError:
    worker_id: int
    error: str
    traceback_text: str
    role: WorkerRole | None = None


@dataclass(frozen=True)
class BVSResult:
    emitted_counts: tuple[int, ...]
    tile_size: int
    tiles: int
    clips: int


@dataclass(frozen=True)
class RIFEResult:
    count: int


@dataclass(frozen=True)
class SRResult:
    frame_ids: tuple[int, ...]
    output_slots: tuple[int, ...]


ResultPayload: TypeAlias = BVSResult | RIFEResult | SRResult


@dataclass(frozen=True)
class TaskResult:
    worker_id: int
    task_id: int
    kind: TaskKind
    seconds: float
    payload: ResultPayload
    gpu_seconds: float | None = None


WorkerMessage: TypeAlias = (
    WorkerReady | TaskStarted | TaskComputeDone | TaskResult | TaskError
)
