"""Internal scheduler task metadata and locality-aware queue helpers."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Deque

from .task_protocol import FrameHandle, TaskKind


@dataclass(frozen=True)
class RifeJob:
    frame0: FrameHandle
    frame1: FrameHandle
    targets: tuple[int, ...]
    timesteps: tuple[float, ...]


@dataclass(frozen=True)
class BVSActive:
    assignments: tuple[tuple[int, int], ...]
    outputs: tuple[FrameHandle, ...]


@dataclass(frozen=True)
class RIFEActive:
    targets: tuple[int, ...]
    outputs: tuple[FrameHandle, ...]
    release_on_result: tuple[FrameHandle, ...]


@dataclass(frozen=True)
class SRActive:
    frame_id: int
    output_slot: int
    release_on_result: tuple[FrameHandle, ...]


@dataclass
class ActiveTask:
    task_id: int
    kind: TaskKind
    submitted_at: float
    started_at: float | None
    meta: object
    compute_done: bool = False


def pop_preferred_frame(
    heap: list[tuple[int, FrameHandle]],
    worker_id: int,
    local_only: bool,
) -> tuple[int, FrameHandle] | None:
    if not heap:
        return None

    best_index: int | None = None
    best_frame_id: int | None = None
    for index, (frame_id, handle) in enumerate(heap):
        if handle.worker_id != worker_id:
            continue
        if best_frame_id is None or frame_id < best_frame_id:
            best_index = index
            best_frame_id = frame_id

    if best_index is None:
        if local_only:
            return None
        return heapq.heappop(heap)

    item = heap[best_index]
    last = heap.pop()
    if best_index < len(heap):
        heap[best_index] = last
        heapq.heapify(heap)
    return item


def pop_preferred_rife(
    jobs: Deque[RifeJob],
    worker_id: int,
    free_slots: int,
    local_only: bool,
) -> RifeJob | None:
    if not jobs:
        return None

    best_index: int | None = None
    best_score = -1
    for index, job in enumerate(jobs):
        if len(job.targets) > free_slots:
            continue

        score = int(job.frame0.worker_id == worker_id)
        score += int(job.frame1.worker_id == worker_id)
        if local_only and score == 0:
            continue

        if score > best_score:
            best_index = index
            best_score = score
            if score == 2:
                break

    if best_index is None:
        return None

    jobs.rotate(-best_index)
    job = jobs.popleft()
    jobs.rotate(best_index)
    return job
