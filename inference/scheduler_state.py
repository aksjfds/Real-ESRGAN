"""Mutable scheduling state and task-submission policy."""

from __future__ import annotations

from collections import deque
import heapq
import time
from typing import Sequence

import numpy as np

from .clip_source import ClipSource
from .gpu_workers import UnifiedGPUWorkers
from .scheduler_types import (
    ActiveTask,
    BVSActive,
    RIFEActive,
    RifeJob,
    SRActive,
    pop_preferred_frame,
    pop_preferred_rife,
)
from .task_protocol import FrameHandle, TaskKind
from .timeline import TimelinePlanner


class SchedulerState:
    def __init__(
        self,
        workers: UnifiedGPUWorkers,
        planner: TimelinePlanner,
        clip_source: ClipSource,
        gpu_ids: Sequence[int],
        batch_size: int,
        bvs_headroom: int,
        expected_output: int,
    ) -> None:
        self.workers = workers
        self.planner = planner
        self.clip_source = clip_source
        self.gpu_ids = [int(value) for value in gpu_ids]
        self.batch_size = int(batch_size)
        self.bvs_headroom = int(bvs_headroom)
        self.expected_output = int(expected_output)

        self.pending: dict[int, int] = {}
        self.next_output = 0
        self.next_source_id = 0
        self.next_interval = 0
        self.next_task_id = 0
        self.bvs_eof = False
        self.final_interval_flushed = False

        self.claimed_targets: set[int] = set()
        self.restored_sources: dict[int, FrameHandle] = {}
        self.restored_heap: list[tuple[int, FrameHandle]] = []
        self.rife_queue = deque()
        self.active: list[ActiveTask | None] = [
            None for _ in self.gpu_ids
        ]

        self.bvs_seconds = 0.0
        self.rife_seconds = 0.0
        self.sr_seconds = 0.0
        self.bvs_clips = 0
        self.bvs_tiles = 0
        self.rife_frames = 0
        self.selected_tiles: list[int] = []
        self.bvs_jobs = 0
        self.rife_jobs = 0
        self.sr_jobs = 0

        self.gpu_seconds_by_kind = {
            TaskKind.BVS: 0.0,
            TaskKind.RIFE: 0.0,
            TaskKind.SR: 0.0,
        }
        self.gpu_seconds_by_worker = [0.0 for _ in self.gpu_ids]
        self.gpu_timing_samples = 0

        self.timeout_by_kind = {
            TaskKind.BVS: 300.0,
            TaskKind.RIFE: 180.0,
            TaskKind.SR: 120.0,
        }

    def next_id(self) -> int:
        value = self.next_task_id
        self.next_task_id += 1
        return value

    def record_gpu_timing(
        self,
        worker_id: int,
        kind: TaskKind,
        seconds: float | None,
    ) -> None:
        if seconds is None:
            return
        value = max(0.0, float(seconds))
        self.gpu_seconds_by_kind[kind] += value
        self.gpu_seconds_by_worker[worker_id] += value
        self.gpu_timing_samples += 1

    def claim_targets(self, targets: Sequence[int]) -> None:
        for target in targets:
            target = int(target)
            if target in self.claimed_targets:
                raise RuntimeError(
                    f"Timeline produced duplicate target frame {target}"
                )
            self.claimed_targets.add(target)

    def add_target(
        self,
        frame_id: int,
        handle: FrameHandle,
        *,
        claim: bool,
    ) -> None:
        if claim:
            self.claim_targets((frame_id,))
        heapq.heappush(
            self.restored_heap,
            (int(frame_id), handle),
        )

    def promote_intervals(self, final: bool = False) -> bool:
        made_progress = False

        while (
            self.next_interval in self.restored_sources
            and self.next_interval + 1 in self.restored_sources
        ):
            current = self.restored_sources.pop(self.next_interval)
            nxt = self.restored_sources[self.next_interval + 1]
            plan = self.planner.plan_interval(
                self.next_interval,
                self.workers.frame_view(current),
                self.workers.frame_view(nxt),
            )
            self.claim_targets(plan.direct_targets)
            self.claim_targets(plan.rife_targets)

            current_uses = len(plan.direct_targets)
            if plan.rife_targets:
                current_uses += 1

            if current_uses == 0:
                self.workers.release_handle(current)
            else:
                self.workers.retain(current, current_uses - 1)
                for target in plan.direct_targets:
                    self.add_target(target, current, claim=False)

                if plan.rife_targets:
                    self.workers.retain(nxt)
                    self.rife_queue.append(
                        RifeJob(
                            frame0=current,
                            frame1=nxt,
                            targets=plan.rife_targets,
                            timesteps=plan.timesteps,
                        )
                    )

            self.next_interval += 1
            made_progress = True

        if (
            final
            and not self.final_interval_flushed
            and self.next_source_id > 0
            and self.next_interval == self.next_source_id - 1
            and self.next_interval in self.restored_sources
        ):
            current = self.restored_sources.pop(self.next_interval)
            targets = self.planner.final_targets(self.next_interval)
            self.claim_targets(targets)

            if not targets:
                self.workers.release_handle(current)
            else:
                self.workers.retain(current, len(targets) - 1)
                for target in targets:
                    self.add_target(target, current, claim=False)

            self.next_interval += 1
            self.final_interval_flushed = True
            made_progress = True

        return made_progress

    def schedule_bvs(self, worker_id: int) -> bool:
        if self.bvs_eof:
            return False
        if (
            self.workers.available_frame_slots(worker_id)
            < self.bvs_headroom
        ):
            return False

        groups: list[tuple[Sequence[np.ndarray], int, int]] = []
        assignments: list[tuple[int, int]] = []
        for _ in range(self.batch_size):
            raw_task = self.clip_source.next_task()
            if raw_task is None:
                self.bvs_eof = True
                break

            frames, emit_start, emit_end = raw_task
            emit_count = max(
                0,
                int(emit_end) - int(emit_start),
            )
            groups.append((frames, emit_start, emit_end))
            assignments.append((self.next_source_id, emit_count))
            self.next_source_id += emit_count

        if not groups:
            return False

        task_id = self.next_id()
        outputs = self.workers.submit_bvs(
            worker_id,
            task_id,
            groups,
        )
        self.active[worker_id] = ActiveTask(
            task_id=task_id,
            kind=TaskKind.BVS,
            submitted_at=time.monotonic(),
            started_at=None,
            meta=BVSActive(
                assignments=tuple(assignments),
                outputs=outputs,
            ),
        )
        self.bvs_jobs += len(groups)
        return True

    def schedule_rife(
        self,
        worker_id: int,
        *,
        local_only: bool,
    ) -> bool:
        job = pop_preferred_rife(
            self.rife_queue,
            worker_id,
            self.workers.available_frame_slots(worker_id),
            local_only,
        )
        if job is None:
            return False

        task_id = self.next_id()
        outputs, deferred = self.workers.submit_rife(
            worker_id,
            task_id,
            job.frame0,
            job.frame1,
            job.timesteps,
        )
        self.active[worker_id] = ActiveTask(
            task_id=task_id,
            kind=TaskKind.RIFE,
            submitted_at=time.monotonic(),
            started_at=None,
            meta=RIFEActive(
                targets=job.targets,
                outputs=outputs,
                release_on_result=deferred,
            ),
        )
        self.rife_jobs += 1
        return True

    def schedule_sr(
        self,
        worker_id: int,
        *,
        local_only: bool,
    ) -> bool:
        if not self.workers.can_submit_sr(worker_id):
            return False

        item = pop_preferred_frame(
            self.restored_heap,
            worker_id,
            local_only,
        )
        if item is None:
            return False

        frame_id, handle = item
        task_id = self.next_id()
        deferred = self.workers.submit_sr(
            worker_id,
            task_id,
            frame_id,
            handle,
        )
        self.active[worker_id] = ActiveTask(
            task_id=task_id,
            kind=TaskKind.SR,
            submitted_at=time.monotonic(),
            started_at=None,
            meta=SRActive(
                frame_id=frame_id,
                release_on_result=deferred,
            ),
        )
        self.sr_jobs += 1
        return True

    def bvs_running(self) -> bool:
        return any(
            item is not None and item.kind is TaskKind.BVS
            for item in self.active
        )

    def schedule_idle_worker(self, worker_id: int) -> bool:
        if self.active[worker_id] is not None:
            return False

        if (
            not self.bvs_eof
            and not self.bvs_running()
            and self.schedule_bvs(worker_id)
        ):
            return True
        if self.schedule_rife(worker_id, local_only=True):
            return True
        if self.schedule_sr(worker_id, local_only=True):
            return True
        if self.schedule_rife(worker_id, local_only=False):
            return True
        if self.schedule_sr(worker_id, local_only=False):
            return True
        if not self.bvs_eof and self.schedule_bvs(worker_id):
            return True
        return False

    def all_idle(self) -> bool:
        return all(item is None for item in self.active)

    def generation_done(self) -> bool:
        return (
            self.bvs_eof
            and self.final_interval_flushed
            and not self.restored_sources
            and not self.rife_queue
            and not self.restored_heap
            and self.all_idle()
        )
