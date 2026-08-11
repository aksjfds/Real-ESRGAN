"""Result handling, watchdogs, and the event-driven scheduler loop."""

from __future__ import annotations

import queue
import time

from .scheduler_state import SchedulerState
from .scheduler_types import BVSActive, RIFEActive, SRActive
from .task_protocol import (
    BVSResult,
    RIFEResult,
    SRResult,
    TaskComputeDone,
    TaskError,
    TaskKind,
    TaskResult,
    TaskStarted,
)


def _handle_started(
    state: SchedulerState,
    message: TaskStarted,
) -> None:
    active = state.active_for_kind(message.worker_id, message.kind)
    if (
        active is None
        or active.task_id != message.task_id
        or active.kind != message.kind
    ):
        raise RuntimeError(
            "Unexpected worker start: "
            f"gpu={message.worker_id} "
            f"task={message.task_id}/{message.kind.value}"
        )
    active.started_at = time.monotonic()


def _handle_compute_done(
    state: SchedulerState,
    message: TaskComputeDone,
) -> None:
    state.mark_compute_done(
        message.worker_id,
        message.task_id,
        message.kind,
    )


def _handle_bvs_result(
    state: SchedulerState,
    message: TaskResult,
) -> None:
    if not isinstance(message.payload, BVSResult):
        raise RuntimeError("BVS worker returned an invalid payload")

    active = state.active_for_kind(message.worker_id, message.kind)
    if active is None or not isinstance(active.meta, BVSActive):
        raise RuntimeError("BVS scheduler metadata mismatch")

    meta = active.meta
    emitted_counts = message.payload.emitted_counts
    if len(meta.assignments) != len(emitted_counts):
        raise RuntimeError(
            "BVS worker returned mismatched assignment count"
        )

    cursor = 0
    for (
        output_start,
        expected_count,
    ), emitted_count in zip(
        meta.assignments,
        emitted_counts,
    ):
        if emitted_count != expected_count:
            raise RuntimeError(
                f"BVS worker emitted {emitted_count} frames, "
                f"expected {expected_count}"
            )

        end_cursor = cursor + emitted_count
        handles = meta.outputs[cursor:end_cursor]
        if len(handles) != emitted_count:
            raise RuntimeError(
                "BVS output-handle accounting mismatch"
            )

        for offset, handle in enumerate(handles):
            source_id = output_start + offset
            if source_id in state.restored_sources:
                raise RuntimeError(
                    f"Duplicate restored source frame {source_id}"
                )
            state.restored_sources[source_id] = handle
        cursor = end_cursor

    if cursor != len(meta.outputs):
        raise RuntimeError(
            "BVS worker left unassigned output handles"
        )

    state.bvs_seconds += message.seconds
    state.bvs_clips += message.payload.clips
    state.bvs_tiles += message.payload.tiles
    if message.payload.tile_size > 0:
        state.selected_tiles.append(message.payload.tile_size)


def _handle_rife_result(
    state: SchedulerState,
    message: TaskResult,
) -> None:
    if not isinstance(message.payload, RIFEResult):
        raise RuntimeError("RIFE worker returned an invalid payload")

    active = state.active_for_kind(message.worker_id, message.kind)
    if active is None or not isinstance(active.meta, RIFEActive):
        raise RuntimeError("RIFE scheduler metadata mismatch")

    meta = active.meta
    state.workers.release_handles(meta.release_on_result)
    if message.payload.count != len(meta.targets):
        raise RuntimeError(
            f"RIFE worker returned {message.payload.count} frames "
            f"for {len(meta.targets)} targets"
        )
    if len(meta.outputs) != len(meta.targets):
        raise RuntimeError(
            "RIFE output-handle accounting mismatch"
        )

    for target, handle in zip(meta.targets, meta.outputs):
        state.add_target(target, handle, claim=False)

    state.rife_seconds += message.seconds
    state.rife_frames += message.payload.count


def _handle_sr_result(
    state: SchedulerState,
    message: TaskResult,
) -> None:
    if not isinstance(message.payload, SRResult):
        raise RuntimeError("SR worker returned an invalid payload")

    active = state.active_for_kind(message.worker_id, message.kind)
    if active is None or not isinstance(active.meta, SRActive):
        raise RuntimeError("SR scheduler metadata mismatch")

    meta = active.meta
    state.workers.release_handles(meta.release_on_result)
    if message.payload.frame_id != meta.frame_id:
        raise RuntimeError(
            "SR worker returned an unexpected frame id"
        )
    if message.payload.output_slot != meta.output_slot:
        raise RuntimeError(
            "SR worker returned an unexpected output slot"
        )

    state.pending[meta.frame_id] = (
        message.worker_id,
        meta.output_slot,
    )
    state.sr_seconds += message.seconds


def _handle_result(
    state: SchedulerState,
    message: TaskResult,
) -> None:
    active = state.active_for_kind(message.worker_id, message.kind)
    if (
        active is None
        or active.task_id != message.task_id
        or active.kind != message.kind
    ):
        raise RuntimeError(
            "Unexpected worker result: "
            f"gpu={message.worker_id} "
            f"task={message.task_id}/{message.kind.value}"
        )
    if not active.compute_done:
        raise RuntimeError(
            "Worker returned a result before its compute boundary: "
            f"gpu={message.worker_id} "
            f"task={message.task_id}/{message.kind.value}"
        )

    state.record_gpu_timing(
        message.worker_id,
        message.kind,
        message.gpu_seconds,
    )

    if message.kind == TaskKind.BVS:
        _handle_bvs_result(state, message)
    elif message.kind == TaskKind.RIFE:
        _handle_rife_result(state, message)
    elif message.kind == TaskKind.SR:
        _handle_sr_result(state, message)
    else:
        raise RuntimeError(
            f"Unknown result kind: {message.kind!r}"
        )

    state.clear_active(message.worker_id, message.kind)


def _watchdog(
    state: SchedulerState,
    now: float,
) -> None:
    for worker_id in range(len(state.gpu_ids)):
        if not state.workers.is_alive(worker_id):
            raise RuntimeError(
                f"cuda:{state.gpu_ids[worker_id]} worker process "
                "exited unexpectedly"
            )

        for active in (
            state.temporal_active[worker_id],
            state.sr_active[worker_id],
        ):
            if active is None:
                continue

            reference = (
                active.started_at
                if active.started_at is not None
                else active.submitted_at
            )
            limit = state.timeout_by_kind[active.kind]
            if now - reference <= limit:
                continue

            if active.compute_done:
                phase = "draining transport"
            elif active.started_at is not None:
                phase = "running compute"
            else:
                phase = "queued"
            raise TimeoutError(
                f"cuda:{state.gpu_ids[worker_id]} "
                f"{active.kind.value.upper()} task {active.task_id} "
                f"stalled while {phase} for {now - reference:.1f}s"
            )


def _schedule_idle_worker_without_rife(
    state: SchedulerState,
    worker_id: int,
) -> bool:
    """Keep BVS parallel until the restored-frame backlog reaches one emit batch.

    A task whose CUDA compute is complete no longer blocks the opposite role:
    BVS/SR transport drain can overlap the other role's CUDA compute. Heavy
    model compute itself remains exclusive per physical GPU for predictable
    VRAM use and to avoid inter-process kernel contention.
    """
    if state.compute_busy(worker_id):
        return False

    if (
        not state.bvs_eof
        and not state.bvs_running()
        and state.schedule_bvs(worker_id)
    ):
        return True

    sr_trigger = max(
        1,
        int(state.clip_source.clip_length)
        - 2 * int(state.clip_source.overlap),
    )
    other_bvs_running = state.bvs_running()
    sr_backlog_ready = (
        state.bvs_eof
        or (other_bvs_running and len(state.restored_heap) >= sr_trigger)
    )

    if sr_backlog_ready:
        if state.schedule_sr(worker_id, local_only=True):
            return True
        if state.schedule_sr(worker_id, local_only=False):
            return True

    if not state.bvs_eof and state.schedule_bvs(worker_id):
        return True

    if state.schedule_sr(worker_id, local_only=True):
        return True
    if state.schedule_sr(worker_id, local_only=False):
        return True
    return False


def run_scheduler(
    state: SchedulerState,
    pump,
    expected_output: int,
    _frame_output_slots: int,
) -> float:
    scheduler_wait = 0.0

    while True:
        pump.check()
        made_progress = False

        while True:
            try:
                message = state.workers.result(False)
            except queue.Empty:
                break

            if isinstance(message, TaskError):
                raise RuntimeError(
                    f"GPU worker {message.worker_id} failed: "
                    f"{message.error}\n{message.traceback_text}"
                )
            if isinstance(message, TaskStarted):
                _handle_started(state, message)
                made_progress = True
                continue
            if isinstance(message, TaskComputeDone):
                _handle_compute_done(state, message)
                made_progress = True
                continue
            if not isinstance(message, TaskResult):
                raise RuntimeError(
                    "Unexpected unified GPU message: "
                    f"{type(message).__name__}"
                )

            _handle_result(state, message)
            made_progress = True

        if state.promote_intervals(
            final=state.bvs_eof and not state.bvs_running()
        ):
            made_progress = True

        while state.next_output in state.pending:
            worker_id, output_slot = state.pending.pop(state.next_output)
            pump.put(
                state.next_output,
                worker_id,
                output_slot,
            )
            state.next_output += 1
            made_progress = True

        for worker_id in range(len(state.gpu_ids)):
            if state.planner.rife_enabled:
                scheduled = state.schedule_idle_worker(worker_id)
            else:
                scheduled = _schedule_idle_worker_without_rife(
                    state,
                    worker_id,
                )
            if scheduled:
                made_progress = True

        now = time.monotonic()
        _watchdog(state, now)

        if state.generation_done():
            break

        if not made_progress:
            wait_started = time.monotonic()
            state.workers.wait_for_event(1.0)
            scheduler_wait += time.monotonic() - wait_started

    if state.pending:
        raise RuntimeError(
            f"Pipeline ended with {len(state.pending)} "
            "out-of-order SR frame(s); "
            f"next expected frame is {state.next_output}"
        )

    if len(state.claimed_targets) != expected_output:
        missing = sorted(
            set(range(expected_output)) - state.claimed_targets
        )[:8]
        raise RuntimeError(
            f"Timeline claimed {len(state.claimed_targets)}/"
            f"{expected_output} targets; missing={missing}"
        )

    if state.next_output != expected_output:
        raise RuntimeError(
            f"Pipeline output mismatch: emitted={state.next_output}, "
            f"expected={expected_output}"
        )

    live_handles = state.workers.live_frame_handles()
    if live_handles:
        raise RuntimeError(
            f"Pipeline ended with {live_handles} live FrameHandle slot(s)"
        )

    return scheduler_wait
