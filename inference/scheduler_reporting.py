"""Console reporting for the modular video scheduler."""

from __future__ import annotations

from .task_protocol import TaskKind


def print_run_header(
    *,
    input_path,
    in_w: int,
    in_h: int,
    info,
    out_w: int,
    out_h: int,
    output_fps: float,
    video_codec: str,
    output_pix_fmt: str,
    mode: str,
    start: str,
    end: str,
    duration: float,
    expected_source: int,
    expected_output: int,
    model: str,
    scale_text: str,
    strength: float,
    clip_length: int,
    tile_size: int,
    batch_size: int,
    rife_enabled: bool,
    source_fps: float,
    gpu_timing: bool,
) -> None:
    print("=== Real-ESRGAN ===", flush=True)
    print(
        f"Input   : {input_path.name} | {in_w}x{in_h} | "
        f"{info.fps:.3f} fps | {info.bit_depth}-bit ({info.pix_fmt})",
        flush=True,
    )
    print(
        f"Output  : {out_w}x{out_h} | {output_fps:.3f} fps | "
        f"{info.bit_depth}-bit ({output_pix_fmt}) | {video_codec}",
        flush=True,
    )
    print(
        f"Range   : {mode} | {start} -> {end} | {duration:.3f}s | "
        f"{expected_source} source / {expected_output} output frames",
        flush=True,
    )
    print(f"Model   : {model} | {scale_text}", flush=True)
    print(
        "Denoise : BasicVSR++ NTIRE Track 1 | "
        f"strength={strength:.2f} | clip={clip_length} | "
        f"tile={tile_size}(OOM fallback) | batch={batch_size}",
        flush=True,
    )

    if rife_enabled:
        print(
            f"Interp  : Practical-RIFE 4.25 | "
            f"{source_fps:.3f} -> {output_fps:.3f} fps | "
            "arbitrary timestep | scene-cut/duplicate guard | FP16",
            flush=True,
        )
    else:
        print(
            "Interp  : target FPS equals source FPS; "
            "RIFE inference bypassed",
            flush=True,
        )

    print(
        "GPU     : one spawned process per device | "
        "permanent CUDA affinity | BVS + RIFE + SR",
        flush=True,
    )
    timing_text = "on" if gpu_timing else "off"
    print(
        "Mode    : typed independent tasks | FrameHandle locality | "
        f"event-driven IPC | gpu_timing={timing_text}",
        flush=True,
    )
    print(flush=True)


def print_completed(
    *,
    base,
    processed: int,
    start: float,
    actual_duration: float,
    fps: float,
    elapsed: float,
    worker_model_time: float,
    decode_elapsed: float,
    state,
    scheduler_wait: float,
    pump,
    flush_time: float,
    audio_time: float,
    selected_tile: int,
    clip_length: int,
    batch_size: int,
    strength: float,
    scene_cuts: int,
    size_mib: float,
    bitrate: float,
    output_path,
) -> None:
    print("\n=== Completed ===", flush=True)
    print(
        f"Frames  : {processed} | {base.format_seconds(start)} -> "
        f"{base.format_seconds(start + actual_duration)} | "
        f"duration={actual_duration:.3f}s",
        flush=True,
    )
    print(
        f"Speed   : {fps:.3f} frame/s | processing={elapsed:.1f}s",
        flush=True,
    )
    print(
        f"Timing  : gpu_models={worker_model_time:.1f}s | "
        f"decode={decode_elapsed:.1f}s | "
        f"basicvsr={state.bvs_seconds:.1f}s/"
        f"{state.bvs_clips} clips | "
        f"rife={state.rife_seconds:.1f}s/"
        f"{state.rife_frames} generated | "
        f"sr={state.sr_seconds:.1f}s/{state.sr_jobs} frames | "
        f"scheduler_wait={scheduler_wait:.1f}s | "
        f"resize={pump.resize_seconds:.1f}s | "
        f"write={pump.write_seconds:.1f}s | "
        f"flush={flush_time:.1f}s | audio={audio_time:.1f}s",
        flush=True,
    )
    if state.gpu_timing_samples:
        per_worker = " | ".join(
            f"cuda:{gpu_id}={seconds:.1f}s"
            for gpu_id, seconds in zip(
                state.gpu_ids,
                state.gpu_seconds_by_worker,
            )
        )
        print(
            "GPU time: "
            f"BVS={state.gpu_seconds_by_kind[TaskKind.BVS]:.1f}s | "
            f"RIFE={state.gpu_seconds_by_kind[TaskKind.RIFE]:.1f}s | "
            f"SR={state.gpu_seconds_by_kind[TaskKind.SR]:.1f}s | "
            + per_worker,
            flush=True,
        )
    print(
        f"BasicVSR: tile={selected_tile} | clip={clip_length} | "
        f"batch={batch_size} | strength={strength:.2f} | "
        f"tiles={state.bvs_tiles} | scene_cuts={scene_cuts}",
        flush=True,
    )
    print(
        f"Scheduler: bvs_jobs={state.bvs_jobs} | "
        f"rife_jobs={state.rife_jobs} | sr_jobs={state.sr_jobs} | "
        f"source_frames={state.next_source_id} | output_frames={processed}",
        flush=True,
    )
    print(
        f"File    : {size_mib:.2f} MiB | {bitrate:.2f} Mb/s",
        flush=True,
    )
    print(f"Output  : {output_path}", flush=True)
