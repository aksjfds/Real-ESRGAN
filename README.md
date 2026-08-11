# Real-ESRGAN 动画视频推理

当前 v6.3 流水线：

```text
FFmpeg 解码
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧
→ Real-ESRGAN full-frame 超分
→ Lanczos4 最终倍率调整
→ HEVC / AV1 编码
```

## 版本

- v4.3：早期代码基线。
- v5.8：固定 BVS 参数 + selective channels_last。
- v6.0：加入 Practical-RIFE 4.25。
- v6.1：一张 GPU 一个常驻 `spawn` 子进程，BVS / RIFE / SR 独立 task。
- v6.2：typed task protocol、FrameHandle 引用计数、locality-aware 调度。
- v6.3：去除当前活动路径的运行时 monkey-patch，显式 BVS/RIFE/SR runtime，事件驱动 IPC，RIFE compact H2D/direct-slot 输出，原生 GPU timing 和显式 progress/output runtime。

## 当前结构

- `inference.py`：当前 CLI；只暴露 `RIFE_FPS`，不再构造或删除遗留 `--fps` 参数。
- `inference/scheduler.py`：CPU 视频编排，不直接执行 CUDA 模型。
- `inference/scheduler_state.py` / `scheduler_loop.py`：任务策略、状态、结果处理、watchdog 和事件驱动等待。
- `inference/task_protocol.py`：BVS / RIFE / SR typed task/result protocol。
- `inference/gpu_transport.py`：一 GPU 一进程、共享内存、Pipe 控制通道、worker 生命周期。
- `inference/gpu_workers.py` / `frame_pool.py`：FrameHandle、slot 引用计数和 locality-aware 数据传递。
- `inference/gpu_worker_process.py`：固定 CUDA device context 的 worker 主循环。
- `inference/gpu_task_handlers.py`：GPU task handler registry。
- `inference/optimized_basicvsrpp.py`：显式 BasicVSR++ 执行优化类；替代 `v54_runtime` monkey-patch。
- `inference/optimized_rife425.py`：RIFE 4.25 compact H2D 和 direct shared-slot 输出。
- `inference/sr_runtime.py`：显式 Real-ESRGAN uint8 CUDA helper。
- `inference/output_runtime.py`：输出 pump、Lanczos4 resize、交互/批处理 progress。
- `inference/clip_source.py` / `timeline.py` / `scene_metrics.py`：CPU clip、目标时间轴、场景检测。

旧 `pipeline.py`、`v51_runtime.py`、`v54_runtime.py`、`progress_log.py`、`gpu_timing.py` 保留用于历史兼容，但当前 v6.3 主路径不再依赖它们。

## BasicVSR++ 固定参数

```text
bvs_tile_size=640
bvs_clip_length=13
bvs_batch_size=1
bvs_strength=1.0
clip_overlap=2
tile_pad=32
scene_threshold=0.30
```

若 `tile=640` OOM，只向 `384 / 320 / 256` 回退。当前版本不包含 `torch.compile`。

## RIFE 4.25 / 输出帧率

Notebook 只保留：

```python
RIFE_FPS = 60
```

`RIFE_FPS` 同时是插帧目标帧率和最终输出帧率，并且必须大于或等于源视频帧率。等于源帧率时绕过 RIFE；高于源帧率时按目标 CFR 时间轴生成真实 arbitrary timestep。

场景保护：

- `scene_difference <= 0.002`：近重复/静止，不调用 RIFE。
- `scene_difference >= 0.30`：场景切换，不跨镜头插帧。

RIFE 8-bit 输入现在保持 `uint8` 通过 H2D，再在 CUDA 上转换/归一化；生成的 8-bit 帧从 CUDA 直接复制到预留 shared-memory slot，不再经过额外 CPU ndarray 中间副本。10-bit 路径保持保守兼容实现。

## 多 GPU 稳定边界

```text
Main Scheduler
├─ cuda:0 spawn process → BVS / RIFE / SR handlers
└─ cuda:1 spawn process → BVS / RIFE / SR handlers
```

每个 GPU 子进程在整个运行期间保持固定 `torch.cuda.device(...)` context。当前主路径不存在共享 CUDA `ThreadPoolExecutor`，也不在任务级反复 `torch.cuda.set_device()`。

BVS、RIFE、SR 始终是独立 task。中间帧通过 `FrameHandle(worker_id, slot, generation)` 引用共享 slot；优先在已有数据的 GPU 上执行下一阶段，只有负载均衡需要时才进行跨 GPU CPU shared-memory copy。

控制 IPC 使用 per-worker Pipe + `multiprocessing.connection.wait()`；worker task queue 使用 `SimpleQueue`。scheduler 没有固定 10ms polling sleep，输出 slot 释放也会主动唤醒 scheduler。

启动前在 Linux 上检查 `/dev/shm` 可用容量，避免 shared-memory 不足时运行到中途才出现不明确故障。

## GPU timing

`--gpu-timing` 现在直接在 GPU worker 内围绕 BVS/RIFE/SR handler 使用 CUDA Event，并通过 typed `TaskResult` 返回统计。关闭时不创建 CUDA Event；开启时因显式同步会产生 profiling 开销。

## Kaggle 参数

```python
MODEL = "realesr-animevideov3"
SCALE = 2
RIFE_FPS = 60
GPU_IDS = "0,1"
BVS_TILE_SIZE = 640
BVS_CLIP_LENGTH = 13
BVS_BATCH_SIZE = 1
BVS_STRENGTH = 1.0
```

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1` / `libaom-av1`
- GPU AV1：`av1_nvenc`
- H.264：`libx264` / `h264_nvenc`

第三方许可见 `THIRD_PARTY_NOTICES.md`。
