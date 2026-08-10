# Real-ESRGAN 动画视频推理

当前 v6.2 流水线：

```text
FFmpeg 解码
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧
→ Real-ESRGAN full-frame 超分
→ Lanczos4 最终倍率调整
→ HEVC / AV1 编码
```

## 版本

- v4.3 代码基线：`90208548939f7b59ae08ff7db7f338b41b703e22`
- v5.8：固定 BVS 参数 + selective channels_last。
- v6.0：加入 Practical-RIFE 4.25。
- v6.1：多 GPU 重构为一张 GPU 一个常驻 `spawn` 子进程，BVS / RIFE / SR 独立 task。
- v6.2：调度模块拆分、typed task protocol、FrameHandle 引用计数与 locality-aware 调度；同 GPU 中间帧不再往返主进程复制。

## 结构

- `inference.py`：统一 CLI，BasicVSR++ / RIFE 参数校验。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/basicvsrpp.py`：BasicVSR++ NTIRE Track 1 模型。
- `inference/rife425.py`：Practical-RIFE 4.25 inference-only 模型与权重缓存。
- `inference/v52_scheduler.py`：兼容入口，转发到模块化 scheduler。
- `inference/scheduler.py`：视频编排；不直接执行 CUDA 模型。
- `inference/scheduler_state.py` / `scheduler_loop.py`：调度状态、策略、结果处理与 watchdog。
- `inference/task_protocol.py`：BVS / RIFE / SR typed task/result protocol。
- `inference/gpu_transport.py` / `gpu_worker_process.py`：一 GPU 一进程、共享内存与进程生命周期。
- `inference/gpu_workers.py` / `frame_pool.py`：FrameHandle、slot 引用计数与 locality-aware 数据传递。
- `inference/clip_source.py` / `timeline.py` / `scene_metrics.py`：CPU clip、目标时间轴和场景检测。
- `inference/gpu_task_handlers.py`：GPU task handler registry。
- `inference/v54_runtime.py`：BVS warp grid cache、临时张量削减、8-bit 紧凑 H2D、selective channels_last。
- `inference/pipeline.py`：输出泵、FFmpeg 解码等基础能力。
- `encode/`：HEVC / AV1 编码后端。

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

不使用 `SOURCE_PROFILE` 或 autotuner。若 `tile=640` OOM，只向 `384 / 320 / 256` 回退。

## RIFE 4.25 / 输出帧率

Notebook 只保留一个帧率参数：

```python
RIFE_FPS = 60
```

不提供独立的 `FPS = "source"` / `--fps` 配置。`RIFE_FPS` 同时是 RIFE 目标帧率和最终输出帧率，并且必须大于或等于源视频帧率。

当 `RIFE_FPS` 等于源帧率时绕过 RIFE；高于源帧率时，根据目标 CFR 时间轴计算真实 interpolation timestep。例如 24 → 60 FPS 直接生成 60 FPS 时间点所需的中间帧，不先升到 120 FPS 再丢帧。

RIFE 位于 BasicVSR++ 之后、Real-ESRGAN 之前：

- BasicVSR++ 仍只处理源帧率。
- RIFE 只处理源分辨率恢复帧。
- Real-ESRGAN 接收真实目标 FPS 帧流。
- Writer 直接以目标帧率接收帧，不由 FFmpeg `fps` filter 补重复帧。

### 场景切换和重复帧

- `scene_difference <= 0.002`：视为近重复/静止，不调用 RIFE。
- `scene_difference >= 0.30`：视为场景切换，不跨镜头插帧。

### 模型与 checkpoint

Practical-RIFE 4.25 首次使用时下载 `RIFEv4.25_0919.zip`，提取 `flownet.pkl` 并缓存到：

```text
~/.cache/realesrgan/rife-v4.25/flownet.pkl
```

加载时只过滤官方 checkpoint 中已知的 training-only `teacher.*` / `caltime.*` 参数，剩余 IFNet 推理权重执行 strict 校验。

## v6.2 多 GPU 调度

稳定边界固定为：

```text
Main Scheduler
├─ cuda:0 spawn process → BVS / RIFE / SR handlers
└─ cuda:1 spawn process → BVS / RIFE / SR handlers
```

每个 GPU 子进程在整个运行期间保持固定 CUDA affinity；不存在共享 CUDA `ThreadPoolExecutor`。BVS、RIFE、SR 是三个独立 task，新增 GPU 功能应新增 task/handler，而不是修改进程模型。

中间恢复帧由 `FrameHandle(worker_id, slot, generation)` 引用共享内存 slot。引用计数保证 slot 只有在所有消费者完成后才能复用，并检测 stale handle / refcount underflow。

调度器优先把 RIFE / SR 派给已经持有输入 FrameHandle 的 GPU：

```text
同 GPU：FrameHandle → worker-local shared slot → 下一 task
跨 GPU：仅在负载均衡需要时复制一次到目标 worker input slot
```

因此 v6.2 去除了 v6.1 `take_frames(copy=True) → 主进程 ndarray → 再复制回 worker` 的固定中间往返，同时保留跨 GPU 动态负载均衡。

worker 会报告 `STARTED / RESULT / ERROR`；scheduler 检查进程存活和分阶段 timeout，并每 30 秒输出 `[gpu-status]`，避免静默卡在 0%。

## v5.8 selective channels_last

BasicVSR++ 仅对主要 Conv2d-heavy 区域转换 Conv2d 权重 memory format：

- `feat_extract`
- `backward_1`
- `forward_1`
- `backward_2`
- `forward_2`
- `reconstruction`

SPyNet、deformable alignment / `deform_conv2d`、`conv_offset`、PixelShuffle、`conv_hr`、`conv_last` 保持原路径。当前版本不包含 `torch.compile`。

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
