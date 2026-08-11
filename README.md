# Real-ESRGAN 动画视频推理

当前 v6.5 流水线：

```text
FFmpeg 解码
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧（可关闭）
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
- v6.3：去除活动路径 monkey-patch，显式 runtime、事件驱动 IPC、RIFE direct-slot、原生 GPU timing。
- v6.4：OutputPump/Pipe fail-fast、BVS CUDA direct-slot、scene signature cache、runtime API 边界和更严格 task/runtime 类型协议。
- v6.5：`RIFE_FPS=0` 关闭插帧并保持源帧率；RIFE 优先使用仓库内模型归档；离线 `[progress]` 首次 60 秒后开始并每 60 秒打印一次。

## 当前结构

- `inference.py`：当前 CLI；只暴露 `RIFE_FPS`，不提供遗留 `--fps` 参数。
- `inference/scheduler.py`：CPU 视频编排，不直接执行 CUDA 模型。
- `inference/scheduler_state.py` / `scheduler_loop.py`：任务策略、状态、结果处理、watchdog 和事件驱动等待。
- `inference/task_protocol.py`：BVS / RIFE / SR typed task/result protocol。
- `inference/worker_protocols.py`：GPU runtime 结构化接口。
- `inference/runtime_api.py`：当前调度路径与 legacy `runtime.py` 之间的窄兼容边界。
- `inference/basicvsrpp_api.py` / `rife425_api.py`：模型内部兼容边界，活动 runtime 不直接访问私有符号。
- `inference/gpu_transport.py`：基础一 GPU 一进程、共享内存和 Pipe 生命周期。
- `inference/stable_gpu_transport.py`：Pipe EOF/startup fail-fast 语义。
- `inference/gpu_workers.py` / `frame_pool.py`：FrameHandle、slot 引用计数和 locality-aware 数据传递。
- `inference/gpu_worker_process.py`：固定 CUDA device context 的 worker 主循环。
- `inference/gpu_task_handlers.py`：GPU task handler registry 和 direct-slot transport。
- `inference/bvs_runtime.py`：当前 BasicVSR++ uint8 CUDA direct-slot runtime。
- `inference/optimized_basicvsrpp.py`：v5.8 BasicVSR++ 数值/执行优化基础类。
- `inference/optimized_rife425.py`：RIFE 4.25 compact H2D 和 direct shared-slot 输出。
- `inference/sr_runtime.py`：显式 Real-ESRGAN uint8 CUDA helper。
- `inference/output_runtime.py`：fail-fast 输出 pump、Lanczos4 resize、交互/批处理 progress。
- `inference/clip_source.py` / `timeline.py` / `scene_metrics.py`：CPU clip、目标时间轴和缓存 scene signature。

旧 `pipeline.py`、`v51_runtime.py`、`v54_runtime.py`、`progress_log.py`、`gpu_timing.py` 保留历史兼容；当前 v6.5 主路径不依赖这些旧安装器。

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

8-bit BVS 结果保持 CUDA `uint8` 到 handler，由 handler 只把需要 emit 的帧直接复制到预留 FrameHandle shared-memory slot；不再先构造完整 CPU clip ndarray 再做第二次 CPU memcpy。10-bit 路径保持保守兼容实现。

## RIFE 4.25 / 输出帧率

Notebook 参数：

```python
RIFE_FPS = 60
```

语义：

- `RIFE_FPS = 0`：完全关闭 RIFE，最终输出保持源视频帧率。
- `RIFE_FPS == 源帧率`：不需要生成中间帧，RIFE 自动旁路。
- `RIFE_FPS > 源帧率`：按目标 CFR 时间轴生成 arbitrary timestep 中间帧。
- 除 `0` 外，不允许 `RIFE_FPS < 源帧率`。

RIFE 权重解析优先检查：

```text
inference/weights/RIFEv4.25_0919.zip
```

仓库归档存在时直接从该 ZIP 提取 `flownet.pkl` 到本地 cache，不发起网络下载；归档缺失时保留 Practical-RIFE 官方 Google Drive fallback。

场景保护：

- `scene_difference <= 0.002`：近重复/静止，不调用 RIFE。
- `scene_difference >= 0.30`：场景切换，不跨镜头插帧。

v6.4 将原有 scene metric 拆成可缓存 `SceneSignature`。计算顺序仍为 `float32 → 64×64 INTER_AREA → luma → histogram`，阈值数学不变；连续帧只复用已经计算过的 signature，避免重复整帧 float32 转换。

RIFE 8-bit 输入保持 `uint8` 通过 H2D，再在 CUDA 上转换/归一化；生成的 8-bit 帧从 CUDA 直接复制到预留 shared-memory slot。10-bit 路径保持保守兼容实现。

## 多 GPU 稳定边界

```text
Main Scheduler
├─ cuda:0 spawn process → BVS / RIFE / SR handlers
└─ cuda:1 spawn process → BVS / RIFE / SR handlers
```

每个 GPU 子进程在整个运行期间保持固定 `torch.cuda.device(...)` context。当前主路径不存在共享 CUDA `ThreadPoolExecutor`，也不在任务级反复 `torch.cuda.set_device()`。

BVS、RIFE、SR 始终是独立 task。中间帧通过 `FrameHandle(worker_id, slot, generation)` 引用共享 slot；优先在已有数据的 GPU 上执行下一阶段，只有负载均衡需要时才进行跨 GPU CPU shared-memory copy。

控制 IPC 使用 per-worker Pipe + `multiprocessing.connection.wait()`；worker task queue 使用 `SimpleQueue`。结果 Pipe 对端关闭时显式处理 EOF，并结合 worker sentinel 进入 fail-fast 路径，不再把 EOF 当作普通 `recv()` 失败。

OutputPump 的有界队列使用 timeout + error check 循环。若 resize/writer 线程异常，主 scheduler 会立即收到错误，不会永久阻塞在满队列 `put()`。

启动前在 Linux 上检查 `/dev/shm` 可用容量，避免 shared-memory 不足运行到中途才出现不明确故障。

## 离线进度

非交互式运行时只打印 `[progress]`。首次周期进度在运行 60 秒后输出，之后每 60 秒一次；结束时可额外输出最终 `done` 状态。不会再输出 `[gpu-status]`。

## GPU timing

`--gpu-timing` 直接在 GPU worker 内围绕 BVS/RIFE/SR handler 使用 CUDA Event，并通过 typed `TaskResult` 返回统计。关闭时不创建 CUDA Event；开启时因显式同步会产生 profiling 开销。

## Kaggle 参数

```python
MODEL = "realesr-animevideov3"
SCALE = 2
RIFE_FPS = 60  # 0 = 关闭 RIFE
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
