# Real-ESRGAN 动画视频推理

当前 v6.8 流水线：

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
- v6.5：`RIFE_FPS=0` 关闭插帧并保持源帧率；RIFE 优先使用仓库内模型归档；离线 `[progress]` 使用固定 heartbeat。
- v6.6：BVS emit group 使用一次 batch D2H，再复制到 FrameHandle shared-memory slots，避免逐帧 blocking D2H。
- v6.7：RIFE 关闭时恢复 BVS/SR 高水位调度，避免 SR 过早抢占第二条 BVS GPU lane；同输出路径增加单实例进程锁；Kaggle Notebook 将子进程 stdout/stderr 合并后由 notebook kernel 单路转发，避免离线日志重复。
- v6.8：每张 GPU 拆为常驻 temporal(BVS/RIFE) 与 SR 两个 `spawn` 进程；任务生命周期拆为 CUDA compute / transport drain 两阶段，保持同卡重模型 compute 互斥但允许另一阶段在 D2H/CPU drain 期间接管 compute lane；BVS/RIFE/SR 使用可复用 pinned H2D/D2H staging 与独立 copy stream，BVS/RIFE 合并批量 D2H；SR 使用显式双 shared-output slot 解耦 resize/encoder 反压，`/dev/shm` 紧张时自动回退单 slot。

## 当前结构

- `inference.py`：当前 CLI；只暴露 `RIFE_FPS`，不提供遗留 `--fps` 参数，并在入口持有同输出路径单实例锁。
- `inference/scheduler.py`：CPU 视频编排，不直接执行 CUDA 模型。
- `inference/scheduler_state.py` / `scheduler_loop.py`：任务策略、compute/drain 状态、结果处理、watchdog 和事件驱动等待。
- `inference/task_protocol.py`：BVS / RIFE / SR typed task/result protocol，以及 compute-boundary 消息。
- `inference/worker_protocols.py`：GPU runtime 结构化接口。
- `inference/runtime_api.py`：当前调度路径与 legacy `runtime.py` 之间的窄兼容边界。
- `inference/basicvsrpp_api.py` / `rife425_api.py`：模型内部兼容边界，活动 runtime 不直接访问私有符号。
- `inference/gpu_transport.py`：每张 GPU 的 temporal/SR 常驻进程、共享内存、SR 双输出 slot 和 Pipe 生命周期。
- `inference/stable_gpu_transport.py`：Pipe EOF/startup fail-fast 语义。
- `inference/gpu_workers.py` / `frame_pool.py`：FrameHandle、slot 引用计数和 locality-aware 数据传递。
- `inference/gpu_worker_process.py`：固定 CUDA device context 的 worker 主循环和 compute-boundary 通知。
- `inference/gpu_task_handlers.py`：GPU task handler registry、compute/drain 边界和 transport 调用。
- `inference/frame_transport.py`：可复用 pinned H2D/D2H staging、独立 CUDA copy stream、batch D2H 与严格 shape/dtype/slot 校验。
- `inference/bvs_runtime.py`：当前 BasicVSR++ uint8 CUDA runtime。
- `inference/optimized_basicvsrpp.py`：v5.8 BasicVSR++ 数值/执行优化基础类。
- `inference/optimized_rife425.py`：RIFE 4.25 compact pinned H2D、batched D2H 和 shared-slot 输出。
- `inference/sr_runtime.py`：显式 Real-ESRGAN uint8 CUDA helper。
- `inference/output_runtime.py`：fail-fast 输出 pump、SR shared-output slot 消费、Lanczos4 resize、交互/批处理 progress。
- `inference/clip_source.py` / `timeline.py` / `scene_metrics.py`：CPU clip、目标时间轴和缓存 scene signature。
- `inference/run_lock.py`：同一输出路径的进程级单实例锁，避免重复运行争抢 GPU/输出文件。

旧 `pipeline.py`、`v51_runtime.py`、`v54_runtime.py`、`progress_log.py`、`gpu_timing.py` 保留历史兼容；当前 v6.8 主路径不依赖这些旧安装器。

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

8-bit BVS 结果保持 CUDA `uint8` 到 handler；多个 emit group 在 CUDA 侧合并为连续 batch 后执行一次 pinned D2H，再由 CPU `np.copyto` 写入预留 FrameHandle shared-memory slots。10-bit 路径保持保守兼容实现。

当 RIFE 关闭时，scheduler 使用一个正常 BVS emit batch 作为 SR backlog 高水位：`clip_length - 2 * overlap`，当前固定参数下为 `9`。在 backlog 未达到高水位时优先保持两张 GPU 继续生产 BVS；达到高水位或 BVS EOF 后再让空闲 GPU 消化 SR，从而避免 SR 一出现就长期抢占第二条 BVS lane。

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

RIFE 8-bit 输入保持 `uint8`，通过可复用 pinned H2D staging 与独立 copy stream 进入 CUDA；生成帧在 CUDA 侧打包后执行一次 batched pinned D2H，再写入预留 shared-memory slots。10-bit 路径保持保守兼容实现。

## 多 GPU 稳定边界

```text
Main Scheduler
├─ cuda:0 temporal spawn process → BVS / RIFE
│  └─ cuda:0 SR spawn process     → Real-ESRGAN
└─ cuda:1 temporal spawn process → BVS / RIFE
   └─ cuda:1 SR spawn process     → Real-ESRGAN
```

每个 GPU 子进程在整个运行期间保持固定 `torch.cuda.device(...)` context。当前主路径不存在共享 CUDA `ThreadPoolExecutor`，也不在任务级反复 `torch.cuda.set_device()`。

BVS、RIFE、SR 始终是独立 task。同一物理 GPU 的 heavy model compute 保持互斥；worker 在 CUDA Event 同步确认 compute boundary 后发送 `TaskComputeDone`，scheduler 才允许同卡另一 role 开始 heavy compute，因此 transport drain 可以与另一阶段 compute 重叠，但不会无控制地并发两个重模型 kernel。

中间帧通过 `FrameHandle(worker_id, slot, generation)` 引用共享 slot；优先在已有数据的 GPU 上执行下一阶段，只有负载均衡需要时才进行跨 GPU CPU shared-memory copy。

SR 每张 GPU 优先使用两个显式 shared-output slot。OutputPump 可持有旧 slot 做 Lanczos4/编码，同时 SR worker 写另一个 slot；若 `/dev/shm` 不足以维持双 slot，则启动时自动回退到单 slot。

控制 IPC 使用 per-role Pipe + `multiprocessing.connection.wait()`；worker task queue 使用 `SimpleQueue`。结果 Pipe 对端关闭时显式处理 EOF，并结合 worker sentinel 进入 fail-fast 路径，不再把 EOF 当作普通 `recv()` 失败。

OutputPump 的有界队列使用 timeout + error check 循环。若 resize/writer 线程异常，主 scheduler 会立即收到错误，不会永久阻塞在满队列 `put()`。

启动前在 Linux 上检查 `/dev/shm` 可用容量，避免 shared-memory 不足运行到中途才出现不明确故障。

同一 `--output` 路径在 POSIX 上通过 `flock(LOCK_EX | LOCK_NB)` 只允许一个 inference 主进程持有；第二个重复运行会在加载 GPU 模型前直接报错，不会与第一个任务争抢 GPU 或同时写同一输出文件。

## 离线进度

非交互式运行时只打印 `[progress]`：启动时立即打印一条，之后每 60 秒一次；结束时可额外输出最终 `done` 状态。不会再输出 `[gpu-status]`。

Kaggle Notebook 使用 `subprocess.Popen(..., stdout=PIPE, stderr=STDOUT)` 捕获 inference 子进程的单一合并输出流，再由 notebook kernel 顺序转发到 stdout，避免子进程输出同时进入多个持久化日志通道。

## GPU timing

`--gpu-timing` 在 GPU worker 内用 CUDA Event 记录 BVS/RIFE/SR compute boundary，并通过 typed `TaskResult` 返回统计。关闭时不记录 elapsed GPU timing；compute-boundary Event 仍用于保证同卡 heavy compute 互斥和安全 handoff。

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