# Real-ESRGAN 动画视频推理

当前 v6.10 流水线：

```text
FFmpeg 解码
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧（可关闭）
→ Real-ESRGAN full-frame 超分
→ Lanczos4 最终倍率调整
→ HEVC / AV1 编码
```

## 项目原则（必须遵守）

**稳定 + 规范 + 低耦合 + 尽量无性能损失。** 所有修改必须基于当前真实代码和实际问题，优先采用可回退、可验证、职责边界清晰的实现，避免无必要重构、额外拷贝、同步、资源占用和性能退化；不得为了局部性能提升破坏稳定性、正确性、兼容性或代码结构。

修改完成后直接提交到 `master`，不得擅自创建分支、PR、tag 或其他远端引用；除非用户明确要求，否则不要拆成多轮提交、不要遗留“后续再优化”的已知问题，应尽量一次性完成本轮明确范围内的修改。

开发中的版本默认标记为 `[Dev] 🔧`，只有经过用户实际测试并明确确认后，才能标记为 `[Release] ✅`。

**AI / 新对话读取规则：**任何 AI 助手或自动化代理在新的对话中首次读取本 README 时，必须先读取并记住本节，并在该对话后续所有涉及本仓库的分析、修改、提交和版本管理中持续遵守；除非用户明确修改本原则，不得自行弱化、忽略或覆盖。

## 版本

- v6.10 - 2ca1203 [Release] ✅：shared-memory 直传、CUDA Lanczos、SR 微批与有序调度。
- v6.9 - f116bff [Dev] 🔧：复用 CUDA Event，优化 H2D/D2H 与连续 slot 拷贝。
- v6.8 - f3f35f9 [Dev] 🔧：temporal/SR 分进程，compute 与 transport drain 解耦。
- v6.7 - 5a15af7 [Dev] 🔧：恢复 BVS/SR 高水位调度，增加单实例锁与 Kaggle 日志转发。
- v6.6 - fc61dcf [Dev] 🔧：BVS emit 改为 batch D2H，减少逐帧阻塞拷贝。
- v6.5 - ecce8f1 [Dev] 🔧：支持关闭 RIFE，并加入固定离线 progress heartbeat。
- v6.4 - ea40744 [Dev] 🔧：增强 fail-fast、scene cache 与 runtime API 边界。
- v6.3 - 5474432 [Dev] 🔧：去除活动路径 monkey-patch，改为显式 runtime 与事件驱动 IPC。
- v6.2 - 0c2109d [Dev] 🔧：加入 typed task、FrameHandle 引用计数与 locality-aware 调度。
- v6.1 - e66ae69 [Dev] 🔧：一张 GPU 一个常驻进程，BVS/RIFE/SR 独立 task。
- v6.0 - 63e28d8 [Dev] 🔧：加入 Practical-RIFE 4.25。
- v5.8 - f5eca71 [Dev] 🔧：固定 BVS 参数并加入 selective channels_last。
- v4.3 - 9020854 [Dev] 🔧：早期代码基线。

## 当前结构

- `inference.py`：当前 CLI；只暴露 `RIFE_FPS`，不提供遗留 `--fps` 参数，并在入口持有同输出路径单实例锁。
- `inference/scheduler.py`：CPU 视频编排，不直接执行 CUDA 模型。
- `inference/scheduler_state.py` / `scheduler_loop.py`：任务策略、compute/drain 状态、结果处理、watchdog 和事件驱动等待。
- `inference/task_protocol.py`：BVS / RIFE / SR typed task/result protocol，以及 compute-boundary 消息。
- `inference/worker_protocols.py`：GPU runtime 结构化接口；活动 RIFE 接口只暴露 CUDA batch 计算，不暴露 scheduler/transport callback。
- `inference/runtime_api.py`：当前调度路径与 legacy `runtime.py` 之间的窄兼容边界。
- `inference/basicvsrpp_api.py` / `rife425_api.py`：模型内部兼容边界，活动 runtime 不直接访问私有符号。
- `inference/gpu_transport.py`：每张 GPU 的 temporal/SR 常驻进程、共享内存、SR 双输出 slot 和 Pipe 生命周期。
- `inference/stable_gpu_transport.py`：Pipe EOF/startup fail-fast 语义。
- `inference/gpu_workers.py` / `frame_pool.py`：FrameHandle、slot 引用计数、locality-aware 数据传递和连续 slot 优先分配。
- `inference/gpu_worker_process.py`：固定 CUDA device context 的 worker 主循环、常驻 compute-boundary Event；对 worker 使用的长期 shared-memory view 尝试 CUDA host registration，退出时在 `SharedMemory.close()` 前注销。
- `inference/gpu_task_handlers.py`：GPU task handler registry、compute/drain 边界；先排 D2H copy stream，再发布 compute boundary，最后只等待真正需要消费的 D2H completion。
- `inference/frame_transport.py`：`cudaHostRegister` shared-memory fast path、注册失败 staging fallback、可复用 pinned H2D/D2H、常驻 CUDA Event、独立 copy stream、direct/batched D2H 和严格 shape/dtype/slot 校验。
- `inference/bvs_runtime.py`：当前 BasicVSR++ uint8 CUDA runtime。
- `inference/optimized_basicvsrpp.py`：v5.8 BasicVSR++ 数值/执行优化基础类。
- `inference/optimized_rife425.py`：RIFE 4.25 compact H2D 与 CUDA batch 生成；`interpolate_into()` 只作为兼容 API 保留。
- `inference/sr_runtime.py`：显式 Real-ESRGAN uint8 CUDA helper。
- `inference/output_runtime.py`：fail-fast 输出 pump、SR shared-output slot 消费、Lanczos4 resize、交互/批处理 progress。
- `inference/clip_source.py` / `timeline.py` / `scene_metrics.py`：CPU clip、目标时间轴和缓存 scene signature。
- `inference/run_lock.py`：同一输出路径的进程级单实例锁，避免重复运行争抢 GPU/输出文件。

旧 `pipeline.py`、`v51_runtime.py`、`v54_runtime.py`、`progress_log.py`、`gpu_timing.py` 保留历史兼容；当前 v6.10 主路径不依赖这些旧安装器。

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

8-bit BVS 结果保持 CUDA `uint8` 到 handler；多个 emit group 在 CUDA 侧合并为连续 batch。FrameSlotPool 优先分配连续输出 slot；对应 shared-memory 映射成功 CUDA host registration 时，packed batch 通过一次异步 D2H 直接写入连续 shared-memory slice，不再经过 pinned staging → `np.copyto`。若 host registration 不可用或 slots 已碎片化，则自动回退 v6.9 的 pinned staging / scatter 路径。10-bit 路径保持保守兼容实现。

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

RIFE 8-bit 输入保持 `uint8`；若来源 shared slot 已注册为 CUDA pinned host memory，则直接异步 H2D，不再先复制到 pinned staging。RIFE runtime 只生成 packed CUDA batch；连续输出 slot 注册成功时，handler 将 batch 直接异步 D2H 到对应 shared-memory slice。10-bit 路径因 CUDA batch 使用中间 `int32` 后再转 `uint16`，继续保守使用 host conversion fallback。

## 多 GPU 稳定边界

```text
Main Scheduler
├─ cuda:0 temporal spawn process → BVS / RIFE
│  └─ cuda:0 SR spawn process     → Real-ESRGAN
└─ cuda:1 temporal spawn process → BVS / RIFE
   └─ cuda:1 SR spawn process     → Real-ESRGAN
```

每个 GPU 子进程在整个运行期间保持固定 `torch.cuda.device(...)` context。当前主路径不存在共享 CUDA `ThreadPoolExecutor`，也不在任务级反复 `torch.cuda.set_device()`。

BVS、RIFE、SR 始终是独立 task。同一物理 GPU 的 heavy model compute 保持互斥。v6.10 handler 在 compute boundary 前先把依赖当前 producer stream 的 D2H 排入独立 copy stream；copy stream 只等待当时已经提交的 model/packing work。worker 随后同步常驻 compute-boundary Event 并发送 `TaskComputeDone`，scheduler 才允许同卡另一 role 开始 heavy compute。因此 D2H 可在 boundary 到达后立即启动并与后续另一阶段 compute 重叠，但不会无控制地并发两个重模型 kernel。

中间帧通过 `FrameHandle(worker_id, slot, generation)` 引用共享 slot；优先在已有数据的 GPU 上执行下一阶段，只有负载均衡需要时才进行跨 GPU CPU shared-memory copy。

SR 每张 GPU 优先使用两个显式 shared-output slot。OutputPump 可持有旧 slot 做 Lanczos4/编码，同时 SR worker 写另一个 slot；若 `/dev/shm` 不足以维持双 slot，则启动时自动回退到单 slot。

控制 IPC 使用 per-role Pipe + `multiprocessing.connection.wait()`；worker task queue 使用 `SimpleQueue`。结果 Pipe 对端关闭时显式处理 EOF，并结合 worker sentinel 进入 fail-fast 路径，不再把 EOF 当作普通 `recv()` 失败。

OutputPump 的有界队列使用 timeout + error check 循环。若 resize/writer 线程异常，主 scheduler 会立即收到错误，不会永久阻塞在满队列 `put()`。

启动前在 Linux 上检查 `/dev/shm` 可用容量，避免 shared-memory 不足运行到中途才出现不明确故障。

同一 `--output` 路径在 POSIX 上通过 `flock(LOCK_EX | LOCK_NB)` 只允许一个 inference 主进程持有；第二个重复运行会在加载 GPU 模型前直接报错，不会与第一个任务争抢 GPU 或同时写同一输出文件。

## 传输边界

当前跨进程 frame pool 仍基于 POSIX shared memory，但每个 CUDA worker 会对自己长期使用的 input / frame-output / SR-output 映射尝试 `cudaHostRegister()`。注册成功后，这些映射由 CUDA 视为 page-locked host memory：8-bit H2D 可以直接从 shared slot 异步 DMA，8-bit D2H 也可以直接写 shared output，因此活动主路径不再需要 shared-memory ↔ pinned-staging 的第二次 CPU memcpy。

`cudaHostRegister()` 是启动期、长期持有的优化，而不是逐帧注册。若当前 CUDA/OS/驱动不支持注册，或注册失败，worker 只打印一次对应映射的 fallback 信息并继续使用 v6.9 可复用 pinned staging，不改变功能正确性。由于 page-locked memory 是有限系统资源，v6.10 只注册推理进程已经固定分配的 frame pools，不创建额外同尺寸 pinned 副本。

连续 FrameHandle slot 会直接形成一个 NumPy/Torch shared-memory slice，一次 packed CUDA batch 对应一次异步 D2H。只有 frame pool 碎片化导致无法取得连续 run 时，才回退 staging + scatter；slot allocator 会优先寻找足够长的连续空闲区来减少这种情况。

## 离线进度

非交互式运行时只打印 `[progress]`：启动时立即打印一条，之后每 60 秒一次；结束时可额外输出最终 `done` 状态。不会再输出 `[gpu-status]`。

Kaggle Notebook 使用 `subprocess.Popen(..., stdout=PIPE, stderr=STDOUT)` 捕获 inference 子进程的单一合并输出流，再由 notebook kernel 顺序转发到 stdout，避免子进程输出同时进入多个持久化日志通道。

## GPU timing

`--gpu-timing` 在 GPU worker 内用常驻 CUDA Event 记录 BVS/RIFE/SR compute boundary，并通过 typed `TaskResult` 返回统计。关闭时不记录 elapsed GPU timing；compute-boundary Event 仍用于保证同卡 heavy compute 互斥和安全 handoff，但不再为每个 task 动态创建 Event。D2H completion 使用独立常驻 Event，只在结果真正要被 host/output consumer 使用前同步一次。

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