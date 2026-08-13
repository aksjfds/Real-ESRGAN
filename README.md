# APISR 动画视频推理

## 当前版本流水线

当前 **APISR v8.9 [Dev] 🔧** 以 `master` v8.9 为完整基线，仅替换 SR 模型后端，并保留 master 的 BVS / RIFE / 调度 / shared-memory / CUDA 传输 / NPP / 编码 / 音频架构。

```text
视频增强（可关闭）：
FFmpeg 解码
→ FFmpeg 轻度 deband（Notebook 默认 0.006，可设 0 关闭）
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧（可关闭）
→ APISR 4× GRL GAN full-frame 超分
→ NPP Lanczos CUDA 最终倍率调整（CPU Lanczos4 fallback）
→ NVENC（Notebook 可选 av1_nvenc / hevc_nvenc；默认 av1_nvenc）
→ ffprobe 成品验证（codec / profile / 位深 / 分辨率 / FPS / 音轨）

音频增强（可关闭，默认开启 FFmpeg 版本）：
原始音频
├→ FFmpeg DSP
│  → 55 Hz high-pass
│  → 500 Hz -0.8 dB / 3 kHz +0.8 dB / 11 kHz high-shelf +0.8 dB
│  → 1.8:1 gentle compression
│  → de-esser
│  → EBU R128 loudnorm（-16 LUFS / -1.5 dBTP）
│  → 48 kHz AAC
│  → Audio 1: Enhanced (FFmpeg)，默认音轨
└→ stream copy
   → Audio 2: Original，原音轨
```

`DEBAND_STRENGTH=0` 时不执行 deband；Notebook 默认 `0.006`，只在 FFmpeg 解码后、BasicVSR++ 前处理。CLI 默认仍为 `0`，保持旧调用兼容。

v8.9 的 `ClipSource` 使用单槽有界异步预取：后台线程提前执行下一次 FFmpeg 读帧、scene signature 与 BVS clip 组装，`Queue(maxsize=1)` 只缓存 1 个待调度 clip；不改变帧顺序、scene-cut 判定、BVS/RIFE/SR 数学路径或编码参数。

APISR 分支使用官方 **v0.1.0 4× GRL GAN** 权重，并固定到该 Release 对应源码 commit `fabe8332413bc7f4024e6db39141c68692e88ea5`。官方 APISR 对 GRL 的 FP16 推理仍注明存在问题，因此 SR 模型固定使用 FP32。

APISR 首次运行会把源码和权重缓存到 `~/.cache/realesrgan/apisr/`：源码按 Git blob manifest 校验，官方权重按大小和 SHA256 校验；POSIX 下使用独立 cache file lock，避免多进程首次启动时竞争写入。可用 `APISR_CACHE_DIR` 改缓存根目录；`APISR_SOURCE_DIR` 可显式指定本地 APISR 源码覆盖。

SR worker 在发送 `WorkerReady` 前，会对实际输入分辨率执行一次 full-frame、batch=1 GRL warmup：用于提前发现显存不足并预热 GRL 动态 resolution table/mask cache。不会自动缩小输入或静默改为 tile，以避免改变数学路径。

8-bit 与 10-bit SR 都使用 CUDA micro-batch 路径。10-bit 最终倍率调整优先调用 NPP `nppiResize_16u_C3R_Ctx`；NPP 不可用或运行失败时仍回退 CPU Lanczos4。SR micro-batch OOM 时只将对应 worker 锁定到 batch=1；batch=1 full-frame OOM 会在 worker 启动 warmup 阶段直接失败。

`AUDIO_ENHANCE=False` 时不执行 DSP，只保留原音轨并优先 stream copy。

## 项目原则（必须遵守）

**稳定 + 规范 + 低耦合 + 尽量无性能损失。** 所有修改必须基于当前真实代码和实际问题，优先采用可回退、可验证、职责边界清晰的实现，避免无必要重构、额外拷贝、同步、资源占用和性能退化；不得为了局部性能提升破坏稳定性、正确性、兼容性或代码结构。

修改完成后直接提交到 `APISR`，不得擅自创建分支、PR、tag 或其他远端引用；除非用户明确要求，否则不要拆成多轮提交、不要遗留“后续再优化”的已知问题，应尽量一次性完成本轮明确范围内的修改。

开发中的版本默认标记为 `[Dev] 🔧`，只有经过用户实际测试并明确确认后，才能标记为 `[Release] ✅`。

**AI / 新对话读取规则：**任何 AI 助手或自动化代理在新的对话中首次读取本 README 时，必须先读取并记住本节，并在该对话后续所有涉及本仓库的分析、修改、提交和版本管理中持续遵守；除非用户明确修改本原则，不得自行弱化、忽略或覆盖。

## 版本历史

- **APISR v8.9 [Dev] 🔧**：以 `master` v8.9 为完整基线，将 Real-ESRGAN SR 后端替换为 APISR 4× GRL GAN；后续补齐真正 BCHW micro-batch、GRL 动态 resolution table/mask 单槽缓存、显式 SR backend 边界、v0.1.0 源码/权重一致性与完整性校验、并发 cache lock、实际尺寸 batch=1 warmup，以及 uint16 CUDA/NPP SR 路径；GRL 继续使用 FP32。
- **v8.9 [Dev] 🔧**：`ClipSource` 增加单槽有界异步预取；后台线程在当前 GPU 任务执行期间提前完成下一 BVS clip 的 FFmpeg 读帧、scene signature 与组帧，`Queue(maxsize=1)` 提供背压并限制额外内存，不改变视频处理数学路径与输出顺序。
- **v8.8 [Dev] 🔧**：修复编码配置/日志一致性；AV1 VBR/CBR/CONSTQP 参数语义、源位深 runtime probe、最终 ffprobe 验证、NVENC AQ 配置等统一。
- **v8.7 [Dev] 🔧**：Notebook 恢复 `hevc_nvenc`，与 `av1_nvenc` 二选一；继续默认 AV1。
- **v8.6 [Dev] 🔧**：在 FFmpeg 解码后、BasicVSR++ 前加入可选轻度 deband；Notebook 默认 `0.006`，CLI 默认 `0`。
- **v8.5 [Release] ✅**：Notebook 使用 AV1 NVENC-only 高质量配置；P7/HQ、VBR CQ18、fullres multipass、AQ、B-ref、GOP 与 8/10-bit 输出控制。
- **v8.2 - 09274ad [Dev] 🔧**：FFmpeg 增强音频作为 Audio 1 默认音轨，原音频作为 Audio 2。
- **v8.1 - b212afa [Dev] 🔧**：音频回退到 v8.0 FFmpeg DSP。
- **v8.0 - d0908ea [Dev] 🔧**：视频/音频增强可独立开关，加入 FFmpeg 音频 DSP 模块。
- **v7.0 - 5ec12df [Release] ✅**：单 GPU 版本，目标 RTX 4090，其余延续 v6.10。
- **v6.10 - 2ca1203 [Release] ✅**：shared-memory 直传、CUDA Lanczos、SR micro-batch 与有序调度。
- **v6.9 - f116bff [Dev] 🔧**：复用 CUDA Event，优化 H2D/D2H 与 contiguous slot copy。
- **v6.8 - f3f35f9 [Dev] 🔧**：temporal/SR 独立进程，compute 与 transport drain 解耦。
- **v6.7 - 5a15af7 [Dev] 🔧**：恢复 BVS/SR high-water scheduling、单实例锁和 Kaggle 日志转发。
- **v6.6 - fc61dcf [Dev] 🔧**：BVS emit batch D2H。
- **v6.5 - ecce8f1 [Dev] 🔧**：RIFE 可关闭，修复离线进度 heartbeat。
- **v6.4 - ea40744 [Dev] 🔧**：加强 fail-fast、scene cache、runtime API 边界。
- **v6.3 - 5474432 [Dev] 🔧**：移除活动路径 monkey patch，显式 runtime + event-driven IPC。
- **v6.2 - 0c2109d [Dev] 🔧**：typed task、FrameHandle refcount、locality-aware scheduling。
- **v6.1 - e66ae69 [Dev] 🔧**：每 GPU 常驻 worker，BVS/RIFE/SR 独立任务类型。
- **v6.0 - 63e28d8 [Dev] 🔧**：Practical-RIFE 4.25。
- **v5.8 - f5eca71 [Dev] 🔧**：固定 BVS 参数 + selective channels_last。
- **v4.3 - 9020854 [Dev] 🔧**：早期代码基线。

## 当前结构

- `realesrgan.ipynb`：Kaggle 入口；3 个代码单元（环境 / 配置 / 执行），默认 `MODEL="APISR_GRL"`。
- `inference.py`：视频增强 CLI 与总入口。
- `inference/apisr_backend.py`：APISR v0.1.0 GRL 注册、源码/权重缓存、完整性校验、FP32 模型加载、resolution cache 与 warmup。
- `inference/scheduler.py`：CPU 总编排、编码 fail-fast probe、最终音频边界与成品验证。
- `inference/clip_source.py`：CPU scene-aware BVS clip 组装与单槽预取。
- `inference/scheduler_state.py` / `scheduler_loop.py`：调度状态、任务策略、结果处理与 watchdog。
- `inference/task_protocol.py` / `worker_protocols.py`：BVS/RIFE/SR typed task/result 与 worker API。
- `inference/gpu_transport.py` / `gpu_workers.py` / `gpu_worker_process.py`：常驻 GPU workers、shared memory、FrameHandle 与进程生命周期。
- `inference/frame_transport.py`：`cudaHostRegister`、pinned fallback、异步 H2D/D2H 与 copy stream。
- `inference/bvs_runtime.py` / `optimized_basicvsrpp.py`：BasicVSR++ runtime。
- `inference/rife425.py` / `optimized_rife425.py`：Practical-RIFE 4.25 runtime。
- `inference/sr_runtime.py`：通用 uint8/uint16 SR CUDA 输入/输出路径；APISR 使用 FP32 contiguous 模型输入。
- `inference/npp_resize.py`：NPP uint8/uint16 C3 Lanczos CUDA resize。
- `inference/output_runtime.py`：输出 pump、编码与进度日志。
- `audio/runtime.py`：FFmpeg dialogue-focused DSP、增强/原始双音轨 mux、音频旁路。
- `audio/process.py`：`VIDEO_ENHANCE=False` 时的视频 stream-copy / 音频处理入口。
- `tests/test_apisr_backend.py`：APISR backend 的轻量 CPU 回归测试。

## 模型与资源

### 仓库内置

- BasicVSR++ NTIRE Track 1 权重分片；运行时合并并验证。
- Practical-RIFE 4.25 模型压缩包/运行时资源按 master 逻辑处理。

### APISR 运行时资源

| 文件 | 用途 | 来源 |
|---|---|---|
| `4x_APISR_GRL_GAN_generator.pth` | APISR 4× GRL GAN SR 权重 | APISR 官方 Release v0.1.0 |
| APISR source commit `fabe8332413bc7f4024e6db39141c68692e88ea5` | 与 v0.1.0 权重对应的 GRL 架构源码 | `Kiteretsu77/APISR` |

默认缓存根目录：`~/.cache/realesrgan/apisr/`。

## 编码

Notebook 默认：

- `VIDEO_CODEC="av1_nvenc"`；可改为 `hevc_nvenc`
- 两种 NVENC 共用：P7 / CQ18 / `ENCODE_GPU=0`
- AV1：Main / HQ / VBR CQ18 / fullres multipass / Spatial AQ / B-ref / GOP
- AV1 Temporal AQ 默认关闭，不与 Spatial AQ 同时启用
- `AV1_BIT_DEPTH=8`：8-bit 源保持 8-bit；10-bit 源保持 10-bit
- `AV1_BIT_DEPTH=10`：8-bit 源提升为 10-bit；10-bit 源保持 10-bit
- HEVC 位深跟随推理帧：8-bit → 8-bit，10-bit → 10-bit
- AV1/HEVC/H.264 在模型加载前执行真实 encoder runtime probe；最终 mux 后执行 ffprobe 验证

底层仍保留 H.264、软件 HEVC 和 CPU AV1 编码后端用于 CLI 兼容，不在 Notebook 暴露。
