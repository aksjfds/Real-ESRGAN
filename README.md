# APISR 动画视频推理

## 当前版本流水线

当前 **APISR v8.14 [Dev] 🔧** 以 `master` v8.9 为完整基线，仅替换 SR 模型后端，并保留 master 的 BVS / RIFE / 调度 / shared-memory / CUDA 传输 / NPP / 编码 / 音频架构。

```text
视频增强（可关闭）：
FFmpeg 解码
→ FFmpeg 轻度 deband（Notebook 默认 0.006，可设 0 关闭）
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧（可关闭）
→ APISR 4× GRL GAN full-frame 超分（4×重建尾部 OOM 时流式条带 fallback）
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

APISR 分支使用官方 **v0.1.0 4× GRL GAN** 权重，并固定到该 Release 对应源码 commit `fabe8332413bc7f4024e6db39141c68692e88ea5`。GRL 按官方限制继续使用 FP32；SR worker 的 FP32 matmul 使用 IEEE 精度，不强制启用 TF32，cuDNN convolution 仍允许 TF32 以保留卷积性能。

APISR 官方 **v0.1.0 4× GRL GAN** 权重直接存放在仓库 `inference/weights/4x_APISR_GRL_GAN_generator.pth`；默认推理读取仓库内置文件并在 worker 启动前校验固定大小与 SHA256，不再运行时下载模型权重；`--model-path` 仍可显式覆盖。APISR 源码继续按固定 commit 首次缓存到 `~/.cache/realesrgan/apisr/source/`，只提取 GRL 推理需要的文件并按 Git blob manifest 校验。

APISR v0.1.0 上游 GRL 使用通用顶层包名 `architecture.*`，且其源码会修改 `sys.path`。本分支在导入前保存并在导入后完整恢复 `sys.path` 与 `architecture.*` 的 `sys.modules` 状态，避免第三方源码 import side effect 泄漏到主运行时。

SR worker 在发送 `WorkerReady` 前执行一次**完整 batch=1 SR startup probe**。APISR 始终以完整原始帧运行 GRL Transformer 主干；若官方 `nearest+conv` 的第二级 2× 上采样重建尾部发生真实 CUDA OOM，仅将已经完成主干后的 2× 64-channel feature 沿较长边流式分条执行 `nearest → conv_up2 → conv_hr → conv_last`，不重新计算 Transformer、不降低 1080p 输入分辨率。尾部连续 3 个 3×3 Conv 的 4×输出感受半径为 3 px，因此每条在 2× feature 上保留精确覆盖所需的 2 px halo，只写回 core；fallback 从 2 条开始，仍 OOM 才递增并在该 worker 保持已通过的最小条数。batch>1 OOM 继续由既有 micro-batch→batch=1 路径处理。NPP、D2H/shared-memory 和最终倍率调整不变。

8-bit 始终使用 CUDA micro-batch 路径。10-bit 会先执行真实 CUDA `uint16` capability probe；只有当前 PyTorch/CUDA 组合确实支持所需 H2D / float conversion / uint16 quantize / D2H 时才启用 uint16 CUDA fast path，否则自动使用稳定的 FP32 GPU 推理 + CPU uint16 输出 fallback。支持 fast path 时，最终倍率调整优先调用 NPP `nppiResize_16u_C3R_Ctx`；NPP 不可用或运行失败时回退 CPU Lanczos4。SR micro-batch OOM 时只将对应 worker 锁定到 batch=1。

`AUDIO_ENHANCE=False` 时不执行 DSP，只保留原音轨并优先 stream copy。

## 项目原则（必须遵守）

**稳定 + 规范 + 低耦合 + 尽量无性能损失。** 所有修改必须基于当前真实代码和实际问题，优先采用可回退、可验证、职责边界清晰的实现，避免无必要重构、额外拷贝、同步、资源占用和性能退化；不得为了局部性能提升破坏稳定性、正确性、兼容性或代码结构。

修改完成后直接提交到 `APISR`，不得擅自创建分支、PR、tag 或其他远端引用；除非用户明确要求，否则不要拆成多轮提交、不要遗留“后续再优化”的已知问题，应尽量一次性完成本轮明确范围内的修改。

开发中的版本默认标记为 `[Dev] 🔧`，只有经过用户实际测试并明确确认后，才能标记为 `[Release] ✅`。

**AI / 新对话读取规则：**任何 AI 助手或自动化代理在新的对话中首次读取本 README 时，必须先读取并记住本节，并在该对话后续所有涉及本仓库的分析、修改、提交和版本管理中持续遵守；除非用户明确修改本原则，不得自行弱化、忽略或覆盖。

### APISR 静态审计闭环规则（必须遵守）

APISR 的基准是当前 `master`。静态审计只把以下两类内容视为 APISR 分支待修问题：

1. `APISR` 相对 `master` 的实际代码差异所引入的问题；
2. APISR backend 与 master 既有接口交互后新产生的正确性、稳定性、兼容性、耦合或性能问题。

与 `master` 字节一致的既有实现、兼容代码、文件或资源，不得仅因为名称仍含 Real-ESRGAN、当前 APISR active path 未使用，或存在可进一步重构空间，就重复列为 APISR 缺陷；只有存在可复现的 APISR 交互故障或用户明确要求同步清理 master 基线时才处理。

每次宣告“静态审计完成”前，必须一次性检查并记录以下边界：**模型拓扑/权重一致性、输入输出与精度、显存与 OOM、H2D/D2H/shared-memory/NPP、并发与生命周期、import/global state、下载/cache/完整性、依赖与许可证、失败/fallback、CLI/Notebook/文档、回归测试，以及 APISR↔master 接口。**

同一 commit 在上述清单已经完成且代码未变化时，不得通过换审查角度继续把“可选重构/性能猜测”包装成新缺陷。后续新增问题必须至少满足其一：**代码发生新变化、出现新的真实运行证据、上游/运行环境约束发生变化，或能指出此前闭环清单中遗漏的具体失败路径。** 无 profiling/画质 A/B 数据的性能想法只能列为实验项，不得列为必须修改的问题。

## 版本历史

- **APISR v8.14 [Dev] 🔧**：修复 1920×1080 FP32 GRL 在约 15 GB GPU 上于 `conv_up2` 申请约 7.9 GiB 导致的 startup OOM。完整 Transformer 主干仍只计算一次；仅对已经生成的 2× reconstruction feature 自适应流式执行 4× `nearest+conv` 尾部，2 px feature halo 精确覆盖尾部 3 个 3×3 Conv 的边界依赖。启动 probe 自动选出能通过的最少条数并复用；不降采样输入、不改 master 调度/transport/NPP，也不把整网按 tile 重算。
- **APISR v8.13 [Dev] 🔧**：将官方 v0.1.0 `4x_APISR_GRL_GAN_generator.pth`（6,479,400 bytes，SHA256 `56fff250139563dea59c4ca81af19cc098d94dc3abaad23640f14cec488e5da1`）直接纳入 `inference/weights/`；默认模型路径读取仓库内置权重并在启动前强校验，不再运行时下载 APISR 模型权重；保留 `--model-path` 覆盖与 APISR 源码固定版本缓存。
- **APISR v8.12 [Dev] 🔧**：完成文件/目录结构清理：Kaggle Notebook 由 `realesrgan.ipynb` 统一命名为 `apisr.ipynb`；BasicVSR++ 分片 checkpoint 模块改为职责明确的 `basicvsrpp_checkpoint.py`；根入口直接使用 `scheduler.py`；删除已退出 active path 的旧 `pipeline` / `balanced_pipeline` / `progress_log`、v5.1/v5.4 runtime 兼容层、`v52_scheduler.py`、`stable_gpu_transport.py` 以及 master 专用 banding debug Notebook。不重排当前 active runtime 子目录，不改变推理数学路径或性能关键路径。
- **APISR v8.11 [Dev] 🔧**：完成静态审计闭环：第三方 GRL import 同时恢复 `sys.path` 与 `architecture.*` 的 `sys.modules` 状态；APISR source/weight 下载加入硬性字节上限并补回归测试；README 固化 APISR↔master 审计边界，避免把 master 原样继承内容反复误报为 APISR 缺陷；补充上游 GPL-3.0 / academic-use disclaimer 提示。
- **APISR v8.10 [Dev] 🔧**：在 v8.9 APISR backend 基础上完成运行边界收口：完整 SR startup probe 取代 model-only warmup；10-bit CUDA `uint16` 改为 capability-gated fast path 并提供稳定 fallback；`frame_transport` 泛化为 uint8/uint16 共用路径，删除 APISR handler 重复 transport；APISR FP32 matmul 恢复 IEEE 语义；完整恢复第三方源码 import 前后的 `sys.path`；下载增加超时/重试并只安全提取 pinned GRL runtime 文件；Notebook 默认执行 CPU 回归测试。
- **APISR v8.9 [Dev] 🔧**：以 `master` v8.9 为完整基线，将 Real-ESRGAN SR 后端替换为 APISR 4× GRL GAN；补齐真正 BCHW micro-batch、GRL 动态 resolution table/mask 单槽缓存、显式 SR backend 边界、v0.1.0 源码/权重一致性与完整性校验、并发 cache lock 和 uint16 CUDA/NPP 初始路径；GRL 使用 FP32。
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

- `apisr.ipynb`：Kaggle 入口；3 个代码单元（环境 / 配置 / 执行），默认 `MODEL="APISR_GRL"`，环境单元执行 APISR CPU 回归测试。
- `inference.py`：视频增强 CLI 与总入口。
- `inference/apisr_backend.py`：APISR v0.1.0 GRL 注册、最小源码缓存、仓库内置权重完整性校验、FP32 模型加载与 resolution cache。
- `inference/scheduler.py`：CPU 总编排、编码 fail-fast probe、最终音频边界与成品验证。
- `inference/clip_source.py`：CPU scene-aware BVS clip 组装与单槽预取。
- `inference/scheduler_state.py` / `scheduler_loop.py`：调度状态、任务策略、结果处理与 watchdog。
- `inference/task_protocol.py` / `worker_protocols.py`：BVS/RIFE/SR typed task/result 与 worker API。
- `inference/gpu_transport.py` / `gpu_workers.py` / `gpu_worker_process.py`：常驻 GPU workers、shared memory、FrameHandle、SR startup probe 与进程生命周期。
- `inference/frame_transport.py`：`cudaHostRegister`、pinned fallback、uint8/uint16 异步 H2D/D2H 与 copy stream。
- `inference/basicvsrpp_checkpoint.py`：BasicVSR++ 仓库分片权重的运行时合并、完整性验证与临时兼容路径管理。
- `inference/bvs_runtime.py` / `optimized_basicvsrpp.py`：BasicVSR++ runtime。
- `inference/rife425.py` / `optimized_rife425.py`：Practical-RIFE 4.25 runtime。
- `inference/sr_runtime.py`：通用 uint8/uint16 SR CUDA 路径与 CUDA uint16 capability probe；APISR 使用 FP32 contiguous 模型输入。
- `inference/npp_resize.py`：NPP uint8/uint16 C3 Lanczos CUDA resize。
- `inference/output_runtime.py`：输出 pump、编码与进度日志。
- `audio/runtime.py`：FFmpeg dialogue-focused DSP、增强/原始双音轨 mux、音频旁路。
- `audio/process.py`：`VIDEO_ENHANCE=False` 时的视频 stream-copy / 音频处理入口。
- `tests/test_apisr_backend.py` / `tests/test_sr_runtime.py`：APISR backend、源码隔离/完整性、dtype capability 的轻量 CPU 回归测试。

## 模型与资源

### 仓库内置

- APISR v0.1.0 `4x_APISR_GRL_GAN_generator.pth`；默认 SR 权重，启动前校验固定大小与 SHA256。
- BasicVSR++ NTIRE Track 1 权重分片；运行时合并并验证。
- Practical-RIFE 4.25 模型压缩包/运行时资源按 master 逻辑处理。

### APISR 源码运行时资源

| 文件 | 用途 | 来源 |
|---|---|---|
| APISR source commit `fabe8332413bc7f4024e6db39141c68692e88ea5` | 与 v0.1.0 权重对应的 GRL 架构源码 | `Kiteretsu77/APISR` |

APISR 源码默认缓存根目录：`~/.cache/realesrgan/apisr/`；模型权重不进入运行时 cache。

**上游使用与许可提示：**APISR 上游将项目以 GPL-3.0 发布，并在 README 中声明项目仅供学术用途（academic use only）且其 disclaimer 适用。本分支运行时下载的 APISR 源码以及仓库内置的 APISR 官方权重仍受上游许可、disclaimer 与权重相关条款约束；详见 `THIRD_PARTY_NOTICES.md`。

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
