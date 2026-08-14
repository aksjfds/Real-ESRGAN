# Real-ESRGAN 动画视频推理

## 当前版本流水线

当前 v8.10 [Dev] 流水线：

```text
视频增强（可关闭）：
FFmpeg 解码
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧（可关闭）
→ Real-ESRGAN full-frame 超分
→ NPP Lanczos CUDA 最终倍率调整（CPU Lanczos4 fallback）
→ NVENC（Notebook 可选 av1_nvenc / hevc_nvenc；默认 av1_nvenc）
→ 原始音轨 stream copy mux（若源视频有音频；不解码、不滤镜、不重编码）
→ ffprobe 成品验证（codec / profile / 位深 / 分辨率 / FPS / 音轨）
```

v8.10 已移除 deband 与音频处理：CLI / Notebook 不再提供 `DEBAND_STRENGTH`、`--deband-strength`、`AUDIO_ENHANCE`、`--audio-enhance`、`--audio-codec` 或 `--audio-bitrate`；不再执行音频 DSP、AAC 重编码或失败后的 AAC fallback。源视频若有音频，只复制第一条原始音轨到最终文件；若 stream copy 与 MP4 容器不兼容则直接报错，不自动转码。

v8.9 的 `ClipSource` 使用单槽有界异步预取：后台线程提前执行下一次 FFmpeg 读帧、scene signature 与 BVS clip 组装，`Queue(maxsize=1)` 只缓存 1 个待调度 clip；不改变帧顺序、scene-cut 判定、BVS/RIFE/SR 数学路径或编码参数。

v8.10 Notebook 暴露 `av1_nvenc` / `hevc_nvenc` 两种 GPU 编码器，默认仍为 `av1_nvenc`。两者共用 `PRESET`、`CQ`、`ENCODE_GPU`。AV1 的 profile 不再作为可变参数暴露；旧 CLI 的 `--av1-profile main` 仅作为兼容占位，成品会验证为 Main；VBR 使用 `CQ`，CBR 要求非零 bitrate 并设置匹配的 maxrate/bufsize，CONSTQP 使用 `QP`。`AV1_BIT_DEPTH` 表示 AV1 最低输出位深：设为 `8` 时 8-bit 片源保持 8-bit，设为 `10` 时 8-bit 片源编码为 10-bit；10-bit 片源始终保持 10-bit。启动日志中的 `encode=` 表示编码器输入/目标像素格式，任务完成后以最终文件的 ffprobe 结果作为实际输出事实。

AV1 与 HEVC/H.264 backend 都会在模型下载/加载前，使用与源视频相同位深的测试帧执行一次真实编码 probe；GPU、驱动、10-bit、lookahead、AQ 等参数组合不兼容时会提前失败。NVENC 默认只启用 Spatial AQ，Temporal AQ 默认关闭；AV1 参数校验禁止同时开启两种 AQ，CONSTQP 则要求两种 AQ 都关闭，以保证 QP 语义不被 AQ 改写。

## 项目原则（必须遵守）

**稳定 + 规范 + 低耦合 + 尽量无性能损失。** 所有修改必须基于当前真实代码和实际问题，优先采用可回退、可验证、职责边界清晰的实现，避免无必要重构、额外拷贝、同步、资源占用和性能退化；不得为了局部性能提升破坏稳定性、正确性、兼容性或代码结构。

修改完成后直接提交到 `master`，不得擅自创建分支、PR、tag 或其他远端引用；除非用户明确要求，否则不要拆成多轮提交、不要遗留“后续再优化”的已知问题，应尽量一次性完成本轮明确范围内的修改。

开发中的版本默认标记为 `[Dev] 🔧`，只有经过用户实际测试并明确确认后，才能标记为 `[Release] ✅`。

**AI / 新对话读取规则：**任何 AI 助手或自动化代理在新的对话中首次读取本 README 时，必须先读取并记住本节，并在该对话后续所有涉及本仓库的分析、修改、提交和版本管理中持续遵守；除非用户明确修改本原则，不得自行弱化、忽略或覆盖。

### master 静态审计闭环规则（必须遵守）

静态审计必须以**当前 `master` HEAD / 明确指定 commit、用户本轮明确范围，以及该 commit 相对其直接基线的实际代码变化**为上下文；不得脱离当前代码和变更事实无限扩张审查范围。对于 active path 中能够指出明确失败路径的历史问题，即使该行代码本轮未修改，也仍可作为真实缺陷处理；但同一 commit 已完成闭环审计后，不得仅因为换了审查角度、命名方式或理想架构偏好，就把相同事实重新包装成“新问题”。

每次宣告“静态审计完成”前，必须一次性检查并记录以下边界：**FFmpeg 解码、BasicVSR++、RIFE、SR 模型与权重、输入输出/位深/精度、最终 resize、H2D/D2H/shared-memory/NPP、GPU 调度与 OOM、并发/生命周期/资源释放、import/global state、下载/cache/完整性、依赖/许可证、失败/fallback/fail-fast、编码/原音轨 stream-copy mux/ffprobe、CLI/Notebook/文档、回归测试，以及各 runtime/API/进程边界。** 对本轮实际未触及但属于共享基础设施的区域，也要确认是否被本轮变化间接影响；确认无影响后记录为已覆盖，不应留到下一轮再作为“新发现”。

同一 commit 在上述清单已经完整审过且代码未变化时，后续再次审查只允许新增以下四类问题：**① 代码发生新变化；② 出现新的真实运行/测试/日志证据；③ 上游依赖、CUDA/PyTorch/FFmpeg/驱动或其他运行环境约束发生变化；④ 能指出上一次闭环清单中遗漏的具体、可复现失败路径。** 若不满足这些条件，应直接说明“未发现新的静态缺陷”，不得为了继续产出内容而扩大问题定义。

审查结论必须区分三类：**必须修改**（有明确正确性、稳定性、兼容性、安全性、数据完整性或可复现故障风险）、**仅运行验证**（静态代码无法判定，必须依赖真实 GPU/视频/编码环境验证）、**实验/可选优化**（重构、性能猜测、可读性或架构偏好）。实验/可选项不得混入必须修改清单。

涉及性能的建议必须有 profiler、FPS、显存峰值、吞吐或画质 A/B 等实际数据支撑；在没有数据时，`torch.compile`、TF32/FP16/BF16、channels-last、tile、batch 调整、额外缓存、异步化等只能作为实验候选，不得作为“需要修改的问题”。任何性能修改仍必须服从“稳定 + 规范 + 低耦合 + 尽量无性能损失”，不得以理论加速为理由引入额外同步、复制、显存峰值、数值误差或回退复杂度。

如果用户要求“按审查建议一次性修改”，必须先完成上述完整闭环审计，再在**一个明确范围、一个提交**中处理所有能静态确认的问题；完成后不得立即用同一份未变化代码再开启一轮增量找错。GitHub 或其他审查工具的结果只能作为证据来源，不能替代对当前代码、测试和运行边界的整体判断。

## 版本历史

- v8.10 [Dev] 🔧：移除 deband 和音频处理功能；删除相关 CLI / Notebook 参数、FFmpeg DSP/AAC 转码与 fallback、独立 `audio` 处理入口和色带调试 Notebook。源音频仅保留第一条原始音轨并使用 stream copy mux；`VIDEO_ENHANCE=False` 时 Notebook 直接 stream copy 原视频/音频。
- v8.9 [Dev] 🔧：`ClipSource` 增加单槽有界异步预取；后台线程在当前 GPU 任务执行期间提前完成下一 BVS clip 的 FFmpeg 读帧、scene signature 与组帧，`Queue(maxsize=1)` 提供背压并限制额外内存，不改变视频处理数学路径与输出顺序。
- v8.8 [Dev] 🔧：修复编码配置/日志一致性：AV1 VBR/CBR/CONSTQP 使用正确 FFmpeg 参数，CONSTQP 要求 AQ 关闭；移除 AV1 profile 的伪可配置语义并保留旧 CLI 兼容，成品验证为 Main；AV1/HEVC/H.264 增加源位深一致的 runtime probe；输出位深日志移除全局状态并显式传递；任务完成后用 ffprobe 验证成品 codec/profile/位深/分辨率/FPS/音轨；NVENC 默认改为仅 Spatial AQ，避免 Spatial/Temporal AQ 同时开启；清理活动路径中的陈旧版本文案。
- v8.7 [Dev] 🔧：Notebook 恢复 `hevc_nvenc`，与 `av1_nvenc` 二选一；继续默认 AV1，HEVC 复用现有 NVENC 高质量 backend。
- v8.6 [Dev] 🔧：在 FFmpeg 解码后、BasicVSR++ 前加入可选轻度 deband；新增 `DEBAND_STRENGTH` / `--deband-strength`，Notebook 默认 `0.006`，CLI 默认 `0` 保持旧行为。
- v8.5 [Release] ✅：Notebook 使用 AV1 NVENC-only 高质量配置；P7/HQ、VBR CQ18、fullres multipass、AQ、B-ref 与 GOP，并新增 `AV1_BIT_DEPTH=8/10` 最低输出位深控制；10-bit 片源禁止降为 8-bit。
- v8.2 - 09274ad [Dev] 🔧：FFmpeg 增强音轨作为默认 Audio 1，同时保留原始音轨作为 Audio 2。
- v8.1 - b212afa [Dev] 🔧：音频回退到 v8.0 FFmpeg DSP；保留视频链和后续非音频改动。
- v8.0 - d0908ea [Dev] 🔧：Notebook 独立视频/音频增强开关，音频 FFmpeg DSP 独立模块。
- v7.0 - 5ec12df [Release] ✅：单 GPU（目标 RTX 4090），其余沿用 v6.10 推理链路。
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

- `realesrgan.ipynb`：Kaggle 入口；3 个代码单元（环境 / 配置 / 执行），v8.10 Notebook 可选 `av1_nvenc` / `hevc_nvenc`，当前默认 `DUAL_GPU=True`。`VIDEO_ENHANCE=False` 时仅 stream copy 原视频和原音频。
- `inference.py`：视频增强 CLI 与总入口。
- `inference/runtime.py` / `runtime_api.py`：基础视频探测、FFmpeg 解码、模型加载兼容边界与原音轨 stream-copy mux。
- `inference/scheduler.py`：CPU 总编排、编码 fail-fast probe、最终原音轨 mux 与成品验证。
- `inference/clip_source.py`：CPU scene-aware BVS clip 组装；v8.9 使用后台线程单槽预取下一 clip。
- `inference/scheduler_state.py` / `scheduler_loop.py`：调度状态、任务策略、结果处理与 watchdog。
- `inference/task_protocol.py` / `worker_protocols.py`：BVS / RIFE / SR typed task/result 与 worker API。
- `inference/gpu_transport.py` / `gpu_workers.py` / `gpu_worker_process.py`：常驻 GPU worker、shared memory、FrameHandle 与进程生命周期。
- `inference/frame_transport.py`：`cudaHostRegister`、pinned fallback、异步 H2D/D2H 与 copy stream。
- `inference/bvs_runtime.py` / `optimized_basicvsrpp.py`：BasicVSR++ runtime。
- `inference/rife425.py` / `optimized_rife425.py`：Practical-RIFE 4.25 runtime。
- `inference/sr_runtime.py`：Real-ESRGAN SR runtime。
- `inference/output_runtime.py`：NPP Lanczos、输出 pump、编码与进度日志。
- `inference/weights/`：仓库内置 BasicVSR++ 权重分片与 RIFE 模型压缩包。

## 模型与资源

### 仓库 `inference/weights/` 已包含

| 文件 | 用途 | 官方来源 / 下载 |
|---|---|---|
| `basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth.part01` | BasicVSR++ NTIRE 2021 Decompression Track 1 权重第 1 分片 | [OpenMMLab 完整权重](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth) |
| `basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth.part02` | 同一 BasicVSR++ 权重第 2 分片；运行时合并并校验 SHA256 前缀 `7b2eba02` | [OpenMMLab 完整权重](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth) |
| `RIFEv4.25_0919.zip` | Practical-RIFE 4.25 模型归档；运行时提取 `flownet.pkl` | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) / [官方 Google Drive](https://drive.google.com/uc?export=download&id=1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg) |

### 运行时下载

| 模型 / 文件 | 用途 | 下载链接 |
|---|---|---|
| `realesr-animevideov3.pth` | 当前 Notebook 默认 Real-ESRGAN 动画视频超分模型，原生 4× | [Real-ESRGAN 官方 Release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) |

当前默认流水线只需要上述模型资源。若手动修改 `MODEL` 使用其他 Real-ESRGAN 模型，程序会按代码中的官方 URL 下载对应权重；README 不重复罗列非默认模型。

## 编码

v8.10 Notebook 默认：

- `VIDEO_CODEC="av1_nvenc"`；可改为 `hevc_nvenc`
- 两种 NVENC 共用：P7 / CQ18 / `ENCODE_GPU=0`
- AV1：Main / HQ / VBR CQ18 / fullres multipass / Spatial AQ strength=8 / B-ref / GOP
- AV1 默认 `TEMPORAL_AQ=0`，并禁止与 Spatial AQ 同时开启
- `AV1_BIT_DEPTH=8`：8-bit 片源输出 8-bit；10-bit 片源仍输出 10-bit
- `AV1_BIT_DEPTH=10`：8-bit / 10-bit 片源都输出 10-bit
- HEVC：HQ / VBR CQ18 / fullres multipass / Spatial AQ strength=8 / lookahead 32 / B-frame 3
- HEVC 位深：8-bit 推理帧输出 8-bit；10-bit 推理帧输出 10-bit
- AV1/HEVC/H.264 编码在模型加载前执行真实 runtime probe；成品 mux 后执行 ffprobe 验证
- 源视频若有音频，最终文件仅 stream copy 第一条原始音轨，不执行任何 DSP 或音频重编码

底层仍保留 H.264/软件 HEVC/CPU AV1 编码后端，用于 CLI 兼容，不在 v8.10 Notebook 暴露。
