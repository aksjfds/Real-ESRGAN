# Real-ESRGAN 动画视频推理

## 当前版本流水线

当前 v8.5 流水线：

```text
视频增强（可关闭）：
FFmpeg 解码
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧（可关闭）
→ Real-ESRGAN full-frame 超分
→ NPP Lanczos CUDA 最终倍率调整（CPU Lanczos4 fallback）
→ AV1 NVENC（Notebook 默认：P7 / HQ / CQ18 / 10-bit P010 / fullres multipass）

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

`AUDIO_ENHANCE=False` 时不执行 DSP，继续只保留原音轨并优先 stream copy。

v8.5 Notebook 只暴露 `av1_nvenc` 参数；底层旧编码后端仍保留用于兼容。AV1 NVENC 的 10-bit 使用 AV1 Main profile + `p010le`，不存在 HEVC 式 `Main10` profile。

## 项目原则（必须遵守）

**稳定 + 规范 + 低耦合 + 尽量无性能损失。** 所有修改必须基于当前真实代码和实际问题，优先采用可回退、可验证、职责边界清晰的实现，避免无必要重构、额外拷贝、同步、资源占用和性能退化；不得为了局部性能提升破坏稳定性、正确性、兼容性或代码结构。

修改完成后直接提交到 `master`，不得擅自创建分支、PR、tag 或其他远端引用；除非用户明确要求，否则不要拆成多轮提交、不要遗留“后续再优化”的已知问题，应尽量一次性完成本轮明确范围内的修改。

开发中的版本默认标记为 `[Dev] 🔧`，只有经过用户实际测试并明确确认后，才能标记为 `[Release] ✅`。

**AI / 新对话读取规则：**任何 AI 助手或自动化代理在新的对话中首次读取本 README 时，必须先读取并记住本节，并在该对话后续所有涉及本仓库的分析、修改、提交和版本管理中持续遵守；除非用户明确修改本原则，不得自行弱化、忽略或覆盖。

## 版本历史

- v8.5 [Dev] 🔧：Notebook 切换为 AV1 NVENC-only 高质量配置；AV1 Main + P010 10-bit、P7/HQ、VBR CQ18、fullres multipass、AQ、B-ref 与 GOP。
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

- `realesrgan.ipynb`：Kaggle 入口；视频、音频参数分离，v8.5 Notebook 只暴露 AV1 NVENC 高质量参数，当前默认 `DUAL_GPU=True`、`AUDIO_ENHANCE=True`。
- `inference.py`：视频增强 CLI 与总入口。
- `inference/scheduler.py`：CPU 总编排与最终音频边界。
- `inference/scheduler_state.py` / `scheduler_loop.py`：调度状态、任务策略、结果处理与 watchdog。
- `inference/task_protocol.py` / `worker_protocols.py`：BVS / RIFE / SR typed task/result 与 worker API。
- `inference/gpu_transport.py` / `gpu_workers.py` / `gpu_worker_process.py`：常驻 GPU worker、shared memory、FrameHandle 与进程生命周期。
- `inference/frame_transport.py`：`cudaHostRegister`、pinned fallback、异步 H2D/D2H 与 copy stream。
- `inference/bvs_runtime.py` / `optimized_basicvsrpp.py`：BasicVSR++ runtime。
- `inference/rife425.py` / `optimized_rife425.py`：Practical-RIFE 4.25 runtime。
- `inference/sr_runtime.py`：Real-ESRGAN SR runtime。
- `inference/output_runtime.py`：NPP Lanczos、输出 pump、编码与进度日志。
- `inference/weights/`：仓库内置 BasicVSR++ 权重分片与 RIFE 模型压缩包。
- `audio/runtime.py`：FFmpeg dialogue-focused DSP、增强/原始双音轨 mux、音频旁路。
- `audio/process.py`：`VIDEO_ENHANCE=False` 时的视频 stream-copy / 音频处理入口。

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

当前默认流水线只需要上述模型资源。若手动修改 `MODEL` 使用其他 Real-ESRGAN 模型，程序会按代码中的官方 URL 下载对应权重；README 不重复罗列非默认模型。当前 FFmpeg 音频增强不需要额外神经网络模型或权重。

## 编码

v8.5 Notebook 默认：

- GPU AV1：`av1_nvenc`
- 10-bit：AV1 Main + `p010le`
- 质量：P7 / HQ / VBR CQ18 / fullres multipass
- AQ：Spatial + Temporal，strength=8
- B-frame：3，`b_ref_mode=middle`
- Lookahead：28（SDK 13.1 在 3 个 B 帧时允许的最大值）
- GOP：240（60 fps 时 4 秒）

底层仍保留旧 HEVC/H.264/CPU AV1 编码后端，用于 CLI 兼容，不在 v8.5 Notebook 暴露。
