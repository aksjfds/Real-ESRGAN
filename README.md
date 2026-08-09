# Real-ESRGAN 动画视频推理

仓库保留一套以画质优先的 full-frame 视频推理流程，并按职责拆分为源视频恢复、Real-ESRGAN 推理、流水线调度与编码。

## 版本

- v4.2 代码基线：`0ae09726e28500f94c9a1db012eb83382fa0731e`
- v5.0：加入可选 BasicVSR++ NTIRE 压缩视频恢复；Real-ESRGAN 仍保持 full-frame。

## 结构

- `inference.py`：统一命令行入口。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/runtime.py`：视频探测/解码、8/10-bit 精度保持、模型加载与单帧 full-frame 推理等基础能力。
- `inference/basicvsrpp.py`：BasicVSR++ NTIRE Track 1 同分辨率视频恢复、时序 clip、场景切换保护与显存自适应分块。
- `inference/pipeline.py`：BasicVSR++ 预取、多 GPU full-frame 流水线、共享内存帧传输、输出顺序控制以及 Lanczos/编码并行。
- `inference/models/`：`SRVGGNetCompact` 等推理模型结构。
- `inference/weights/`：推理权重缓存目录；BasicVSR++ checkpoint 首次使用 B/C 档时自动下载。
- `encode/runtime.py`：编码后端选择和 CLI 参数。
- `encode/hevc.py`：H.264/HEVC 编码实现。
- `encode/av1.py`：AV1 编码实现。
- `requirements.txt`：运行依赖。

## 源质量档

`--source-profile` 提供三档，默认 `A`：

- `A`：关闭 BasicVSR++，路径与 v4.2 相同，适合干净/高质量片源。
- `B`：BasicVSR++ NTIRE Track 1，`strength=0.25`、7 帧 clip，适合轻度压缩、轻噪声和轻微时序不稳定。
- `C`：BasicVSR++ NTIRE Track 1，`strength=0.50`、9 帧 clip，适合明显压缩和噪声。

B/C 都使用 2 帧 clip overlap、scene-cut 检测、FP16 优先、32 像素空间上下文。空间 tile 默认 512；OOM 时自动依次回退到 384/320/256。tile 只属于 BasicVSR++ 前置恢复，**不会把 Real-ESRGAN 改成 tile 推理**。

BasicVSR++ 输出与输入保持相同分辨率和整数位深：8-bit 输入返回 8-bit，10-bit 解码路径返回 16-bit 容器中的高精度 RGB，再交给现有 Real-ESRGAN 位深路径。

## 推理策略

Real-ESRGAN 固定使用 full-frame，不包含 tile、batch 自动调参或 tile 融合逻辑。

- CUDA：自动使用 FP16 + channels-last。
- CPU：自动使用 FP32；但 BasicVSR++ B/C 档要求 CUDA。
- 输入分辨率固定保持源视频分辨率。
- 模型始终输出原生倍率。
- 当目标倍率小于模型原生倍率时，完整帧推理结束后仅做一次全帧 Lanczos4 缩放。
- 8-bit 源输出 8-bit；10-bit 源使用高精度 RGB 推理并输出 10-bit。

## 流水线

`A` 档保持原 v4.2 流水线：

- 输入帧和模型原生倍率输出通过共享内存传递，`multiprocessing.Queue` 只传递 frame ID / worker ID 等小型元数据。
- 某张 GPU 完成一帧后立即接收下一帧，不等待其它 GPU、Lanczos 缩放和编码。
- 全帧 Lanczos4 与 FFmpeg/NVENC 写入在独立输出线程执行，与后续 GPU 推理重叠。

`B/C` 档进一步按 GPU 数量调度：

- 2 张及以上 GPU：第一张 GPU 专用于 BasicVSR++，其余 GPU 专用于 Real-ESRGAN full-frame；两个重计算阶段通过有界内存预取队列重叠，避免模型激活互相抢显存。
- 单 GPU：BasicVSR++ 与 Real-ESRGAN 使用同一设备，采用串行 feed，优先保证显存稳定和画质；Lanczos/编码线程仍保持并行。
- FFmpeg/NVENC 编码路径不因开启 BasicVSR++ 改写。

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1`，缺失时自动尝试 `libaom-av1`
- GPU AV1：`av1_nvenc`
- 兼容 H.264：`libx264` / `h264_nvenc`

具体 codec 参数、encoder 检测和视频 writer 都集中在 `encode/`。

## 使用

在 Kaggle 中打开根目录 `realesrgan.ipynb`，设置 `SOURCE_PROFILE = "A" / "B" / "C"` 后运行。建议先使用 `TEST_SECONDS = 10` 验证显存、BasicVSR++ tile fallback、编码、GPU 利用率和音画同步，再改为 `0` 处理完整视频。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。BasicVSR++ 适配代码的第三方说明见 `THIRD_PARTY_NOTICES.md`。
