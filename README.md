# Real-ESRGAN 动画视频推理

仓库只保留一套 AnimeVideo-v3 视频推理流程，并按职责拆分为推理与编码两部分。

## 结构

- `inference.py`：统一命令行入口。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/runtime.py`：视频解码、Real-ESRGAN 模型、多 GPU worker、tile、融合、进度与音频封装。
- `inference/bitdepth.py`：自动检测并保持 8-bit / 10-bit 源视频精度。
- `inference/models/`：`SRVGGNetCompact` 等推理模型结构。
- `encode/runtime.py`：编码后端选择和 CLI 参数。
- `encode/hevc.py`：H.264/HEVC 编码实现，包括 `libx264`、`libx265`、`h264_nvenc`、`hevc_nvenc`。
- `encode/av1.py`：AV1 编码实现，包括 `libsvtav1`、`libaom-av1`、`av1_nvenc`。
- `requirements.txt`：运行依赖。

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1`，缺失时自动尝试 `libaom-av1`
- GPU AV1：`av1_nvenc`（需要 FFmpeg、驱动和 GPU 都支持 AV1 NVENC）
- 兼容 H.264：`libx264` / `h264_nvenc`
- 位深自动保持：8-bit 源输出 8-bit；10-bit 源使用高精度 RGB 推理并输出 10-bit。

推理目录不包含具体视频编码器实现；所有 codec 参数、encoder 检测和视频 writer 都集中在 `encode/`。

## 使用

在 Kaggle 中打开根目录 `realesrgan.ipynb`，修改输入路径和参数后按顺序运行。建议先使用 `TEST_SECONDS = 10` 验证显存、编码和音画同步，再改为 `0` 处理完整视频。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。
