# Real-ESRGAN 动画视频推理

仓库只保留一套 AnimeVideo-v3 视频推理流程，并按职责拆分为推理与编码两部分。

## 结构

- `inference.py`：统一命令行入口。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/runtime.py`：Real-ESRGAN 视频推理、多 GPU worker、tile、解码、音频封装等核心逻辑。
- `inference/models/`：`SRVGGNetCompact` 等推理所需模型结构。
- `encode/runtime.py`：AV1 编码扩展；原有 H.264/HEVC 仍直接使用推理核心中的 writer。
- `requirements.txt`：运行依赖。

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1`，缺失时自动尝试 `libaom-av1`
- GPU AV1：`av1_nvenc`（需要 FFmpeg、驱动和 GPU 都支持 AV1 NVENC）

H.264 的 `libx264` / `h264_nvenc` 仍保留兼容。

## 使用

在 Kaggle 中打开根目录 `realesrgan.ipynb`，修改输入路径和参数后按顺序运行。建议先使用 `TEST_SECONDS = 10` 验证显存、编码和音画同步，再改为 `0` 处理完整视频。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。
