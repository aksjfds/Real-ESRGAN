# Real-ESRGAN 动画视频推理

仓库只保留一套 AnimeVideo-v3 视频推理流程，并按职责拆分为推理与编码两部分。

## 结构

- `inference.py`：统一命令行入口。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/runtime.py`：视频探测/解码、8/10-bit 精度保持、Real-ESRGAN、多 GPU worker、原生倍率重建、全帧 Lanczos4 缩放、进度与音频封装。
- `inference/autotune.py`：`FULL_FRAME=False` 时，在实际 GPU 上自动寻找质量安全的 tile + batch 最快组合。
- `inference/models/`：`SRVGGNetCompact` 等推理模型结构。
- `inference/weights/`：随仓库提供的推理权重。
- `encode/runtime.py`：编码后端选择和 CLI 参数。
- `encode/hevc.py`：H.264/HEVC 编码实现，包括 `libx264`、`libx265`、`h264_nvenc`、`hevc_nvenc`。
- `encode/av1.py`：AV1 编码实现，包括 `libsvtav1`、`libaom-av1`、`av1_nvenc`。
- `requirements.txt`：运行依赖。

## 推理模式

Notebook 只使用一个 `FULL_FRAME` 参数控制推理方式：

- `FULL_FRAME=True`：画质优先，固定整帧推理，不进行 tile / batch 自动搜索。
- `FULL_FRAME=False`：启用自动 tile + batch 搜索；tile 不会主动选择小于 512 的值，并自动过滤 OOM 组合。
- `MAX_TILE_SIZE` / `MAX_BATCH_SIZE` 控制自动搜索上限；`TILE_SIZE` / `BATCH_SIZE` 作为额外候选和 CPU 回退值。

## 输出缩放

模型始终运行其原生倍率。以 AnimeVideo-v3 的 `SCALE=2` 为例：

```text
AnimeVideo-v3 原生 x4
        ↓
完整 x4 帧（tile 模式先在 x4 完成融合）
        ↓
一次全帧 Lanczos4
        ↓
最终 x2
```

不再在 worker 内对每个 frame/tile 使用 bicubic 从 x4 缩回 x2，从而避免逐 tile 重采样造成额外的线条软化和边界差异。

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
