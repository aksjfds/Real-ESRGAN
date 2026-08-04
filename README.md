# Real-ESRGAN 动画视频推理

本项目提供面向 Kaggle T4 × 2 的动画视频增强流程：先用 BasicVSR++ 修复压缩视频，
再用官方 `realesr-animevideov3` checkpoint 超分辨率。每张 GPU 常驻一份 Real-ESRGAN
模型，并行批量处理图块。

## 文件

- `realesrgan.py`：视频推理 CLI
- `realesrgan.ipynb`：Kaggle Notebook
- `requirements.txt`：Python 依赖

## 当前处理链

1. FFmpeg 按源帧率解码 RGB 帧。
2. BasicVSR++ 按时间片修复压缩失真。
3. Real-ESRGAN 使用带上下文的 tile 推理，模型采用官方前向路径。
4. 在原生 4× 尺寸完整拼接 FP32 图块。
5. 对完整帧执行一次 Lanczos4，得到指定输出倍率。
6. 以 `rgb48le` 送入 FFmpeg，默认使用 `hevc_nvenc` 编码，并封装原音频。

已移除未启用的输入预缩放、全局 pre-pad、TTA、Shift Ensemble、残差改写、
Base Correction、Back Projection、Dehalo、Range Limit、Anime4K、Descale/getnative
及遮罩版流程。

## 使用

在 Kaggle 中打开 `realesrgan.ipynb`，修改输入、输出、测试起点和所需参数后顺序运行。
建议先保持 `TEST_SECONDS = 10` 检查画质、显存、编码和音画同步，再改为 `0` 处理完整视频。

Notebook 默认启用 tile 覆盖检查。T4 显存不足时，先把
`BASICVSRPP_TILE_SIZE` 从 512 降至 256；Real-ESRGAN OOM 时，再降低 `BATCH_SIZE`，
最后降低 `TILE_SIZE`。

本仓库中的 Real-ESRGAN 代码遵循 BSD 3-Clause `LICENSE`；BasicVSR++ 相关说明见
`THIRD_PARTY_NOTICES.md`。
