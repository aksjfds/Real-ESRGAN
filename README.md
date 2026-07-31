# Real-ESRGAN 动画视频推理

项目仅保留两套 Kaggle 视频推理流程。

## 快速版

- `realesrgan.py`
- `realesrgan.ipynb`
- `requirements.txt`

使用 PyTorch `realesr-animevideov3`，支持 T4 × 2 常驻模型、整帧或重叠图块、
任意起点测试、原帧率、音频封装和 HEVC NVENC。Notebook 默认使用
`hevc_nvenc`、`CQ 18`、`p7`，输出 8-bit `yuv420p`。

## 遮罩版

- `realesrgan_mask.py`
- `realesrgan_mask.vpy`
- `realesrgan_mask.ipynb`
- `requirements_mask.txt`

先用 DPIR `drunet_color`（默认强度 1.2）在原尺寸上轻度降噪；降噪后的画面再分别送入
`realesr-animevideov3` 线条分支、NNEDI3 平面/色度分支和 Kirsch 主轮廓遮罩，
软融合后在 16-bit YUV444 中做两阶段 vszip deband，并加入逐帧变化的 Gaussian
颗粒。双 GPU 模式会将连续帧分成两个独立任务，每张 T4 常驻一份 DPIR 和
Real-ESRGAN 模型，NNEDI3 与最终后处理在 CPU 上运行。最终同样输出 8-bit
`hevc_nvenc`（CQ 14、p7）。该流程质量更可调，但会明显慢于快速版；动态颗粒
也会增加编码体积。

## 使用

在 Kaggle 中打开对应 Notebook，修改输入路径、测试起点和参数后按顺序运行。
第一次运行会下载相应模型；遮罩版的官方 DPIR 模型包约 460 MiB。先用
`TEST_SECONDS = 10` 验证效果、显存、编码与音画同步，再改为 `0` 处理完整视频。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。vs-mlrt 模型从其官方
发布地址按需下载；DPIR 模型和 vs-mlrt 来自同一官方发布源，NNEDI3 使用
`vapoursynth-znedi3`，deband/颗粒使用 vsjetpack 的 `vszip` 包装器，请同时
遵守相应项目许可。
