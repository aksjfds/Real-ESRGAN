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

主体线条使用 `realesr-animevideov3`，平面与色度使用 NNEDI3，通过
降噪 Kirsch 主轮廓软遮罩融合。双 GPU 模式会将连续帧分成两个独立任务，每张
T4 常驻一份 Real-ESRGAN 模型，NNEDI3 在 CPU 上处理平面分支。最终同样输出
8-bit `hevc_nvenc`（CQ 14、p7）。该流程质量更可调，但比快速版慢。

## 使用

在 Kaggle 中打开对应 Notebook，修改输入路径、测试起点和参数后按顺序运行。
第一次运行会下载相应模型。先用 `TEST_SECONDS = 10` 验证效果、显存、编码与
音画同步，再改为 `0` 处理完整视频。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。vs-mlrt 模型从其官方
发布地址按需下载；NNEDI3 使用 `vapoursynth-znedi3`，请同时遵守相应项目许可。
