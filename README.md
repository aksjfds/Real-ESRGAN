# Real-ESRGAN 动画视频推理

项目仅保留两套 Kaggle 视频推理流程。

## 快速版

- `realesrgan.py`
- `realesrgan.ipynb`
- `requirements.txt`

使用官方 PyTorch `realesr-animevideov3` checkpoint，并以 `strict=True` 加载。
支持 T4 × 2 每卡一份常驻模型、整帧或带上下文的单次写入 tile、任意起点测试、
原帧率、音频封装和 HEVC NVENC。增强管线内部保持浮点，编码器输入为
`rgb48le`；Notebook 默认的 `hevc_nvenc + OUTPUT_PIX_FMT=auto` 输出 `p010le`。

`baseline` 是重构后的官方模型路径；`safe` 另外启用 X8、一次反投影、Lanczos4
最终缩放和轻度局部范围约束。X8 会将主模型调用次数提高到 8 倍，优先画质而非速度。
可选 getnative/Descale 有独立安装单元；组件不完整时不会用普通 resize 冒充。
Waifu2x、SwinIR、SCUNet、Restormer 和 Real-CUGAN 只保留显式能力边界，当前未接入
经验证的官方常驻后端，选择相关预设或参数会直接报错而不会静默替换。

### 快速版新增参数

| 分组 | 参数 |
| --- | --- |
| 预设 | `--quality-preset baseline\|safe\|compressed-anime\|blurred-anime\|max` |
| 原生分析 | `--native-analysis`、`--native-samples`、`--native-min-height`、`--native-max-height`、`--native-kernels`、`--native-confidence`、`--native-height`、`--native-kernel`、`--descale` |
| 单一前处理 | `--preprocess`、`--preprocess-strength`、`--preprocess-model-path`、`--preprocess-auto-apply`、`--scunet-model` |
| 集成/残差 | `--tta`、`--tta-batch-size`、`--shift-ensemble`、`--residual-mode`、`--residual-strength`、`--residual-flat-strength`、`--residual-edge-strength`、`--residual-edge-low`、`--residual-edge-high`、`--base-correction` |
| CUGAN | `--cugan-ensemble`、`--cugan-model-path`、`--cugan-scale`、`--cugan-alpha`、`--cugan-global-weight`、`--cugan-mask-mode` |
| 反投影 | `--back-projection-iterations`、`--back-projection-strength`、`--back-projection-kernel`、`--back-projection-clamp` |
| tile | `--tile-pad`、`--pre-pad`、`--tile-verify-coverage`；旧 `--overlap` 仅映射到 `tile-pad` |
| 后处理 | `--anime4k`、`--anime4k-shader-dir`、`--anime4k-preset`、`--anime4k-strength`、`--anime4k-shaders`、`--dehalo-strength`、`--dehalo-radius`、`--range-limit`、`--range-radius`、`--overshoot`、`--undershoot` |
| 编码 | `--output-pix-fmt auto\|yuv420p\|yuv420p10le\|p010le` |

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
