# Real-ESRGAN 动画视频推理

仓库提供一套以画质优先的动画视频处理流程：BasicVSR++ 同分辨率时序恢复 → Real-ESRGAN full-frame 超分 → Lanczos4 最终倍率调整 → 视频编码。

## 版本

- v4.3 代码基线：`90208548939f7b59ae08ff7db7f338b41b703e22`
- v5.8 代码基线：`f5eca71cc10e6b105310cef762bd46469d173e8a`

## 结构

- `inference.py`：统一命令行入口与 BasicVSR++ 固定参数校验。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/runtime.py`：视频探测/解码、8/10-bit 精度保持、模型加载与单帧 full-frame 推理。
- `inference/basicvsrpp.py`：BasicVSR++ NTIRE Track 1 模型、时序 clip、场景切换保护与显存分块。
- `inference/v54_runtime.py`：BasicVSR++ 执行层优化，包括 warp grid 缓存、临时张量削减、8-bit 紧凑 H2D 和 selective channels_last。
- `inference/v52_scheduler.py`：双 GPU 及以上的 BasicVSR++ / Real-ESRGAN 动态调度。
- `inference/v51_runtime.py`：Real-ESRGAN 共享内存传输和稳定 ETA 等执行优化。
- `inference/checkpoint_parts.py`：合并并校验仓库内 BasicVSR++ checkpoint 分片。
- `inference/pipeline.py`：基础 full-frame SR、共享内存、顺序输出、Lanczos/编码能力。
- `inference/balanced_pipeline.py`：创建每张 GPU 对应的 BasicVSR++ 模型实例和 clip worker。
- `inference/models/`：推理模型结构。
- `inference/weights/`：Real-ESRGAN 权重与 BasicVSR++ Track 1 checkpoint 分片。
- `encode/`：HEVC / AV1 等编码后端。

## BasicVSR++ 固定参数

BasicVSR++ 不再使用 `SOURCE_PROFILE`、A-E 档位或启动时 autotuner。参数直接由 Notebook / CLI 提供，并原样传入推理：

- `--bvs-tile-size`：空间 tile，默认 `640`。
- `--bvs-clip-length`：时序 clip 长度，默认 `13`。
- `--bvs-batch-size`：每张 GPU 单个 BVS 任务包含的独立 clip 数，默认 `1`。
- `--bvs-strength`：恢复结果 residual blend 强度，默认 `1.0`。

默认配置：

```text
bvs_tile_size=640
bvs_clip_length=13
bvs_batch_size=1
bvs_strength=1.0
```

`strength` 使用 `original + strength * (enhanced - original)`。当前固定 `clip_overlap=2`、`tile_pad=32`、`scene_threshold=0.30`，BasicVSR++ 优先使用 FP16。若指定 tile 发生 CUDA OOM，只允许向 `384 / 320 / 256` 回退以保证任务能够继续运行，不做性能搜索。

当前动态调度路径要求至少两张 CUDA GPU。每张 GPU 都加载独立 BasicVSR++ 实例，并在 BVS clip 与 full-frame Real-ESRGAN 任务之间动态分配工作。

## v5.8 selective channels_last

v5.8 在固定参数路径上增加 BasicVSR++ selective channels_last 优化。实现只改变主卷积区域 `Conv2d` 权重的 memory format，不修改模型结构、checkpoint 权重数值或恢复参数。

使用 PyTorch 的：

```python
torch.nn.utils.convert_conv2d_weight_memory_format(
    module,
    torch.channels_last,
)
```

当前转换范围：

- `feat_extract`
- `backward_1`
- `forward_1`
- `backward_2`
- `forward_2`
- `reconstruction`

明确保持原路径、不做 selective channels_last 转换的区域：

- SPyNet
- deformable alignment / `deform_conv2d`
- `conv_offset`
- PixelShuffle / upsample 卷积
- `conv_hr`
- `conv_last`

该优化只在 FP16 CUDA 且 GPU Compute Capability >= 7.0 时启用。当前版本不包含 `torch.compile`。

## 数据路径

- 8-bit BasicVSR++：源 clip 保持 uint8 传入 GPU，在 CUDA 上转 float32 并归一化；tile 拼接、strength blend、clamp、round 和 uint8 量化均在 GPU 完成，最后只传回最终恢复帧。
- 10-bit BasicVSR++：保留高精度原路径。
- Real-ESRGAN：固定 full-frame FP16 + channels_last；不做 Real-ESRGAN tile/batch 自动调参。
- BasicVSR++：v5.8 对主 Conv2d-heavy restoration blocks 使用 selective channels_last；复杂时序/光流/可变形卷积区域保持原执行路径。
- 目标倍率小于模型原生倍率时，完整帧超分后只做一次全帧 Lanczos4 缩放。

## 进度与日志

交互式 Notebook 使用 tqdm 动态进度条；Kaggle Batch / Save & Run 使用独立的持久化 `[progress]` 日志。稳定 FPS 与 ETA 基于已完成输出帧计算。

## BasicVSR++ 权重

官方 NTIRE 2021 Compressed Video Enhancement Track 1 checkpoint 拆成两个普通 Git 文件放在 `inference/weights/`：

- `basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth.part01`
- `basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth.part02`

启动时 `inference/checkpoint_parts.py` 将分片写入系统临时目录中的完整 `.pth`，校验 SHA256 前缀 `7b2eba02` 后加载。正常推理不依赖运行时联网下载，也不需要 Git LFS。

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1` / `libaom-av1`
- GPU AV1：`av1_nvenc`
- H.264：`libx264` / `h264_nvenc`

## Kaggle 使用

在根目录 `realesrgan.ipynb` 的参数单元格直接设置：

```python
BVS_TILE_SIZE = 640
BVS_CLIP_LENGTH = 13
BVS_BATCH_SIZE = 1
BVS_STRENGTH = 1.0
```

Notebook 默认使用 `GPU_IDS = "0,1"`。修改参数后运行最后一个调用单元格即可。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。BasicVSR++ 适配代码的第三方说明见 `THIRD_PARTY_NOTICES.md`。
