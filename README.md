# Real-ESRGAN 动画视频推理

当前 v6.0 流水线：

```text
FFmpeg 解码
→ BasicVSR++ NTIRE Track 1 同分辨率时序恢复
→ Practical-RIFE 4.25 任意 timestep 插帧（目标帧率由 RIFE_FPS 唯一决定）
→ Real-ESRGAN full-frame 超分
→ Lanczos4 最终倍率调整
→ HEVC / AV1 编码
```

## 版本

- v4.3 代码基线：`90208548939f7b59ae08ff7db7f338b41b703e22`
- v5.8：固定 BVS 参数 + selective channels_last
- v6.0：在 BasicVSR++ 与 Real-ESRGAN 之间加入 Practical-RIFE 4.25

## 结构

- `inference.py`：统一 CLI，BasicVSR++ / RIFE 参数校验。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/basicvsrpp.py`：BasicVSR++ NTIRE Track 1。
- `inference/rife425.py`：Practical-RIFE 4.25 inference-only IFNet、任意 timestep 推理和官方模型缓存。
- `inference/v52_scheduler.py`：BVS / RIFE / Real-ESRGAN 动态多 GPU 调度。
- `inference/v54_runtime.py`：BVS warp grid cache、临时张量削减、8-bit 紧凑 H2D、selective channels_last。
- `inference/pipeline.py`：共享内存、输出泵、FFmpeg 解码等基础能力。
- `encode/`：HEVC / AV1 编码后端。

## BasicVSR++ 固定参数

不使用 `SOURCE_PROFILE` 或 autotuner：

```text
bvs_tile_size=640
bvs_clip_length=13
bvs_batch_size=1
bvs_strength=1.0
clip_overlap=2
tile_pad=32
scene_threshold=0.30
```

若 `tile=640` OOM，只向 `384 / 320 / 256` 回退。

## RIFE 4.25 / 输出帧率

Notebook 只保留一个帧率参数：

```python
RIFE_FPS = 60
```

不再提供独立的 `FPS = "source"` / `--fps` 配置。`RIFE_FPS` 同时是 RIFE 目标帧率和最终视频输出帧率，并且必须 **大于或等于源视频帧率**；若小于源帧率，程序会在加载 GPU 模型之前直接报错。

当 `RIFE_FPS` 等于源帧率时，不需要生成中间帧；当它高于源帧率时，v6.0 根据目标 CFR 时间轴计算真实 interpolation timestep。例如 24 → 60 FPS 会直接生成目标 60 FPS 时间点所需的中间帧，不先升到 120 FPS 再丢帧。

RIFE 位于 BasicVSR++ 之后、Real-ESRGAN 之前，因此：

- BasicVSR++ 仍只处理源帧率，不因 60 FPS 增加 2.5 倍工作量。
- RIFE 只处理源分辨率帧，不在 4K 上插帧。
- Real-ESRGAN 接收真正的目标 FPS 帧流。
- Writer 在 RIFE 提升帧率时直接以目标帧率接收帧，不再由 FFmpeg `fps` filter 补重复帧。

### 场景切换和重复帧

相邻恢复帧的 `scene_difference <= 0.002` 时视为近重复/静止，保持静止时间段，不调用 RIFE。

若相邻恢复帧通过现有 `scene_difference >= 0.30` 判定为场景切换，则不跨镜头插帧，而是在下一个真实源帧时间点前保持上一镜头。

### 模型与 checkpoint

使用 Practical-RIFE 4.25。首次启用时通过 `gdown` 从 Practical-RIFE 官方 Google Drive 模型链接下载 `RIFEv4.25_0919.zip`，只提取 `flownet.pkl`，缓存到：

```text
~/.cache/realesrgan/rife-v4.25/flownet.pkl
```

官方 4.25 checkpoint 可能包含 `teacher.*` 训练/蒸馏分支权重。v6.0 推理加载器会明确过滤这些 training-only tensor，然后对剩余 IFNet 推理权重执行 strict 校验；不会再因为 `teacher.*` 被当作 unexpected keys 而启动失败。

RIFE 使用官方 CLI 同样支持的 FP16 路径，以利用 T4 Tensor Core；模型结构、推理权重和目标 timestep 不变。

## 多 GPU 调度

v6.0 不增加新的异步 producer/queue。RIFE 与当前 BVS clip task 融合：

```text
GPU task: BasicVSR++ clip
          → 对该 clip 的 emitted source intervals 做 RIFE
          → 返回目标 FPS 帧
          → 原 v5.8 scheduler 继续分配 Real-ESRGAN
```

这样保留 v5.8 已验证的 BVS/SR 动态调度骨架；RIFE 不会和同一 GPU 上的 BVS 同时运行，也不会改动 BVS 的 clip/tile/batch/strength。

## v5.8 selective channels_last

BasicVSR++ 仅对主要 Conv2d-heavy 区域转换 Conv2d 权重 memory format：

- `feat_extract`
- `backward_1`
- `forward_1`
- `backward_2`
- `forward_2`
- `reconstruction`

SPyNet、deformable alignment / `deform_conv2d`、`conv_offset`、PixelShuffle、`conv_hr`、`conv_last` 保持原路径。当前版本不包含 `torch.compile`。

## Kaggle 参数

```python
MODEL = "realesr-animevideov3"
SCALE = 2
RIFE_FPS = 60

GPU_IDS = "0,1"

BVS_TILE_SIZE = 640
BVS_CLIP_LENGTH = 13
BVS_BATCH_SIZE = 1
BVS_STRENGTH = 1.0
```

首次 RIFE 运行需要网络下载官方 4.25 模型；同一环境后续运行使用缓存。

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1` / `libaom-av1`
- GPU AV1：`av1_nvenc`
- H.264：`libx264` / `h264_nvenc`

第三方许可见 `THIRD_PARTY_NOTICES.md`。
