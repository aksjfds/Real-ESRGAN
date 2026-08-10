# Real-ESRGAN 动画视频推理

仓库保留一套以画质优先的 full-frame 视频推理流程，并按职责拆分为源视频恢复、Real-ESRGAN 推理、流水线调度与编码。

## 版本

- v4.2 代码基线：`0ae09726e28500f94c9a1db012eb83382fa0731e`
- v5.1 代码基线：`e582920ee1eead8c2237a054cf098326db9f1e3f`

## 结构

- `inference.py`：统一命令行入口。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/runtime.py`：视频探测/解码、8/10-bit 精度保持、模型加载与单帧 full-frame 推理等基础能力。
- `inference/source_profiles.py`：BasicVSR++ A-E 源质量档的统一配置。
- `inference/basicvsrpp.py`：BasicVSR++ NTIRE Track 1 同分辨率视频恢复、时序 clip、场景切换保护与显存自适应分块。
- `inference/basicvsrpp_autotune.py`：B-E 的快速质量约束吞吐 autotuner，只搜索不降低画质的执行参数并缓存结果。
- `inference/v51_runtime.py`：v5.1 的质量保持型执行优化，包括 8-bit GPU 端 BasicVSR++ 拼接/混合/量化、SR 直接写共享输出缓冲和稳定 ETA。
- `inference/checkpoint_parts.py`：将仓库内 BasicVSR++ checkpoint 分片临时合并、校验后交给 PyTorch 加载。
- `inference/pipeline.py`：基础 full-frame SR、共享内存、顺序输出、Lanczos/编码流水线。
- `inference/balanced_pipeline.py`：B-E 多 GPU 负载均衡；BasicVSR++ clip 并行后让全部 GPU 回到 full-frame SR。
- `inference/models/`：`SRVGGNetCompact` 等推理模型结构。
- `inference/weights/`：Real-ESRGAN 权重与 BasicVSR++ Track 1 的两个仓库分片。
- `encode/runtime.py`：编码后端选择和 CLI 参数。
- `encode/hevc.py`：H.264/HEVC 编码实现。
- `encode/av1.py`：AV1 编码实现。
- `requirements.txt`：运行依赖。

## 源质量档

`--source-profile` 提供五档，默认 `A`：

- `A`：关闭 BasicVSR++，适合干净/高质量片源。
- `B`：`strength=0.25`、7 帧 clip，轻度恢复。
- `C`：`strength=0.50`、9 帧 clip，中等恢复。
- `D`：`strength=0.75`、11 帧 clip，强恢复。
- `E`：`strength=1.00`、13 帧 clip，完整使用 BasicVSR++ 输出，即满强度恢复。

`strength` 是 `original + strength * (enhanced - original)` 的 residual blend。B-E 固定使用各档 clip length、2 帧 clip overlap、scene-cut 检测、FP16 优先和 32 像素空间上下文。autotuner 不会改变这些画质参数。

BasicVSR++ 输出与输入保持相同分辨率和整数位深：8-bit 输入返回 8-bit，10-bit 解码路径返回 16-bit 容器中的高精度 RGB，再交给现有 Real-ESRGAN 位深路径。

## BasicVSR++ autotune

B-E 启动后会在实际视频推理前使用合成 clip 做快速 benchmark，不消费源视频帧。目标不是占满显存，而是在画质约束不变的前提下选择实测吞吐最快的执行参数：

- 画质硬约束：`strength`、clip length、clip overlap、tile pad 和 scene-cut 阈值全部固定；Real-ESRGAN 的 full-frame、模型、Lanczos 和编码路径完全不由该 autotuner 修改。
- tile：只向 512 基线以上搜索。候选根据实际源分辨率压缩成少量不同分块拓扑的代表点；例如 1920×1080 通常只测试 `512/640/960/1080`。
- 快速排名：先用 5 帧短 clip 在完整源分辨率上跑真实 tiled BasicVSR++；最终只让排名靠前的候选使用当前档位真实 clip length 再确认。
- clip batch：16 GiB 级 GPU（包括 Tesla T4）固定 `batch=1`；总显存至少 24 GiB 才尝试 batch=2，至少 48 GiB 且 batch=2 确实更快时才进一步尝试 batch=3。
- 显存只作为安全约束：候选至少保留 `max(2 GiB, 20% 总显存)` 的全局空闲余量，额外占用显存不会提高评分。
- 运行期不再每个 clip batch 无条件调用 `torch.cuda.empty_cache()`；只有全局空闲显存低于安全阈值时才释放未占用 allocator cache。

v5.1 的 8-bit BasicVSR++ 数据路径与旧版本性能特征不同，因此 autotune cache 升级到 `~/.cache/realesrgan/basicvsrpp-autotune-v4.json`。10-bit 仍走原有高精度处理路径。

## v5.1 数据路径优化

画质算法保持不变，只减少中间数据搬运和 CPU 内存复制：

- 8-bit BasicVSR++：tile 输出保持在 GPU 上完成完整 clip 拼接；`strength` residual blend、clamp、round 和 uint8 量化也在 GPU 上完成，最后只把最终 uint8 恢复帧传回 CPU。这样避免逐 tile 把 float32 结果传回 CPU，再由 CPU 拼接/混合/量化。
- 8-bit Real-ESRGAN：模型仍按原有 full-frame FP16 + channels-last 路径计算；量化后的 CUDA uint8 输出直接复制到对应 worker 的共享输出缓冲，不再先生成 worker-local NumPy 大帧再 `np.copyto` 一次。
- 10-bit：BasicVSR++ 和 Real-ESRGAN 都保留 v5.0 原路径，不为了性能修改高位深计算与转换语义。
- Lanczos4、编码参数、A-E profile、scene-cut、tile pad 和 clip overlap 均未修改。

双 GPU B-E 仍采用阶段互斥的 balanced 调度：两张 GPU 并行处理 BasicVSR++ clip，随后两张 GPU 共同处理 Real-ESRGAN full-frame。v5.1 不强行让同一 GPU 同时执行两个大模型，避免 CUDA/显存竞争导致吞吐下降或 OOM；输出线程继续与 GPU 阶段并行执行 Lanczos4 和 FFmpeg/NVENC 写入。

## 进度与 ETA

v5.1 不再同时显示 tqdm 的短窗口瞬时速度和另一套累计 FPS。进度条只显示一套基于已完成输出帧的稳定速度：前期使用自首帧以来的平均值，运行足够久后使用最长约 120 秒的窗口，并用同一速度计算 ETA。这样可以覆盖 BasicVSR++ 的批次节奏，减少 ETA 大幅跳动。

## BasicVSR++ 权重

官方 NTIRE 2021 Compressed Video Enhancement Track 1 checkpoint 已拆成两个普通 Git 文件放在 `inference/weights/`：

- `basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth.part01`
- `basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth.part02`

B-E 启动时，`inference/checkpoint_parts.py` 会按顺序将两个分片写入系统临时目录中的完整 `.pth`，计算 SHA256 并要求前缀为 `7b2eba02`；加载期间各 BasicVSR++ 实例复用同一个临时文件，进程退出时自动删除。正常推理不再依赖运行时联网下载，也不需要 Git LFS。

## 推理策略

Real-ESRGAN 固定使用 full-frame，不包含 tile、batch 自动调参或 tile 融合逻辑。BasicVSR++ autotune 只作用于前置同分辨率恢复阶段。

- CUDA：自动使用 FP16 + channels-last。
- CPU：自动使用 FP32；但 BasicVSR++ B-E 档要求 CUDA。
- 输入分辨率固定保持源视频分辨率。
- 模型始终输出原生倍率。
- 当目标倍率小于模型原生倍率时，完整帧推理结束后仅做一次全帧 Lanczos4 缩放。
- 8-bit 源输出 8-bit；10-bit 源使用高精度 RGB 推理并输出 10-bit。

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1`，缺失时自动尝试 `libaom-av1`
- GPU AV1：`av1_nvenc`
- 兼容 H.264：`libx264` / `h264_nvenc`

具体 codec 参数、encoder 检测和视频 writer 都集中在 `encode/`。

## 使用

在 Kaggle 中打开根目录 `realesrgan.ipynb`，设置 `SOURCE_PROFILE = "A" / "B" / "C" / "D" / "E"` 后运行。Notebook 默认测试区间为 5:35 开始、15 秒；第一次 B-E 建议观察 `[autotuner]` 结果、双 GPU 利用率、稳定 FPS/ETA、编码和音画同步；相同环境后续运行会优先复用 autotune cache。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。BasicVSR++ 适配代码的第三方说明见 `THIRD_PARTY_NOTICES.md`。