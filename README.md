# Real-ESRGAN 动画视频推理

仓库保留一套以画质优先的 full-frame 视频推理流程，并按职责拆分为源视频恢复、Real-ESRGAN 推理、流水线调度与编码。

## 版本

- v4.2 代码基线：`0ae09726e28500f94c9a1db012eb83382fa0731e`
- v5.0：加入可选 BasicVSR++ NTIRE 压缩视频恢复；Real-ESRGAN 仍保持 full-frame。

## 结构

- `inference.py`：统一命令行入口。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/runtime.py`：视频探测/解码、8/10-bit 精度保持、模型加载与单帧 full-frame 推理等基础能力。
- `inference/source_profiles.py`：BasicVSR++ A-E 源质量档的统一配置。
- `inference/basicvsrpp.py`：BasicVSR++ NTIRE Track 1 同分辨率视频恢复、时序 clip、场景切换保护与显存自适应分块。
- `inference/basicvsrpp_autotune.py`：B-E 的快速质量约束吞吐 autotuner，只搜索不降低画质的执行参数并缓存结果。
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

- `A`：关闭 BasicVSR++，路径与 v4.2 相同，适合干净/高质量片源。
- `B`：`strength=0.25`、7 帧 clip，轻度恢复。
- `C`：`strength=0.50`、9 帧 clip，中等恢复。
- `D`：`strength=0.75`、11 帧 clip，强恢复。
- `E`：`strength=1.00`、13 帧 clip，完整使用 BasicVSR++ 输出，即满强度恢复。

`strength` 是 `original + strength * (enhanced - original)` 的 residual blend。B-E 固定使用各档 clip length、2 帧 clip overlap、scene-cut 检测、FP16 优先和 32 像素空间上下文。autotuner 不会改变这些画质参数。

BasicVSR++ 输出与输入保持相同分辨率和整数位深：8-bit 输入返回 8-bit，10-bit 解码路径返回 16-bit 容器中的高精度 RGB，再交给现有 Real-ESRGAN 位深路径。

## BasicVSR++ autotune

B-E 启动后会在实际视频推理前使用合成 clip 做快速 benchmark，不消费源视频帧。目标不是占满显存，而是在画质约束不变的前提下选择实测吞吐最快的执行参数：

- 画质硬约束：`strength`、clip length、clip overlap、tile pad 和 scene-cut 阈值全部固定；Real-ESRGAN 的 full-frame、模型、Lanczos 和编码路径完全不由该 autotuner 修改。
- tile：只向 512 基线以上搜索。候选根据实际源分辨率压缩成少量不同分块拓扑的代表点；例如 1920×1080 通常只测试 `512/640/960/1080`，不再逐个测试十几个接近的 tile。
- 快速排名：先用 5 帧短 clip 在完整源分辨率上跑真实 tiled BasicVSR++，包含实际切块、padding、D2H 和拼接，只用于排序候选。
- 最终确认：只让快速排名前两名使用当前 A-E 档的真实 clip length 再跑完整源分辨率 benchmark；基线只有在快速结果非常接近时才额外参加确认。
- clip batch：16 GiB 级 GPU（包括 Tesla T4）固定 `batch=1`，不再浪费启动时间测试高显存但通常更慢的 batch=2；总显存至少 24 GiB 才尝试 batch=2，至少 48 GiB 且 batch=2 确实更快时才进一步尝试 batch=3。
- 显存只作为安全约束：候选至少保留 `max(2 GiB, 20% 总显存)` 的全局空闲余量，额外占用显存不会提高评分。
- CUDA benchmark 前只做一次小尺寸 warm-up，并在计时前后同步 GPU，避免 lazy initialization 和异步 kernel launch 扭曲测量。
- 运行期不再每个 clip batch 无条件调用 `torch.cuda.empty_cache()`；只有全局空闲显存低于安全阈值时才释放未占用 allocator cache。
- 多 GPU B-E 会让同型号 GPU 复用相同 autotune cache；第二张相同 GPU 不重复搜索。

结果缓存到 `~/.cache/realesrgan/basicvsrpp-autotune-v3.json`。cache key 使用 GPU 型号、总显存、PyTorch/CUDA、源分辨率以及完整 profile 质量参数，不再把容易波动的当前 free-VRAM 桶写入 key；读取缓存时仍会根据记录的峰值显存重新做安全检查。

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

## 流水线

`A` 档保持原 v4.2 流水线：

- 输入帧和模型原生倍率输出通过共享内存传递，`multiprocessing.Queue` 只传递 frame ID / worker ID 等小型元数据。
- 某张 GPU 完成一帧后立即接收下一帧，不等待其它 GPU、Lanczos 缩放和编码。
- 全帧 Lanczos4 与 FFmpeg/NVENC 写入在独立输出线程执行，与后续 GPU 推理重叠。

`B-E` 档按 GPU 数量调度：

- 2 张及以上 GPU：不再固定“GPU0 只跑 BasicVSR++、GPU1 只跑 SR”。相邻且具有相同 overlap 语义的 BasicVSR++ clips 会分配给多张 GPU 并行恢复；得到一批完整原分辨率帧后，全部请求 GPU 再共同运行 Real-ESRGAN full-frame。
- 在显存和实测吞吐确有收益时，大显存 GPU 可以 batch 处理多个互相独立的 clip；不同 clip 不会互相传播时序信息。
- 这种 balanced phase 调度用于修复 BasicVSR++ 比 AnimeVideo-v3 更慢时的 producer bottleneck，避免 SR GPU 长时间等待上游出帧。
- 为避免两个大模型在同一 GPU 上同时抢计算资源，多 GPU B-E 不使用后台 BasicVSR++ prefetch；clip 批次和 SR 批次按需衔接。
- 单 GPU 同样使用 autotune 后的 tile/batch，但 BasicVSR++ 与 Real-ESRGAN 仍按阶段串行使用 GPU。
- FFmpeg/NVENC 编码和 Lanczos 输出线程保持现有实现。

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1`，缺失时自动尝试 `libaom-av1`
- GPU AV1：`av1_nvenc`
- 兼容 H.264：`libx264` / `h264_nvenc`

具体 codec 参数、encoder 检测和视频 writer 都集中在 `encode/`。

## 使用

在 Kaggle 中打开根目录 `realesrgan.ipynb`，设置 `SOURCE_PROFILE = "A" / "B" / "C" / "D" / "E"` 后运行。Notebook 默认测试区间为 5:35 开始、15 秒；第一次 B-E 建议先观察 `[autotuner]` 结果、显存、双 GPU 利用率、编码和音画同步；同一环境后续运行会优先复用 autotune cache。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。BasicVSR++ 适配代码的第三方说明见 `THIRD_PARTY_NOTICES.md`。