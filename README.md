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
- `inference/basicvsrpp_autotune.py`：B-E 的质量约束吞吐 autotuner，自动搜索 tile、clip length 与 clip batch，并缓存结果。
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
- `B`：`strength=0.25`、7 帧最低 clip 基线，轻度恢复。
- `C`：`strength=0.50`、9 帧最低 clip 基线，中等恢复。
- `D`：`strength=0.75`、11 帧最低 clip 基线，强恢复。
- `E`：`strength=1.00`、13 帧最低 clip 基线，完整使用 BasicVSR++ 输出，即满强度恢复。

`strength` 是 `original + strength * (enhanced - original)` 的 residual blend。B-E 固定使用 2 帧 clip overlap、scene-cut 检测、FP16 优先和 32 像素空间上下文。autotuner 不会减小各档时序基线，也不会改变 `strength`、`overlap`、`tile_pad` 或 scene-cut 阈值。

BasicVSR++ 输出与输入保持相同分辨率和整数位深：8-bit 输入返回 8-bit，10-bit 解码路径返回 16-bit 容器中的高精度 RGB，再交给现有 Real-ESRGAN 位深路径。

## BasicVSR++ autotune

B-E 启动后会在实际视频推理前使用合成 clip 做 benchmark，不消费源视频帧。目标不是占满显存，而是在画质约束不变的前提下选择实测吞吐最快的执行参数：

- tile：从 512 基线开始，测试通用候选和由实际源分辨率推导出的分块拓扑临界点；不会因为连续两个较大 tile 变慢就提前停止。1080p 会覆盖类似 512/540/640/768/896/960/1024/1080/1152/1280/1536 的候选。运行期仍保留 384/320/256 的 OOM fallback。
- tile benchmark：先按完整源画面的真实 tile 形状和数量做拓扑筛选，再让基线和估算最快的候选实际跑完整源分辨率 tiled BasicVSR++，计入 padding、D2H 和拼接成本。
- clip length：只允许从 profile 基线向上搜索；B 最多尝试到 13 帧，C/D/E 最多尝试到 15 帧。只有完整源分辨率实测有效 FPS 更快才采用更长 clip。
- clip batch：使用 BasicVSR++ 原生 `N,T,3,H,W` 的 N 维并行独立 clips。默认尝试 1/2；显存至少 24 GiB 时才额外尝试 3。额外显存占用本身不会提高候选评分，batch 必须带来至少约 2% 的实测吞吐提升才采用。
- 质量硬约束：`strength`、clip overlap、tile pad、scene-cut 阈值和各档最低 clip 不参与降质式调优；Real-ESRGAN 的 full-frame、模型、Lanczos 和编码路径完全不由该 autotuner 修改。
- 显存仅作为安全约束：候选至少保留 `max(2 GiB, 20% 总显存)` 的全局空闲余量；评分只看有效输出帧吞吐量。
- 运行期不再每个 clip batch 无条件调用 `torch.cuda.empty_cache()`；只有全局空闲显存低于同一安全阈值时才释放未占用 allocator cache。
- 多 GPU B-E 会让每张 restoration GPU 使用自己的 autotune 结果；每张卡可以一次处理多个独立 clip，再按原始顺序输出。

结果缓存到 `~/.cache/realesrgan/basicvsrpp-autotune-v2.json`，cache key 包含 GPU 型号、可用/总显存档位、PyTorch/CUDA 版本、源分辨率与 profile 质量基线。相同环境后续运行通常直接复用结果。

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
- autotuner 可以进一步让单张 GPU 在显存和吞吐确有收益时以 batch 形式同时处理 2/3 个独立 clip；不同 clip 不会互相传播时序信息。
- 这种 balanced phase 调度用于修复 BasicVSR++ 比 AnimeVideo-v3 更慢时的 producer bottleneck，避免 SR GPU 长时间等待上游出帧。
- 为避免两个大模型在同一 GPU 上同时抢计算资源，多 GPU B-E 不使用后台 BasicVSR++ prefetch；clip 批次和 SR 批次按需衔接。
- 单 GPU 同样使用 autotune 后的 tile/clip/batch，但 BasicVSR++ 与 Real-ESRGAN 仍按阶段串行使用 GPU。
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

在 Kaggle 中打开根目录 `realesrgan.ipynb`，设置 `SOURCE_PROFILE = "A" / "B" / "C" / "D" / "E"` 后运行。Notebook 默认测试区间为 5:35 开始、15 秒；第一次 B-E 建议先观察 autotune 结果、显存、双 GPU 利用率、编码和音画同步；同一环境后续运行会优先复用 autotune cache。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。BasicVSR++ 适配代码的第三方说明见 `THIRD_PARTY_NOTICES.md`。
