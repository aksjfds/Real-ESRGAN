# Kaggle T4×2 视频推理

入口为 `inference_realesrgan_video_kaggle.py`，Notebook 为
`notebooks/Real_ESRGAN_Kaggle_T4x2.ipynb`。

## 设计与原入口的差异

- 原 `inference_realesrgan_video.py` 以子视频为并行单位，每个进程各自加载模型；
  `inference_realesrgan_video_fast.py` 增加了帧批量，但仍会按进程复制模型，而且使用硬边 tile。
- 新入口固定每个 GPU 一个长期存活的工作进程和一份模型。每帧图块分配到所有 GPU，
  每张卡按 `--batch-size` 批量推理，不会逐帧或逐图块重新加载/复制模型。
- 图块以 `tile_size - overlap` 为步长，边缘用反射填充；输出用二维渐变权重融合。
- 主进程通过 ffmpeg 解码为固定帧率 rawvideo，完成融合后用 x264/x265 编码。
  最终单独裁剪并封装原视频的音频，以测试起点为零点重建时间戳。
- 进度条有独立线程按 `--progress-interval` 强制刷新；默认 60 秒。

## 推荐的 T4×2 起始参数

动漫视频建议：

```text
--model realesr-animevideov3 --scale 2 --fp16
--tile-size 256 --overlap 32 --batch-size 4 --gpu-ids 0,1
--video-codec libx264 --crf 18 --preset medium
--audio-codec aac --audio-bitrate 192k
--start-time 0 --test-seconds 10 --progress-interval 60
```

`--input-width 0 --input-height 0` 保持源尺寸；只指定一个维度会保持宽高比。
完成 10 秒测试后将 `--test-seconds` 设为 `0`，即可从 `--start-time` 处理到末尾。

## OOM 降级顺序

1. `batch-size: 4 → 2 → 1`。
2. `tile-size: 256 → 192 → 128`，对应 `overlap: 32 → 24 → 16`。
3. 设置 `--input-width 1280 --input-height 0` 限制推理输入。
4. 必要时降低输出倍率。FP16 通常更省显存，不应作为 OOM 时首先关闭的选项。

RRDB 大模型（`RealESRGAN_x4plus`、`RealESRGAN_x2plus`）应从 batch 1、tile 128/192
开始；小型动漫模型可从 batch 4、tile 256 开始。

## 兼容和同步说明

- 仓库 `requirements.txt` 固定了旧版 torch/torchvision。Kaggle 中不要安装它，Notebook
  保留镜像自带且互相匹配的 CUDA 版本，只安装 `requirements-video-kaggle.txt`，随后用
  `pip install -e . --no-deps` 安装本仓库。
- BasicSR 1.4.2 使用了新版 torchvision 已删除的 `functional_tensor` 模块；新入口在导入
  BasicSR 前提供最小兼容别名。权重加载也显式使用 `weights_only=True`，并兼容旧 PyTorch。
- 10 秒测试默认使用 AAC 重新编码音频，可精确裁剪并重置时间戳。完整视频可使用
  `--audio-codec copy` 避免音频重编码；若 MP4 不支持源音频编码，脚本自动回退 AAC。
- 解码端明确转换为固定帧率，因此 CFR 视频可稳定同步。VFR 输入会被转换为探测到的平均
  帧率；极端 VFR 素材应先转为 CFR，或在完整运行前重点检查 10 秒样本的口型/节拍同步。
- GPU 利用率呈周期性下降时，常见瓶颈是 CPU 图块融合或 libx264。可先将 preset 改为
  `fast`/`veryfast`；不要增加每卡进程数，否则会再次复制模型并争抢显存。
