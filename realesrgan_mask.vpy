"""VapourSynth Real-ESRGAN masked anime upscaler.

Line regions:
    realesr-animevideov3 (native 4x, resized to requested 2x/3x)
Flat regions:
    ArtCNN R16F96 luma 2x + ArtCNN R16F96 Chroma reconstruction

The script is intended to be launched by ``realesrgan_mask.py``.  All
capitalised settings below can also be injected with ``vspipe --arg``.
"""

from __future__ import annotations

import hashlib
import sys
from fractions import Fraction
from pathlib import Path

import vapoursynth as vs


core = vs.core


def setting(name: str, default: str) -> str:
    return str(globals().get(name, default))


def setting_int(name: str, default: int) -> int:
    return int(setting(name, str(default)))


def setting_float(name: str, default: float) -> float:
    return float(setting(name, str(default)))


def setting_bool(name: str, default: bool) -> bool:
    value = setting(name, "1" if default else "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false or 1/0, got {value!r}")


INPUT = setting("INPUT", "")
VSMLRT_PATH = setting("VSMLRT_PATH", "")
REALESRGAN_MODEL = setting("REALESRGAN_MODEL", "")
ARTCNN_LUMA_MODEL = setting("ARTCNN_LUMA_MODEL", "")
ARTCNN_CHROMA_MODEL = setting("ARTCNN_CHROMA_MODEL", "")
ENGINE_DIR = setting("ENGINE_DIR", "./vs-engine-cache")
BACKEND = setting("BACKEND", "ort_cuda").lower()
GPU_IDS = tuple(int(item) for item in setting("GPU_IDS", "0").split(",") if item.strip())

START_FRAME = setting_int("START_FRAME", 0)
FRAME_COUNT = setting_int("FRAME_COUNT", 0)
OUTPUT_SCALE = setting_float("OUTPUT_SCALE", 2.0)
OUTPUT_FPS_NUM = setting_int("OUTPUT_FPS_NUM", 0)
OUTPUT_FPS_DEN = setting_int("OUTPUT_FPS_DEN", 1)
INPUT_WIDTH = setting_int("INPUT_WIDTH", 0)
INPUT_HEIGHT = setting_int("INPUT_HEIGHT", 0)

TILE_SIZE = setting_int("TILE_SIZE", 384)
OVERLAP = setting_int("OVERLAP", 24)
FP16 = setting_bool("FP16", True)
NUM_STREAMS = setting_int("NUM_STREAMS", 2)
BATCH_SIZE = setting_int("BATCH_SIZE", 1)
PREFER_NHWC = setting_bool("PREFER_NHWC", True)
WORKSPACE_GIB = setting_float("WORKSPACE_GIB", 4.0)
USE_CUDA_GRAPH = setting_bool("USE_CUDA_GRAPH", True)
CORE_THREADS = setting_int("CORE_THREADS", 8)
CACHE_MIB = setting_int("CACHE_MIB", 2048)

MATRIX = setting("MATRIX", "709")
TRANSFER = setting("TRANSFER", "709")
PRIMARIES = setting("PRIMARIES", "709")
INPUT_RANGE = setting("INPUT_RANGE", "limited")
OUTPUT_RANGE = setting("OUTPUT_RANGE", "limited")

MASK_LOW = setting_float("MASK_LOW", 0.040)
MASK_HIGH = setting_float("MASK_HIGH", 0.140)
MASK_DARK_BOOST = setting_float("MASK_DARK_BOOST", 0.5)
MASK_STRENGTH = setting_float("MASK_STRENGTH", 1.0)
MASK_INFLATE = setting_int("MASK_INFLATE", 1)
MASK_FEATHER = setting_int("MASK_FEATHER", 1)
MASK_PREFILTER = setting_int("MASK_PREFILTER", 2)
MASK_SUPPORT_LOW = setting_float("MASK_SUPPORT_LOW", 0.012)
MASK_SUPPORT_HIGH = setting_float("MASK_SUPPORT_HIGH", 0.035)
MASK_SUPPORT_RADIUS = setting_int("MASK_SUPPORT_RADIUS", 2)


def validate_settings() -> None:
    required_paths = {
        "INPUT": INPUT,
        "VSMLRT_PATH": VSMLRT_PATH,
        "REALESRGAN_MODEL": REALESRGAN_MODEL,
        "ARTCNN_LUMA_MODEL": ARTCNN_LUMA_MODEL,
        "ARTCNN_CHROMA_MODEL": ARTCNN_CHROMA_MODEL,
    }
    for name, value in required_paths.items():
        if not value:
            raise ValueError(f"{name} is required")
        if name != "VSMLRT_PATH" and not Path(value).is_file():
            raise FileNotFoundError(f"{name} not found: {value}")
    if not Path(VSMLRT_PATH).is_file():
        raise FileNotFoundError(f"VSMLRT_PATH not found: {VSMLRT_PATH}")
    if not GPU_IDS:
        raise ValueError("GPU_IDS must contain at least one CUDA device")
    if len(GPU_IDS) != 1:
        raise ValueError("This pipeline is configured for exactly one CUDA device")
    if OUTPUT_SCALE not in {2.0, 3.0, 4.0}:
        raise ValueError("OUTPUT_SCALE must be 2, 3, or 4")
    if START_FRAME < 0 or FRAME_COUNT < 0:
        raise ValueError("START_FRAME and FRAME_COUNT must be non-negative")
    if OUTPUT_FPS_DEN <= 0:
        raise ValueError("OUTPUT_FPS_DEN must be positive")
    if INPUT_WIDTH < 0 or INPUT_HEIGHT < 0:
        raise ValueError("INPUT_WIDTH and INPUT_HEIGHT must be non-negative")
    if TILE_SIZE < 0 or OVERLAP < 0:
        raise ValueError("TILE_SIZE and OVERLAP must be 0 or positive")
    if TILE_SIZE and OVERLAP * 2 >= TILE_SIZE:
        raise ValueError("OVERLAP must be smaller than half of TILE_SIZE")
    if NUM_STREAMS <= 0 or BATCH_SIZE <= 0:
        raise ValueError("NUM_STREAMS and BATCH_SIZE must be positive")
    if WORKSPACE_GIB <= 0 or CORE_THREADS <= 0 or CACHE_MIB <= 0:
        raise ValueError("Workspace, core threads, and cache size must be positive")
    if not 0 <= MASK_LOW < MASK_HIGH <= 1:
        raise ValueError("MASK thresholds must satisfy 0 <= MASK_LOW < MASK_HIGH <= 1")
    if not 0 <= MASK_SUPPORT_LOW < MASK_SUPPORT_HIGH <= 1:
        raise ValueError(
            "MASK support thresholds must satisfy "
            "0 <= MASK_SUPPORT_LOW < MASK_SUPPORT_HIGH <= 1"
        )
    if MASK_SUPPORT_RADIUS < 0:
        raise ValueError("MASK_SUPPORT_RADIUS must be non-negative")
    if MASK_STRENGTH < 0 or MASK_DARK_BOOST < 0:
        raise ValueError("MASK_STRENGTH and MASK_DARK_BOOST must be non-negative")
    if MASK_INFLATE < 0 or MASK_FEATHER < 0 or MASK_PREFILTER < 0:
        raise ValueError("Mask morphology settings must be non-negative")
    if BACKEND not in {"trt", "ort_cuda"}:
        raise ValueError("BACKEND must be trt or ort_cuda")


validate_settings()
sys.path.insert(0, str(Path(VSMLRT_PATH).resolve().parent))
import vsmlrt  # noqa: E402

core.num_threads = CORE_THREADS
core.max_cache_size = CACHE_MIB
Path(ENGINE_DIR).mkdir(parents=True, exist_ok=True)


def make_backend(device_id: int):
    if BACKEND == "trt":
        return vsmlrt.Backend.TRT(
            fp16=FP16,
            output_format=1 if FP16 else 0,
            device_id=device_id,
            workspace=max(1, int(WORKSPACE_GIB * 1024**3)),
            use_cuda_graph=USE_CUDA_GRAPH,
            num_streams=NUM_STREAMS,
            static_shape=True,
            engine_folder=str(Path(ENGINE_DIR).resolve()),
            builder_optimization_level=3,
        )
    return vsmlrt.Backend.ORT_CUDA(
        fp16=FP16,
        device_id=device_id,
        num_streams=NUM_STREAMS,
        cudnn_benchmark=True,
        prefer_nhwc=PREFER_NHWC,
        output_format=1 if FP16 else 0,
        # ORT CUDA graph capture is not safe for this multi-model graph on T4:
        # ArtCNN convolution setup can call cuDNN while the stream is capturing.
        use_cuda_graph=False,
    )


def resolve_input_size(clip: vs.VideoNode) -> tuple[int, int]:
    width, height = INPUT_WIDTH, INPUT_HEIGHT
    if width == 0 and height == 0:
        return clip.width, clip.height
    if width == 0:
        width = round(clip.width * height / clip.height)
    elif height == 0:
        height = round(clip.height * width / clip.width)
    width = max(2, width + width % 2)
    height = max(2, height + height % 2)
    return width, height


def change_fps(clip: vs.VideoNode, target: Fraction) -> vs.VideoNode:
    source = Fraction(clip.fps_num, clip.fps_den)
    if source == target:
        return clip
    length = max(1, round(clip.num_frames * target / source))
    template = core.std.BlankClip(
        clip,
        length=length,
        fpsnum=target.numerator,
        fpsden=target.denominator,
        keep=True,
    )

    def select_frame(n: int) -> vs.VideoNode:
        index = min(clip.num_frames - 1, int(n * source / target))
        return clip[index]

    return core.std.FrameEval(template, select_frame)


def max_clips(clips: list[vs.VideoNode]) -> vs.VideoNode:
    result = clips[0]
    for candidate in clips[1:]:
        result = core.std.Expr([result, candidate], "x y max")
    return result


def kirsch_mask(y: vs.VideoNode) -> vs.VideoNode:
    # Denoise only the mask source. The neural model inputs remain untouched.
    # Median removes isolated bright/dark specks, then the wider low-pass rejects
    # grain and compression texture while retaining persistent character outlines.
    if MASK_PREFILTER > 0:
        mask_source = core.std.Median(y)
        mask_source = core.std.BoxBlur(
            mask_source,
            hradius=MASK_PREFILTER,
            hpasses=1,
            vradius=MASK_PREFILTER,
            vpasses=1,
        )
    else:
        mask_source = y

    kernels = (
        [5, 5, 5, -3, 0, -3, -3, -3, -3],
        [5, 5, -3, 5, 0, -3, -3, -3, -3],
        [5, -3, -3, 5, 0, -3, 5, -3, -3],
        [-3, -3, -3, 5, 0, -3, 5, 5, -3],
        [-3, -3, -3, -3, 0, -3, 5, 5, 5],
        [-3, -3, -3, -3, 0, 5, -3, 5, 5],
        [-3, -3, 5, -3, 0, 5, -3, -3, 5],
        [-3, 5, 5, -3, 0, 5, -3, -3, -3],
    )
    responses = [
        core.std.Convolution(mask_source, matrix=kernel, divisor=15.0, saturate=True)
        for kernel in kernels
    ]
    edge = max_clips(responses)

    # A main contour has spatial support along neighbouring pixels. Isolated
    # Kirsch responses from grain do not, so gate the fine response with a
    # low-frequency support map before any dilation.
    if MASK_SUPPORT_RADIUS > 0:
        support = core.std.BoxBlur(
            edge,
            hradius=MASK_SUPPORT_RADIUS,
            hpasses=1,
            vradius=MASK_SUPPORT_RADIUS,
            vpasses=1,
        )
    else:
        support = edge
    support_gate = core.std.Expr(
        support,
        f"x {MASK_SUPPORT_LOW} - "
        f"{MASK_SUPPORT_HIGH - MASK_SUPPORT_LOW} / 0 max 1 min",
    )

    # Keep a modest dark-scene compensation, but no longer amplify every weak
    # dark-area response as aggressively as the old noise-sensitive mask.
    compensated = core.std.Expr(
        [edge, y],
        f"x 1 {MASK_DARK_BOOST} 1 y - * + *",
    )
    ramp = core.std.Expr(
        compensated,
        f"x {MASK_LOW} - {MASK_HIGH - MASK_LOW} / 0 max 1 min "
        f"{MASK_STRENGTH} * 0 max 1 min",
    )
    ramp = core.std.Expr([ramp, support_gate], "x y *")
    for _ in range(MASK_INFLATE):
        ramp = core.std.Inflate(ramp)
    if MASK_FEATHER > 0:
        ramp = core.std.BoxBlur(
            ramp,
            hradius=MASK_FEATHER,
            hpasses=1,
            vradius=MASK_FEATHER,
            vpasses=1,
        )
    return ramp


def inference_options(clip: vs.VideoNode) -> tuple[tuple[int, int], tuple[int, int]]:
    if TILE_SIZE == 0:
        return (clip.width, clip.height), (0, 0)
    return (TILE_SIZE, TILE_SIZE), (OVERLAP, OVERLAP)


def run_model(
    clip: vs.VideoNode,
    model_path: str,
    backend,
    *,
    flexible: bool = False,
) -> vs.VideoNode | list[vs.VideoNode]:
    tilesize, overlap = inference_options(clip)
    function = vsmlrt.flexible_inference if flexible else vsmlrt.inference
    return function(
        clip,
        str(Path(model_path).resolve()),
        tilesize=tilesize,
        overlap=overlap,
        backend=backend,
        input_name=None,
        batch_size=BATCH_SIZE,
    )


def process_device(clip: vs.VideoNode, device_id: int) -> vs.VideoNode:
    backend = make_backend(device_id)
    target_width = int(round(clip.width * OUTPUT_SCALE))
    target_height = int(round(clip.height * OUTPUT_SCALE))
    # Keep the mask in fp32, but feed fp16 data to fp16 neural networks. ORT
    # otherwise keeps fp32 model I/O even when the internal weights are fp16.
    mask_yuv444 = core.resize.Bilinear(
        clip,
        format=vs.YUV444PS,
        range_in_s=INPUT_RANGE,
        range_s="full",
    )
    mask_y = core.std.ShufflePlanes(mask_yuv444, 0, vs.GRAY)
    model_yuv444 = (
        core.resize.Point(mask_yuv444, format=vs.YUV444PH)
        if FP16
        else mask_yuv444
    )
    model_y = core.std.ShufflePlanes(model_yuv444, 0, vs.GRAY)
    rgb = core.resize.Bicubic(
        clip,
        format=vs.RGBH if FP16 else vs.RGBS,
        matrix_in_s=MATRIX,
        range_in_s=INPUT_RANGE,
    )

    line4 = run_model(rgb, REALESRGAN_MODEL, backend)
    assert isinstance(line4, vs.VideoNode)
    # std.MaskedMerge does not accept half-float RGBH/GRAYH. Keep neural
    # inference in fp16, then convert only the final model outputs to fp32.
    line_scaled = (
        core.resize.Point(line4, format=vs.RGBS)
        if OUTPUT_SCALE == 4.0 and FP16
        else line4
        if OUTPUT_SCALE == 4.0
        else core.resize.Lanczos(
            line4,
            width=target_width,
            height=target_height,
            format=vs.RGBS,
            filter_param_a=4,
        )
    )

    art_y2 = run_model(model_y, ARTCNN_LUMA_MODEL, backend)
    assert isinstance(art_y2, vs.VideoNode)

    chroma_input = core.std.Expr(model_yuv444, ["", "x 0.5 +"])
    chroma_planes = run_model(
        chroma_input,
        ARTCNN_CHROMA_MODEL,
        backend,
        flexible=True,
    )
    assert isinstance(chroma_planes, list) and len(chroma_planes) == 2
    art_chroma = core.std.ShufflePlanes(
        [chroma_input, chroma_planes[0], chroma_planes[1]],
        [0, 0, 0],
        vs.YUV,
    )
    art_chroma = core.std.Expr(art_chroma, ["", "x 0.5 -"])
    art_chroma2 = core.resize.Spline36(
        art_chroma,
        width=clip.width * 2,
        height=clip.height * 2,
    )
    art_yuv2 = core.std.ShufflePlanes(
        [art_y2, art_chroma2, art_chroma2],
        [0, 1, 2],
        vs.YUV,
    )
    art_yuv_scaled = (
        art_yuv2
        if OUTPUT_SCALE == 2.0
        else core.resize.Lanczos(
            art_yuv2,
            width=target_width,
            height=target_height,
            filter_param_a=4,
        )
    )
    art_rgb_scaled = core.resize.Bicubic(
        art_yuv_scaled,
        format=vs.RGBS,
        matrix_in_s=MATRIX,
        range_in_s="full",
    )

    mask = kirsch_mask(mask_y)
    mask_scaled = core.resize.Bicubic(
        mask,
        width=target_width,
        height=target_height,
        format=vs.GRAYS,
        filter_param_a=0,
        filter_param_b=0.5,
    )
    merged = core.std.MaskedMerge(art_rgb_scaled, line_scaled, mask_scaled)
    print(
        f"[gpu] cuda:{device_id} resident filters ready: "
        "RealESRGAN animevideov3 + ArtCNN R16F96 luma/chroma",
        file=sys.stderr,
        flush=True,
    )
    return merged


source_path = Path(INPUT).resolve()
index_name = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:20] + ".ffindex"
index_path = Path(ENGINE_DIR).resolve() / index_name
source = core.ffms2.Source(source=str(source_path), cachefile=str(index_path))
if source.format is None:
    raise ValueError("Variable-format video is not supported")
if source.format.color_family != vs.YUV:
    raise ValueError("This video pipeline expects a YUV input video")
if START_FRAME >= source.num_frames:
    raise ValueError(f"START_FRAME {START_FRAME} is outside the {source.num_frames}-frame clip")

end_frame = source.num_frames if FRAME_COUNT == 0 else min(source.num_frames, START_FRAME + FRAME_COUNT)
source = source[START_FRAME:end_frame]
input_width, input_height = resolve_input_size(source)
if (source.width, source.height) != (input_width, input_height):
    source = core.resize.Spline36(source, width=input_width, height=input_height)

source_rate = Fraction(source.fps_num, source.fps_den)
output_rate = (
    source_rate
    if OUTPUT_FPS_NUM <= 0
    else Fraction(OUTPUT_FPS_NUM, OUTPUT_FPS_DEN)
)
if output_rate < source_rate:
    source = change_fps(source, output_rate)

hybrid = process_device(source, GPU_IDS[0])

if output_rate > source_rate:
    hybrid = change_fps(hybrid, output_rate)

output = core.resize.Bicubic(
    hybrid,
    format=vs.YUV420P8,
    matrix_s=MATRIX,
    range_s=OUTPUT_RANGE,
    dither_type="error_diffusion",
)

# Stable metadata for Y4M/FFmpeg. Values cover the defaults used by the runner.
matrix_props = {"709": 1, "470bg": 5, "170m": 6, "2020ncl": 9}
transfer_props = {"709": 1, "470m": 4, "470bg": 5, "170m": 6, "2020_10": 14}
primaries_props = {"709": 1, "470m": 4, "470bg": 5, "170m": 6, "2020": 9}
output = core.std.SetFrameProps(
    output,
    _Matrix=matrix_props.get(MATRIX, 1),
    _Transfer=transfer_props.get(TRANSFER, 1),
    _Primaries=primaries_props.get(PRIMARIES, 1),
    _Range=1 if OUTPUT_RANGE == "limited" else 0,
    _FieldBased=0,
)
output.set_output()
