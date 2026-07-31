"""VapourSynth DPIR + Real-ESRGAN masked anime upscaler.

Before enlargement:
    DPIR drunet_color denoising
Line regions:
    realesr-animevideov3 (native 4x, resized to requested 2x/3x)
Flat regions:
    NNEDI3 enlargement for luma and chroma
After enlargement:
    two-stage vszip debanding and dynamic Gaussian grain

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
DPIR_MODEL = setting("DPIR_MODEL", "")
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
CORE_THREADS = setting_int("CORE_THREADS", 2)
CACHE_MIB = setting_int("CACHE_MIB", 2048)
DPIR_STRENGTH = setting_float("DPIR_STRENGTH", 1.2)
DPIR_TILE_SIZE = setting_int("DPIR_TILE_SIZE", 256)
DPIR_OVERLAP = setting_int("DPIR_OVERLAP", 16)
DEBAND_ENABLED = setting_bool("DEBAND_ENABLED", True)
DEBAND1_RADIUS = setting_int("DEBAND1_RADIUS", 24)
DEBAND1_THRESHOLD = setting_float("DEBAND1_THRESHOLD", 64.0)
DEBAND2_RADIUS = setting_int("DEBAND2_RADIUS", 12)
DEBAND2_THRESHOLD = setting_float("DEBAND2_THRESHOLD", 32.0)
GRAIN_LUMA = setting_float("GRAIN_LUMA", 32.0)
GRAIN_CHROMA = setting_float("GRAIN_CHROMA", 16.0)
GRAIN_SEED = setting_int("GRAIN_SEED", 333)
NNEDI3_NSIZE = setting_int("NNEDI3_NSIZE", 0)
NNEDI3_NNS = setting_int("NNEDI3_NNS", 2)
NNEDI3_QUAL = setting_int("NNEDI3_QUAL", 2)
NNEDI3_PSCRN = setting_int("NNEDI3_PSCRN", 2)

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
        "DPIR_MODEL": DPIR_MODEL,
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
    if DPIR_STRENGTH < 0:
        raise ValueError("DPIR_STRENGTH must be non-negative")
    if DPIR_TILE_SIZE < 0 or DPIR_OVERLAP < 0:
        raise ValueError("DPIR_TILE_SIZE and DPIR_OVERLAP must be non-negative")
    if DPIR_TILE_SIZE and DPIR_TILE_SIZE % 8:
        raise ValueError("DPIR_TILE_SIZE must be a multiple of 8")
    if DPIR_TILE_SIZE and DPIR_OVERLAP * 2 >= DPIR_TILE_SIZE:
        raise ValueError("DPIR_OVERLAP must be smaller than half of DPIR_TILE_SIZE")
    if DEBAND1_RADIUS <= 0 or DEBAND2_RADIUS <= 0:
        raise ValueError("DEBAND radii must be positive")
    if DEBAND1_THRESHOLD < 0 or DEBAND2_THRESHOLD < 0:
        raise ValueError("DEBAND thresholds must be non-negative")
    if GRAIN_LUMA < 0 or GRAIN_CHROMA < 0 or GRAIN_SEED < 0:
        raise ValueError("Grain strengths and seed must be non-negative")
    if NUM_STREAMS <= 0 or BATCH_SIZE <= 0:
        raise ValueError("NUM_STREAMS and BATCH_SIZE must be positive")
    if WORKSPACE_GIB <= 0 or CORE_THREADS <= 0 or CACHE_MIB <= 0:
        raise ValueError("Workspace, core threads, and cache size must be positive")
    if NNEDI3_NSIZE not in {0, 4}:
        raise ValueError("NNEDI3_NSIZE must be 0 or 4 for enlargement")
    if NNEDI3_NNS not in {0, 1, 2, 3, 4}:
        raise ValueError("NNEDI3_NNS must be between 0 and 4")
    if NNEDI3_QUAL not in {1, 2}:
        raise ValueError("NNEDI3_QUAL must be 1 or 2")
    if NNEDI3_PSCRN not in {0, 1, 2, 3, 4}:
        raise ValueError("NNEDI3_PSCRN must be between 0 and 4")
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
try:
    from vsdeband import f3k_deband  # noqa: E402
except ImportError as error:
    raise RuntimeError(
        "vsjetpack is required for two-stage debanding; install requirements_mask.txt"
    ) from error

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
        # Keep ORT CUDA graph disabled for compatibility with the current
        # VapourSynth scheduling path.
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
    # This extra prefilter affects only mask extraction; all upscale branches
    # otherwise receive the same unmodified DPIR result. Median removes isolated
    # specks, then the wider low-pass rejects residual grain while retaining
    # persistent character outlines.
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


def dpir_inference_options(clip: vs.VideoNode) -> tuple[tuple[int, int], tuple[int, int]]:
    if DPIR_TILE_SIZE == 0:
        return (clip.width, clip.height), (0, 0)
    return (DPIR_TILE_SIZE, DPIR_TILE_SIZE), (DPIR_OVERLAP, DPIR_OVERLAP)


def run_dpir(clip: vs.VideoNode, backend) -> vs.VideoNode:
    """Run the official color DRUNet wrapper before any enlargement."""
    if DPIR_STRENGTH == 0:
        return clip
    model_path = Path(DPIR_MODEL).resolve()
    if model_path.parent.name != "dpir":
        raise RuntimeError(
            f"Unexpected DPIR model layout: {model_path}; expected .../dpir/drunet_color.onnx"
        )
    # The high-level wrapper adds DPIR's second sigma input and applies the
    # TensorRT precision workaround required by older TensorRT releases.
    vsmlrt.models_path = str(model_path.parent.parent)
    tilesize, overlap = dpir_inference_options(clip)
    result = vsmlrt.DPIR(
        clip,
        strength=DPIR_STRENGTH,
        tilesize=tilesize,
        overlap=overlap,
        model=vsmlrt.DPIRModel.drunet_color,
        backend=backend,
    )
    return result


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


def nnedi3_function():
    """Resolve the modern znedi3 namespace with an old-plugin fallback."""
    try:
        return core.znedi3.nnedi3
    except AttributeError:
        try:
            return core.nnedi3.nnedi3
        except AttributeError as error:
            raise RuntimeError(
                "NNEDI3 plugin is unavailable; install vapoursynth-znedi3"
            ) from error


def nnedi3_double(clip: vs.VideoNode) -> vs.VideoNode:
    """Double both dimensions by applying NNEDI3 vertically, then horizontally."""
    nnedi3 = nnedi3_function()
    options = {
        "field": 1,
        "dh": True,
        "nsize": NNEDI3_NSIZE,
        "nns": NNEDI3_NNS,
        "qual": NNEDI3_QUAL,
        "pscrn": NNEDI3_PSCRN,
    }
    vertical = nnedi3(clip, **options)
    transposed = core.std.Transpose(vertical)
    horizontal = nnedi3(transposed, **options)
    return core.std.Transpose(horizontal)


def nnedi3_scale(clip: vs.VideoNode, width: int, height: int) -> vs.VideoNode:
    """Enlarge to 2x/3x/4x; NNEDI3 itself operates in powers of two."""
    passes = 1 if OUTPUT_SCALE == 2.0 else 2
    result = clip
    for _ in range(passes):
        result = nnedi3_double(result)
    if (result.width, result.height) != (width, height):
        result = core.resize.Lanczos(
            result,
            width=width,
            height=height,
            filter_param_a=4,
        )
    return result


def finish_deband_and_grain(clip: vs.VideoNode) -> vs.VideoNode:
    """Deband twice in 16-bit 4:4:4, then add frame-varying Gaussian grain."""
    if not DEBAND_ENABLED:
        return clip

    # Coarse pass: remove wider gradients without adding grain yet.
    stage1 = f3k_deband(
        clip,
        radius=DEBAND1_RADIUS,
        thr=[DEBAND1_THRESHOLD] * 3,
        grain=0,
        sample_mode=f3k_deband.SampleMode.SQUARE,
        dynamic_grain=False,
        blur_first=True,
        seed=GRAIN_SEED + START_FRAME,
        random=f3k_deband.RandomAlgo.UNIFORM,
    )

    # Fine pass: use row/column mean sampling and Gaussian dynamic grain.
    # The START_FRAME offset prevents the second GPU chunk from repeating the
    # exact same seeded noise sequence at the join.
    return f3k_deband(
        stage1,
        radius=DEBAND2_RADIUS,
        thr=[DEBAND2_THRESHOLD] * 3,
        grain=[GRAIN_LUMA, GRAIN_CHROMA, GRAIN_CHROMA],
        sample_mode=f3k_deband.SampleMode.COL_ROW_MEAN,
        dynamic_grain=True,
        blur_first=True,
        seed=GRAIN_SEED + START_FRAME + 1,
        random=(
            f3k_deband.RandomAlgo.UNIFORM,
            f3k_deband.RandomAlgo.GAUSSIAN(0.5),
        ),
    )


def process_device(clip: vs.VideoNode, device_id: int) -> vs.VideoNode:
    # Separate backend objects keep both ONNX sessions resident on this GPU.
    # The graph dependency guarantees DPIR completes before either upscale path.
    dpir_backend = make_backend(device_id)
    upscale_backend = make_backend(device_id)
    target_width = int(round(clip.width * OUTPUT_SCALE))
    target_height = int(round(clip.height * OUTPUT_SCALE))
    rgb = core.resize.Bicubic(
        clip,
        format=vs.RGBH if FP16 else vs.RGBS,
        matrix_in_s=MATRIX,
        range_in_s=INPUT_RANGE,
    )
    denoised_rgb = run_dpir(rgb, dpir_backend)

    # Derive every downstream branch from the same DPIR output so the mask,
    # Real-ESRGAN lines and NNEDI3 flat regions remain spatially consistent.
    mask_yuv444 = core.resize.Bilinear(
        denoised_rgb,
        format=vs.YUV444PS,
        matrix_s=MATRIX,
        range_s="full",
    )
    mask_y = core.std.ShufflePlanes(mask_yuv444, 0, vs.GRAY)
    flat_yuv444 = core.resize.Bicubic(
        denoised_rgb,
        format=vs.YUV444P16,
        matrix_s=MATRIX,
        range_s="full",
    )

    line4 = run_model(denoised_rgb, REALESRGAN_MODEL, upscale_backend)
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

    flat_yuv_scaled = nnedi3_scale(flat_yuv444, target_width, target_height)
    flat_rgb_scaled = core.resize.Bicubic(
        flat_yuv_scaled,
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
    merged = core.std.MaskedMerge(flat_rgb_scaled, line_scaled, mask_scaled)
    print(
        f"[gpu] cuda:{device_id} resident filters ready: "
        "DPIR drunet_color + RealESRGAN animevideov3 + NNEDI3 flat/chroma",
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

post_yuv444 = core.resize.Bicubic(
    hybrid,
    format=vs.YUV444P16,
    matrix_s=MATRIX,
    range_s="full",
    dither_type="error_diffusion",
)
post_yuv444 = finish_deband_and_grain(post_yuv444)

output = core.resize.Bicubic(
    post_yuv444,
    format=vs.YUV420P8,
    range_in_s="full",
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
