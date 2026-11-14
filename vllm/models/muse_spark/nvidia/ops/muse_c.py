# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT-built `_muse_C` extension with Muse Spark specific CUDA kernels.

The sources live in ``../csrc`` and are compiled on first use with
``torch.utils.cpp_extension.load``; importing this module never builds.
The kernels target SM12x (NVFP4 instructions) and are registered in the
``_muse_C`` torch library. It also carries the two attention ops of
vllm-project/vllm#62132 for builds whose ``torch.ops._C`` lacks them; the
model picks between the two with ``common.ops.resolve_compiled_op``. To add a kernel, put its ``.cu`` file in ``csrc``,
declare it in ``muse_c_ops.h``, register it in ``muse_c_bindings.cpp`` and
append the file to ``_SOURCES``.

The build needs nvcc and ninja, which the stock runtime images do not ship, so
callers ask ``preflight()`` first and only ``load()`` when it passes. Once a
build has succeeded, later processes load the binary without either of them.
"""

import contextlib
import fcntl
import functools
import hashlib
import os
import re
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils import cpp_extension

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_EXT_NAME = "_muse_C"
_CSRC_DIR = Path(__file__).resolve().parent.parent / "csrc"
# Compiled into the extension.
_SOURCES = (
    "muse_c_bindings.cpp",
    "muse_nvfp4_experts_quant.cu",
    "muse_nvfp4_split_gemm1.cu",
    "muse_moe_route.cu",
    "muse_nvfp4_decode_moe.cu",
    "muse_spark_expert_prefetch.cu",
    "muse_spark_l2_prefetch.cu",
    "muse_spark_mul_tanh_soft_cap.cu",
    "muse_spark_qknorm_rope_kv_insert.cu",
)
_CSRC_SUFFIXES = (".cpp", ".cu", ".cuh", ".h")
# The NVFP4 e2m1 conversion instructions need an sm_120 capable nvcc.
_MIN_NVCC_VERSION = (12, 8)

# Written, with the library file name in it, after a build has succeeded: the
# library file alone may be a half-written one of a killed builder.
_BUILT_MARKER = "built"
# Serializes the builders of one build directory across processes. The kernel
# drops an flock when its holder dies, however it dies.
_BUILD_FLOCK = "build.flock"
# torch's own build lock (cpp_extension.FileBaton). Its owner is the process
# that created the file and a killed one never removes it, so every later
# loader would wait for it forever.
_TORCH_BUILD_LOCK = "lock"

_load_lock = threading.Lock()
_loaded = False
# A failed load is final for the process: every layer would otherwise try again.
_load_error: Exception | None = None


def _nvcc_path() -> str:
    cuda_home = cpp_extension.CUDA_HOME
    nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else ""
    if not os.path.isfile(nvcc):
        raise RuntimeError(
            f"Building the {_EXT_NAME} extension needs the CUDA toolkit, but "
            "nvcc was not found. Install a CUDA toolkit "
            f"(>= {_MIN_NVCC_VERSION[0]}.{_MIN_NVCC_VERSION[1]}) and point "
            "CUDA_HOME at it, or put nvcc on PATH."
        )
    return nvcc


def _nvcc_version(nvcc: str) -> tuple[int, int]:
    out = subprocess.run(
        [nvcc, "--version"], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    match = re.search(r"release (\d+)\.(\d+)", out)
    if match is None:
        raise RuntimeError(f"Cannot parse the nvcc version from {nvcc!r}: {out!r}")
    version = (int(match.group(1)), int(match.group(2)))
    if version < _MIN_NVCC_VERSION:
        raise RuntimeError(
            f"The {_EXT_NAME} extension needs nvcc >= "
            f"{_MIN_NVCC_VERSION[0]}.{_MIN_NVCC_VERSION[1]}, found "
            f"{version[0]}.{version[1]} at {nvcc}."
        )
    return version


def _gencode_archs(nvcc_version: tuple[int, int]) -> list[str]:
    # Same arch selection as vLLM's SM12x NVFP4 kernels: CUDA 13 compiles the
    # 12.0f family, which also covers sm_121.
    if nvcc_version >= (13, 0):
        return ["120f"]
    if nvcc_version >= (12, 9):
        return ["120a", "121a"]
    return ["120a"]


def _gencode_flags(nvcc_version: tuple[int, int]) -> list[str]:
    return [
        f"-gencode=arch=compute_{a},code=sm_{a}" for a in _gencode_archs(nvcc_version)
    ]


def _uncovered_device(nvcc_version: tuple[int, int]) -> str | None:
    """Why the binary of this nvcc would not run on the GPU in this process, if so.

    The cubins carry no PTX, and an arch-specific one (sm_120a, sm_121a) runs on
    that capability only, so a 12.8 toolkit (sm_120a alone) builds an extension that
    fails at its first kernel launch on an SM121 GPU (GB10) with "no kernel image".
    """
    # NVML, so that the CUDA context is not created here.
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        return None
    archs = _gencode_archs(nvcc_version)
    if "120f" in archs or f"12{capability.minor}a" in archs:
        return None
    return (
        f"nvcc {nvcc_version[0]}.{nvcc_version[1]} builds the {_EXT_NAME} extension "
        f"for {', '.join(f'sm_{a}' for a in archs)} only, which does not run on "
        f"this GPU (compute capability {capability.major}.{capability.minor}); "
        "nvcc >= 12.9 is needed"
    )


def _build_flags(nvcc_version: tuple[int, int]) -> tuple[list[str], list[str]]:
    cflags = ["-O3"]
    # Match the flags of the stock NVFP4 experts quantization: no
    # --use_fast_math, and torch's COMMON_NVCC_FLAGS minus the
    # -D__CUDA_NO_HALF*__ defines (vLLM's CMake removes them for CUDA >= 12;
    # the kernels need the half/bf16 conversion operators).
    cuda_cflags = [
        "-O3",
        *_gencode_flags(nvcc_version),
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    return cflags, cuda_cflags


def _build_directory(nvcc: str, cflags: list[str], cuda_cflags: list[str]) -> str:
    # Key the build directory on everything that affects the binary, so that
    # workers with different sources, torch or nvcc never share (or clobber)
    # one build. The digest includes the nvcc path, so a cache copied to a host
    # with another CUDA_HOME is rebuilt, not reused. VLLM_CACHE_ROOT must be
    # node-local: the build lock is an fcntl.flock, which some network
    # filesystems do not honor across hosts.
    digest = hashlib.sha256()
    for path in sorted(_CSRC_DIR.iterdir()):
        if path.suffix in _CSRC_SUFFIXES:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    digest.update(repr((_SOURCES, cflags, cuda_cflags)).encode())
    digest.update(f"{torch.__version__}|{nvcc}".encode())
    return os.path.join(envs.VLLM_CACHE_ROOT, "muse_c", digest.hexdigest()[:16])


def _built_library(build_dir: str) -> str | None:
    """The library of a finished build in ``build_dir``, if there is one."""
    try:
        name = Path(build_dir, _BUILT_MARKER).read_text().strip()
    except OSError:
        return None
    library = os.path.join(build_dir, name)
    return library if name and os.path.isfile(library) else None


@functools.cache
def preflight() -> str | None:
    """Why ``load()`` cannot work on this host, or None if it can.

    Cheap and without side effects, so that callers can choose a path that
    does not need the extension before they commit to one that does. A host
    with a finished build needs no ninja, but still nvcc: the build directory
    is keyed on it. The GPU must be one that the nvcc builds for: a toolkit older
    than 12.9 emits no code for an SM121 GPU.
    """
    try:
        nvcc = _nvcc_path()
        nvcc_version = _nvcc_version(nvcc)
    except Exception as e:
        return str(e)
    if (reason := _uncovered_device(nvcc_version)) is not None:
        return reason
    cflags, cuda_cflags = _build_flags(nvcc_version)
    build_dir = _build_directory(nvcc, cflags, cuda_cflags)
    if _built_library(build_dir) is None and not cpp_extension.is_ninja_available():
        return f"Building the {_EXT_NAME} extension needs ninja, which was not found."
    return None


@contextlib.contextmanager
def _build_exclusive(build_dir: str) -> Iterator[None]:
    """Hold the build directory against the other processes that load into it."""
    with open(os.path.join(build_dir, _BUILD_FLOCK), "a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(
                "Another process is building the %s extension in %s; waiting "
                "for it.",
                _EXT_NAME,
                build_dir,
            )
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _load_locked() -> None:
    nvcc = _nvcc_path()
    nvcc_version = _nvcc_version(nvcc)
    # For the callers that load without the preflight: an extension that would
    # fail at its first launch is an error here, where the callers fall back.
    if (reason := _uncovered_device(nvcc_version)) is not None:
        raise RuntimeError(reason)
    cflags, cuda_cflags = _build_flags(nvcc_version)
    build_dir = _build_directory(nvcc, cflags, cuda_cflags)
    os.makedirs(build_dir, exist_ok=True)
    # The library only registers torch ops (no Python module).
    library = _built_library(build_dir)
    if library is not None:
        torch.ops.load_library(library)
        return
    with _build_exclusive(build_dir):
        # A process that held the directory before us may have finished.
        library = _built_library(build_dir)
        if library is not None:
            torch.ops.load_library(library)
        else:
            _build_and_load(build_dir, nvcc_version, cflags, cuda_cflags)


def _build_and_load(
    build_dir: str,
    nvcc_version: tuple[int, int],
    cflags: list[str],
    cuda_cflags: list[str],
) -> None:
    """Compile into ``build_dir``, which the caller holds exclusively."""
    # Every builder is under the flock, so a torch lock file that is still
    # there belongs to one that died.
    try:
        os.remove(os.path.join(build_dir, _TORCH_BUILD_LOCK))
        logger.warning(
            "Removed a stale build lock of a killed %s build in %s.",
            _EXT_NAME,
            build_dir,
        )
    except FileNotFoundError:
        pass
    logger.info(
        "Building the %s extension (nvcc %d.%d, build dir %s); this takes a "
        "minute.",
        _EXT_NAME,
        *nvcc_version,
        build_dir,
    )
    # With is_python_module=False the call loads the library through
    # torch.ops.load_library and returns its path.
    library = cpp_extension.load(
        name=_EXT_NAME,
        sources=[str(_CSRC_DIR / src) for src in _SOURCES],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        build_directory=build_dir,
        with_cuda=True,
        is_python_module=False,
        verbose=False,
    )
    Path(build_dir, _BUILT_MARKER).write_text(os.path.basename(library))


_fakes_registered: set[str] = set()


def _op_defined(name: str) -> bool:
    try:
        getattr(torch.ops._muse_C, name)
    except (AttributeError, RuntimeError):
        return False
    return True


def _register_fakes() -> None:
    """Fake (meta) implementations, so torch.compile can trace the ops.

    The attention ops only mutate their arguments and return nothing, like the
    stock ``torch.ops._C`` versions they mirror. Each fake is registered once
    per process, and only if the loaded library defines the op.
    """
    if "muse_spark_mul_tanh_soft_cap" not in _fakes_registered and _op_defined(
        "muse_spark_mul_tanh_soft_cap"
    ):

        @torch.library.register_fake("_muse_C::muse_spark_mul_tanh_soft_cap")
        def _muse_spark_mul_tanh_soft_cap_fake(
            out: torch.Tensor,
            input: torch.Tensor,
            multiplier: float,
            soft_cap_temp: float,
        ) -> None:
            return None

        _fakes_registered.add("muse_spark_mul_tanh_soft_cap")

    if (
        "fused_muse_spark_qknorm_rope_kv_insert" not in _fakes_registered
        and _op_defined("fused_muse_spark_qknorm_rope_kv_insert")
    ):

        @torch.library.register_fake("_muse_C::fused_muse_spark_qknorm_rope_kv_insert")
        def _fused_muse_spark_qknorm_rope_kv_insert_fake(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int,
            eps: float,
            cos_sin_cache: torch.Tensor | None,
            rotary_dim: int,
            slot_mapping: torch.Tensor | None,
            kv_cache: torch.Tensor | None,
            block_size: int,
            kv_cache_dtype: str,
            k_scale: torch.Tensor | None = None,
            v_scale: torch.Tensor | None = None,
            q_fp8_out: torch.Tensor | None = None,
            q_scale: torch.Tensor | None = None,
        ) -> None:
            return None

        _fakes_registered.add("fused_muse_spark_qknorm_rope_kv_insert")


def _register_prefill_moe_fakes() -> None:
    """Fakes of the prefill MoE ops (they only write their arguments)."""
    if "gather_scaled_fp4_experts_quant" not in _fakes_registered and _op_defined(
        "gather_scaled_fp4_experts_quant"
    ):

        @torch.library.register_fake("_muse_C::gather_scaled_fp4_experts_quant")
        def _gather_scaled_fp4_experts_quant_fake(
            output: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            row_map: torch.Tensor,
            input_global_scale: torch.Tensor,
            input_offset_by_experts: torch.Tensor,
            output_scale_offset_by_experts: torch.Tensor,
        ) -> None:
            return None

        _fakes_registered.add("gather_scaled_fp4_experts_quant")

    if "moe_route_local" not in _fakes_registered and _op_defined("moe_route_local"):

        @torch.library.register_fake("_muse_C::moe_route_local")
        def _moe_route_local_fake(
            topk_ids: torch.Tensor,
            expert_map: torch.Tensor | None,
            num_local_experts: int,
            n: int,
            k: int,
            expert_offsets: torch.Tensor,
            blockscale_offsets: torch.Tensor,
            problem_sizes1: torch.Tensor,
            problem_sizes2: torch.Tensor,
            a_map: torch.Tensor,
            c_map: torch.Tensor,
        ) -> None:
            return None

        _fakes_registered.add("moe_route_local")


def _register_decode_fakes() -> None:
    """Fakes of the NVFP4 decode MoE ops (they only write their arguments)."""
    if "nvfp4_decode_moe" in _fakes_registered or not _op_defined("nvfp4_decode_moe"):
        return

    @torch.library.register_fake("_muse_C::nvfp4_decode_moe")
    def _nvfp4_decode_moe_fake(
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        w13: torch.Tensor,
        sf13: torch.Tensor,
        w2: torch.Tensor,
        sf2: torch.Tensor,
        prm: torch.Tensor,
        gs: torch.Tensor,
        up_first: bool,
        rows: torch.Tensor,
        perm_idx: torch.Tensor,
        hq_data: torch.Tensor,
        hq_sf: torch.Tensor,
        plan: torch.Tensor,
        ctl: torch.Tensor,
        dbg_acc: torch.Tensor | None = None,
        flags: int = 0,
        xq_data: torch.Tensor | None = None,
        xq_sf: torch.Tensor | None = None,
    ) -> None:
        return None

    @torch.library.register_fake("_muse_C::nvfp4_decode_act_quant")
    def _nvfp4_decode_act_quant_fake(
        hidden: torch.Tensor, gs: torch.Tensor, data: torch.Tensor, sf: torch.Tensor
    ) -> None:
        return None

    _fakes_registered.add("nvfp4_decode_moe")


def _register_l2_prefetch_fake() -> None:
    """Fake of the L2 prefetch op (it only hints the cache)."""
    if "l2_prefetch" in _fakes_registered or not _op_defined("l2_prefetch"):
        return

    @torch.library.register_fake("_muse_C::l2_prefetch")
    def _l2_prefetch_fake(
        tensors: list[torch.Tensor],
        num_ctas: int,
        chunk_kib: int,
        evict_last: bool,
    ) -> None:
        return None

    _fakes_registered.add("l2_prefetch")


def _register_expert_prefetch_fake() -> None:
    """Fake of the expert prefetch op (it only hints the cache)."""
    if "expert_prefetch" in _fakes_registered or not _op_defined("expert_prefetch"):
        return

    @torch.library.register_fake("_muse_C::expert_prefetch")
    def _expert_prefetch_fake(
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        w13: torch.Tensor,
        sf13: torch.Tensor,
        w2: torch.Tensor,
        sf2: torch.Tensor,
        up_first: bool,
        num_ctas: int,
        budget_bytes: int,
        pdl: bool,
        dump: torch.Tensor | None = None,
    ) -> None:
        return None

    _fakes_registered.add("expert_prefetch")


def load() -> None:
    """Load the `_muse_C` library, building it on the first use of a cache dir.

    Safe to call from several threads and processes: one builds, the others
    wait for it and load the same binary. Raises if the library cannot be
    built (see ``preflight``), and keeps raising: a failure is not retried.
    """
    global _loaded, _load_error
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        if _load_error is not None:
            raise RuntimeError(
                f"The {_EXT_NAME} extension failed to load earlier in this process."
            ) from _load_error
        try:
            _load_locked()
        except Exception as e:
            _load_error = e
            raise
        _loaded = True
    # Outside the failure-latching path: a missing fake must not poison load().
    try:
        _register_fakes()
        _register_decode_fakes()
        _register_prefill_moe_fakes()
        _register_l2_prefetch_fake()
        _register_expert_prefetch_fake()
    except Exception:
        logger.warning("Could not register _muse_C fake ops", exc_info=True)


def _allocate_experts_quant_outputs(
    m_numtopk: int, k: int, topk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Output buffers of the experts quantization of ``m_numtopk`` rows of width
    ``k``, sized like the stock wrapper does."""
    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization (see the stock wrapper).
    MAX_TOKENS_PER_EXPERT = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE to set this value."
    )
    scales_k = k // 16
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    output = torch.empty(m_numtopk, k // 2, device=device, dtype=torch.uint8)
    output_scales = torch.empty(
        MAX_TOKENS_PER_EXPERT * topk, padded_k, dtype=torch.int32, device=device
    )
    return output, output_scales


def moe_route_local(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    num_local_experts: int,
    intermediate_size: int,
    hidden_size: int,
) -> tuple[torch.Tensor, ...]:
    """Routing data of the CUTLASS grouped GEMMs of one expert-parallel rank.

    What ``remap_topk_to_local`` and ``vllm._custom_ops.get_cutlass_moe_mm_data``
    (the NVFP4 flavour, with block-scale offsets) give together, from the global
    ids, in three short kernels and with no work per remote route.

    Args:
        topk_ids: [num_tokens, top_k] int32 global expert ids, contiguous
        expert_map: [num_global_experts] int32 local id or -1, or None if the ids
            are local already
    Returns:
        ``(expert_offsets, blockscale_offsets, problem_sizes1, problem_sizes2,
        a_map, c_map)``, all int32: the first row of every expert [E + 1], the
        same with every expert padded to whole 128-row tiles [E + 1], the
        (m, 2n, k) and (m, k, n) problem sizes of the two GEMMs [E, 3], the token
        of every local row (-1 past the local rows) [num_routes], and the row of
        every route (-1 for a remote one) [num_routes].
    """
    load()
    device = topk_ids.device
    num_routes = topk_ids.numel()
    expert_offsets = torch.empty(
        num_local_experts + 1, dtype=torch.int32, device=device
    )
    blockscale_offsets = torch.empty_like(expert_offsets)
    problem_sizes1 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=device
    )
    problem_sizes2 = torch.empty_like(problem_sizes1)
    a_map = torch.empty(num_routes, dtype=torch.int32, device=device)
    c_map = torch.empty_like(a_map)
    torch.ops._muse_C.moe_route_local(
        topk_ids,
        expert_map,
        num_local_experts,
        intermediate_size,
        hidden_size,
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
    )
    return (
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
    )


def is_loaded() -> bool:
    """Whether ``load()`` has succeeded in this process. Never builds."""
    return _loaded


def gather_scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    row_map: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``vllm._custom_ops.scaled_fp4_experts_quant`` of ``input_tensor[row_map]``.

    The permuted activation matrix is never built: row ``r`` of the result is the
    quantization of ``input_tensor[row_map[r]]`` with the scale and the
    scale-factor layout of the expert that owns row ``r``, and only the first
    ``expert_offsets[-1]`` rows (the local routes) are computed. The bytes are
    those of the stock op on the shuffled copy.

    Args:
        input_tensor: Activations [num_tokens, k], bf16 or fp16, k % 256 == 0
        row_map: [num_routes] int32 source row of every permuted row; only the
            entries below ``expert_offsets[-1]`` are read
        input_global_scale: A per-expert scaling factor [n_experts]
        expert_offsets: The expert offsets tensor [n_experts+1]
        blockscale_offsets: The blockscale offsets tensor [n_experts+1]
    Outputs:
        output: The quantized tensor in NVFP4 [num_routes, k/2]
        output_scales: The blockscale tensor in FP8-E4M3
    """
    load()
    assert input_tensor.ndim == 2, (
        f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    )
    num_routes = row_map.shape[0]
    k = input_tensor.shape[1]
    padded_k = (k // 16 + (4 - 1)) // 4
    num_experts = input_global_scale.shape[0]
    # output is uint8 and packed fp4 values
    output = torch.empty(
        num_routes, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    # Each expert's scales start at a multiple of 128 rows.
    output_scales = torch.empty(
        num_routes + 128 * num_experts,
        padded_k,
        dtype=torch.int32,
        device=input_tensor.device,
    )
    torch.ops._muse_C.gather_scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        row_map,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    return output, output_scales.view(torch.float8_e4m3fn)


def mul_and_silu_scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Mul+SiLU+NVFP4 quantization for MoE intermediate activations.

    Same as ``vllm._custom_ops.silu_and_mul_scaled_fp4_experts_quant`` for an
    input whose halves are [up | gate] instead of [gate | up]: it quantizes
    ``up * silu(gate)`` with bit-identical results.

    Args:
        input_tensor: The input tensor with up || gate layout [m_topk, k*2]
        input_global_scale: A per-expert scaling factor [n_experts]
        expert_offsets: The expert offsets tensor [n_experts+1]
        blockscale_offsets: The blockscale offsets tensor [n_experts+1]
        topk: Number of top-k experts selected
    Outputs:
        output: The quantized tensor in NVFP4 [m_topk, k/2]
        output_scales: The blockscale tensor in FP8-E4M3
    """
    load()
    assert input_tensor.ndim == 2, (
        f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    )
    m_numtopk, k_times_2 = input_tensor.shape
    assert k_times_2 % 2 == 0, "input width must be even (up || gate layout)"
    output, output_scales = _allocate_experts_quant_outputs(
        m_numtopk, k_times_2 // 2, topk, input_tensor.device
    )
    torch.ops._muse_C.mul_and_silu_scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales


def silu_and_mul_split_scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
    gate_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU+Mul+NVFP4 quantization of the output of the exact GEMM1.

    The input holds the two halves of the expert outputs as stacked matrices,
    the first half in the rows ``[0, m_topk)`` and the second in
    ``[m_topk, 2 * m_topk)``; the first half is the gate if ``gate_first``,
    else the up half. The result is bit-identical to the stock
    ``silu_and_mul_scaled_fp4_experts_quant`` on the same values laid out
    [gate | up].

    Args:
        input_tensor: The stacked halves [2 * m_topk, k]
        input_global_scale: A per-expert scaling factor [n_experts]
        expert_offsets: The expert offsets tensor [n_experts+1] (of m_topk rows)
        blockscale_offsets: The blockscale offsets tensor [n_experts+1]
        topk: Number of top-k experts selected
        gate_first: Whether the gate is the first half
    Outputs:
        output: The quantized tensor in NVFP4 [m_topk, k/2]
        output_scales: The blockscale tensor in FP8-E4M3
    """
    load()
    assert input_tensor.ndim == 2, (
        f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    )
    rows, k = input_tensor.shape
    assert rows % 2 == 0, "input must hold two stacked halves"
    output, output_scales = _allocate_experts_quant_outputs(
        rows // 2, k, topk, input_tensor.device
    )
    torch.ops._muse_C.silu_and_mul_split_scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
        gate_first,
    )
    return output, output_scales.view(torch.float8_e4m3fn)


def split_gemm1_prepare(
    a_fp4: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tables and activations of the exact GEMM1: ``(a_fp4_x2, expert_offsets_x2,
    blockscale_offsets_x2, problem_sizes_x2)``.

    The grouped GEMM reads ``w13`` [E, 2N, K / 2] (and its swizzled block scales)
    as ``[2E, N, K / 2]``: group ``2e + h`` is half ``h`` of expert ``e``, with
    its own alpha. Both halves multiply the same rows, which are stored twice in
    ``a_fp4_x2`` [2R, K / 2] (the output of a group lands at the offset of its
    input); the activation scale factors are shared. ``problem_sizes`` is the
    ``[E, 3]`` table of the GEMM with ``2N`` columns. Only the rows that experts
    own are copied.
    """
    load()
    num_experts = problem_sizes.shape[0]
    rows, row_bytes = a_fp4.shape
    a_fp4_x2 = torch.empty(2 * rows, row_bytes, dtype=torch.uint8, device=a_fp4.device)
    expert_offsets_x2 = torch.empty(
        2 * num_experts, dtype=torch.int32, device=a_fp4.device
    )
    blockscale_offsets_x2 = torch.empty_like(expert_offsets_x2)
    problem_sizes_x2 = torch.empty(
        2 * num_experts, 3, dtype=torch.int32, device=a_fp4.device
    )
    torch.ops._muse_C.split_gemm1_prepare(
        a_fp4,
        expert_offsets,
        blockscale_offsets,
        problem_sizes,
        a_fp4_x2,
        expert_offsets_x2,
        blockscale_offsets_x2,
        problem_sizes_x2,
    )
    return a_fp4_x2, expert_offsets_x2, blockscale_offsets_x2, problem_sizes_x2
