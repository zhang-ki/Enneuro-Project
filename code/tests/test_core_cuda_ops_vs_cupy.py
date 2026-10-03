"""Compare RawKernel operators with the original CuPy-backed Function path.

The two implementations use the same EnNeuro ``Function.__call__`` path and
the same CUDA Tensor inputs. Only the ``forward`` implementation differs:

* current operators: handwritten CUDA C++ through ``cupy.RawKernel``;
* original reference operators: the original ``x0 + x1``/``x0 - x1``/
  ``x0 * x1``/``x0 / x1`` CuPy-backed expressions.

Correctness tests:

    conda run -n neuro python -m pytest code/tests/test_core_cuda_ops_vs_cupy.py -v

Direct timing comparison:

    conda run -n neuro python code/tests/test_core_cuda_ops_vs_cupy.py
"""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pytest


try:
    import cupy as cp
except ImportError as exc:
    if __name__ == "__main__":
        raise SystemExit(f"CuPy is required: {exc}") from exc
    pytest.skip("CuPy is required for CUDA operator tests", allow_module_level=True)

try:
    DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
except Exception as exc:
    if __name__ == "__main__":
        raise SystemExit(f"CUDA runtime is unavailable: {exc}") from exc
    pytest.skip(f"CUDA runtime is unavailable: {exc}", allow_module_level=True)

if DEVICE_COUNT < 1:
    if __name__ == "__main__":
        raise SystemExit("No CUDA device is available")
    pytest.skip("No CUDA device is available", allow_module_level=True)


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eneuro.base.core import (  # noqa: E402
    Add,
    Config,
    Div,
    Function,
    Mul,
    Sub,
    Tensor,
    _binary_raw_kernels,
)


class OriginalAdd(Function):
    def forward(self, x0, x1):
        return x0 + x1


class OriginalSub(Function):
    def forward(self, x0, x1):
        return x0 - x1


class OriginalMul(Function):
    def forward(self, x0, x1):
        return x0 * x1


class OriginalDiv(Function):
    def forward(self, x0, x1):
        return x0 / x1


OPERATORS = (
    ("add", Add, OriginalAdd),
    ("sub", Sub, OriginalSub),
    ("mul", Mul, OriginalMul),
    ("div", Div, OriginalDiv),
)


def _run(operator, lhs, rhs):
    """Run an operator through the same Tensor/Function call path."""
    with Config.no_grad():
        return operator()(Tensor(lhs, device="cuda"), Tensor(rhs, device="cuda")).data


@pytest.mark.parametrize("name, raw_operator, original_operator", OPERATORS)
@pytest.mark.parametrize("case", ("contiguous", "broadcast"))
def test_rawkernel_matches_original_cupy_function(
    name, raw_operator, original_operator, case
):
    """RawKernel and the original CuPy-backed Function produce equal results."""
    if case == "contiguous":
        lhs = cp.asarray([[1.5, -2.0, 4.0], [6.0, 3.0, -8.0]], dtype=cp.float32)
        rhs = cp.asarray([[2.0, 0.5, -1.0], [3.0, 1.5, 4.0]], dtype=cp.float32)
    else:
        lhs = cp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=cp.float32)
        rhs = cp.asarray([[[2.0, 4.0, 8.0]]], dtype=cp.float32)

    raw_result = _run(raw_operator, lhs, rhs)
    original_result = _run(original_operator, lhs, rhs)

    assert isinstance(raw_result, cp.ndarray)
    assert raw_result.shape == original_result.shape
    assert raw_result.dtype == original_result.dtype
    cp.testing.assert_allclose(raw_result, original_result, rtol=1e-6, atol=1e-6)
    assert ("float", name) in _binary_raw_kernels


@pytest.mark.parametrize("name, raw_operator, original_operator", OPERATORS)
def test_rawkernel_matches_original_dtype_promotion(
    name, raw_operator, original_operator
):
    """Mixed float32/int16 promotion matches the original Function path."""
    lhs = cp.asarray([1.5, 3.0, 7.5], dtype=cp.float32)
    rhs = cp.asarray([1, 2, 3], dtype=cp.int16)

    raw_result = _run(raw_operator, lhs, rhs)
    original_result = _run(original_operator, lhs, rhs)

    assert raw_result.dtype == original_result.dtype
    np.testing.assert_allclose(
        cp.asnumpy(raw_result), cp.asnumpy(original_result)
    )


def _elapsed_ms(call, repeats):
    """Measure end-to-end wall time, including the shared Function path."""
    call()
    cp.cuda.Stream.null.synchronize()
    start = perf_counter()
    for _ in range(repeats):
        call()
    cp.cuda.Stream.null.synchronize()
    return (perf_counter() - start) * 1000.0 / repeats


def run_comparison(size=1_000_000, repeats=200):
    """Print RawKernel versus original CuPy-backed Function timings."""
    random = cp.random.RandomState(0)
    lhs = random.uniform(-2.0, 2.0, size).astype(cp.float32)
    rhs = random.uniform(0.5, 2.0, size).astype(cp.float32)
    lhs_tensor = Tensor(lhs, device="cuda")
    rhs_tensor = Tensor(rhs, device="cuda")

    device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    print(f"CUDA device: {device_name}")
    print(f"elements={size:,}, dtype=float32, repeats={repeats}")
    print("both sides use EnNeuro Function.__call__ and the same CUDA Tensor inputs")
    print("raw: handwritten RawKernel; original: CuPy-backed forward expression")
    print(f"{'operator':<10}{'match':<8}{'raw ms':>12}{'original ms':>16}{'raw/original':>16}")

    for name, raw_operator, original_operator in OPERATORS:
        with Config.no_grad():
            raw_call = lambda: raw_operator()(lhs_tensor, rhs_tensor).data
            original_call = lambda: original_operator()(lhs_tensor, rhs_tensor).data

            raw_result = raw_call()
            original_result = original_call()
            cp.testing.assert_allclose(
                raw_result, original_result, rtol=1e-6, atol=1e-6
            )

            raw_ms = _elapsed_ms(raw_call, repeats)
            original_ms = _elapsed_ms(original_call, repeats)

        ratio = raw_ms / original_ms if original_ms else float("inf")
        print(
            f"{name:<10}{'yes':<8}{raw_ms:>12.4f}{original_ms:>16.4f}"
            f"{ratio:>16.2f}x"
        )


if __name__ == "__main__":
    run_comparison()
