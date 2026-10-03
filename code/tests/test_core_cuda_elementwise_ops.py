"""CUDA/CuPy coverage for the core elementwise arithmetic operators.

These tests intentionally exercise the public Function/Tensor path with CuPy
arrays.  They are skipped as a module when CuPy or a CUDA device is not
available, which keeps CPU-only test environments usable.
"""

from pathlib import Path
import sys

import numpy as np
import pytest


cp = pytest.importorskip("cupy", reason="CuPy is required for CUDA operator tests")

try:
    _device_count = cp.cuda.runtime.getDeviceCount()
except Exception as exc:  # CuPy uses backend-specific exception classes here.
    pytest.skip(f"CUDA runtime is unavailable: {exc}", allow_module_level=True)

if _device_count < 1:
    pytest.skip("No CUDA device is available", allow_module_level=True)

# Make the repository package importable when pytest is launched from D:\Enneuro.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eneuro.base.core import (  # noqa: E402
    Add,
    Div,
    Mul,
    Sub,
    Tensor,
    _binary_raw_kernels,
)


OPERATORS = (
    ("add", Add, cp.add),
    ("sub", Sub, cp.subtract),
    ("mul", Mul, cp.multiply),
    ("div", Div, cp.divide),
)


@pytest.mark.parametrize("name, operator, reference", OPERATORS)
def test_cuda_operator_result_stays_on_cupy_and_matches_reference(
    name, operator, reference
):
    """Each operator computes the expected values on the CUDA backend."""
    lhs = cp.asarray([1.5, -2.0, 4.0], dtype=cp.float32)
    rhs = cp.asarray([2.0, 0.5, -1.0], dtype=cp.float32)

    result = operator()(Tensor(lhs, device="cuda"), Tensor(rhs, device="cuda"))

    assert result.device == "cuda"
    assert isinstance(result.data, cp.ndarray)
    cp.testing.assert_allclose(result.data, reference(lhs, rhs))
    assert ("float", name) in _binary_raw_kernels


@pytest.mark.parametrize("_, operator, reference", OPERATORS)
def test_cuda_operator_supports_broadcasting(_, operator, reference):
    """Broadcasting follows CuPy/NumPy semantics for all four operators."""
    lhs = cp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=cp.float32)
    rhs = cp.asarray([[[2.0, 4.0, 8.0]]], dtype=cp.float32)

    result = operator()(Tensor(lhs, device="cuda"), Tensor(rhs, device="cuda"))

    assert result.data.shape == (2, 2, 3)
    cp.testing.assert_allclose(result.data, reference(lhs, rhs))


@pytest.mark.parametrize("_, operator, reference", OPERATORS)
def test_cuda_operator_supports_noncontiguous_negative_stride_views(
    _, operator, reference
):
    """Python-side normalization handles noncontiguous and negative strides."""
    lhs = cp.arange(1, 25, dtype=cp.float32).reshape(4, 6).T[::-1, ::2]
    rhs = cp.arange(1, 7, dtype=cp.float32).reshape(6, 1)

    result = operator()(Tensor(lhs, device="cuda"), Tensor(rhs, device="cuda"))

    cp.testing.assert_allclose(result.data, reference(lhs, rhs))


@pytest.mark.parametrize("_, operator, reference", OPERATORS)
def test_cuda_operator_matches_cupy_mixed_dtype_promotion(_, operator, reference):
    """RawKernel output dtype and values match CuPy for mixed numeric inputs."""
    lhs = cp.asarray([1.5, 3.0, 7.5], dtype=cp.float32)
    rhs = cp.asarray([1, 2, 3], dtype=cp.int16)

    result = operator()(Tensor(lhs, device="cuda"), Tensor(rhs, device="cuda"))
    expected = reference(lhs, rhs)

    assert result.data.dtype == expected.dtype
    cp.testing.assert_allclose(result.data, expected)


@pytest.mark.parametrize("_, operator, reference", OPERATORS)
@pytest.mark.parametrize(
    "dtype",
    [cp.float16, cp.float32, cp.float64, cp.int32, cp.complex64],
    ids=["float16", "float32", "float64", "int32", "complex64"],
)
def test_cuda_operator_preserves_cupy_dtype_promotion(_, operator, reference, dtype):
    """CUDA results use the dtype promotion selected by CuPy for each op."""
    lhs = cp.asarray([6, 9, 12], dtype=dtype)
    rhs = cp.asarray([2, 3, 4], dtype=dtype)

    result = operator()(Tensor(lhs, device="cuda"), Tensor(rhs, device="cuda"))
    expected = reference(lhs, rhs)

    # Division intentionally follows true-division promotion (which may be
    # wider than cp.result_type(lhs, rhs) for integral inputs).
    assert result.data.dtype == expected.dtype
    np.testing.assert_allclose(cp.asnumpy(result.data), cp.asnumpy(expected))
