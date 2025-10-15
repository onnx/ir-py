# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the tensor_adapters module."""

from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest

import ml_dtypes
import numpy as np
import parameterized
import torch

import onnx_ir as ir
from onnx_ir import tensor_adapters


def skip_if_no(module_name: str):
    """Decorator to skip a test if a module is not installed."""
    if importlib.util.find_spec(module_name) is None:
        return unittest.skip(f"{module_name} not installed")
    return lambda func: func


@skip_if_no("torch")
class TorchTensorTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (torch.bfloat16, ml_dtypes.bfloat16),
            (torch.bool, np.bool_),
            (torch.complex128, np.complex128),
            (torch.complex64, np.complex64),
            (torch.float16, np.float16),
            (torch.float32, np.float32),
            (torch.float64, np.float64),
            (torch.float8_e4m3fn, ml_dtypes.float8_e4m3fn),
            (torch.float8_e4m3fnuz, ml_dtypes.float8_e4m3fnuz),
            (torch.float8_e5m2, ml_dtypes.float8_e5m2),
            (torch.float8_e5m2fnuz, ml_dtypes.float8_e5m2fnuz),
            (torch.float8_e8m0fnu, ml_dtypes.float8_e8m0fnu),
            (torch.int16, np.int16),
            (torch.int32, np.int32),
            (torch.int64, np.int64),
            (torch.int8, np.int8),
            (torch.uint16, np.uint16),
            (torch.uint32, np.uint32),
            (torch.uint64, np.uint64),
            (torch.uint8, np.uint8),
        ],
    )
    def test_numpy_returns_correct_dtype(self, dtype: torch.dtype, np_dtype):
        tensor = tensor_adapters.TorchTensor(torch.tensor([1], dtype=dtype))
        self.assertEqual(tensor.numpy().dtype, np_dtype)
        self.assertEqual(tensor.__array__().dtype, np_dtype)
        self.assertEqual(np.array(tensor).dtype, np_dtype)

    @parameterized.parameterized.expand(
        [
            (torch.bfloat16,),
            (torch.bool,),
            (torch.complex128,),
            (torch.complex64,),
            (torch.float16,),
            (torch.float32,),
            (torch.float64,),
            (torch.float8_e4m3fn,),
            (torch.float8_e4m3fnuz,),
            (torch.float8_e5m2,),
            (torch.float8_e5m2fnuz,),
            (torch.float8_e8m0fnu,),
            (torch.int16,),
            (torch.int32,),
            (torch.int64,),
            (torch.int8,),
            (torch.uint16,),
            (torch.uint32,),
            (torch.uint64,),
            (torch.uint8,),
        ],
    )
    def test_tobytes(self, dtype: torch.dtype):
        tensor = tensor_adapters.TorchTensor(torch.tensor([1], dtype=dtype))
        self.assertEqual(tensor.tobytes(), tensor.numpy().tobytes())

    def test_tofile_method_exists_and_works(self):
        """Test that tofile() method exists and works correctly."""
        torch_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor = tensor_adapters.TorchTensor(torch_tensor)

        # Test with BytesIO buffer
        buffer = io.BytesIO()
        tensor.tofile(buffer)
        result_bytes = buffer.getvalue()

        expected_bytes = tensor.tobytes()
        self.assertEqual(result_bytes, expected_bytes)

    @parameterized.parameterized.expand(
        [
            (torch.bfloat16,),
            (torch.bool,),
            (torch.complex128,),
            (torch.complex64,),
            (torch.float16,),
            (torch.float32,),
            (torch.float64,),
            (torch.float8_e4m3fn,),
            (torch.float8_e4m3fnuz,),
            (torch.float8_e5m2,),
            (torch.float8_e5m2fnuz,),
            (torch.int16,),
            (torch.int32,),
            (torch.int64,),
            (torch.int8,),
            (torch.uint16,),
            (torch.uint32,),
            (torch.uint64,),
            (torch.uint8,),
        ],
    )
    def test_tofile(self, dtype: torch.dtype):
        """Test tofile() method for various data types."""
        torch_tensor = torch.tensor([1], dtype=dtype)
        tensor = tensor_adapters.TorchTensor(torch_tensor)

        with tempfile.NamedTemporaryFile() as temp_file:
            tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        expected_bytes = tensor.tobytes()
        self.assertEqual(result_bytes, expected_bytes)


class TorchDtypeConversionTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (ir.DataType.BFLOAT16, torch.bfloat16),
            (ir.DataType.BOOL, torch.bool),
            (ir.DataType.COMPLEX128, torch.complex128),
            (ir.DataType.COMPLEX64, torch.complex64),
            (ir.DataType.FLOAT16, torch.float16),
            (ir.DataType.FLOAT, torch.float32),
            (ir.DataType.DOUBLE, torch.float64),
            (ir.DataType.FLOAT8E4M3FN, torch.float8_e4m3fn),
            (ir.DataType.FLOAT8E4M3FNUZ, torch.float8_e4m3fnuz),
            (ir.DataType.FLOAT8E5M2, torch.float8_e5m2),
            (ir.DataType.FLOAT8E5M2FNUZ, torch.float8_e5m2fnuz),
            (ir.DataType.FLOAT8E8M0, torch.float8_e8m0fnu),  # Requires PyTorch 2.7+
            (ir.DataType.INT16, torch.int16),
            (ir.DataType.INT32, torch.int32),
            (ir.DataType.INT64, torch.int64),
            (ir.DataType.INT8, torch.int8),
            (ir.DataType.UINT8, torch.uint8),
            (ir.DataType.UINT16, torch.uint16),
            (ir.DataType.UINT32, torch.uint32),
            (ir.DataType.UINT64, torch.uint64),
        ]
    )
    def test_to_torch_dtype(self, onnx_dtype: ir.DataType, expected_torch_dtype: torch.dtype):
        actual = tensor_adapters.to_torch_dtype(onnx_dtype)
        self.assertEqual(actual, expected_torch_dtype)

    @parameterized.parameterized.expand(
        [
            (torch.bfloat16, ir.DataType.BFLOAT16),
            (torch.bool, ir.DataType.BOOL),
            (torch.complex128, ir.DataType.COMPLEX128),
            (torch.complex64, ir.DataType.COMPLEX64),
            (torch.float16, ir.DataType.FLOAT16),
            (torch.float32, ir.DataType.FLOAT),
            (torch.float64, ir.DataType.DOUBLE),
            (torch.float8_e4m3fn, ir.DataType.FLOAT8E4M3FN),
            (torch.float8_e4m3fnuz, ir.DataType.FLOAT8E4M3FNUZ),
            (torch.float8_e5m2, ir.DataType.FLOAT8E5M2),
            (torch.float8_e5m2fnuz, ir.DataType.FLOAT8E5M2FNUZ),
            (torch.float8_e8m0fnu, ir.DataType.FLOAT8E8M0),  # Requires PyTorch 2.7+
            (torch.int16, ir.DataType.INT16),
            (torch.int32, ir.DataType.INT32),
            (torch.int64, ir.DataType.INT64),
            (torch.int8, ir.DataType.INT8),
            (torch.uint8, ir.DataType.UINT8),
            (torch.uint16, ir.DataType.UINT16),
            (torch.uint32, ir.DataType.UINT32),
            (torch.uint64, ir.DataType.UINT64),
        ]
    )
    def test_from_torch_dtype(
        self, torch_dtype: torch.dtype, expected_onnx_dtype: ir.DataType
    ):
        actual = tensor_adapters.from_torch_dtype(torch_dtype)
        self.assertEqual(actual, expected_onnx_dtype)

    def test_tofile_non_contiguous(self):
        base = torch.arange(0, 64, dtype=torch.int32).reshape(8, 8)
        sliced = base[:, ::2]  # Stride in last dim -> non-contiguous
        self.assertFalse(sliced.is_contiguous())
        tensor = tensor_adapters.TorchTensor(sliced)
        # Ensure bytes correspond to the contiguous clone inside implementation
        expected_manual = sliced.contiguous().numpy().tobytes()
        with tempfile.NamedTemporaryFile() as tmp:
            tensor.tofile(tmp)
            tmp.seek(0)
            data = tmp.read()
        self.assertEqual(data, expected_manual)
        self.assertEqual(tensor.tobytes(), expected_manual)


if __name__ == "__main__":
    unittest.main()
