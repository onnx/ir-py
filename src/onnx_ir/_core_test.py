# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N802
from __future__ import annotations

import copy
import io
import pathlib
import tempfile
import unittest
import unittest.mock
from typing import Any

import ml_dtypes
import numpy as np
import onnx
import onnx.external_data_helper
import parameterized
import torch

import onnx_ir as ir
from onnx_ir import _core, _type_casting


class TensorTest(unittest.TestCase):
    def test_initialize(self):
        tensor = _core.Tensor(
            np.random.rand(1, 2).astype(np.float32),
            dtype=ir.DataType.FLOAT,
            shape=_core.Shape((1, 2)),
            name="test",
        )
        self.assertEqual(tensor.name, "test")
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        self.assertEqual(tensor.shape, _core.Shape((1, 2)))
        np.testing.assert_array_equal(tensor, tensor)

    def test_init_raises_when_value_is_not_array(self):
        with self.assertRaises(TypeError):
            _core.Tensor(42)

    def test_init_requires_type_when_value_is_not_np_array(self):
        torch_tensor = torch.tensor(42)
        with self.assertRaises(ValueError):
            _core.Tensor(torch_tensor)

    @parameterized.parameterized.expand(
        [
            ("bfloat16", np.uint16, ir.DataType.BFLOAT16),
            (
                "float8e4m3fn",
                np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)})),
                ir.DataType.FLOAT8E4M3FN,
            ),
            ("float8e4m3fnuz", np.uint8, ir.DataType.FLOAT8E4M3FNUZ),
            ("float8e5m2", np.uint8, ir.DataType.FLOAT8E5M2),
            ("float8e5m2fnuz", np.uint8, ir.DataType.FLOAT8E5M2FNUZ),
            ("float8e8m0", np.uint8, ir.DataType.FLOAT8E8M0),
            ("int2", np.int8, ir.DataType.INT2),
            ("int2_uint8", np.uint8, ir.DataType.INT2),
            ("int4", np.int8, ir.DataType.INT4),
            ("int4_uint8", np.uint8, ir.DataType.INT4),
            ("uint2", np.uint8, ir.DataType.UINT2),
            ("uint4", np.uint8, ir.DataType.UINT4),
            ("float4e2m1", np.uint8, ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_init_with_non_native_numpy_dtype(self, _: str, np_dtype, dtype: ir.DataType):
        array = np.array([0b1, 0b11], dtype=np_dtype)
        tensor = _core.Tensor(array, dtype=dtype)
        self.assertEqual(tensor.dtype, dtype)
        np.testing.assert_array_equal(tensor, array.view(dtype.numpy()))

    def test_initialize_with_just_np_array(self):
        array = np.random.rand(1, 2)
        tensor = _core.Tensor(array)
        np.testing.assert_array_equal(tensor, array)

    @parameterized.parameterized.expand(
        [
            ("bfloat16", ml_dtypes.bfloat16(0.5)),
            ("float32", np.float32(0.5)),
            ("bool", np.bool(True)),
        ]
    )
    def test_initialize_with_np_number(self, _: str, number: np.generic):
        tensor = _core.Tensor(number)
        np.testing.assert_equal(tensor.numpy(), np.array(number), strict=True)

    def test_initialize_raises_when_numpy_dtype_doesnt_match(self):
        array = np.random.rand(1, 2).astype(np.float32)
        with self.assertRaises(TypeError):
            _core.Tensor(array, dtype=ir.DataType.INT64)

    def test_initialize_supports_custom_dtype(self):
        custom_dtype = np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)}))
        array = np.random.rand(1, 2).astype(custom_dtype)
        _core.Tensor(array, dtype=ir.DataType.FLOAT8E4M3FN)

    def test_initialize_raises_when_numpy_dtype_doesnt_match_custom_dtype(self):
        custom_dtype = np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)}))
        array = np.random.rand(1, 2).astype(custom_dtype)
        with self.assertRaises(TypeError):
            _core.Tensor(array, dtype=ir.DataType.BFLOAT16)

    def test_initialize_with_torch_tensor(self):
        array = np.random.rand(1, 2).astype(np.int64)
        np_tensor = _core.Tensor(array)
        torch_tensor = _core.Tensor(torch.tensor(array), dtype=ir.DataType.INT64)
        np.testing.assert_array_equal(torch_tensor, array)
        np.testing.assert_array_equal(torch_tensor, np_tensor)

    def test_dlpack_np_to_torch(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        torch_tensor = torch.from_dlpack(tensor)
        np.testing.assert_array_equal(torch_tensor, array)

    def test_dlpack_torch_to_np(self):
        torch_tensor = torch.rand(1, 2)
        tensor = _core.Tensor(torch_tensor, dtype=ir.DataType.FLOAT)
        array = np.from_dlpack(tensor)
        np.testing.assert_array_equal(array, torch_tensor)

    def test_repr(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertIsInstance(repr(tensor), str)

    def test_dtype_returns_data_type_enum(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)

    def test_shape(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertEqual(tensor.shape, _core.Shape((1, 2)))

    def test_numpy_returns_np_array(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        np.testing.assert_equal(tensor.numpy(), array)

    def test_numpy_returns_data_when_dtype_is_not_supported(self):
        array = np.array([1], dtype=np.uint8)
        tensor = _core.Tensor(array, dtype=ir.DataType.INT4)
        np.testing.assert_equal(tensor.numpy(), array)

    def test_tobytes(self):
        array = np.random.rand(1, 2).astype(np.float32)
        torch_tensor = torch.tensor(array)
        tensor = _core.Tensor(torch_tensor, dtype=ir.DataType.FLOAT)
        self.assertEqual(tensor.tobytes(), array.tobytes())

    def test_tobytes_returns_packed_data_for_int2(self):
        array = np.array([-2, -1, 0, 1, 1, -2, 1], dtype=np.int8)
        # Test array size not divisible by 4
        assert len(array) % 4 != 0
        tensor = _core.Tensor(array, dtype=ir.DataType.INT2)
        # -2, -1, 0, 1 => [0b10, 0b11, 0b00, 0b01] => 0b01001110 = 0x4E
        # 1, -2, 1, 0 (padding) => [0b01, 0b10, 0b01, 0b00] => 0b00011001 = 0x19
        self.assertEqual(tensor.tobytes(), b"\x4e\x19")

    def test_tobytes_returns_packed_data_for_int2_ml_dtypes(self):
        array = np.array([-2, -1, 0, 1, 1, -2, 1], dtype=ml_dtypes.int2)
        # Test array size not divisible by 4
        assert len(array) % 4 != 0
        tensor = _core.Tensor(array, dtype=ir.DataType.INT2)
        self.assertEqual(tensor.tobytes(), b"\x4e\x19")

    def test_tobytes_returns_packed_data_for_uint2(self):
        array = np.array([0, 1, 2, 3, 3, 2, 1], dtype=np.uint8)
        # Test array size not divisible by 4
        assert len(array) % 4 != 0
        tensor = _core.Tensor(array, dtype=ir.DataType.UINT2)
        # 0, 1, 2, 3 => 0b11100100 = 0xE4
        # 3, 2, 1, 0 (padding) => 0b00011011 = 0x1B
        self.assertEqual(tensor.tobytes(), b"\xe4\x1b")

    def test_tobytes_returns_packed_data_for_uint2_ml_dtypes(self):
        array = np.array([0, 1, 2, 3, 3, 2, 1], dtype=ml_dtypes.uint2)
        # Test array size not divisible by 4
        assert len(array) % 4 != 0
        tensor = _core.Tensor(array, dtype=ir.DataType.UINT2)
        self.assertEqual(tensor.tobytes(), b"\xe4\x1b")

    def test_tobytes_returns_packed_data_for_int4(self):
        array = np.array([-8, -1, 0, 1, 2, 7, 1], dtype=np.int8)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.INT4)
        self.assertEqual(tensor.tobytes(), b"\xf8\x10r\x01")

    def test_tobytes_returns_packed_data_for_int4_ml_dtypes(self):
        array = np.array([-8, -1, 0, 1, 2, 7, 1], dtype=ml_dtypes.int4)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.INT4)
        self.assertEqual(tensor.tobytes(), b"\xf8\x10r\x01")

    def test_tobytes_returns_packed_data_for_uint4(self):
        array = np.array([0, 1, 2, 7, 15], dtype=np.uint8)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.UINT4)
        self.assertEqual(tensor.tobytes(), b"\x10r\x0f")

    def test_tobytes_returns_packed_data_for_uint4_ml_dtypes(self):
        array = np.array([0, 1, 2, 7, 15], dtype=ml_dtypes.uint4)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.UINT4)
        self.assertEqual(tensor.tobytes(), b"\x10r\x0f")

    def test_tobytes_returns_packed_data_for_float4e2m1(self):
        array = np.array([0, 1, 2, 7, 15], dtype=np.uint8)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.FLOAT4E2M1)
        self.assertEqual(tensor.tobytes(), b"\x10r\x0f")

    def test_tobytes_returns_packed_data_for_float4e2m1_ml_dtypes(self):
        array = np.array([0, 1, 2, 7, 15], dtype=np.uint8)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.FLOAT4E2M1)
        self.assertEqual(tensor.tobytes(), b"\x10r\x0f")

    def test_metadata(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        tensor.meta["test"] = 1
        self.assertEqual(tensor.meta["test"], 1)
        tensor.metadata_props["test"] = "any string"
        self.assertEqual(tensor.metadata_props["test"], "any string")

    def test_tobytes_big_endian_handling(self):
        """Test that tobytes() correctly handles byte order conversion on big endian systems."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = _core.Tensor(array)

        # Mock _IS_LITTLE_ENDIAN to simulate big endian system
        with unittest.mock.patch("onnx_ir._core._IS_LITTLE_ENDIAN", False):
            result_bytes = tensor.tobytes()

        # Verify that the result is in little endian format regardless of system endianness
        expected_bytes = array.astype(array.dtype.newbyteorder("<")).tobytes()
        self.assertEqual(result_bytes, expected_bytes)

    def test_tobytes_packed_types_big_endian_handling(self):
        """Test that tobytes() handles byte order conversion for packed 4-bit types."""
        array = np.array([0, 1, 2, 7, 15], dtype=np.uint8)
        tensor = _core.Tensor(array, dtype=ir.DataType.UINT4)

        # Mock _IS_LITTLE_ENDIAN to simulate big endian system
        with unittest.mock.patch("onnx_ir._core._IS_LITTLE_ENDIAN", False):
            result_bytes = tensor.tobytes()

        # For packed types, the result should be the same as the packed data in little endian
        packed_array = _type_casting.pack_4bitx2(array.view(ir.DataType.UINT4.numpy()))
        expected_bytes = packed_array.astype(packed_array.dtype.newbyteorder("<")).tobytes()
        self.assertEqual(result_bytes, expected_bytes)

    def test_tofile_with_fileno_numpy_array(self):
        """Test tofile() with file-like object that has fileno() method and numpy array."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = _core.Tensor(array)

        with tempfile.NamedTemporaryFile() as temp_file:
            tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        self.assertEqual(result_bytes, array.tobytes())

    def test_tofile_with_fileno_non_numpy_array(self):
        """Test tofile() with file-like object that has fileno() method but non-numpy array."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        torch_tensor = torch.tensor(array)
        tensor = _core.Tensor(torch_tensor, dtype=ir.DataType.FLOAT)

        with tempfile.NamedTemporaryFile() as temp_file:
            tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Should use tobytes() path since _raw is not a numpy array
        self.assertEqual(result_bytes, tensor.tobytes())

    def test_tofile_without_fileno(self):
        """Test tofile() with file-like object without fileno() method."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = _core.Tensor(array)

        buffer = io.BytesIO()
        tensor.tofile(buffer)
        result_bytes = buffer.getvalue()

        self.assertEqual(result_bytes, tensor.tobytes())

    def test_tofile_packed_types_with_fileno(self):
        """Test tofile() with packed types and file with fileno()."""
        array = np.array([0, 1, 2, 7, 15], dtype=np.uint8)
        tensor = _core.Tensor(array, dtype=ir.DataType.UINT4)

        with tempfile.NamedTemporaryFile() as temp_file:
            tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Should be the same as tobytes() for packed types
        self.assertEqual(result_bytes, tensor.tobytes())

    def test_tofile_big_endian_handling_with_fileno(self):
        """Test tofile() big endian handling when file has fileno() method."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = _core.Tensor(array)

        with tempfile.NamedTemporaryFile() as temp_file:
            # Mock _IS_LITTLE_ENDIAN to simulate big endian system
            with unittest.mock.patch("onnx_ir._core._IS_LITTLE_ENDIAN", False):
                tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Should still produce little endian output
        expected_bytes = array.astype(array.dtype.newbyteorder("<")).tobytes()
        self.assertEqual(result_bytes, expected_bytes)

    def test_tofile_empty_tensor(self):
        """Test tofile() with an empty tensor."""
        # Test with numpy empty array
        empty_array = np.array([], dtype=np.float32)
        tensor = _core.Tensor(empty_array)

        with tempfile.NamedTemporaryFile() as temp_file:
            tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Empty tensor should write empty bytes
        self.assertEqual(result_bytes, b"")
        self.assertEqual(result_bytes, tensor.tobytes())

    def test_tofile_empty_tensor_torch(self):
        """Test tofile() with an empty torch tensor."""
        # Test with torch empty tensor
        empty_torch_tensor = torch.tensor([], dtype=torch.float32)
        tensor = _core.Tensor(empty_torch_tensor, dtype=ir.DataType.FLOAT)

        with tempfile.NamedTemporaryFile() as temp_file:
            tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Empty tensor should write empty bytes
        self.assertEqual(result_bytes, b"")
        self.assertEqual(result_bytes, tensor.tobytes())

    def test_tofile_consecutive_writes_same_file(self):
        """Test tofile() with three tensors writing consecutively to the same file."""
        # Create three different tensors
        array1 = np.array([1.0, 2.0], dtype=np.float32)
        array2 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        array3 = np.array([6.0], dtype=np.float32)

        tensor1 = _core.Tensor(array1)
        tensor2 = _core.Tensor(array2)
        tensor3 = _core.Tensor(array3)

        with tempfile.NamedTemporaryFile() as temp_file:
            # Write three tensors consecutively
            tensor1.tofile(temp_file)
            tensor2.tofile(temp_file)
            tensor3.tofile(temp_file)

            # Read the entire file
            temp_file.seek(0)
            result_bytes = temp_file.read()

        # The file should contain all three tensors' data concatenated
        expected_bytes = array1.tobytes() + array2.tobytes() + array3.tobytes()
        self.assertEqual(result_bytes, expected_bytes)

        # Verify each part
        bytes1 = array1.tobytes()
        bytes2 = array2.tobytes()
        bytes3 = array3.tobytes()

        self.assertEqual(result_bytes[: len(bytes1)], bytes1)
        self.assertEqual(result_bytes[len(bytes1) : len(bytes1) + len(bytes2)], bytes2)
        self.assertEqual(result_bytes[len(bytes1) + len(bytes2) :], bytes3)

    def test_tofile_consecutive_writes_mixed_types(self):
        """Test tofile() with mixed tensor types (numpy and torch) writing consecutively."""
        # Create tensors with different underlying types
        numpy_array = np.array([1.0, 2.0], dtype=np.float32)
        torch_array = np.array([3.0, 4.0], dtype=np.float32)
        torch_tensor_raw = torch.tensor(torch_array)

        numpy_tensor = _core.Tensor(numpy_array)
        torch_tensor = _core.Tensor(torch_tensor_raw, dtype=ir.DataType.FLOAT)

        with tempfile.NamedTemporaryFile() as temp_file:
            # Write numpy tensor first, then torch tensor
            numpy_tensor.tofile(temp_file)
            torch_tensor.tofile(temp_file)

            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Should be equivalent to concatenating their tobytes()
        expected_bytes = numpy_tensor.tobytes() + torch_tensor.tobytes()
        self.assertEqual(result_bytes, expected_bytes)

    def test_tofile_consecutive_writes_packed_types(self):
        """Test tofile() with packed tensor types writing consecutively."""
        # Create packed tensors
        array1 = np.array([0, 1, 2, 7], dtype=np.uint8)
        array2 = np.array([8, 9, 10, 15], dtype=np.uint8)

        tensor1 = _core.Tensor(array1, dtype=ir.DataType.UINT4)
        tensor2 = _core.Tensor(array2, dtype=ir.DataType.UINT4)

        with tempfile.NamedTemporaryFile() as temp_file:
            # Write packed tensors consecutively
            tensor1.tofile(temp_file)
            tensor2.tofile(temp_file)

            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Should be equivalent to concatenating their tobytes()
        expected_bytes = tensor1.tobytes() + tensor2.tobytes()
        self.assertEqual(result_bytes, expected_bytes)


def _to_external_tensor(tensor_proto, dir: str, filename: str):
    onnx.external_data_helper.set_external_data(tensor_proto, location=filename)
    path = pathlib.Path(dir) / filename
    with open(path, "wb") as f:
        f.write(tensor_proto.raw_data)
    tensor_proto.ClearField("raw_data")
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL


class ExternalTensorTest(unittest.TestCase):
    """Test the memory mapped external tensor class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.external_data_name = "test_model.bin"
        self.base_path = self.temp_dir.name
        self.data = np.random.rand(2, 42).astype(np.float32)
        self.data_float16 = np.random.rand(2, 42).astype(np.float16)
        self.model = self._simple_model_with_external(
            self.base_path, self.external_data_name, self.data
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _simple_model_with_external(
        self, base_path: str, external_data_name: str, data: np.ndarray
    ) -> onnx.ModelProto:
        input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None])
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None])
        raw_data = data.tobytes()
        tensor = onnx.helper.make_tensor(
            "input", onnx.TensorProto.FLOAT, data.shape, raw_data, raw=True
        )
        raw_data2 = self.data_float16.tobytes()
        tensor2 = onnx.helper.make_tensor(
            "input2", onnx.TensorProto.FLOAT16, data.shape, raw_data2, raw=True
        )
        onnx.external_data_helper.set_external_data(
            tensor, external_data_name, offset=0, length=len(raw_data)
        )
        onnx.external_data_helper.set_external_data(
            tensor2, external_data_name, offset=len(raw_data), length=len(raw_data2)
        )

        node = onnx.helper.make_node("Identity", inputs=["input"], outputs=["output"])
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [node], "test_graph", [input], [output], initializer=[tensor, tensor2]
            )
        )
        tensor.ClearField("raw_data")
        tensor2.ClearField("raw_data")
        # Save the data to disk
        with open(pathlib.Path(base_path) / external_data_name, "wb") as f:
            f.write(raw_data)
            f.write(raw_data2)
        return model

    def test_initialize(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        np.testing.assert_equal(tensor, self.data)
        # Ensure repeated reads are consistent
        np.testing.assert_equal(tensor, self.data)

    def test_release_does_not_invalidate_tensor(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        self.assertEqual(tensor.tobytes(), self.data.tobytes())
        # Release tensor
        tensor.release()
        self.assertEqual(tensor.raw, None)
        # Tensor can be re-loaded after release
        self.assertEqual(tensor.tobytes(), self.data.tobytes())

    def test_initialize_with_relative_path(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            name="input",
            shape=_core.Shape(external_tensor.dims),
            base_dir=pathlib.Path(self.base_path),
        )
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        np.testing.assert_equal(tensor, self.data)
        # Ensure repeated reads are consistent
        np.testing.assert_equal(tensor, self.data)

    def test_totypes_returns_correct_data_in(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        external_tensor2 = self.model.graph.initializer[1]
        external_info2 = onnx.external_data_helper.ExternalDataInfo(external_tensor2)
        tensor2 = _core.ExternalTensor(
            external_info2.location,
            offset=external_info2.offset,
            length=external_info2.length,
            dtype=ir.DataType.FLOAT16,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor2.dims),
        )
        self.assertEqual(tensor.tobytes(), self.data.tobytes())
        self.assertEqual(tensor2.tobytes(), self.data_float16.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(tensor.tobytes(), self.data.tobytes())
        self.assertEqual(tensor2.tobytes(), self.data_float16.tobytes())

    @parameterized.parameterized.expand(
        [
            ("FLOAT", ir.DataType.FLOAT),
            ("BOOL", ir.DataType.BOOL),
            ("FLOAT16", ir.DataType.FLOAT16),
            ("DOUBLE", ir.DataType.DOUBLE),
        ]
    )
    def test_external_tensor(self, _: str, dtype: ir.DataType):
        expected_array = np.array(
            [[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]]
        ).astype(dtype.numpy())
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array, dtype=dtype))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    def test_external_tensor_bfloat16(self):
        expected_array = np.array(
            [[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]]
        ).astype(ml_dtypes.bfloat16)
        tensor_proto = ir.serde.serialize_tensor(
            ir.Tensor(expected_array.view(np.uint16), dtype=ir.DataType.BFLOAT16)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(
                tensor.numpy().view(ml_dtypes.bfloat16), expected_array
            )
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    @parameterized.parameterized.expand(
        [
            (
                "FLOAT8E4M3FN",
                ir.DataType.FLOAT8E4M3FN,
                ml_dtypes.float8_e4m3fn,
            ),
            (
                "FLOAT8E4M3FNUZ",
                ir.DataType.FLOAT8E4M3FNUZ,
                ml_dtypes.float8_e4m3fnuz,
            ),
            (
                "FLOAT8E5M2",
                ir.DataType.FLOAT8E5M2,
                ml_dtypes.float8_e5m2,
            ),
            (
                "FLOAT8E5M2FNUZ",
                ir.DataType.FLOAT8E5M2FNUZ,
                ml_dtypes.float8_e5m2fnuz,
            ),
            (
                "FLOAT8E8M0",
                ir.DataType.FLOAT8E8M0,
                ml_dtypes.float8_e8m0fnu,
            ),
        ]
    )
    def test_external_tensor_float8(self, _: str, dtype: ir.DataType, np_dtype):
        # FLOAT8E8M0 has different precision characteristics (8 exponent bits, 0 mantissa bits)
        # It can only represent powers of 2 and special values
        if dtype == ir.DataType.FLOAT8E8M0:
            expected_array = np.array(
                [[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]]
            ).astype(np_dtype)
            tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array, dtype=dtype))
        else:
            expected_array = np.array(
                [[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 40.0, 2.0]]
            ).astype(np_dtype)
            tensor_proto = ir.serde.serialize_tensor(
                ir.Tensor(expected_array.view(np.uint8), dtype=dtype)
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy().view(np_dtype), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    @parameterized.parameterized.expand(
        [
            ("INT8", ir.DataType.INT8),
            ("INT16", ir.DataType.INT16),
            ("INT32", ir.DataType.INT32),
            ("INT64", ir.DataType.INT64),
            ("INT4", ir.DataType.INT4),
        ]
    )
    def test_external_tensor_int(self, _: str, dtype: ir.DataType):
        expected_array = np.array([[-8, 0, 1, 7]]).astype(dtype.numpy())
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array, dtype=dtype))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    @parameterized.parameterized.expand(
        [
            ("UINT8", ir.DataType.UINT8),
            ("UINT16", ir.DataType.UINT16),
            ("UINT32", ir.DataType.UINT32),
            ("UINT64", ir.DataType.UINT64),
            ("UINT4", ir.DataType.UINT4),
        ]
    )
    def test_external_tensor_uint(self, _: str, dtype: ir.DataType):
        expected_array = np.array([[0, 1, 15]]).astype(dtype.numpy())
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array, dtype=dtype))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    @parameterized.parameterized.expand(
        [
            ("COMPLEX64", np.complex64),
            ("COMPLEX128", np.complex128),
        ]
    )
    def test_external_tensor_complex(self, _: str, np_dtype: np.dtype):
        expected_array = np.array([[0.0 + 1j, 0.2 - 1j, 0.3]], dtype=np_dtype)
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    def test_external_tensor_float4e2m1(self):
        expected_array = np.array([0, 1, 2, 7, 15]).view(ml_dtypes.float4_e2m1fn)
        tensor_proto = ir.serde.serialize_tensor(
            ir.Tensor(expected_array, dtype=ir.DataType.FLOAT4E2M1)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    def test_external_tensor_empty_tensor(self):
        expected_array = np.array([], dtype=np.float32)
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    def test_tofile_basic(self):
        """Test ExternalTensor.tofile() with basic functionality."""
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )

        # Test writing to BytesIO
        output = io.BytesIO()
        tensor.tofile(output)
        output.seek(0)
        written_data = output.read()

        # Verify the written data matches expected
        expected_data = self.data.tobytes()
        self.assertEqual(written_data, expected_data)

    def test_tofile_with_offset(self):
        """Test ExternalTensor.tofile() with offset handling."""
        # Use the second tensor which has an offset
        external_tensor2 = self.model.graph.initializer[1]
        external_info2 = onnx.external_data_helper.ExternalDataInfo(external_tensor2)
        tensor2 = _core.ExternalTensor(
            external_info2.location,
            offset=external_info2.offset,
            length=external_info2.length,
            dtype=ir.DataType.FLOAT16,
            base_dir=self.base_path,
            name="input2",
            shape=_core.Shape(external_tensor2.dims),
        )

        # Test writing to BytesIO
        output = io.BytesIO()
        tensor2.tofile(output)
        output.seek(0)
        written_data = output.read()

        # Verify the written data matches expected
        expected_data = self.data_float16.tobytes()
        self.assertEqual(written_data, expected_data)

    def test_tofile_with_file_object(self):
        """Test ExternalTensor.tofile() writing to a file."""
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )

        with tempfile.NamedTemporaryFile() as temp_file:
            tensor.tofile(temp_file)
            temp_file.seek(0)
            written_data = temp_file.read()

            # Verify the written data matches expected
            expected_data = self.data.tobytes()
            self.assertEqual(written_data, expected_data)

    def test_tofile_empty_tensor(self):
        """Test ExternalTensor.tofile() with empty tensor."""
        expected_array = np.array([], dtype=np.float32)
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)

            self.assertIsInstance(tensor, _core.ExternalTensor)

            # Test writing empty tensor to BytesIO
            output = io.BytesIO()
            tensor.tofile(output)
            output.seek(0)
            written_data = output.read()

            # Should write empty bytes
            self.assertEqual(written_data, b"")
            del tensor

    def test_tofile_large_chunks(self):
        """Test ExternalTensor.tofile() handles large data with chunking."""
        # Create a larger array to test the chunking mechanism
        large_data = np.random.rand(1100, 1100).astype(np.float32)
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(large_data))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "large_tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)

            self.assertIsInstance(tensor, _core.ExternalTensor)

            # Test writing to BytesIO
            output = io.BytesIO()
            tensor.tofile(output)
            output.seek(0)
            written_data = output.read()

            # Verify the written data matches expected
            expected_data = large_data.tobytes()
            self.assertEqual(written_data, expected_data)
            del tensor

    def test_tofile_invalidated_tensor_raises_error(self):
        """Test that tofile() raises error on invalidated tensor."""
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )

        # Invalidate the tensor
        tensor.invalidate()

        # Should raise ValueError when trying to write
        output = io.BytesIO()
        with self.assertRaisesRegex(ValueError, "invalidated"):
            tensor.tofile(output)

    def test_tofile_consecutive_writes(self):
        """Test ExternalTensor.tofile() with consecutive writes to same file."""
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )

        # Write tensor three times consecutively to BytesIO
        output = io.BytesIO()
        tensor.tofile(output)
        tensor.tofile(output)
        tensor.tofile(output)

        output.seek(0)
        written_data = output.read()

        # Should have written the data three times
        expected_data = self.data.tobytes()
        expected_triple = expected_data + expected_data + expected_data
        self.assertEqual(written_data, expected_triple)


class SymbolicDimTest(unittest.TestCase):
    def test_init_raises_when_value_is_int(self):
        # Static dimensions should be python integers
        with self.assertRaises(TypeError):
            _core.SymbolicDim(42)

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_equality_with_other_dimensions(self, _: str, value: Any):
        dim1 = _core.SymbolicDim(value)
        dim2 = _core.SymbolicDim(value)
        self.assertEqual(dim1, dim2)

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_equality_with_python_values(self, _: str, value: Any):
        dim = _core.SymbolicDim(value)
        self.assertEqual(dim, value)
        self.assertIn(value, [dim])
        self.assertIn(dim, [value])

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_it_is_hashable(self, _: str, value: Any):
        dim = _core.SymbolicDim(value)
        self.assertEqual(hash(dim), hash(value))
        self.assertIn(dim, {dim})
        self.assertIn(dim, {value})


class ShapeTest(unittest.TestCase):
    def test_init_raises_when_denotations_and_dims_have_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "denotations"):
            _core.Shape([42], ["DATA_CHANNEL", "BATCH"])

    def test_int_dimensions_are_python_ints(self):
        shape = _core.Shape([42])
        self.assertIsInstance(shape[0], int)

    def test_str_dimensions_are_symbolic_dims(self):
        shape = _core.Shape(["any string"])
        self.assertIsInstance(shape[0], _core.SymbolicDim)

    def test_none_dimensions_are_symbolic_dims(self):
        shape = _core.Shape([None])
        self.assertIsInstance(shape[0], _core.SymbolicDim)

    def test_init_raises_when_dims_is_not_a_list(self):
        with self.assertRaises(TypeError):
            _core.Shape(42)

    def test_init_converts_np_shape_to_tuple(self):
        dims = np.array([42, 42])
        shape = _core.Shape(dims)
        self.assertEqual(shape.dims, tuple(dims))

    def test_init_converts_np_int_to_python_int(self):
        dims = [np.int32(42)]
        shape = _core.Shape(dims)
        self.assertIsInstance(shape[0], int)
        self.assertNotIsInstance(shape[0], np.int32)
        self.assertIsInstance(shape.dims[0], int)

    @parameterized.parameterized.expand(
        [
            ("empty", (), ()),
            ("1d", (42,), (42,)),
            ("int", (42, 42), (42, 42)),
            ("str", ("any string", "any string"), ("any string", "any string")),
            ("None", (None, None), (None, None)),
        ]
    )
    def test_eq_with_other_shapes(
        self, _: str, dims_1: tuple[Any, ...], dims_2: tuple[Any, ...]
    ):
        shape_1 = _core.Shape(dims_1)
        shape_2 = _core.Shape(dims_2)
        self.assertEqual(shape_1, shape_2)

    @parameterized.parameterized.expand(
        [
            ("empty", ()),
            ("1d", (42,)),
            ("int", (42, 42)),
            ("str", ("any string", "any string")),
            ("None", (None, None)),
        ]
    )
    def test_eq_with_tuple(self, _: str, dims: tuple[Any, ...]):
        shape = _core.Shape(dims)
        self.assertEqual(shape, dims)

    @parameterized.parameterized.expand(
        [
            ("empty", []),
            (
                "1d",
                [
                    42,
                ],
            ),
            ("int", [42, 42]),
            ("str", ["any string", "any string"]),
            ("None", [None, None]),
        ]
    )
    def test_eq_with_list(self, _: str, dims: list[Any]):
        shape = _core.Shape(dims)
        self.assertEqual(shape, dims)

    def test_eq_with_np_shape(self):
        dims = (42,)
        array = np.zeros(dims)
        shape = _core.Shape(dims)
        self.assertEqual(shape, array.shape)

    @parameterized.parameterized.expand(
        [
            ("empty", (), (1,)),
            ("d", (42,), (0,)),
            ("rank", (42, 42), (42, 42, 42)),
            ("str", ("any string",), (42,)),
            ("None", (None, None), (None, 42)),
        ]
    )
    def test_ne_with_other_shapes(
        self, _: str, dims_1: tuple[Any, ...], dims_2: tuple[Any, ...]
    ):
        shape_1 = _core.Shape(dims_1)
        shape_2 = _core.Shape(dims_2)
        self.assertNotEqual(shape_1, shape_2)

    def test_ne_with_random_object(self):
        shape = _core.Shape((42,))
        self.assertNotEqual(shape, 42)

    def test_setitem_raises_when_shape_is_frozen(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",), frozen=True)
        with self.assertRaisesRegex(TypeError, "frozen"):
            shape[0] = 1

        with self.assertRaisesRegex(TypeError, "frozen"):
            shape[0] = "some_string"

    def test_getitem(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",))
        self.assertEqual(shape[0], 42)

    def test_getitem_accepts_a_slice(self):
        shape = _core.Shape([1, 2, 3, 4])
        self.assertEqual(shape[1:3], (2, 3))

    @parameterized.parameterized.expand(
        [
            ("int", 42),
            ("str", "any string"),
            ("None", None),
            ("SymbolicDim", _core.SymbolicDim("any string")),
        ]
    )
    def test_setitem(self, _: str, value):
        shape = _core.Shape([0])
        shape[0] = value
        dim = shape[0]
        if isinstance(dim, _core.SymbolicDim):
            self.assertEqual(dim.value, value)
        else:
            self.assertEqual(dim, value)

    def test_len(self):
        shape = _core.Shape([42, "any string"])
        self.assertEqual(len(shape), 2)

    def test_get_denotation(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",))
        self.assertEqual(shape.get_denotation(0), "DATA_CHANNEL")

    def test_set_denotation(self):
        shape = _core.Shape([42, 0], ["DATA_CHANNEL", "BATCH"])
        shape.set_denotation(1, "UPDATED")
        self.assertEqual(shape.get_denotation(1), "UPDATED")

    def test_set_denotation_is_still_possible_when_shape_is_frozen(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",), frozen=True)
        shape.set_denotation(0, "UPDATED")
        self.assertEqual(shape.get_denotation(0), "UPDATED")

    def test_is_static(self):
        dim_from_numpy = np.array([42]).shape[0]
        np_int = np.int32(42)
        shape = _core.Shape([42, "any string", dim_from_numpy, np_int])
        self.assertTrue(shape.is_static(0))
        self.assertFalse(shape.is_static(1))
        self.assertTrue(shape.is_static(2))
        self.assertTrue(shape.is_static(3))
        self.assertFalse(shape.is_static())

    def test_is_static_raises_when_index_out_of_range(self):
        shape = _core.Shape([42])
        with self.assertRaises(IndexError):
            shape.is_static(1)

    def test_is_static_on_whole_shape(self):
        shape = _core.Shape([42, "any string"])
        self.assertFalse(shape.is_static())
        shape = _core.Shape([42, 42])
        self.assertTrue(shape.is_static())

    def test_is_static_on_empty_shape(self):
        shape = _core.Shape(())
        self.assertTrue(shape.is_static())

    def test_is_dynamic(self):
        dim_from_numpy = np.array([42]).shape[0]
        np_int = np.int32(42)
        shape = _core.Shape([42, "any string", dim_from_numpy, np_int])
        self.assertFalse(shape.is_dynamic(0))
        self.assertTrue(shape.is_dynamic(1))
        self.assertFalse(shape.is_dynamic(2))
        self.assertFalse(shape.is_dynamic(3))
        self.assertTrue(shape.is_dynamic())

    def test_is_dynamic_raises_when_index_out_of_range(self):
        shape = _core.Shape([42])
        with self.assertRaises(IndexError):
            shape.is_dynamic(1)

    def test_is_dynamic_on_whole_shape(self):
        shape = _core.Shape([42, "any string"])
        self.assertTrue(shape.is_dynamic())
        shape = _core.Shape([42, 42])
        self.assertFalse(shape.is_dynamic())

    def test_is_dynamic_on_empty_shape(self):
        shape = _core.Shape(())
        self.assertFalse(shape.is_dynamic())

    def test_is_unknown_dim(self):
        shape = _core.Shape([42, None, "any string", None])
        self.assertFalse(shape.is_unknown_dim(0))  # integer dimension is not unknown
        self.assertTrue(shape.is_unknown_dim(1))  # None dimension is unknown
        self.assertFalse(
            shape.is_unknown_dim(2)
        )  # string dimension is not unknown (it's symbolic)
        self.assertTrue(shape.is_unknown_dim(3))  # None dimension is unknown

    def test_is_unknown_dim_raises_when_index_out_of_range(self):
        shape = _core.Shape([42])
        with self.assertRaises(IndexError):
            shape.is_unknown_dim(1)

    def test_has_unknown_dim(self):
        # Shape with unknown dimensions
        shape = _core.Shape([42, None, "any string"])
        self.assertTrue(shape.has_unknown_dim())

        # Shape with only None dimensions
        shape = _core.Shape([None, None])
        self.assertTrue(shape.has_unknown_dim())

        # Shape with no unknown dimensions (static and symbolic)
        shape = _core.Shape([42, "any string", 64])
        self.assertFalse(shape.has_unknown_dim())

        # Shape with only static dimensions
        shape = _core.Shape([42, 64, 128])
        self.assertFalse(shape.has_unknown_dim())

        # Shape with only symbolic dimensions
        shape = _core.Shape(["batch", "height", "width"])
        self.assertFalse(shape.has_unknown_dim())

    def test_has_unknown_dim_on_empty_shape(self):
        shape = _core.Shape(())
        self.assertFalse(shape.has_unknown_dim())


class ValueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = _core.Value(name="v0")
        self.v1 = _core.Value(name="v1")
        self.node = _core.Node(
            "test", "TestOp", inputs=(self.v0, self.v1, self.v1), num_outputs=2
        )

    def test_initialize(self):
        _ = _core.Value()

    def test_it_is_hashable(self):
        value = _core.Value()
        self.assertIsInstance(hash(value), int)
        self.assertIn(value, {value})

    def test_meta(self):
        value = _core.Value()
        value.meta["test"] = 1
        self.assertEqual(value.meta["test"], 1)
        value.metadata_props["test"] = "any string"
        self.assertEqual(value.metadata_props["test"], "any string")

    def test_producer(self):
        self.assertEqual(self.v0.producer(), None)
        self.assertEqual(self.v1.producer(), None)
        self.assertEqual(self.node.outputs[0].producer(), self.node)
        self.assertEqual(self.node.outputs[1].producer(), self.node)

    def test_consumers(self):
        self.assertEqual(self.v0.consumers(), (self.node,))
        self.assertEqual(self.v1.consumers(), (self.node,))
        self.assertEqual(self.node.outputs[0].consumers(), ())
        self.assertEqual(self.node.outputs[1].consumers(), ())

    def test_name_setter_updates_const_value_name(self):
        """Test that setting a Value's name also updates the const_value's name if it exists."""
        tensor = ir.tensor([1, 2, 3], name="original_tensor_name")
        value = _core.Value(name="original_value_name", const_value=tensor)

        # Verify initial state
        self.assertEqual(value.name, "original_value_name")
        self.assertEqual(value.const_value.name, "original_tensor_name")

        # Update the value's name and verify const_value name is also updated
        value.name = "new_name"
        self.assertEqual(value.name, "new_name")
        self.assertEqual(value.const_value.name, "new_name")

        # Test setting name to None
        value.name = None
        self.assertIsNone(value.name)
        self.assertIsNone(value.const_value.name)

    def test_name_setter_without_const_value(self):
        """Test that setting a Value's name works normally when no const_value exists."""
        value = _core.Value(name="original_name")

        # Verify initial state
        self.assertEqual(value.name, "original_name")
        self.assertIsNone(value.const_value)

        # Update the name
        value.name = "new_name"
        self.assertEqual(value.name, "new_name")

        # Set to None
        value.name = None
        self.assertIsNone(value.name)

    def test_initializer_name_setter_raises_when_set_to_none(self):
        """Test that setting an initializer value's name to None raises ValueError."""
        tensor = ir.tensor([1, 2, 3])
        value = _core.Value(name="initializer1", const_value=tensor)
        _core.Graph(inputs=(), outputs=(), nodes=(), initializers=[value])

        # Verify the value is an initializer
        self.assertTrue(value.is_initializer())

        # Attempt to set name to None should raise ValueError
        with self.assertRaisesRegex(
            ValueError,
            "Initializer value cannot have name set to None. Please pop\\(\\) the value from initializers first",
        ):
            value.name = None

        # Name should remain unchanged
        self.assertEqual(value.name, "initializer1")

    def test_initializer_name_setter_updates_graph_initializers_dict(self):
        """Test that renaming an initializer value updates the graph's initializers dictionary."""
        tensor = ir.tensor([1, 2, 3])
        value = _core.Value(name="old_name", const_value=tensor)
        graph = _core.Graph(inputs=(), outputs=(), nodes=(), initializers=[value])

        # Verify initial state
        self.assertTrue(value.is_initializer())
        self.assertIn("old_name", graph.initializers)
        self.assertIs(graph.initializers["old_name"], value)
        self.assertEqual(value.name, "old_name")

        # Rename the value and verify the graph's initializers dict is updated
        value.name = "new_name"

        # Old key should be removed, new key should be added
        self.assertNotIn("old_name", graph.initializers)
        self.assertIn("new_name", graph.initializers)
        self.assertIs(graph.initializers["new_name"], value)
        self.assertEqual(value.name, "new_name")
        self.assertEqual(value.const_value.name, "new_name")

    def test_non_initializer_name_setter_works_normally(self):
        """Test that name changes work normally for values that are not initializers."""
        # Test regular value (not part of any graph)
        tensor = ir.tensor([1, 2, 3])
        value = _core.Value(name="original_name", const_value=tensor)

        self.assertFalse(value.is_initializer())

        # Should be able to change name without issues
        value.name = "new_name"
        self.assertEqual(value.name, "new_name")
        self.assertEqual(value.const_value.name, "new_name")

        # Should be able to set to None without issues
        value.name = None
        self.assertIsNone(value.name)
        self.assertIsNone(value.const_value.name)

        # Test graph input
        input_value = _core.Value(name="input1")
        _core.Graph(inputs=[input_value], outputs=(), nodes=())

        self.assertTrue(input_value.is_graph_input())
        self.assertFalse(input_value.is_initializer())

        # Should be able to rename input without issues
        input_value.name = "renamed_input"
        self.assertEqual(input_value.name, "renamed_input")

    def test_merge_shapes_with_equal_dimensions(self):
        value = _core.Value(shape=_core.Shape([1, 2, 3]))
        shape2 = _core.Shape([1, 2, 3])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_merge_shapes_with_symbolic_dimensions_equal(self):
        value = _core.Value(shape=_core.Shape(["batch", "seq_len", 3]))
        shape2 = _core.Shape(["batch", "seq_len", 3])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, ["batch", "seq_len", 3])

    def test_merge_shapes_prefers_concrete_over_symbolic_from_shape1(self):
        value = _core.Value(shape=_core.Shape([64, 128, 3]))
        shape2 = _core.Shape(["batch", "seq_len", 3])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, [64, 128, 3])

    def test_merge_shapes_prefers_concrete_over_symbolic_from_shape2(self):
        value = _core.Value(shape=_core.Shape(["batch", "seq_len", 3]))
        shape2 = _core.Shape([64, 128, 3])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, [64, 128, 3])

    def test_merge_shapes_with_none_dimensions_prefers_named_symbolic(self):
        value = _core.Value(shape=_core.Shape([None, 128, 3]))
        shape2 = _core.Shape(["batch", 128, 3])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, ["batch", 128, 3])

    def test_merge_shapes_with_none_dimensions_keeps_named_symbolic_from_shape1(self):
        value = _core.Value(shape=_core.Shape(["batch", 128, 3]))
        shape2 = _core.Shape([None, 128, 3])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, ["batch", 128, 3])

    def test_merge_shapes_with_conflicting_concrete_dimensions_raises(self):
        value = _core.Value(shape=_core.Shape([64, 128, 3]))
        shape2 = _core.Shape([32, 128, 3])
        with self.assertRaisesRegex(ValueError, "Conflicting dimensions"):
            value.merge_shapes(shape2)

    def test_merge_shapes_with_mixed_dimensions(self):
        value = _core.Value(shape=_core.Shape([64, "seq_len", 3, None]))
        shape2 = _core.Shape(["batch", 128, 3, "hidden"])
        value.merge_shapes(shape2)
        # 64 (concrete) wins over "batch" (symbolic)
        # 128 (concrete) wins over "seq_len" (symbolic)
        # 3 (equal) stays the same
        # "hidden" (named symbolic) wins over None
        self.assertEqual(value.shape[0], 64)
        self.assertEqual(value.shape[1], 128)
        self.assertEqual(value.shape[2], 3)
        self.assertEqual(value.shape[3], "hidden")

    def test_merge_shapes_with_different_named_symbolic_dimensions_takes_shape1(self):
        # When merging two shapes with different named symbolic dimensions,
        # the first shape's dimension is taken following the documented precedence rule.
        value = _core.Value(shape=_core.Shape(["batch", 128]))
        shape2 = _core.Shape(["sequence", 128])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, ["batch", 128])

    def test_merge_shapes_with_empty_shapes(self):
        value = _core.Value(shape=_core.Shape([]))
        shape2 = _core.Shape([])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, [])

    def test_merge_shapes_raises_on_different_ranks(self):
        value = _core.Value(shape=_core.Shape([1, 2, 3]))
        shape2 = _core.Shape([1, 2])
        with self.assertRaisesRegex(ValueError, "same rank"):
            value.merge_shapes(shape2)

    def test_merge_shapes_when_other_is_none_keeps_shape_unchanged(self):
        original_shape = _core.Shape([1, 2, 3])
        value = _core.Value(shape=original_shape)
        value.merge_shapes(None)
        self.assertEqual(value.shape, [1, 2, 3])
        # Verify it's still the same instance when not frozen
        self.assertIs(value.shape, original_shape)

    def test_merge_shapes_modifies_value_shape_in_place_if_not_frozen(self):
        original_shape = _core.Shape([1, 2, 3])
        value = _core.Value(shape=original_shape)
        shape2 = _core.Shape([1, 2, 3])
        value.merge_shapes(shape2)
        # Verify the value's shape is updated in place
        self.assertIs(value.shape, original_shape)

    def test_merge_shapes_when_value_has_no_shape_creates_copy(self):
        value = _core.Value()
        shape2 = _core.Shape([1, 2, 3])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, [1, 2, 3])
        # Verify it's a copy, not the same instance
        self.assertIsNot(value.shape, shape2)

    def test_merge_shapes_with_frozen_shape_creates_new_shape(self):
        frozen_shape = _core.Shape([1, 2, 3], frozen=True)
        value = _core.Value(shape=frozen_shape)
        shape2 = _core.Shape([1, 2, 4])
        # Should raise because conflicting concrete dimensions
        with self.assertRaisesRegex(ValueError, "Conflicting dimensions"):
            value.merge_shapes(shape2)

    def test_merge_shapes_with_symbolic_dim_objects(self):
        sym1 = _core.SymbolicDim("batch")
        sym2 = _core.SymbolicDim("batch")
        value = _core.Value(shape=_core.Shape([sym1, 128]))
        shape2 = _core.Shape([sym2, 128])
        value.merge_shapes(shape2)
        self.assertEqual(value.shape, ["batch", 128])

    def test_merge_shapes_with_none_symbolic_dims(self):
        value = _core.Value(shape=_core.Shape([None, None, 3]))
        shape2 = _core.Shape([None, None, 3])
        value.merge_shapes(shape2)
        self.assertTrue(value.shape.is_unknown_dim(0))
        self.assertTrue(value.shape.is_unknown_dim(1))
        self.assertEqual(value.shape[2], 3)

    # TODO(justinchuby): Test all methods


class SetValueMagicHandlerTest(unittest.TestCase):
    """Tests for the set_value_magic_handler function and Value arithmetic operations."""

    def setUp(self):
        """Create test values for arithmetic operations."""
        self.value1 = ir.Value(name="value1")
        self.value2 = ir.Value(name="value2")

    def tearDown(self):
        """Reset the handler after each test."""
        ir.set_value_magic_handler(None)

    def test_raises_error_when_no_handler_set(self):
        """Test that arithmetic operations raise an error when no handler is set."""
        with self.assertRaises(ValueError) as cm:
            _ = self.value1 + self.value2
        self.assertIn("No magic handler is set", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = self.value1 - self.value2
        self.assertIn("No magic handler is set", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = self.value1 * self.value2
        self.assertIn("No magic handler is set", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = self.value1 / self.value2
        self.assertIn("No magic handler is set", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = -self.value1
        self.assertIn("No magic handler is set", str(cm.exception))

    def test_sets_and_returns_old_handler(self):
        """Test that the function properly sets and returns the old handler."""
        # Handler should be None initially
        self.assertIsNone(_core.WithArithmeticMethods._magic_handler)

        class MockHandler:
            def Add(self, lhs, rhs):
                return ir.Value(name="add_result")

        handler = MockHandler()
        old_handler = ir.set_value_magic_handler(handler)

        # Old handler should be None
        self.assertIsNone(old_handler)
        # Handler should be set
        self.assertIs(_core.WithArithmeticMethods._magic_handler, handler)

        # Reset handler
        ir.set_value_magic_handler(None)
        self.assertIsNone(_core.WithArithmeticMethods._magic_handler)

    def test_returns_previous_handler(self):
        """Test that the function returns the previous handler when setting a new one."""

        class Handler1:
            pass

        class Handler2:
            pass

        handler1 = Handler1()
        handler2 = Handler2()

        # Set first handler
        old = ir.set_value_magic_handler(handler1)
        self.assertIsNone(old)

        # Set second handler, should return first
        old = ir.set_value_magic_handler(handler2)
        self.assertIs(old, handler1)
        self.assertIs(_core.WithArithmeticMethods._magic_handler, handler2)

    def test_add_handler(self):
        """Test the __add__ magic method."""

        class MockHandler:
            def Add(self, lhs, rhs):
                result = ir.Value(name="add_result")
                result.lhs = lhs
                result.rhs = rhs
                return result

        ir.set_value_magic_handler(MockHandler())
        result = self.value1 + self.value2
        self.assertEqual(result.name, "add_result")
        self.assertIs(result.lhs, self.value1)
        self.assertIs(result.rhs, self.value2)

    def test_sub_handler(self):
        """Test the __sub__ magic method."""

        class MockHandler:
            def Sub(self, lhs, rhs):
                result = ir.Value(name="sub_result")
                result.lhs = lhs
                result.rhs = rhs
                return result

        ir.set_value_magic_handler(MockHandler())
        result = self.value1 - self.value2
        self.assertEqual(result.name, "sub_result")
        self.assertIs(result.lhs, self.value1)
        self.assertIs(result.rhs, self.value2)

    def test_mul_handler(self):
        """Test the __mul__ magic method."""

        class MockHandler:
            def Mul(self, lhs, rhs):
                result = ir.Value(name="mul_result")
                result.lhs = lhs
                result.rhs = rhs
                return result

        ir.set_value_magic_handler(MockHandler())
        result = self.value1 * self.value2
        self.assertEqual(result.name, "mul_result")
        self.assertIs(result.lhs, self.value1)
        self.assertIs(result.rhs, self.value2)

    def test_truediv_handler(self):
        """Test the __truediv__ magic method."""

        class MockHandler:
            def Div(self, lhs, rhs):
                result = ir.Value(name="div_result")
                result.lhs = lhs
                result.rhs = rhs
                return result

        ir.set_value_magic_handler(MockHandler())
        result = self.value1 / self.value2
        self.assertEqual(result.name, "div_result")
        self.assertIs(result.lhs, self.value1)
        self.assertIs(result.rhs, self.value2)

    def test_neg_handler(self):
        """Test the __neg__ magic method."""

        class MockHandler:
            def Neg(self, operand):
                result = ir.Value(name="neg_result")
                result.operand = operand
                return result

        ir.set_value_magic_handler(MockHandler())
        result = -self.value1
        self.assertEqual(result.name, "neg_result")
        self.assertIs(result.operand, self.value1)

    def test_radd_handler(self):
        """Test the __radd__ magic method."""

        class MockHandler:
            def Add(self, lhs, rhs):
                result = ir.Value(name="radd_result")
                result.lhs = lhs
                result.rhs = rhs
                return result

        # Create a mock object that doesn't have __add__
        class MockObject:
            pass

        mock_obj = MockObject()
        ir.set_value_magic_handler(MockHandler())
        # When mock_obj + value1 is called, it should fall back to value1.__radd__(mock_obj)
        # __radd__ calls Add(other, self), so other is lhs and self is rhs
        result = self.value1.__radd__(mock_obj)
        self.assertEqual(result.name, "radd_result")
        self.assertIs(result.lhs, mock_obj)
        self.assertIs(result.rhs, self.value1)

    def test_rsub_handler(self):
        """Test the __rsub__ magic method."""

        class MockHandler:
            def Sub(self, lhs, rhs):
                result = ir.Value(name="rsub_result")
                result.lhs = lhs
                result.rhs = rhs
                return result

        class MockObject:
            pass

        mock_obj = MockObject()
        ir.set_value_magic_handler(MockHandler())
        # __rsub__ calls Sub(other, self), so other is lhs and self is rhs
        result = self.value1.__rsub__(mock_obj)
        self.assertEqual(result.name, "rsub_result")
        self.assertIs(result.lhs, mock_obj)
        self.assertIs(result.rhs, self.value1)

    def test_rmul_handler(self):
        """Test the __rmul__ magic method."""

        class MockHandler:
            def Mul(self, lhs, rhs):
                result = ir.Value(name="rmul_result")
                result.lhs = lhs
                result.rhs = rhs
                return result

        class MockObject:
            pass

        mock_obj = MockObject()
        ir.set_value_magic_handler(MockHandler())
        # __rmul__ calls Mul(other, self), so other is lhs and self is rhs
        result = self.value1.__rmul__(mock_obj)
        self.assertEqual(result.name, "rmul_result")
        self.assertIs(result.lhs, mock_obj)
        self.assertIs(result.rhs, self.value1)

    def test_rtruediv_handler(self):
        """Test the __rtruediv__ magic method."""

        class MockHandler:
            def Div(self, lhs, rhs):
                result = ir.Value(name="rdiv_result")
                result.lhs = lhs
                result.rhs = rhs
                return result

        class MockObject:
            pass

        mock_obj = MockObject()
        ir.set_value_magic_handler(MockHandler())
        # __rtruediv__ calls Div(other, self), so other is lhs and self is rhs
        result = self.value1.__rtruediv__(mock_obj)
        self.assertEqual(result.name, "rdiv_result")
        self.assertIs(result.lhs, mock_obj)
        self.assertIs(result.rhs, self.value1)

    def test_all_handler_methods_required(self):
        """Test that all handler methods are called correctly."""

        class CompleteHandler:
            def __init__(self):
                self.calls = []

            def Add(self, lhs, rhs):
                self.calls.append(("add", lhs, rhs))
                return ir.Value(name="result")

            def Sub(self, lhs, rhs):
                self.calls.append(("sub", lhs, rhs))
                return ir.Value(name="result")

            def Mul(self, lhs, rhs):
                self.calls.append(("mul", lhs, rhs))
                return ir.Value(name="result")

            def Div(self, lhs, rhs):
                self.calls.append(("div", lhs, rhs))
                return ir.Value(name="result")

            def Neg(self, operand):
                self.calls.append(("neg", operand))
                return ir.Value(name="result")

        handler = CompleteHandler()
        ir.set_value_magic_handler(handler)
        _ = self.value1 + self.value2
        _ = self.value1 - self.value2
        _ = self.value1 * self.value2
        _ = self.value1 / self.value2
        _ = -self.value1
        _ = self.value1.__radd__(self.value2)
        _ = self.value1.__rsub__(self.value2)
        _ = self.value1.__rmul__(self.value2)
        _ = self.value1.__rtruediv__(self.value2)

        # Verify all methods were called
        self.assertEqual(len(handler.calls), 9)
        self.assertEqual(handler.calls[0][0], "add")
        self.assertEqual(handler.calls[1][0], "sub")
        self.assertEqual(handler.calls[2][0], "mul")
        self.assertEqual(handler.calls[3][0], "div")
        self.assertEqual(handler.calls[4][0], "neg")
        self.assertEqual(handler.calls[5][0], "add")  # radd uses Add
        self.assertEqual(handler.calls[6][0], "sub")  # rsub uses Sub
        self.assertEqual(handler.calls[7][0], "mul")  # rmul uses Mul
        self.assertEqual(handler.calls[8][0], "div")  # rtruediv uses Div

    def test_handler_swap(self):
        """Test swapping handlers using the return value."""

        class Handler1:
            def Add(self, lhs, rhs):
                return ir.Value(name="handler1_result")

        class Handler2:
            def Add(self, lhs, rhs):
                return ir.Value(name="handler2_result")

        handler1 = Handler1()
        handler2 = Handler2()

        old = ir.set_value_magic_handler(handler1)
        self.assertIsNone(old)
        result1 = self.value1 + self.value2
        self.assertEqual(result1.name, "handler1_result")

        old = ir.set_value_magic_handler(handler2)
        self.assertIs(old, handler1)
        result2 = self.value1 + self.value2
        self.assertEqual(result2.name, "handler2_result")

        # Restore handler1
        old = ir.set_value_magic_handler(handler1)
        self.assertIs(old, handler2)
        result3 = self.value1 + self.value2
        self.assertEqual(result3.name, "handler1_result")

        # Reset to None
        old = ir.set_value_magic_handler(None)
        self.assertIs(old, handler1)

        # Should raise error with no handler
        with self.assertRaises(ValueError):
            _ = self.value1 + self.value2


class NodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = _core.Value(name="v0")
        self.v1 = _core.Value(name="v1")
        self.node = _core.Node(
            "test", "TestOp", inputs=(self.v0, self.v1, self.v1), num_outputs=3
        )
        self.node_a = _core.Node("test", "TestOpA", inputs=[self.node.outputs[0]])
        self.node_b = _core.Node("test", "TestOpB", inputs=self.node.outputs)

    def test_it_is_hashable(self):
        self.assertIsInstance(hash(self.node), int)
        self.assertIn(self.node, {self.node})

    def test_init_with_values(self):
        self.assertEqual(self.node.domain, "test")
        self.assertEqual(self.node.op_type, "TestOp")
        self.assertEqual(self.node.inputs, (self.v0, self.v1, self.v1))
        self.assertEqual(len(self.node.outputs), 3)
        self.assertEqual(self.node.attributes, {})

    def test_init_with_preinitialized_outputs(self):
        out_1 = _core.Value(
            name="out_1",
            shape=_core.Shape([1]),
            type=_core.TensorType(ir.DataType.BFLOAT16),
        )
        out_2 = _core.Value(
            name="out_2",
            shape=_core.Shape([2]),
            type=_core.TensorType(ir.DataType.INT4),
        )
        node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1), outputs=[out_1, out_2])
        self.assertEqual(node.outputs[0].name, "out_1")
        self.assertEqual(node.outputs[0].shape, _core.Shape([1]))
        self.assertEqual(node.outputs[0].dtype, ir.DataType.BFLOAT16)
        self.assertEqual(node.outputs[1].name, "out_2")
        self.assertEqual(node.outputs[1].shape, _core.Shape([2]))
        self.assertEqual(node.outputs[1].dtype, ir.DataType.INT4)
        self.assertIs(node.outputs[0], out_1)
        self.assertIs(node.outputs[1], out_2)
        self.assertIs(node.outputs[0].producer(), node)
        self.assertIs(node.outputs[1].producer(), node)
        self.assertIs(node.outputs[0].index(), 0)
        self.assertIs(node.outputs[1].index(), 1)

    def test_init_raises_when_num_outputs_does_not_match_outputs(self):
        with self.assertRaisesRegex(ValueError, "outputs"):
            _core.Node("test", "TestOp", inputs=(self.v0, self.v1), num_outputs=2, outputs=[])

    def test_init_with_zero_num_outputs(self):
        node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1), num_outputs=0)
        self.assertEqual(node.outputs, ())

    def test_init_with_empty_outputs(self):
        node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1), outputs=[])
        self.assertEqual(node.outputs, ())

    def test_init_produces_one_output_with_unspecified_output_argument(self):
        node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1))
        self.assertEqual(len(node.outputs), 1)

    def test_metadata(self):
        self.node.meta["test"] = 1
        self.assertEqual(self.node.meta["test"], 1)
        self.node.metadata_props["test"] = "any string"
        self.assertEqual(self.node.metadata_props["test"], "any string")

    def test_it_is_added_to_a_graph_if_specified(self):
        graph = _core.Graph(
            (self.v0, self.v1),  # type: ignore
            self.node.outputs,
            nodes=(self.node,),
        )
        self.assertIn(self.node, graph)

    def test_predecessors(self):
        self.assertEqual(self.node.predecessors(), ())
        self.assertEqual(self.node_a.predecessors(), (self.node,))
        self.assertEqual(self.node_b.predecessors(), (self.node,))

    def test_predecessors_are_unique(self):
        # node_b has three inputs from node, but only one predecessor
        self.assertEqual(self.node_b.predecessors(), (self.node,))

    def test_successors(self):
        self.assertEqual(self.node.successors(), (self.node_a, self.node_b))
        self.assertEqual(self.node_a.successors(), ())
        self.assertEqual(self.node_b.successors(), ())

    def test_successors_are_unique(self):
        self.assertEqual(self.node.successors(), (self.node_a, self.node_b))

    def test_domain_normalizes_ai_onnx(self):
        # Node domain is always normalized to "" if it is "ai.onnx"
        node = _core.Node("ai.onnx", "TestOp", inputs=())
        self.assertEqual(node.domain, "")

        node.domain = ""
        self.assertEqual(node.domain, "")

        node.domain = "ai.onnx"
        self.assertEqual(node.domain, "")

    def test_attributes_add(self):
        node = _core.Node("ai.onnx", "TestOp", inputs=())
        node.attributes.add(_core.AttrInt64("test_attr", 1))
        self.assertIn("test_attr", node.attributes)
        self.assertEqual(node.attributes["test_attr"].value, 1)

    def test_attributes_set_raise_with_type_error(self):
        node = _core.Node("ai.onnx", "TestOp", inputs=())
        with self.assertRaises(TypeError):
            node.attributes["test_attr"] = 1
        with self.assertRaises(TypeError):
            node.attributes[1] = _core.AttrInt64("test_attr", 1)

    def test_init_accepts_attribute_mapping(self):
        node = _core.Node(
            "ai.onnx", "TestOp", inputs=(), attributes=[_core.AttrInt64("test_attr", 1)]
        )
        new_node = _core.Node("", "OtherOp", inputs=(), attributes=node.attributes)
        self.assertEqual(new_node.attributes, node.attributes)

    def test_attributes_get_int(self):
        node = _core.Node(
            "ai.onnx", "TestOp", inputs=(), attributes=[_core.AttrInt64("test_attr", 1)]
        )
        self.assertEqual(node.attributes.get_int("test_attr"), 1)
        self.assertIsNone(node.attributes.get_int("non_existent_attr"))
        self.assertEqual(node.attributes.get_int("non_existent_attr", 42), 42)

    def test_attributes_get_float(self):
        node = _core.Node(
            "ai.onnx", "TestOp", inputs=(), attributes=[_core.AttrFloat32("test_attr", 1.0)]
        )
        self.assertEqual(node.attributes.get_float("test_attr"), 1.0)
        self.assertIsNone(node.attributes.get_float("non_existent_attr"))
        self.assertEqual(node.attributes.get_float("non_existent_attr", 42.0), 42.0)

    def test_attributes_get_string(self):
        node = _core.Node(
            "ai.onnx", "TestOp", inputs=(), attributes=[_core.AttrString("test_attr", "value")]
        )
        self.assertEqual(node.attributes.get_string("test_attr"), "value")
        self.assertIsNone(node.attributes.get_string("non_existent_attr"))
        self.assertEqual(node.attributes.get_string("non_existent_attr", "default"), "default")

    def test_attributes_get_tensor(self):
        tensor = ir.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        node = _core.Node(
            "ai.onnx", "TestOp", inputs=(), attributes=[_core.AttrTensor("test_attr", tensor)]
        )
        np.testing.assert_equal(
            node.attributes.get_tensor("test_attr").numpy(), tensor.numpy()
        )
        self.assertIsNone(node.attributes.get_tensor("non_existent_attr"))
        np.testing.assert_equal(
            node.attributes.get_tensor("non_existent_attr", tensor).numpy(), tensor.numpy()
        )

    def test_attributes_get_ints(self):
        node = _core.Node(
            "ai.onnx",
            "TestOp",
            inputs=(),
            attributes=[_core.AttrInt64s("test_attr", [1, 2, 3])],
        )
        self.assertEqual(node.attributes.get_ints("test_attr"), (1, 2, 3))
        self.assertIsNone(node.attributes.get_ints("non_existent_attr"))
        self.assertEqual(node.attributes.get_ints("non_existent_attr", [42]), [42])

    def test_attributes_get_floats(self):
        node = _core.Node(
            "ai.onnx",
            "TestOp",
            inputs=(),
            attributes=[_core.AttrFloat32s("test_attr", [1.0, 2.0, 3.0])],
        )
        self.assertEqual(node.attributes.get_floats("test_attr"), (1.0, 2.0, 3.0))
        self.assertIsNone(node.attributes.get_floats("non_existent_attr"))
        self.assertEqual(node.attributes.get_floats("non_existent_attr", [42.0]), [42.0])

    def test_attributes_get_strings(self):
        node = _core.Node(
            "ai.onnx",
            "TestOp",
            inputs=(),
            attributes=[_core.AttrStrings("test_attr", ["a", "b", "c"])],
        )
        self.assertEqual(node.attributes.get_strings("test_attr"), ("a", "b", "c"))
        self.assertIsNone(node.attributes.get_strings("non_existent_attr"))
        self.assertEqual(
            node.attributes.get_strings("non_existent_attr", ["default"]), ["default"]
        )

    def test_attributes_get_tensors(self):
        tensor1 = ir.Tensor(np.array([1.0, 2.0], dtype=np.float32))
        tensor2 = ir.Tensor(np.array([3.0, 4.0], dtype=np.float32))
        node = _core.Node(
            "ai.onnx",
            "TestOp",
            inputs=(),
            attributes=[_core.AttrTensors("test_attr", [tensor1, tensor2])],
        )
        tensors = node.attributes.get_tensors("test_attr")
        self.assertIsNotNone(tensors)
        self.assertEqual(len(tensors), 2)
        np.testing.assert_equal(tensors[0].numpy(), tensor1.numpy())
        np.testing.assert_equal(tensors[1].numpy(), tensor2.numpy())
        self.assertIsNone(node.attributes.get_tensors("non_existent_attr"))
        np.testing.assert_equal(
            node.attributes.get_tensors("non_existent_attr", [tensor1]), [tensor1]
        )

    def test_resize_inputs_increase_size(self):
        """Test that resize_inputs increases the number of inputs by adding None values."""
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        node = _core.Node("", "TestOp", inputs=(v0, v1), num_outputs=1)

        self.assertEqual(len(node.inputs), 2)
        self.assertIs(node.inputs[0], v0)
        self.assertIs(node.inputs[1], v1)

        # Resize to 4 inputs
        node.resize_inputs(4)

        self.assertEqual(len(node.inputs), 4)
        self.assertIs(node.inputs[0], v0)
        self.assertIs(node.inputs[1], v1)
        self.assertIsNone(node.inputs[2])
        self.assertIsNone(node.inputs[3])

    def test_resize_inputs_decrease_size(self):
        """Test that resize_inputs decreases the number of inputs and removes uses."""
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        v2 = _core.Value(name="v2")
        node = _core.Node("", "TestOp", inputs=(v0, v1, v2), num_outputs=1)

        self.assertEqual(len(node.inputs), 3)
        # Check that node is in v2's uses
        self.assertEqual(len(v2.uses()), 1)
        self.assertIn(_core.Usage(node, 2), v2.uses())

        # Resize to 2 inputs (remove v2)
        node.resize_inputs(2)

        self.assertEqual(len(node.inputs), 2)
        self.assertIs(node.inputs[0], v0)
        self.assertIs(node.inputs[1], v1)
        # Check that node is no longer in v2's uses
        self.assertEqual(len(v2.uses()), 0)

    def test_resize_inputs_same_size(self):
        """Test that resize_inputs does nothing when size is unchanged."""
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        node = _core.Node("", "TestOp", inputs=(v0, v1), num_outputs=1)

        # Resize to same size
        node.resize_inputs(2)

        self.assertEqual(len(node.inputs), 2)
        self.assertIs(node.inputs[0], v0)
        self.assertIs(node.inputs[1], v1)

    def test_resize_inputs_to_zero(self):
        """Test that resize_inputs can reduce inputs to zero."""
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        node = _core.Node("", "TestOp", inputs=(v0, v1), num_outputs=1)

        node.resize_inputs(0)

        self.assertEqual(len(node.inputs), 0)
        self.assertEqual(node.inputs, ())
        # Check that uses are removed
        self.assertEqual(len(v0.uses()), 0)
        self.assertEqual(len(v1.uses()), 0)

    def test_resize_inputs_from_zero(self):
        """Test that resize_inputs can increase from zero inputs."""
        node = _core.Node("", "TestOp", inputs=(), num_outputs=1)

        self.assertEqual(len(node.inputs), 0)

        node.resize_inputs(3)

        self.assertEqual(len(node.inputs), 3)
        self.assertIsNone(node.inputs[0])
        self.assertIsNone(node.inputs[1])
        self.assertIsNone(node.inputs[2])

    def test_resize_inputs_preserves_none_inputs(self):
        """Test that resize_inputs preserves None inputs when decreasing size."""
        v0 = _core.Value(name="v0")
        node = _core.Node("", "TestOp", inputs=(v0, None, None), num_outputs=1)

        node.resize_inputs(2)

        self.assertEqual(len(node.inputs), 2)
        self.assertIs(node.inputs[0], v0)
        self.assertIsNone(node.inputs[1])

    def test_resize_outputs_increase_size(self):
        """Test that resize_outputs increases the number of outputs."""
        v0 = _core.Value(name="v0")
        node = _core.Node("", "TestOp", inputs=(v0,), num_outputs=2)

        self.assertEqual(len(node.outputs), 2)
        old_output_0 = node.outputs[0]
        old_output_1 = node.outputs[1]

        # Resize to 4 outputs
        node.resize_outputs(4)

        self.assertEqual(len(node.outputs), 4)
        # Verify old outputs are preserved
        self.assertIs(node.outputs[0], old_output_0)
        self.assertIs(node.outputs[1], old_output_1)
        # Verify new outputs are created
        self.assertIsNotNone(node.outputs[2])
        self.assertIsNotNone(node.outputs[3])
        # Verify new outputs have correct producer and index
        self.assertIs(node.outputs[2].producer(), node)
        self.assertIs(node.outputs[3].producer(), node)
        self.assertEqual(node.outputs[2].index(), 2)
        self.assertEqual(node.outputs[3].index(), 3)

    def test_resize_outputs_decrease_size(self):
        """Test that resize_outputs decreases the number of outputs when they have no uses."""
        v0 = _core.Value(name="v0")
        node = _core.Node("", "TestOp", inputs=(v0,), num_outputs=3)

        self.assertEqual(len(node.outputs), 3)
        old_output_0 = node.outputs[0]

        # Resize to 1 output
        node.resize_outputs(1)

        self.assertEqual(len(node.outputs), 1)
        self.assertIs(node.outputs[0], old_output_0)

    def test_resize_outputs_decrease_size_raises_when_output_has_uses(self):
        """Test that resize_outputs raises ValueError when removing outputs with uses."""
        v0 = _core.Value(name="v0")
        node = _core.Node("", "TestOp", inputs=(v0,), num_outputs=3)
        # Create a consumer for the third output
        _consumer = _core.Node("", "Consumer", inputs=(node.outputs[2],), num_outputs=1)

        self.assertEqual(len(node.outputs[2].uses()), 1)

        # Try to resize to 2 outputs (remove the third one)
        with self.assertRaisesRegex(ValueError, "Cannot remove output.*because it has uses"):
            node.resize_outputs(2)

        # Verify outputs are unchanged
        self.assertEqual(len(node.outputs), 3)

    def test_resize_outputs_same_size(self):
        """Test that resize_outputs does nothing when size is unchanged."""
        v0 = _core.Value(name="v0")
        node = _core.Node("", "TestOp", inputs=(v0,), num_outputs=2)

        old_outputs = node.outputs

        # Resize to same size
        node.resize_outputs(2)

        self.assertEqual(len(node.outputs), 2)
        self.assertIs(node.outputs[0], old_outputs[0])
        self.assertIs(node.outputs[1], old_outputs[1])

    def test_resize_outputs_to_zero(self):
        """Test that resize_outputs can reduce outputs to zero."""
        v0 = _core.Value(name="v0")
        node = _core.Node("", "TestOp", inputs=(v0,), num_outputs=2)

        node.resize_outputs(0)

        self.assertEqual(len(node.outputs), 0)
        self.assertEqual(node.outputs, ())

    def test_resize_outputs_from_zero(self):
        """Test that resize_outputs can increase from zero outputs."""
        v0 = _core.Value(name="v0")
        node = _core.Node("", "TestOp", inputs=(v0,), num_outputs=0)

        self.assertEqual(len(node.outputs), 0)

        node.resize_outputs(2)

        self.assertEqual(len(node.outputs), 2)
        self.assertIsNotNone(node.outputs[0])
        self.assertIsNotNone(node.outputs[1])
        self.assertIs(node.outputs[0].producer(), node)
        self.assertIs(node.outputs[1].producer(), node)
        self.assertEqual(node.outputs[0].index(), 0)
        self.assertEqual(node.outputs[1].index(), 1)

    def test_resize_outputs_decrease_with_middle_output_having_uses(self):
        """Test that resize_outputs raises when removing a middle output with uses."""
        v0 = _core.Value(name="v0")
        node = _core.Node("", "TestOp", inputs=(v0,), num_outputs=4)
        # Create a consumer for the second output (index 1)
        _consumer = _core.Node("", "Consumer", inputs=(node.outputs[1],), num_outputs=1)

        # Try to resize to 1 output (remove outputs at indices 1, 2, 3)
        with self.assertRaisesRegex(ValueError, "Cannot remove output.*because it has uses"):
            node.resize_outputs(1)

        # Verify outputs are unchanged
        self.assertEqual(len(node.outputs), 4)

    # TODO(justinchuby): Test all methods


class GraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = _core.Value(name="v0")
        self.v1 = _core.Value(name="v1")
        self.node = _core.Node(
            "", "Add", inputs=(self.v0, self.v1), num_outputs=1, name="node_add"
        )
        self.graph = _core.Graph(
            (self.v0, self.v1),
            self.node.outputs,
            nodes=(self.node,),
            opset_imports={"": 1},
        )

    def test_initialize(self):
        self.assertEqual(self.graph.inputs, [self.v0, self.v1])
        self.assertEqual(self.graph.outputs, [*self.node.outputs])
        self.assertEqual(self.graph.opset_imports, {"": 1})
        self.assertEqual(self.graph.initializers, {})
        self.assertIsNone(self.graph.doc_string)

    def test_it_is_hashable(self):
        self.assertIsInstance(hash(self.graph), int)
        self.assertIn(self.graph, {self.graph})

    def test_it_is_iterable_of_nodes(self):
        self.assertEqual(list(self.graph), [self.node])

    def test_node_returns_node_by_name(self):
        self.assertIs(self.graph.node("node_add"), self.node)

    def test_node_returns_node_by_index(self):
        self.assertIs(self.graph.node(0), self.node)

    def test_node_raises_when_node_does_not_exist(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            self.graph.node("non_existent")

    def test_node_raises_when_index_out_of_range(self):
        with self.assertRaises(IndexError):
            self.graph.node(1)

    def test_num_nodes_returns_the_count_of_nodes(self):
        self.assertEqual(self.graph.num_nodes(), 1)
        self.assertEqual(self.graph.num_nodes(), len(self.graph))

    def test_metadata(self):
        self.graph.meta["test"] = 1
        self.assertEqual(self.graph.meta["test"], 1)
        self.graph.metadata_props["test"] = "any string"
        self.assertEqual(self.graph.metadata_props["test"], "any string")

    def test_remove_removes_node_from_graph(self):
        self.graph.remove(self.node)
        self.assertEqual(list(self.graph), [])
        self.assertIsNone(self.node.graph)

    def test_remove_does_not_change_input_users(self):
        self.graph.remove(self.node)
        self.assertEqual(tuple(self.v0.uses()), ((self.node, 0),))
        self.assertEqual(tuple(self.v1.uses()), ((self.node, 1),))

    def test_remove_does_not_change_graph_in_out(self):
        self.graph.remove(self.node)
        self.assertEqual(self.graph.inputs, [self.v0, self.v1])
        self.assertEqual(self.graph.outputs, list(self.node.outputs))

    def test_remove_raises_when_node_does_not_belong_to_graph(self):
        node = _core.Node("", "Add", inputs=(self.v0, self.v1), num_outputs=1)
        with self.assertRaisesRegex(ValueError, "graph"):
            self.graph.remove(node)

    def test_remove_safe_raises_when_node_output_is_graph_output(self):
        with self.assertRaisesRegex(ValueError, "output"):
            self.graph.remove(self.node, safe=True)

    def test_remove_safe_raises_when_node_has_users(self):
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        add_node = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1)
        identity_node = _core.Node("", "Identity", inputs=add_node.outputs, num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            identity_node.outputs,
            nodes=(add_node, identity_node),
            opset_imports={"": 1},
        )
        with self.assertRaisesRegex(ValueError, "used by other nodes"):
            graph.remove(add_node, safe=True)

    def test_remove_safe_removes_uses_of_removed_nodes(self):
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        add_node = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1)
        identity_node = _core.Node("", "Identity", inputs=add_node.outputs, num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            identity_node.outputs,
            nodes=(add_node, identity_node),
            opset_imports={"": 1},
        )
        # Remove add_node and check that it is no longer a consumer of v0 and v1
        sub_node = _core.Node("", "Sub", inputs=(v0, v1), num_outputs=1)
        identity_node.replace_input_with(0, sub_node.outputs[0])
        graph.insert_before(identity_node, sub_node)
        graph.remove(add_node, safe=True)
        self.assertEqual(tuple(v0.uses()), ((sub_node, 0),))
        self.assertEqual(tuple(v1.uses()), ((sub_node, 1),))
        self.assertEqual(tuple(graph), (sub_node, identity_node))
        self.assertEqual(add_node.inputs, (None, None))

    def test_register_initializer(self):
        self.v1.const_value = ir.tensor([1, 2, 3])
        self.graph.register_initializer(self.v1)
        self.assertEqual(self.graph.initializers, {self.v1.name: self.v1})

    def test_register_initializer_raises_when_value_is_not_constant(self):
        with self.assertRaises(ValueError):
            self.graph.register_initializer(self.v0)

    def test_register_initializer_raises_when_a_different_value_is_already_registered(self):
        self.v1.const_value = ir.tensor([1, 2, 3])
        self.graph.register_initializer(self.v1)
        # This is fine
        self.graph.register_initializer(self.v1)
        self.v0.name = "v1"
        with self.assertRaisesRegex(ValueError, "already registered"):
            # Registering a different value with the same name should raise
            self.graph.register_initializer(self.v0)

    def test_register_initializer_raises_when_value_does_not_have_a_name(self):
        self.v1.name = None
        with self.assertRaises(ValueError):
            self.graph.register_initializer(self.v1)

    # TODO(justinchuby): Test graph mutation methods

    # Test topological sort.
    # Graph structure:
    #   nodes: [node, ...]
    #   edges: [(predecessor_node, successor_node), ...]
    #   subgraphs: {node: [subgraph, ...]}

    def test_topological_sort_empty_graph(self):
        graph = _core.Graph(
            inputs=(),
            outputs=(),
            nodes=(),
        )
        graph.sort()
        self.assertEqual(tuple(graph), ())

    def test_topological_sort_linear_dependencies(self):
        # nodes=[1,2,3], edges=[(1,2),(2,3)]
        v0 = _core.Value(name="v0")
        node1 = _core.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(node1.outputs[0],), num_outputs=1)
        node3 = _core.Node("", "Node3", inputs=(node2.outputs[0],), num_outputs=1)
        graph = _core.Graph(
            (v0,),
            node3.outputs,
            nodes=(node3, node2, node1),
        )
        graph.sort()
        sorted_nodes = tuple(graph)
        expected_order = (node1, node2, node3)
        self.assertEqual(sorted_nodes, expected_order)

    def test_topological_sort_independent_subgraphs(self):
        # nodes=[1,2,3,4], edges=[(1,3),(2,4)]
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(v1,), num_outputs=1)
        node3 = _core.Node("", "Node3", inputs=(node1.outputs[0],), num_outputs=1)
        node4 = _core.Node("", "Node4", inputs=(node2.outputs[0],), num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            (node3.outputs[0], node4.outputs[0]),
            nodes=(node4, node3, node2, node1),
        )
        graph.sort()
        sorted_nodes = tuple(graph)
        expected_order = (node2, node4, node1, node3)
        self.assertEqual(sorted_nodes, expected_order)

    def test_topological_sort_shared_successor(self):
        # nodes=[1,2,3], edges=[(1,3),(2,3)]
        v0 = _core.Value(name="v0")
        node1 = _core.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(v0,), num_outputs=1)
        node3 = _core.Node(
            "", "Node3", inputs=(node1.outputs[0], node2.outputs[0]), num_outputs=1
        )
        graph = _core.Graph(
            (v0,),
            (node3.outputs[0],),
            nodes=(node3, node2, node1),
        )
        graph.sort()
        sorted_nodes = tuple(graph)
        expected_order = (node2, node1, node3)
        self.assertEqual(sorted_nodes, expected_order)

    def _create_shared_predecessor_nodes(
        self,
    ) -> tuple[_core.Value, tuple[_core.Node, _core.Node, _core.Node]]:
        # nodes=[0,1,2], edges=[(0,1),(0,2)]
        v0 = _core.Value(name="v0")
        node0 = _core.Node("", "Node0", inputs=(v0,), num_outputs=1)
        node1 = _core.Node("", "Node1", inputs=(node0.outputs[0],), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(node0.outputs[0],), num_outputs=1)
        return v0, (node0, node1, node2)

    @parameterized.parameterized.expand(
        [
            ("012", (0, 1, 2), (0, 1, 2)),
            ("021", (0, 2, 1), (0, 2, 1)),
            ("102", (1, 0, 2), (0, 1, 2)),
            ("120", (1, 2, 0), (0, 1, 2)),
            ("201", (2, 0, 1), (0, 2, 1)),
            ("210", (2, 1, 0), (0, 2, 1)),
        ]
    )
    def test_topological_sort_shared_predecessor(
        self, _: str, initial_order: tuple[int], expected_order: tuple[int]
    ):
        v0, nodes = self._create_shared_predecessor_nodes()
        graph = _core.Graph((v0,), (), nodes=[nodes[i] for i in initial_order])
        graph.sort()
        sorted_nodes = list(graph)
        self.assertEqual(sorted_nodes, [nodes[i] for i in expected_order])

    def test_topological_sort_cycle_detection(self):
        # nodes=[1,2,3], edges=[(1,2),(2,3),(3,2)]
        v0 = _core.Value(name="v0")
        node1 = _core.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(node1.outputs[0], v0), num_outputs=1)
        node3 = _core.Node("", "Node3", inputs=(node2.outputs[0],), num_outputs=1)
        node2.replace_input_with(1, node3.outputs[0])
        graph = _core.Graph(
            (v0,),
            (node3.outputs[0],),
            nodes=(node1, node2, node3),
        )
        with self.assertRaises(ValueError):
            graph.sort()

    def test_topological_sort_subgraph(self):
        # main_graph: nodes=[a,b,c,d,>,if], edges=[(a,>),(b,>),(>,if)], subgraphs={if:[then_graph,else_graph]}
        # then_graph: nodes=[sub], edges=[(c,sub),(d,sub)]
        # else_graph: nodes=[add], edges=[(c,add),(d,add)]
        v0 = _core.Value(name="va")
        v1 = _core.Value(name="vb")
        v2 = _core.Value(name="vc")
        v3 = _core.Value(name="vd")
        node0 = _core.Node("", "a", inputs=(v0,), num_outputs=1)
        node1 = _core.Node("", "b", inputs=(v1,), num_outputs=1)
        node2 = _core.Node("", "c", inputs=(v2,), num_outputs=1)
        node3 = _core.Node("", "d", inputs=(v3,), num_outputs=1)
        node4 = _core.Node(
            "", "sub", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1
        )
        node5 = _core.Node(
            "", "add", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1
        )
        node6 = _core.Node("", ">", inputs=(node0.outputs[0], node1.outputs[0]), num_outputs=1)
        then_graph = _core.Graph(
            inputs=(),
            outputs=(node4.outputs[0],),
            nodes=(node4,),
            name="then_graph",
        )
        else_graph = _core.Graph(
            inputs=(),
            outputs=(node5.outputs[0],),
            nodes=(node5,),
            name="else_graph",
        )
        node7 = _core.Node(
            "",
            "if",
            inputs=(node6.outputs[0],),
            num_outputs=1,
            attributes=[
                ir.AttrGraph("then_branch", then_graph),
                ir.AttrGraph("else_branch", else_graph),
            ],
        )
        main_graph_rev = _core.Graph(
            inputs=(v0, v1, v2, v3),
            outputs=(node7.outputs[0],),
            nodes=(node7, node6, node3, node2, node1, node0),  # if, >, d, c, b, a
            name="main_graph_rev",
        )
        main_graph_rev.sort()
        self.assertEqual(
            tuple(node.op_type for node in tuple(main_graph_rev)),
            ("d", "c", "b", "a", ">", "if"),
        )

    def test_all_nodes_returns_all_nodes(self):
        # Create a graph with a subgraph
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        node0 = _core.Node("", "A", inputs=(v0,), num_outputs=1)
        node1 = _core.Node("", "B", inputs=(v1,), num_outputs=1)
        sub_node = _core.Node(
            "", "Sub", inputs=(node0.outputs[0], node1.outputs[0]), num_outputs=1
        )
        subgraph = _core.Graph(
            inputs=(), outputs=(sub_node.outputs[0],), nodes=(sub_node,), name="subgraph"
        )
        main_node = _core.Node(
            "",
            "If",
            inputs=(node0.outputs[0],),
            attributes=[ir.AttrGraph("then_branch", subgraph)],
        )
        graph = _core.Graph(
            inputs=(v0, v1),
            outputs=(main_node.outputs[0],),
            nodes=(node0, node1, main_node),
            name="main_graph",
        )
        all_nodes = list(graph.all_nodes())
        # Should include node0, node1, main_node, and sub_node
        self.assertIn(node0, all_nodes)
        self.assertIn(node1, all_nodes)
        self.assertIn(main_node, all_nodes)
        self.assertIn(sub_node, all_nodes)
        self.assertEqual(len(all_nodes), 4)

    def test_subgraphs_returns_all_subgraphs(self):
        # Create a graph with two subgraphs
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        node0 = _core.Node("", "A", inputs=(v0,), num_outputs=1)
        node1 = _core.Node("", "B", inputs=(v1,), num_outputs=1)
        sub_node1 = _core.Node("", "Sub1", inputs=(node0.outputs[0],), num_outputs=1)
        sub_node2 = _core.Node("", "Sub2", inputs=(node1.outputs[0],), num_outputs=1)
        subgraph1 = _core.Graph(
            inputs=(), outputs=(sub_node1.outputs[0],), nodes=(sub_node1,), name="subgraph1"
        )
        subgraph2 = _core.Graph(
            inputs=(), outputs=(sub_node2.outputs[0],), nodes=(sub_node2,), name="subgraph2"
        )
        main_node = _core.Node(
            "",
            "If",
            inputs=(node0.outputs[0],),
            attributes=[
                ir.AttrGraph("then_branch", subgraph1),
                ir.AttrGraph("else_branch", subgraph2),
            ],
        )
        graph = _core.Graph(
            inputs=(v0, v1),
            outputs=(main_node.outputs[0],),
            nodes=(node0, node1, main_node),
            name="main_graph",
        )
        subgraphs = list(graph.subgraphs())
        self.assertIn(subgraph1, subgraphs)
        self.assertIn(subgraph2, subgraphs)
        self.assertEqual(len(subgraphs), 2)

    def test_subgraphs_returns_empty_subgraphs(self):
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        node0 = _core.Node("", "A", inputs=(v0,), num_outputs=1)
        subgraph1 = _core.Graph(inputs=(), outputs=(), nodes=(), name="subgraph1")
        main_node = _core.Node(
            "",
            "SomeOp",
            inputs=(node0.outputs[0],),
            attributes=[
                ir.AttrGraph("subgraph", subgraph1),
            ],
        )
        graph = _core.Graph(
            inputs=(v0, v1),
            outputs=(main_node.outputs[0],),
            nodes=(node0, main_node),
            name="main_graph",
        )
        subgraphs = list(graph.subgraphs())
        self.assertIn(subgraph1, subgraphs)
        self.assertEqual(len(subgraphs), 1)


class GraphContainersTest(unittest.TestCase):
    """Test containers for input, output and initializers of a graph."""

    def setUp(self):
        self.graph = _core.Graph(inputs=(), outputs=(), nodes=())
        self.value1 = _core.Value(name="input1")
        self.value2 = _core.Value(name="output1")
        self.value3 = _core.Value(name="initializer1", const_value=ir.tensor([1, 2, 3]))

    def test_initialize(self):
        graph = _core.Graph(
            inputs=(self.value1,),
            outputs=(self.value2,),
            nodes=(),
            initializers=(self.value3,),
        )
        self.assertEqual(graph.inputs, [self.value1])
        self.assertTrue(self.value1.is_graph_input())
        self.assertIs(self.value1.graph, graph)
        self.assertFalse(self.value1.is_graph_output())
        self.assertFalse(self.value1.is_initializer())
        self.assertEqual(graph.outputs, [self.value2])
        self.assertTrue(self.value2.is_graph_output())
        self.assertIs(self.value2.graph, graph)
        self.assertFalse(self.value2.is_graph_input())
        self.assertFalse(self.value2.is_initializer())
        self.assertEqual(graph.initializers, {self.value3.name: self.value3})
        self.assertTrue(self.value3.is_initializer())
        self.assertIs(self.value3.graph, graph)
        self.assertFalse(self.value3.is_graph_input())
        self.assertFalse(self.value3.is_graph_output())

    def test_append_to_inputs(self):
        self.graph.inputs.append(self.value1)
        self.assertIn(self.value1, self.graph.inputs)
        self.assertTrue(self.value1.is_graph_input())
        self.assertIs(self.value1.graph, self.graph)
        self.assertFalse(self.value1.is_graph_output())
        self.assertFalse(self.value1.is_initializer())

    def test_append_input_raises_when_input_belongs_to_another_graph(self):
        other_graph = _core.Graph(inputs=(), outputs=(), nodes=())
        other_graph.inputs.append(self.value1)
        with self.assertRaisesRegex(ValueError, "is already owned by a different graph"):
            self.graph.inputs.append(self.value1)
        # Append is ok after the value is removed from the old graph
        other_graph.inputs.clear()
        self.graph.inputs.append(self.value1)
        self.assertTrue(self.value1.is_graph_input())
        self.assertIs(self.value1.graph, self.graph)

    def test_extend_inputs(self):
        self.graph.inputs.extend([self.value1, self.value2])
        self.assertIn(self.value1, self.graph.inputs)
        self.assertIn(self.value2, self.graph.inputs)
        self.assertTrue(self.value1.is_graph_input())
        self.assertTrue(self.value2.is_graph_input())
        self.assertIs(self.value1.graph, self.graph)
        self.assertIs(self.value2.graph, self.graph)

    def test_pop_from_inputs(self):
        self.graph.inputs.append(self.value1)
        popped = self.graph.inputs.pop()
        self.assertIs(popped, self.value1)
        self.assertNotIn(self.value1, self.graph.inputs)
        self.assertFalse(self.value1.is_graph_input())
        self.assertIsNone(self.value1.graph)

    def test_pop_from_duplicated_inputs(self):
        self.graph.inputs.extend([self.value1, self.value1])
        popped = self.graph.inputs.pop()
        self.assertIs(popped, self.value1)
        self.assertIn(self.value1, self.graph.inputs)
        self.assertTrue(self.value1.is_graph_input())
        self.assertIs(self.value1.graph, self.graph)

    def test_pop_from_inputs_raises_when_empty(self):
        with self.assertRaises(IndexError):
            self.graph.inputs.pop()

    def test_insert_into_inputs(self):
        self.graph.inputs.insert(0, self.value1)
        self.assertIs(self.graph.inputs[0], self.value1)
        self.assertTrue(self.value1.is_graph_input())
        self.assertIs(self.value1.graph, self.graph)

    def test_remove_from_inputs(self):
        self.graph.inputs.append(self.value1)
        self.graph.inputs.remove(self.value1)
        self.assertNotIn(self.value1, self.graph.inputs)
        self.assertFalse(self.value1.is_graph_input())
        self.assertIsNone(self.value1.graph)

    def test_clear_inputs(self):
        self.graph.inputs.extend([self.value1, self.value2])
        self.graph.inputs.clear()
        self.assertEqual(len(self.graph.inputs), 0)
        self.assertFalse(self.value1.is_graph_input())
        self.assertIsNone(self.value1.graph)
        self.assertFalse(self.value2.is_graph_input())
        self.assertIsNone(self.value2.graph)

    def test_clear_duplicated_inputs(self):
        self.graph.inputs.extend([self.value1, self.value1])
        self.graph.inputs.clear()
        self.assertEqual(len(self.graph.inputs), 0)
        self.assertFalse(self.value1.is_graph_input())
        self.assertIsNone(self.value1.graph)

    def test_inputs_set_items(self):
        self.graph.inputs.append(self.value1)
        self.graph.inputs[-1] = self.value2
        self.assertNotIn(self.value1, self.graph.inputs)
        self.assertIn(self.value2, self.graph.inputs)
        self.assertIs(self.graph.inputs[0], self.value2)
        self.assertTrue(self.value2.is_graph_input())
        self.assertIs(self.value2.graph, self.graph)
        self.assertFalse(self.value1.is_graph_input())
        self.assertIsNone(self.value1.graph)

    def test_inputs_set_items_slices(self):
        self.graph.inputs.extend([self.value1, self.value2])
        # Replace with one existing and one new input
        self.graph.inputs[0:2] = [self.value2, self.value3]
        self.assertNotIn(self.value1, self.graph.inputs)
        self.assertIn(self.value2, self.graph.inputs)
        self.assertIn(self.value3, self.graph.inputs)
        self.assertIs(self.value2.graph, self.graph)
        self.assertIs(self.value3.graph, self.graph)
        self.assertTrue(self.value2.is_graph_input())
        self.assertTrue(self.value3.is_graph_input())
        self.assertFalse(self.value1.is_graph_input())
        self.assertIsNone(self.value1.graph)

    def test_take_inputs(self):
        self.graph.inputs.extend([self.value1, self.value2, self.value3])
        inputs = self.graph.inputs[:2]
        self.graph.inputs.clear()
        self.graph.inputs.extend(inputs)
        self.assertEqual(len(self.graph.inputs), 2)
        self.assertEqual(self.graph.inputs, [self.value1, self.value2])
        self.assertTrue(self.value1.is_graph_input())
        self.assertTrue(self.value2.is_graph_input())
        self.assertFalse(self.value3.is_graph_input())
        self.assertIs(self.value1.graph, self.graph)
        self.assertIs(self.value2.graph, self.graph)
        self.assertIsNone(self.value3.graph)

    def test_inputs_copy(self):
        self.graph.inputs.extend([self.value1, self.value2])
        inputs_copy = self.graph.inputs.copy()
        self.assertEqual(inputs_copy, [self.value1, self.value2])
        self.assertIsNot(inputs_copy, self.graph.inputs)
        # Modifying the copy does not affect the original
        inputs_copy.append(self.value3)
        self.assertNotIn(self.value3, self.graph.inputs)
        self.assertIn(self.value3, inputs_copy)

    def test_inputs_append_raises_when_input_is_node_output(self):
        node = ir.node("SomeOp", inputs=[])
        with self.assertRaisesRegex(ValueError, "produced by a node"):
            self.graph.inputs.append(node.outputs[0])

    def test_inputs_extend_raises_when_input_is_node_output(self):
        node = ir.node("SomeOp", inputs=[])
        with self.assertRaisesRegex(ValueError, "produced by a node"):
            self.graph.inputs.extend(node.outputs)

    def test_append_to_outputs(self):
        self.graph.outputs.append(self.value2)
        self.assertIn(self.value2, self.graph.outputs)
        self.assertTrue(self.value2.is_graph_output())

    def test_append_output_raises_when_output_belongs_to_another_graph(self):
        other_graph = _core.Graph(inputs=(), outputs=(), nodes=())
        other_graph.outputs.append(self.value2)
        with self.assertRaisesRegex(ValueError, "is already an output of a different graph"):
            self.graph.outputs.append(self.value2)
        # Append is ok after the value is removed from the old graph
        other_graph.outputs.clear()
        self.graph.outputs.append(self.value2)
        self.assertTrue(self.value2.is_graph_output())
        self.assertIs(self.value2.graph, self.graph)

    def test_extend_outputs(self):
        self.graph.outputs.extend([self.value1, self.value2])
        self.assertIn(self.value1, self.graph.outputs)
        self.assertIn(self.value2, self.graph.outputs)

    def test_pop_from_outputs(self):
        self.graph.outputs.append(self.value2)
        popped = self.graph.outputs.pop()
        self.assertIs(popped, self.value2)
        self.assertNotIn(self.value2, self.graph.outputs)
        self.assertFalse(self.value2.is_graph_output())
        self.assertIsNone(self.value2.graph)

    def test_pop_from_duplicated_outputs(self):
        self.graph.outputs.extend([self.value1, self.value1])
        popped = self.graph.outputs.pop()
        self.assertIs(popped, self.value1)
        self.assertIn(self.value1, self.graph.outputs)
        self.assertTrue(self.value1.is_graph_output())
        self.assertIs(self.value1.graph, self.graph)

    def test_pop_from_outputs_raises_when_empty(self):
        with self.assertRaises(IndexError):
            self.graph.outputs.pop()

    def test_insert_into_outputs(self):
        self.graph.outputs.insert(0, self.value2)
        self.assertIs(self.graph.outputs[0], self.value2)
        self.assertTrue(self.value2.is_graph_output())
        self.assertIs(self.value2.graph, self.graph)

    def test_remove_from_outputs(self):
        self.graph.outputs.append(self.value2)
        self.graph.outputs.remove(self.value2)
        self.assertNotIn(self.value2, self.graph.outputs)
        self.assertFalse(self.value2.is_graph_output())
        self.assertIsNone(self.value2.graph)

    def test_clear_outputs(self):
        self.graph.outputs.extend([self.value1, self.value2])
        self.graph.outputs.clear()
        self.assertEqual(len(self.graph.outputs), 0)
        self.assertFalse(self.value1.is_graph_output())
        self.assertIsNone(self.value1.graph)
        self.assertFalse(self.value2.is_graph_output())
        self.assertIsNone(self.value2.graph)

    def test_clear_duplicated_outputs(self):
        self.graph.outputs.extend([self.value1, self.value1])
        self.graph.outputs.clear()
        self.assertEqual(len(self.graph.outputs), 0)
        self.assertFalse(self.value1.is_graph_output())
        self.assertIsNone(self.value1.graph)

    def test_outputs_set_items(self):
        self.graph.outputs.append(self.value1)
        self.graph.outputs[-1] = self.value2
        self.assertNotIn(self.value1, self.graph.outputs)
        self.assertIn(self.value2, self.graph.outputs)
        self.assertIs(self.graph.outputs[0], self.value2)
        self.assertTrue(self.value2.is_graph_output())
        self.assertIs(self.value2.graph, self.graph)
        self.assertFalse(self.value1.is_graph_output())
        self.assertIsNone(self.value1.graph)

    def test_outputs_set_items_slices(self):
        self.graph.outputs.extend([self.value1, self.value2])
        # Replace with one existing and one new output
        self.graph.outputs[0:2] = [self.value2, self.value3]
        self.assertNotIn(self.value1, self.graph.outputs)
        self.assertIn(self.value2, self.graph.outputs)
        self.assertIn(self.value3, self.graph.outputs)
        self.assertIs(self.value2.graph, self.graph)
        self.assertIs(self.value3.graph, self.graph)
        self.assertTrue(self.value2.is_graph_output())
        self.assertTrue(self.value3.is_graph_output())
        self.assertFalse(self.value1.is_graph_output())
        self.assertIsNone(self.value1.graph)

    def test_take_outputs(self):
        self.graph.outputs.extend([self.value1, self.value2, self.value3])
        outputs = self.graph.outputs[:2]
        self.graph.outputs.clear()
        self.graph.outputs.extend(outputs)
        self.assertEqual(len(self.graph.outputs), 2)
        self.assertEqual(self.graph.outputs, [self.value1, self.value2])
        self.assertTrue(self.value1.is_graph_output())
        self.assertTrue(self.value2.is_graph_output())
        self.assertFalse(self.value3.is_graph_output())
        self.assertIs(self.value1.graph, self.graph)
        self.assertIs(self.value2.graph, self.graph)
        self.assertIsNone(self.value3.graph)

    def test_outputs_copy(self):
        self.graph.outputs.extend([self.value1, self.value2])
        outputs_copy = self.graph.outputs.copy()
        self.assertEqual(outputs_copy, [self.value1, self.value2])
        self.assertIsNot(outputs_copy, self.graph.outputs)
        # Modifying the copy does not affect the original
        outputs_copy.append(self.value3)
        self.assertNotIn(self.value3, self.graph.outputs)
        self.assertIn(self.value3, outputs_copy)

    def test_initializers_setitem(self):
        self.graph.initializers["initializer1"] = self.value3
        self.assertIn("initializer1", self.graph.initializers)
        self.assertTrue(self.value3.is_initializer())
        self.assertIs(self.value3.graph, self.graph)
        # Replace initializer
        self.value1.name = "initializer1"
        self.graph.initializers["initializer1"] = self.value1
        self.assertIn("initializer1", self.graph.initializers)
        self.assertTrue(self.value1.is_initializer())
        self.assertIs(self.value1.graph, self.graph)
        self.assertFalse(self.value3.is_initializer())
        self.assertIsNone(self.value3.graph)

    def test_initializers_setitem_raises_when_key_does_not_match(self):
        with self.assertRaisesRegex(ValueError, "does not match the name of the value"):
            self.graph.initializers["some_key"] = self.value3

    def test_initializers_setitem_raises_when_it_belongs_to_another_graph(self):
        other_graph = _core.Graph(inputs=(), outputs=(), nodes=())
        other_graph.initializers["initializer1"] = self.value3
        with self.assertRaisesRegex(
            ValueError, "is already an initializer of a different graph"
        ):
            self.graph.initializers["initializer1"] = self.value3
        # Set is ok after the value is removed from the old graph
        other_graph.initializers.clear()
        self.graph.initializers["initializer1"] = self.value3
        self.assertIn("initializer1", self.graph.initializers)
        self.assertTrue(self.value3.is_initializer())
        self.assertIs(self.value3.graph, self.graph)

    def test_initializers_setitem_raises_when_value_does_not_have_a_name(self):
        self.value3.name = None
        with self.assertRaises(TypeError):
            self.graph.initializers[None] = self.value3

        with self.assertRaisesRegex(ValueError, "cannot be an empty string"):
            self.graph.initializers[""] = _core.Value(name="")

    def test_initializers_setitem_checks_value_name_match(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.graph.initializers["some_name"] = _core.Value(name="some_other_name")

    def test_initializers_setitem_assigns_key_to_value_name_if_not_set(self):
        value = _core.Value(name=None)
        self.graph.initializers["some_name"] = value
        self.assertEqual(value.name, "some_name")
        self.assertIs(value, self.graph.initializers["some_name"])

        value = _core.Value(name="")
        self.graph.initializers["some_other_name"] = value
        self.assertEqual(value.name, "some_other_name")
        self.assertIs(value, self.graph.initializers["some_other_name"])

    def test_initializers_setitem_checks_value_type(self):
        with self.assertRaisesRegex(TypeError, "must be a Value object"):
            self.graph.initializers["some_name"] = ir.tensor([1, 2, 3], name="some_tensor")

    def test_initializers_setitem_raises_when_value_is_node_output(self):
        node = ir.node("SomeOp", inputs=[])
        with self.assertRaisesRegex(ValueError, "produced by a node"):
            self.graph.initializers["some_name"] = node.outputs[0]

    def test_initializers_add_checks_value_name(self):
        # Initializers should always have a name
        with self.assertRaisesRegex(ValueError, "cannot be an empty string"):
            self.graph.initializers.add(_core.Value(name=""))

        with self.assertRaisesRegex(TypeError, "must be a string"):
            self.graph.initializers.add(_core.Value(name=None))

    def test_initializers_add_checks_value_type(self):
        # Initializers should be of type Value
        with self.assertRaisesRegex(TypeError, "must be a Value object"):
            self.graph.initializers.add(ir.tensor([1, 2, 3], name="some_tensor"))

    def test_delete_initializer(self):
        self.graph.initializers["initializer1"] = self.value3
        del self.graph.initializers["initializer1"]
        self.assertNotIn("initializer1", self.graph.initializers)
        self.assertFalse(self.value3.is_initializer())
        self.assertIsNone(self.value3.graph)

    def test_delete_initializer_raises_when_key_does_not_exist(self):
        with self.assertRaises(KeyError):
            del self.graph.initializers["non_existent"]

    def test_clear_initializers(self):
        self.graph.initializers["initializer1"] = self.value3
        self.graph.initializers.clear()
        self.assertEqual(len(self.graph.initializers), 0)
        self.assertFalse(self.value3.is_initializer())
        self.assertIsNone(self.value3.graph)

    def test_pop_initializer(self):
        self.graph.initializers["initializer1"] = self.value3
        popped = self.graph.initializers.pop("initializer1")
        self.assertEqual(popped, self.value3)
        self.assertNotIn("initializer1", self.graph.initializers)
        self.assertFalse(self.value3.is_initializer())
        self.assertIsNone(self.value3.graph)

    def test_update_initializers(self):
        self.graph.initializers["initializer1"] = self.value3
        new_initializer = _core.Value(name="initializer2")
        self.graph.initializers.update({new_initializer.name: new_initializer})
        self.assertIn(new_initializer.name, self.graph.initializers)
        self.assertTrue(new_initializer.is_initializer())
        self.assertEqual(new_initializer.graph, self.graph)
        self.assertIn("initializer1", self.graph.initializers)
        self.assertTrue(self.value3.is_initializer())
        self.assertEqual(self.value3.graph, self.graph)

    def test_iter_initializers(self):
        self.graph.initializers["initializer1"] = self.value3
        initializers = list(self.graph.initializers.values())
        self.assertEqual(len(initializers), 1)
        self.assertEqual(initializers[0].name, "initializer1")
        self.assertTrue(initializers[0].is_initializer())
        self.assertEqual(initializers[0].graph, self.graph)

    def test_contains_initializer(self):
        self.graph.initializers["initializer1"] = self.value3
        self.assertIn("initializer1", self.graph.initializers)
        self.assertTrue(self.value3.is_initializer())
        self.assertEqual(self.value3.graph, self.graph)

    def test_not_contains_initializer(self):
        self.assertNotIn("non_existent", self.graph.initializers)
        self.assertFalse(self.value3.is_initializer())
        self.assertIsNone(self.value3.graph)

    def test_initializer_can_be_added_as_input(self):
        self.graph.initializers["initializer1"] = self.value3
        self.graph.inputs.append(self.value3)
        self.assertIn(self.value3, self.graph.inputs)
        self.assertTrue(self.value3.is_graph_input())
        self.assertIs(self.value3.graph, self.graph)
        self.assertFalse(self.value3.is_graph_output())
        self.assertTrue(self.value3.is_initializer())

    def test_initializer_can_be_added_as_output(self):
        self.graph.initializers["initializer1"] = self.value3
        self.graph.outputs.append(self.value3)
        self.assertIn(self.value3, self.graph.outputs)
        self.assertTrue(self.value3.is_graph_output())
        self.assertIs(self.value3.graph, self.graph)
        self.assertFalse(self.value3.is_graph_input())
        self.assertTrue(self.value3.is_initializer())


class ModelTest(unittest.TestCase):
    def test_graphs_returns_all_subgraphs(self):
        # main_graph: nodes=[a,b,c,d,>,if], edges=[(a,>),(b,>),(>,if)], subgraphs={if:[then_graph,else_graph]}
        # then_graph: nodes=[sub], edges=[(c,sub),(d,sub)]
        # else_graph: nodes=[add], edges=[(c,add),(d,add)]
        v0 = _core.Value(name="va")
        v1 = _core.Value(name="vb")
        v2 = _core.Value(name="vc")
        v3 = _core.Value(name="vd")
        node0 = _core.Node("", "a", inputs=(v0,), num_outputs=1)
        node1 = _core.Node("", "b", inputs=(v1,), num_outputs=1)
        node2 = _core.Node("", "c", inputs=(v2,), num_outputs=1)
        node3 = _core.Node("", "d", inputs=(v3,), num_outputs=1)
        node4 = _core.Node(
            "", "sub", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1
        )
        node5 = _core.Node(
            "", "add", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1
        )
        node6 = _core.Node("", ">", inputs=(node0.outputs[0], node1.outputs[0]), num_outputs=1)
        then_graph = _core.Graph(
            inputs=(),
            outputs=(node4.outputs[0],),
            nodes=(node4,),
            name="then_graph",
        )
        else_graph = _core.Graph(
            inputs=(),
            outputs=(node5.outputs[0],),
            nodes=(node5,),
            name="else_graph",
        )
        node7 = _core.Node(
            "",
            "if",
            inputs=(node6.outputs[0],),
            num_outputs=1,
            attributes=[
                ir.AttrGraph("then_branch", then_graph),
                ir.AttrGraph("else_branch", else_graph),
            ],
        )
        main_graph = _core.Graph(
            inputs=(v0, v1, v2, v3),
            outputs=(node7.outputs[0],),
            nodes=(node0, node1, node2, node6, node7),
            name="main_graph",
        )
        model = _core.Model(main_graph, ir_version=10)
        self.assertEqual(
            tuple(model.graphs()),
            (main_graph, then_graph, else_graph),
        )


class TypeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("tensor", _core.TensorType(ir.DataType.FLOAT)),
            ("sequence", _core.SequenceType(_core.TensorType(ir.DataType.BOOL))),
            ("optional", _core.OptionalType(_core.TensorType(ir.DataType.FLOAT16))),
            (
                "sequence_optional",
                _core.SequenceType(_core.OptionalType(_core.TensorType(ir.DataType.INT8))),
            ),
            (
                "optional_sequence",
                _core.OptionalType(_core.SequenceType(_core.TensorType(ir.DataType.INT16))),
            ),
        ]
    )
    def test_type_is_hashable(self, _: str, type_: ir.TypeProtocol):
        self.assertIsInstance(hash(type_), int)
        self.assertIn(type_, {type_})  # type: ignore
        # Assert that a different type object can still be matched
        self.assertIn(copy.deepcopy(type_), {type_})  # type: ignore

    def test_type_is_comparable(self):
        self.assertEqual(
            _core.TensorType(ir.DataType.FLOAT), _core.TensorType(ir.DataType.FLOAT)
        )
        self.assertNotEqual(
            _core.TensorType(ir.DataType.FLOAT), _core.TensorType(ir.DataType.FLOAT16)
        )

    @parameterized.parameterized.expand(
        [
            ("tensor", _core.TensorType(ir.DataType.FLOAT)),
            ("sequence", _core.SequenceType(_core.TensorType(ir.DataType.BOOL))),
            ("optional", _core.OptionalType(_core.TensorType(ir.DataType.FLOAT16))),
            (
                "sequence_optional",
                _core.SequenceType(_core.OptionalType(_core.TensorType(ir.DataType.INT8))),
            ),
            (
                "optional_sequence",
                _core.OptionalType(_core.SequenceType(_core.TensorType(ir.DataType.INT16))),
            ),
        ]
    )
    def test_composite_type_is_comparable(self, _: str, type_: ir.TypeProtocol):
        self.assertEqual(type_, type_)
        # Equal even if deep-copied
        self.assertEqual(type_, copy.deepcopy(type_))


class AttrTest(unittest.TestCase):
    """Test the Attr class."""

    def test_init(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42, doc_string="test string")
        self.assertEqual(attr.name, "test")
        self.assertEqual(attr.value, 42)
        self.assertEqual(attr.type, ir.AttributeType.INT)
        self.assertEqual(attr.doc_string, "test string")

    def test_as_float(self):
        attr = _core.Attr("test", ir.AttributeType.FLOAT, 42.0)
        self.assertEqual(attr.as_float(), 42.0)

        attr_int_value = _core.Attr("test", ir.AttributeType.FLOAT, 42)
        self.assertEqual(attr_int_value.as_float(), 42.0)

    def test_as_int(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 0)
        self.assertEqual(attr.as_int(), 0)

    def test_as_string(self):
        attr = _core.Attr("test", ir.AttributeType.STRING, "test string")
        self.assertEqual(attr.as_string(), "test string")

    def test_as_tensor(self):
        attr = _core.Attr("test", ir.AttributeType.TENSOR, ir.tensor([42.0]))
        np.testing.assert_equal(attr.as_tensor().numpy(), np.array([42.0]))

    def test_as_graph(self):
        attr = _core.Attr("test", ir.AttributeType.GRAPH, _core.Graph((), (), nodes=()))
        self.assertIsInstance(attr.as_graph(), _core.Graph)

    def test_as_floats(self):
        attr = _core.Attr("test", ir.AttributeType.FLOATS, [42.0])
        self.assertEqual(tuple(attr.as_floats()), (42.0,))

    def test_as_ints(self):
        attr = _core.Attr("test", ir.AttributeType.INTS, [42])
        self.assertEqual(tuple(attr.as_ints()), (42,))

    def test_as_strings(self):
        attr = _core.Attr("test", ir.AttributeType.STRINGS, ["test string", ""])
        self.assertEqual(attr.as_strings(), ("test string", ""))

    def test_as_tensors(self):
        attr = _core.Attr("test", ir.AttributeType.TENSORS, [ir.tensor([42.0])])
        np.testing.assert_equal(attr.as_tensors()[0].numpy(), np.array([42.0]))

    def test_as_graphs(self):
        attr = _core.Attr("test", ir.AttributeType.GRAPHS, [_core.Graph((), (), nodes=())])
        self.assertIsInstance(attr.as_graphs()[0], _core.Graph)

    def test_as_float_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42)
        with self.assertRaises(TypeError):
            attr.as_float()

    def test_as_int_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.FLOAT, 42.0)
        with self.assertRaises(TypeError):
            attr.as_int()

    def test_as_string_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42)
        with self.assertRaises(TypeError):
            attr.as_string()

    def test_as_tensor_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42)
        with self.assertRaises(TypeError):
            attr.as_tensor()

    def test_as_graph_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42)
        with self.assertRaises(TypeError):
            attr.as_graph()

    def test_as_floats_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42)
        with self.assertRaises(TypeError):
            attr.as_floats()

    def test_as_ints_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.FLOAT, 42.0)
        with self.assertRaises(TypeError):
            attr.as_ints()

    def test_as_strings_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42)
        with self.assertRaises(TypeError):
            attr.as_strings()

    def test_as_tensors_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42)
        with self.assertRaises(TypeError):
            attr.as_tensors()

    def test_as_graphs_type_error(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42)
        with self.assertRaises(TypeError):
            attr.as_graphs()

    def test_meta(self):
        """Test that the meta property returns a MetadataStore and works correctly."""
        attr = _core.Attr("test", ir.AttributeType.INT, 42)

        # Test that meta property returns a MetadataStore
        meta = attr.meta
        self.assertIsInstance(meta, ir._metadata.MetadataStore)

        # Test that the same instance is returned on subsequent calls
        meta2 = attr.meta
        self.assertIs(meta, meta2)

        # Test that we can store and retrieve metadata
        attr.meta["source_line"] = 42
        attr.meta["source_file"] = "test.py"
        self.assertEqual(attr.meta["source_line"], 42)
        self.assertEqual(attr.meta["source_file"], "test.py")

        # Test metadata validity features
        attr.meta.invalidate("source_line")
        self.assertFalse(attr.meta.is_valid("source_line"))
        self.assertTrue(attr.meta.is_valid("source_file"))


class LazyTensorTest(unittest.TestCase):
    def test_lazy_tensor_initialization(self):
        def tensor_fn():
            return ir.tensor([1, 2, 3], dtype=ir.DataType.INT64)

        lazy_tensor = _core.LazyTensor(
            tensor_fn, dtype=ir.DataType.INT64, shape=ir.Shape((3,))
        )
        self.assertEqual(lazy_tensor.dtype, ir.DataType.INT64)
        self.assertEqual(lazy_tensor.shape, (3,))

    def test_lazy_tensor_numpy(self):
        def tensor_fn():
            return ir.tensor([1, 2, 3], dtype=ir.DataType.INT64)

        lazy_tensor = _core.LazyTensor(
            tensor_fn, dtype=ir.DataType.INT64, shape=ir.Shape((3,))
        )
        np.testing.assert_array_equal(lazy_tensor.numpy(), np.array([1, 2, 3]))

    def test_lazy_tensor_tobytes(self):
        def tensor_fn():
            return ir.tensor([1, 2, 3], dtype=ir.DataType.INT64)

        lazy_tensor = _core.LazyTensor(
            tensor_fn, dtype=ir.DataType.INT64, shape=ir.Shape((3,))
        )
        self.assertEqual(
            lazy_tensor.tobytes(),
            b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00",
        )


class PackedTensorTest(unittest.TestCase):
    """Test the PackedTensor class for 4-bit data types."""

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4),
            ("UINT4", ir.DataType.UINT4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_initialize_with_uint8_packed_data(self, _: str, dtype: ir.DataType):
        """Test initializing PackedTensor with pre-packed uint8 data."""
        # Create packed data - 4 elements packed into 2 uint8 values
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)  # [1,2] and [3,4] packed
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape, name="test_packed")

        self.assertEqual(tensor.dtype, dtype)
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.name, "test_packed")
        self.assertIs(tensor.raw, packed_data)

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4),
            ("UINT4", ir.DataType.UINT4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_initialize_with_torch_tensor(self, _: str, dtype: ir.DataType):
        packed_data = torch.tensor([424242], dtype=torch.uint32)
        shape = _core.Shape([2, 4])

        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape, name="test_packed")

        self.assertEqual(tensor.dtype, dtype)
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.name, "test_packed")
        self.assertIs(tensor.raw, packed_data)
        self.assertEqual(tensor.tobytes(), packed_data.numpy(force=True).tobytes())
        np.testing.assert_array_equal(
            tensor.numpy_packed().flatten(), packed_data.numpy(force=True).view(np.uint8)
        )
        np.testing.assert_array_equal(
            tensor.numpy(),
            _type_casting.unpack_4bitx2(
                packed_data.numpy(force=True).view(np.uint8), dims=[2, 4]
            ).view(dtype.numpy()),
        )

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4),
            ("UINT4", ir.DataType.UINT4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_initialize_raises_when_shape_is_incorrect(self, _: str, dtype: ir.DataType):
        """Test initializing PackedTensor with pre-packed uint8 data."""
        # Create packed data - 4 elements packed into 2 uint8 values
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)  # [1,2] and [3,4] packed
        shape = _core.Shape([42])  # Incorrect shape

        with self.assertRaisesRegex(ValueError, "Expected the packed array to be 21 bytes"):
            _core.PackedTensor(packed_data, dtype=dtype, shape=shape, name="test_packed")

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4, ml_dtypes.int4),
            ("UINT4", ir.DataType.UINT4, ml_dtypes.uint4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1, ml_dtypes.float4_e2m1fn),
        ]
    )
    def test_initialize_with_ml_dtypes_raises(self, _: str, dtype: ir.DataType, np_dtype):
        """Test initializing PackedTensor with ml_dtypes arrays."""
        # Create array with ml_dtypes - these will be automatically packed
        if dtype == ir.DataType.INT4:
            array = np.array([-8, -1, 0, 1, 2, 7], dtype=np_dtype)
        else:
            array = np.array([0, 1, 2, 7, 15, 8], dtype=np_dtype)
        shape = _core.Shape(array.shape)

        with self.assertRaisesRegex(TypeError, "PackedTensor expects the value to be packed"):
            _core.PackedTensor(array, dtype=dtype, shape=shape)

    def test_initialize_raises_when_dtype_not_packed(self):
        """Test that PackedTensor raises error for non-packed data types."""
        array = np.array([1, 2, 3, 4], dtype=np.uint8)
        shape = _core.Shape([4])

        with self.assertRaises(TypeError) as cm:
            _core.PackedTensor(array, dtype=ir.DataType.FLOAT, shape=shape)

        self.assertIn(
            "PackedTensor only supports INT2, UINT2, INT4, UINT4, FLOAT4E2M1",
            str(cm.exception),
        )

    def test_initialize_raises_when_value_not_array_compatible(self):
        """Test that PackedTensor raises error for non-array compatible values."""
        with self.assertRaisesRegex(TypeError, "Expected an array compatible object"):
            _core.PackedTensor(42, dtype=ir.DataType.INT4, shape=_core.Shape([1]))

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4, ml_dtypes.int4, [-8, -1, 0, 1], [0xF8, 0x10]),
            ("UINT4", ir.DataType.UINT4, ml_dtypes.uint4, [0, 1, 2, 7], [0x10, 0x72]),
            (
                "FLOAT4E2M1",
                ir.DataType.FLOAT4E2M1,
                ml_dtypes.float4_e2m1fn,
                [0, 1, 2, 3],
                None,
            ),
        ]
    )
    def test_numpy_returns_unpacked_data_for_all_types(
        self, _: str, dtype: ir.DataType, np_dtype, values, packed_bytes
    ):
        """Test that numpy() returns unpacked data for all 4-bit types."""
        values_array = np.array(values, dtype=np_dtype)

        if packed_bytes is not None:
            # Use pre-computed packed bytes for INT4 and UINT4
            packed_data = np.array(packed_bytes, dtype=np.uint8)
        else:
            # Use type casting for FLOAT4E2M1
            packed_data = _type_casting.pack_4bitx2(values_array)

        shape = _core.Shape([len(values)])
        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape)
        result = tensor.numpy()

        np.testing.assert_array_equal(result, values_array)
        self.assertEqual(result.dtype, np_dtype)

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4, ml_dtypes.int4),
            ("UINT4", ir.DataType.UINT4, ml_dtypes.uint4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1, ml_dtypes.float4_e2m1fn),
        ]
    )
    def test_tobytes_for_all_types(self, _: str, dtype: ir.DataType, np_dtype):
        """Test that tobytes() works correctly for all 4-bit types."""
        if dtype == ir.DataType.INT4:
            values = [-8, -1, 0, 1]
        else:
            values = [0, 1, 2, 3]

        values_array = np.array(values, dtype=np_dtype)
        packed_data = _type_casting.pack_4bitx2(values_array)
        shape = _core.Shape([len(values)])

        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape)
        result_bytes = tensor.tobytes()
        expected_bytes = packed_data.tobytes()

        self.assertEqual(result_bytes, expected_bytes)

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4, ml_dtypes.int4),
            ("UINT4", ir.DataType.UINT4, ml_dtypes.uint4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1, ml_dtypes.float4_e2m1fn),
        ]
    )
    def test_odd_sized_arrays_for_all_types(self, _: str, dtype: ir.DataType, np_dtype):
        """Test odd-sized arrays work correctly for all 4-bit types."""
        if dtype == ir.DataType.INT4:
            values = [-8, -1, 0, 1, 2]  # 5 elements
        else:
            values = [0, 1, 2, 3, 4]  # 5 elements

        values_array = np.array(values, dtype=np_dtype)
        packed_data = _type_casting.pack_4bitx2(values_array)
        shape = _core.Shape([len(values)])

        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape)
        result = tensor.numpy()

        np.testing.assert_array_equal(result, values_array)
        self.assertEqual(result.dtype, np_dtype)

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4),
            ("UINT4", ir.DataType.UINT4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_numpy_packed_for_all_types(self, _: str, dtype: ir.DataType):
        """Test that numpy_packed() returns raw packed data for all types."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape)
        result = tensor.numpy_packed()

        np.testing.assert_array_equal(result, packed_data)
        self.assertEqual(result.dtype, np.uint8)

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4),
            ("UINT4", ir.DataType.UINT4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_dlpack_methods_for_all_types(self, _: str, dtype: ir.DataType):
        """Test DLPack methods work for all 4-bit types."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape)

        # Should be able to get DLPack representation
        dlpack_tensor = tensor.__dlpack__()
        self.assertIsNotNone(dlpack_tensor)

        # Should be able to get device info
        device_info = tensor.__dlpack_device__()
        self.assertIsInstance(device_info, tuple)
        self.assertEqual(len(device_info), 2)

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4),
            ("UINT4", ir.DataType.UINT4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_properties_for_all_types(self, _: str, dtype: ir.DataType):
        """Test that properties work correctly for all 4-bit types."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape, name="test")

        # Test basic properties
        self.assertEqual(tensor.dtype, dtype)
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.name, "test")
        self.assertEqual(tensor.size, 4)
        self.assertEqual(tensor.nbytes, 2)  # 4 elements * 0.5 bytes each = 2 bytes
        self.assertTrue(tensor.shape.frozen)
        self.assertIs(tensor.raw, packed_data)

    def test_array_method_returns_unpacked_numpy_array(self):
        """Test that __array__ method returns unpacked numpy array."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)
        result = tensor.__array__()

        expected = np.array([1, 2, 3, 4], dtype=ml_dtypes.uint4)
        np.testing.assert_array_equal(result, expected)

    def test_repr_returns_string_representation(self):
        """Test that __repr__ returns a meaningful string representation."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(
            packed_data, dtype=ir.DataType.UINT4, shape=shape, name="test_tensor"
        )
        result = repr(tensor)

        self.assertIsInstance(result, str)
        self.assertIn("PackedTensor", result)
        self.assertIn("UINT4", result)
        self.assertIn("[4]", result)
        self.assertIn("test_tensor", result)

    def test_properties_are_immutable(self):
        """Test that dtype, shape, and raw properties are immutable."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)

        # Properties should return the correct values
        self.assertEqual(tensor.dtype, ir.DataType.UINT4)
        self.assertEqual(tensor.shape, shape)
        self.assertIs(tensor.raw, packed_data)

    def test_shape_is_frozen_after_initialization(self):
        """Test that the shape is frozen after PackedTensor initialization."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)

        self.assertTrue(tensor.shape.frozen)

    def test_metadata_properties(self):
        """Test metadata and metadata_props properties work correctly."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])
        metadata_props = {"test_key": "test_value"}

        tensor = _core.PackedTensor(
            packed_data, dtype=ir.DataType.UINT4, shape=shape, metadata_props=metadata_props
        )

        # Test metadata_props
        self.assertEqual(tensor.metadata_props["test_key"], "test_value")

        # Test meta store
        tensor.meta["analysis_key"] = 42
        self.assertEqual(tensor.meta["analysis_key"], 42)

    def test_doc_string_property(self):
        """Test doc_string property works correctly."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])
        doc_string = "Test packed tensor documentation"

        tensor = _core.PackedTensor(
            packed_data, dtype=ir.DataType.UINT4, shape=shape, doc_string=doc_string
        )

        self.assertEqual(tensor.doc_string, doc_string)

    def test_size_and_nbytes_properties(self):
        """Test size and nbytes properties are calculated correctly."""
        packed_data = np.array([0x21, 0x43, 0x05], dtype=np.uint8)  # 5 elements packed
        shape = _core.Shape([5])

        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)

        # Size should be the number of elements
        self.assertEqual(tensor.size, 5)

        # nbytes should account for 4-bit elements (0.5 bytes each, rounded up)
        # 5 elements * 0.5 bytes = 2.5 bytes, rounded up to 3 bytes
        expected_nbytes = 3  # math.ceil(5 * 0.5)
        self.assertEqual(tensor.nbytes, expected_nbytes)

    def test_empty_tensor(self):
        """Test PackedTensor with empty data."""
        packed_data = np.array([], dtype=np.uint8)
        shape = _core.Shape([0])

        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)

        self.assertEqual(tensor.size, 0)
        self.assertEqual(tensor.nbytes, 0)
        result = tensor.numpy()
        self.assertEqual(result.size, 0)
        self.assertEqual(result.dtype, ml_dtypes.uint4)

    @parameterized.parameterized.expand(
        [
            ("2D", [2, 3]),
            ("3D", [2, 2, 2]),
            ("4D", [1, 2, 2, 2]),
        ]
    )
    def test_multidimensional_shapes(self, _: str, dims):
        """Test PackedTensor with multidimensional shapes."""
        total_elements = np.prod(dims)
        # Need enough packed bytes for the elements (round up for odd counts)
        packed_size = (total_elements + 1) // 2
        packed_data = np.arange(packed_size, dtype=np.uint8)
        shape = _core.Shape(dims)

        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)
        result = tensor.numpy()

        self.assertEqual(result.shape, tuple(dims))
        self.assertEqual(result.size, total_elements)

    def test_integration_with_regular_tensor_operations(self):
        """Test that PackedTensor integrates well with numpy operations."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)  # [1,2,3,4]
        shape = _core.Shape([4])

        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)

        # Should be able to use with numpy functions
        np_array = np.array(tensor)
        expected = np.array([1, 2, 3, 4], dtype=ml_dtypes.uint4)
        np.testing.assert_array_equal(np_array, expected)

        # Should be able to get numpy array and perform operations
        result = tensor.numpy()
        self.assertEqual(result.sum(), 10)  # 1+2+3+4 = 10

    @parameterized.parameterized.expand(
        [
            ("INT4", ir.DataType.INT4),
            ("UINT4", ir.DataType.UINT4),
            ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_tobytes_big_endian_handling(self, _: str, dtype: ir.DataType):
        """Test that PackedTensor.tobytes() correctly handles byte order conversion."""
        # Create packed data
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])
        tensor = _core.PackedTensor(packed_data, dtype=dtype, shape=shape)

        # Mock _IS_LITTLE_ENDIAN to simulate big endian system
        with unittest.mock.patch("onnx_ir._core._IS_LITTLE_ENDIAN", False):
            result_bytes = tensor.tobytes()

        # Verify that the result is in little endian format regardless of system endianness
        expected_bytes = packed_data.astype(packed_data.dtype.newbyteorder("<")).tobytes()
        self.assertEqual(result_bytes, expected_bytes)

    def test_tofile_packed_tensor(self):
        """Test tofile() method works correctly for PackedTensor."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])
        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)

        with tempfile.NamedTemporaryFile() as temp_file:
            tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Should be the same as tobytes()
        self.assertEqual(result_bytes, tensor.tobytes())

    def test_tofile_packed_tensor_big_endian_handling(self):
        """Test tofile() big endian handling for PackedTensor."""
        packed_data = np.array([0x21, 0x43], dtype=np.uint8)
        shape = _core.Shape([4])
        tensor = _core.PackedTensor(packed_data, dtype=ir.DataType.UINT4, shape=shape)

        with tempfile.NamedTemporaryFile() as temp_file:
            # Mock _IS_LITTLE_ENDIAN to simulate big endian system
            with unittest.mock.patch("onnx_ir._core._IS_LITTLE_ENDIAN", False):
                tensor.tofile(temp_file)
            temp_file.seek(0)
            result_bytes = temp_file.read()

        # Should still produce little endian output
        expected_bytes = packed_data.astype(packed_data.dtype.newbyteorder("<")).tobytes()
        self.assertEqual(result_bytes, expected_bytes)


class StringTensorTest(unittest.TestCase):
    def test_nbytes(self):
        data = np.array([b"A", b"BC", b"D"])
        tensor = _core.StringTensor(data)
        self.assertEqual(tensor.nbytes, 4)

    def test_nbytes_2d(self):
        data = np.array([[b"A", b"BC", b"D"], [b"EFG", b"H", b"I"]])
        tensor = _core.StringTensor(data)
        self.assertEqual(tensor.nbytes, 9)

    def test_nbytes_empty(self):
        data = np.array([])
        tensor = _core.StringTensor(data)
        self.assertEqual(tensor.nbytes, 0)

    def test_nbytes_single(self):
        data = np.array([b"ABC"])
        tensor = _core.StringTensor(data)
        self.assertEqual(tensor.nbytes, 3)


class GraphCloneTest(unittest.TestCase):
    """Test the Graph.clone() method."""

    def test_clone_simple_graph(self):
        """Test cloning a simple graph with basic nodes."""
        v0 = _core.Value(name="input1")
        v1 = _core.Value(name="input2")
        node = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1, name="add_node")
        graph = _core.Graph(
            inputs=(v0, v1),
            outputs=(node.outputs[0],),
            nodes=(node,),
            name="simple_graph",
            doc_string="A simple graph",
        )

        cloned_graph = graph.clone()

        # Verify graph properties are copied
        self.assertEqual(cloned_graph.name, graph.name)
        self.assertEqual(cloned_graph.doc_string, graph.doc_string)
        self.assertIsNot(cloned_graph, graph)

        # Verify nodes are different objects
        self.assertEqual(len(cloned_graph), len(graph))
        cloned_node = cloned_graph[0]
        self.assertIsNot(cloned_node, node)
        self.assertEqual(cloned_node.op_type, node.op_type)
        self.assertEqual(cloned_node.name, node.name)

        # Verify inputs are different objects
        self.assertEqual(len(cloned_graph.inputs), len(graph.inputs))
        self.assertIsNot(cloned_graph.inputs[0], v0)
        self.assertIsNot(cloned_graph.inputs[1], v1)
        self.assertEqual(cloned_graph.inputs[0].name, v0.name)
        self.assertEqual(cloned_graph.inputs[1].name, v1.name)

        # Verify outputs are different objects
        self.assertEqual(len(cloned_graph.outputs), len(graph.outputs))
        self.assertIsNot(cloned_graph.outputs[0], node.outputs[0])

    def test_clone_graph_with_initializers(self):
        """Test cloning a graph with initializers."""
        v0 = _core.Value(name="input1")
        initializer_tensor = ir.tensor([1.0, 2.0, 3.0], name="weights")
        v_init = _core.Value(name="weights", const_value=initializer_tensor)
        node = _core.Node("", "Mul", inputs=(v0, v_init), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
            initializers=(v_init,),
        )

        cloned_graph = graph.clone()

        # Verify initializers are cloned
        self.assertEqual(len(cloned_graph.initializers), len(graph.initializers))
        self.assertIn("weights", cloned_graph.initializers)
        cloned_init = cloned_graph.initializers["weights"]
        self.assertIsNot(cloned_init, v_init)
        self.assertEqual(cloned_init.name, v_init.name)

        # Verify tensor is shared (not deep copied)
        self.assertIsNotNone(cloned_init.const_value)
        self.assertIs(cloned_init.const_value, initializer_tensor)

    def test_clone_graph_with_subgraphs(self):
        """Test cloning a graph with subgraphs (e.g., If node)."""
        v0 = _core.Value(name="condition")
        v1 = _core.Value(name="x")
        v2 = _core.Value(name="y")

        # Create then branch
        add_node = _core.Node("", "Add", inputs=(v1, v2), num_outputs=1)
        then_graph = _core.Graph(
            inputs=(),
            outputs=(add_node.outputs[0],),
            nodes=(add_node,),
            name="then_branch",
        )

        # Create else branch
        sub_node = _core.Node("", "Sub", inputs=(v1, v2), num_outputs=1)
        else_graph = _core.Graph(
            inputs=(),
            outputs=(sub_node.outputs[0],),
            nodes=(sub_node,),
            name="else_branch",
        )

        # Create If node
        if_node = _core.Node(
            "",
            "If",
            inputs=(v0,),
            num_outputs=1,
            attributes=[
                ir.AttrGraph("then_branch", then_graph),
                ir.AttrGraph("else_branch", else_graph),
            ],
        )
        main_graph = _core.Graph(
            inputs=(v0, v1, v2),
            outputs=(if_node.outputs[0],),
            nodes=(if_node,),
            name="main_graph",
        )

        cloned_graph = main_graph.clone()

        # Verify subgraphs are cloned
        cloned_if_node = cloned_graph[0]
        self.assertIsNot(cloned_if_node, if_node)

        cloned_then = cloned_if_node.attributes["then_branch"].value
        cloned_else = cloned_if_node.attributes["else_branch"].value

        self.assertIsNot(cloned_then, then_graph)
        self.assertIsNot(cloned_else, else_graph)
        self.assertEqual(cloned_then.name, then_graph.name)
        self.assertEqual(cloned_else.name, else_graph.name)

        # Verify nodes in subgraphs are cloned
        self.assertEqual(len(cloned_then), len(then_graph))
        cloned_then_node = cloned_then[0]
        self.assertIsNot(cloned_then_node, add_node)
        self.assertEqual(cloned_then_node.op_type, add_node.op_type)

    def test_clone_graph_with_metadata(self):
        """Test cloning a graph with metadata."""
        v0 = _core.Value(name="input")
        node = _core.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )
        graph.metadata_props["prop_key"] = "prop_value"

        cloned_graph = graph.clone()

        # Verify metadata_props are copied
        self.assertEqual(
            cloned_graph.metadata_props["prop_key"], graph.metadata_props["prop_key"]
        )
        self.assertIsNot(cloned_graph.metadata_props, graph.metadata_props)

        # Note: meta is NOT cloned - it's a separate runtime dictionary

    def test_clone_preserves_node_attributes(self):
        """Test that cloning preserves node attributes."""
        v0 = _core.Value(name="input")
        node = _core.Node(
            "",
            "Clip",
            inputs=(v0,),
            num_outputs=1,
            attributes=[
                ir.AttrFloat32("min", -1.0),
                ir.AttrFloat32("max", 1.0),
            ],
        )
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )

        cloned_graph = graph.clone()
        cloned_node = cloned_graph[0]

        # Verify attributes are cloned
        self.assertEqual(len(cloned_node.attributes), len(node.attributes))
        self.assertEqual(cloned_node.attributes["min"].value, -1.0)
        self.assertEqual(cloned_node.attributes["max"].value, 1.0)

    def test_clone_preserves_value_types_and_shapes(self):
        """Test that cloning preserves value types and shapes on inputs."""
        v0 = _core.Value(
            name="input",
            shape=_core.Shape([1, 3, 224, 224]),
            type=_core.TensorType(ir.DataType.FLOAT),
        )
        node = _core.Node("", "Identity", inputs=(v0,), num_outputs=1)

        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )

        cloned_graph = graph.clone()
        cloned_input = cloned_graph.inputs[0]

        # Verify input shapes and types are preserved
        self.assertEqual(cloned_input.shape, v0.shape)
        self.assertEqual(cloned_input.dtype, v0.dtype)

        # Verify the cloned graph has the same structure
        self.assertEqual(len(cloned_graph.inputs), len(graph.inputs))
        self.assertEqual(len(cloned_graph.outputs), len(graph.outputs))

    def test_clone_empty_graph(self):
        """Test cloning an empty graph."""
        graph = _core.Graph(inputs=(), outputs=(), nodes=())
        cloned_graph = graph.clone()

        self.assertIsNot(cloned_graph, graph)
        self.assertEqual(len(cloned_graph.inputs), 0)
        self.assertEqual(len(cloned_graph.outputs), 0)
        self.assertEqual(len(list(cloned_graph)), 0)

    def test_clone_graph_maintains_topology(self):
        """Test that cloning preserves the graph topology."""
        v0 = _core.Value(name="input")
        node1 = _core.Node("", "Add", inputs=(v0, v0), num_outputs=1, name="add1")
        node2 = _core.Node("", "Mul", inputs=(node1.outputs[0], v0), num_outputs=1, name="mul")
        node3 = _core.Node(
            "", "Sub", inputs=(node2.outputs[0], node1.outputs[0]), num_outputs=1, name="sub"
        )

        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node3.outputs[0],),
            nodes=(node1, node2, node3),
        )

        cloned_graph = graph.clone()
        cloned_nodes = list(cloned_graph)

        # Verify nodes are in the same order
        self.assertEqual(len(cloned_nodes), 3)
        self.assertEqual(cloned_nodes[0].name, "add1")
        self.assertEqual(cloned_nodes[1].name, "mul")
        self.assertEqual(cloned_nodes[2].name, "sub")

        # Verify connections are preserved
        # node2 should take input from node1's output
        self.assertIs(cloned_nodes[1].inputs[0], cloned_nodes[0].outputs[0])
        # node3 should take input from node2's output and node1's output
        self.assertIs(cloned_nodes[2].inputs[0], cloned_nodes[1].outputs[0])
        self.assertIs(cloned_nodes[2].inputs[1], cloned_nodes[0].outputs[0])

    def test_clone_with_allow_outer_scope_values_false_raises_error(self):
        # Create main graph
        v0 = _core.Value(name="main_input")

        # Create subgraph that references the outer scope value v0
        sub_node = _core.Node("", "Add", inputs=(v0, v0), name="sub_add")
        subgraph = _core.Graph(
            inputs=(),  # No inputs - references outer scope
            outputs=(sub_node.outputs[0],),
            nodes=(sub_node,),
            name="subgraph",
        )

        # The outer scope value v0 will cause an error
        with self.assertRaisesRegex(RuntimeError, "In clone_graph with args"):
            # The error captured is not the direct error but a wrapped one from clone()
            _ = subgraph.clone()

    def test_clone_with_allow_outer_scope_values(self):
        """Test that outer scope values are preserved when allow_outer_scope_values=True."""
        # Create main graph
        v0 = _core.Value(name="main_input")

        # Create subgraph that references the outer scope value v0
        sub_node = _core.Node("", "Add", inputs=(v0, v0), name="sub_add")
        subgraph = _core.Graph(
            inputs=(),  # No inputs - references outer scope
            outputs=(sub_node.outputs[0],),
            nodes=(sub_node,),
            name="subgraph",
        )

        # Clone the subgraph with allow_outer_scope_values=True
        cloned_subgraph = subgraph.clone(allow_outer_scope_values=True)

        # The outer scope value v0 should NOT be cloned, it should be preserved
        cloned_sub_node = cloned_subgraph[0]
        self.assertIs(cloned_sub_node.inputs[0], v0)
        self.assertIs(cloned_sub_node.inputs[1], v0)

    def test_clone_with_allow_outer_scope_values_mixed_scope(self):
        """Test allow_outer_scope_values with both inner and outer scope values."""
        # Create outer scope values
        outer_v1 = _core.Value(name="outer_value1")
        outer_v2 = _core.Value(name="outer_value2")

        # Create subgraph with its own input and using outer scope values
        inner_input = _core.Value(name="inner_input")
        sub_node1 = _core.Node("", "Add", inputs=(inner_input, outer_v1))
        sub_node2 = _core.Node("", "Mul", inputs=(sub_node1.outputs[0], outer_v2))
        subgraph = _core.Graph(
            inputs=(inner_input,),
            outputs=(sub_node2.outputs[0],),
            nodes=(sub_node1, sub_node2),
            name="mixed_subgraph",
        )

        # Clone with allow_outer_scope_values=True
        cloned_subgraph = subgraph.clone(allow_outer_scope_values=True)

        # Inner input should be cloned
        cloned_inner_input = cloned_subgraph.inputs[0]
        self.assertIsNot(cloned_inner_input, inner_input)
        self.assertEqual(cloned_inner_input.name, inner_input.name)

        # Outer scope values should be preserved
        cloned_node1 = cloned_subgraph[0]
        cloned_node2 = cloned_subgraph[1]

        self.assertIs(cloned_node1.inputs[0], cloned_inner_input)  # Cloned inner
        self.assertIs(cloned_node1.inputs[1], outer_v1)  # Preserved outer
        self.assertIs(cloned_node2.inputs[1], outer_v2)  # Preserved outer

    def test_clone_with_allow_outer_scope_values_nested_subgraphs(self):
        """Test allow_outer_scope_values with nested subgraphs."""
        # Create outer value
        outer_value = _core.Value(name="outer")

        # Create inner subgraph that uses outer value
        inner_node = _core.Node("", "Identity", inputs=(outer_value,))
        inner_subgraph = _core.Graph(
            inputs=(),
            outputs=(inner_node.outputs[0],),
            nodes=(inner_node,),
            name="inner_subgraph",
        )

        # Create middle subgraph with If node containing the inner subgraph
        condition = _core.Value(name="condition")
        if_node = _core.Node(
            "",
            "If",
            inputs=(condition,),
            num_outputs=1,
            attributes=[
                ir.AttrGraph("then_branch", inner_subgraph),
                ir.AttrGraph("else_branch", inner_subgraph),
            ],
        )
        middle_subgraph = _core.Graph(
            inputs=(condition,),
            outputs=(if_node.outputs[0],),
            nodes=(if_node,),
            name="middle_subgraph",
        )

        # Clone with allow_outer_scope_values=True
        cloned_middle = middle_subgraph.clone(allow_outer_scope_values=True)

        # Condition should be cloned (it's an input)
        cloned_condition = cloned_middle.inputs[0]
        self.assertIsNot(cloned_condition, condition)

        # Check nested subgraphs preserve outer values
        cloned_if = cloned_middle[0]
        cloned_then = cloned_if.attributes["then_branch"].value
        cloned_else = cloned_if.attributes["else_branch"].value

        # Inner subgraphs should preserve the outer_value
        self.assertIs(cloned_then[0].inputs[0], outer_value)
        self.assertIs(cloned_else[0].inputs[0], outer_value)

    def test_clone_with_allow_outer_scope_values_with_initializers(self):
        """Test that initializers in the subgraph are cloned, not preserved."""
        # Create outer value
        outer_value = _core.Value(name="outer")

        # Create subgraph with initializer
        initializer_tensor = ir.tensor([1.0, 2.0, 3.0], name="weights")
        v_init = _core.Value(name="weights", const_value=initializer_tensor)
        sub_node = _core.Node("", "Add", inputs=(outer_value, v_init))
        subgraph = _core.Graph(
            inputs=(),
            outputs=(sub_node.outputs[0],),
            nodes=(sub_node,),
            initializers=(v_init,),
            name="subgraph_with_init",
        )

        # Clone with allow_outer_scope_values=True
        cloned_subgraph = subgraph.clone(allow_outer_scope_values=True)

        # Outer value should be preserved
        cloned_node = cloned_subgraph[0]
        self.assertIs(cloned_node.inputs[0], outer_value)

        # Initializer should be cloned (it's local to the subgraph)
        cloned_init = cloned_subgraph.initializers["weights"]
        self.assertIsNot(cloned_init, v_init)
        self.assertEqual(cloned_init.name, v_init.name)
        # But the tensor itself should be shared
        self.assertIs(cloned_init.const_value, initializer_tensor)

    def test_clone_with_allow_outer_scope_values_outer_initializer(self):
        """Test that initializers from outer graph are preserved when allow_outer_scope_values=True."""
        # Create outer graph with initializer
        initializer_tensor = ir.tensor([1.0, 2.0, 3.0], name="outer_weights")
        outer_init = _core.Value(name="outer_weights", const_value=initializer_tensor)
        outer_input = _core.Value(name="outer_input")
        outer_node = _core.Node("", "Add", inputs=(outer_input, outer_init))
        _outer_graph = _core.Graph(
            inputs=(outer_input,),
            outputs=(outer_node.outputs[0],),
            nodes=(outer_node,),
            initializers=(outer_init,),
            name="outer_graph",
        )

        # Create subgraph that references the outer initializer
        inner_value = _core.Value(name="inner_value")
        sub_node = _core.Node("", "Mul", inputs=(inner_value, outer_init))
        subgraph = _core.Graph(
            inputs=(inner_value,),
            outputs=(sub_node.outputs[0],),
            nodes=(sub_node,),
            initializers=(),  # No local initializers, references outer
            name="subgraph",
        )

        # Clone the subgraph with allow_outer_scope_values=True
        cloned_subgraph = subgraph.clone(allow_outer_scope_values=True)

        # The outer initializer should be preserved (not cloned)
        cloned_node = cloned_subgraph[0]
        self.assertIs(cloned_node.inputs[1], outer_init)

        # The inner input should be cloned
        cloned_inner = cloned_subgraph.inputs[0]
        self.assertIsNot(cloned_inner, inner_value)
        self.assertEqual(cloned_inner.name, inner_value.name)

    def test_clone_with_allow_outer_scope_values_node_outputs(self):
        """Test that node outputs within the subgraph are cloned, not preserved."""
        # Create outer value
        outer_value = _core.Value(name="outer")

        # Create subgraph with chain of nodes
        node1 = _core.Node("", "Identity", inputs=(outer_value,))
        node2 = _core.Node("", "Identity", inputs=(node1.outputs[0],))
        subgraph = _core.Graph(
            inputs=(),
            outputs=(node2.outputs[0],),
            nodes=(node1, node2),
            name="chained_subgraph",
        )

        # Clone with allow_outer_scope_values=True
        cloned_subgraph = subgraph.clone(allow_outer_scope_values=True)

        # Outer value should be preserved
        cloned_node1 = cloned_subgraph[0]
        self.assertIs(cloned_node1.inputs[0], outer_value)

        # Internal connection should use cloned values
        cloned_node2 = cloned_subgraph[1]
        self.assertIsNot(cloned_node2.inputs[0], node1.outputs[0])
        self.assertIs(cloned_node2.inputs[0], cloned_node1.outputs[0])


class ModelCloneTest(unittest.TestCase):
    """Test the Model.clone() method."""

    def test_clone_simple_model(self):
        """Test cloning a simple model."""
        v0 = _core.Value(name="input")
        node = _core.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
            name="main_graph",
        )
        model = _core.Model(
            graph,
            ir_version=10,
            producer_name="test_producer",
            producer_version="1.0",
            domain="test.domain",
            model_version=42,
            doc_string="Test model",
        )

        cloned_model = model.clone()

        # Verify model properties are copied
        self.assertIsNot(cloned_model, model)
        self.assertEqual(cloned_model.ir_version, model.ir_version)
        self.assertEqual(cloned_model.producer_name, model.producer_name)
        self.assertEqual(cloned_model.producer_version, model.producer_version)
        self.assertEqual(cloned_model.domain, model.domain)
        self.assertEqual(cloned_model.model_version, model.model_version)
        self.assertEqual(cloned_model.doc_string, model.doc_string)

        # Verify graph is cloned
        self.assertIsNot(cloned_model.graph, model.graph)
        self.assertEqual(cloned_model.graph.name, model.graph.name)

    def test_clone_model_with_functions(self):
        """Test cloning a model with local functions."""
        # Create a simple function
        func_input = _core.Value(name="func_input")
        func_node = _core.Node("", "Identity", inputs=(func_input,), num_outputs=1)
        func_graph = _core.Graph(
            inputs=(func_input,),
            outputs=(func_node.outputs[0],),
            nodes=(func_node,),
            name="func_graph",
        )
        function = _core.Function(
            domain="custom.domain",
            name="CustomFunc",
            graph=func_graph,
            attributes=[],
        )

        # Create main graph
        v0 = _core.Value(name="input")
        node = _core.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )

        model = _core.Model(
            graph,
            ir_version=10,
            functions=[function],
        )

        cloned_model = model.clone()

        # Verify functions are cloned
        self.assertEqual(len(cloned_model.functions), len(model.functions))
        cloned_func = cloned_model.functions[("custom.domain", "CustomFunc", "")]
        original_func = model.functions[("custom.domain", "CustomFunc", "")]

        self.assertIsNot(cloned_func, original_func)
        self.assertEqual(cloned_func.domain, original_func.domain)
        self.assertEqual(cloned_func.name, original_func.name)

    def test_clone_model_with_metadata_props(self):
        """Test cloning a model with metadata properties."""
        v0 = _core.Value(name="input")
        node = _core.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )
        model = _core.Model(graph, ir_version=10)
        model.metadata_props["author"] = "test_author"
        model.metadata_props["version"] = "1.0.0"

        cloned_model = model.clone()

        # Verify metadata_props are copied
        self.assertEqual(cloned_model.metadata_props["author"], model.metadata_props["author"])
        self.assertEqual(
            cloned_model.metadata_props["version"], model.metadata_props["version"]
        )
        self.assertIsNot(cloned_model.metadata_props, model.metadata_props)

    def test_clone_model_with_complex_graph(self):
        """Test cloning a model with a complex graph including subgraphs."""
        v0 = _core.Value(name="cond")
        v1 = _core.Value(name="x")

        # Create subgraph
        sub_node = _core.Node("", "Neg", inputs=(v1,), num_outputs=1)
        then_graph = _core.Graph(
            inputs=(),
            outputs=(sub_node.outputs[0],),
            nodes=(sub_node,),
        )

        # Create main graph with If node
        if_node = _core.Node(
            "",
            "If",
            inputs=(v0,),
            num_outputs=1,
            attributes=[ir.AttrGraph("then_branch", then_graph)],
        )
        main_graph = _core.Graph(
            inputs=(v0, v1),
            outputs=(if_node.outputs[0],),
            nodes=(if_node,),
        )

        model = _core.Model(main_graph, ir_version=10)
        cloned_model = model.clone()

        # Verify subgraphs are cloned
        cloned_if_node = cloned_model.graph[0]
        cloned_then = cloned_if_node.attributes["then_branch"].value
        original_then = if_node.attributes["then_branch"].value

        self.assertIsNot(cloned_then, original_then)
        self.assertEqual(len(cloned_then), len(original_then))

    def test_clone_model_opset_imports(self):
        """Test that cloning preserves opset imports."""
        v0 = _core.Value(name="input")
        node = _core.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
            opset_imports={"": 18, "custom.domain": 1},
        )
        model = _core.Model(graph, ir_version=10)

        cloned_model = model.clone()

        # Verify opset imports are preserved
        self.assertEqual(cloned_model.graph.opset_imports, model.graph.opset_imports)
        self.assertIsNot(cloned_model.graph.opset_imports, model.graph.opset_imports)


class FunctionCloneTest(unittest.TestCase):
    """Test the Function.clone() method."""

    def test_clone_simple_function(self):
        """Test cloning a simple function."""
        v0 = _core.Value(name="input")
        node = _core.Node("", "Abs", inputs=(v0,), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )
        function = _core.Function(
            domain="test.domain",
            name="AbsFunc",
            graph=graph,
            attributes=[],
        )

        cloned_function = function.clone()

        # Verify function properties are copied
        self.assertIsNot(cloned_function, function)
        self.assertEqual(cloned_function.domain, function.domain)
        self.assertEqual(cloned_function.name, function.name)
        self.assertEqual(cloned_function.overload, function.overload)

        # Verify graph is cloned
        self.assertIsNot(cloned_function.graph, function.graph)
        self.assertEqual(len(list(cloned_function.graph)), len(list(function.graph)))

    def test_clone_function_with_attributes(self):
        """Test cloning a function with attributes."""
        v0 = _core.Value(name="input")
        node = _core.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )
        function = _core.Function(
            domain="test.domain",
            name="TestFunc",
            graph=graph,
            attributes=[
                ir.AttrInt64("axis", 0),
                ir.AttrFloat32("scale", 1.0),
            ],
        )

        cloned_function = function.clone()

        # Verify attributes are present (but note: non-graph attributes are shared, not cloned)
        self.assertEqual(len(cloned_function._attributes), len(function._attributes))
        self.assertIn("axis", cloned_function._attributes)
        self.assertIn("scale", cloned_function._attributes)

        # Non-graph attributes are shared in the clone implementation
        cloned_axis_attr = cloned_function._attributes["axis"]
        original_axis_attr = function._attributes["axis"]
        self.assertIs(cloned_axis_attr, original_axis_attr)
        self.assertEqual(cloned_axis_attr.name, original_axis_attr.name)
        self.assertEqual(cloned_axis_attr.value, original_axis_attr.value)

    def test_clone_function_with_overload(self):
        """Test cloning a function with an overload."""
        v0 = _core.Value(name="input")
        node = _core.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph = _core.Graph(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )
        function = _core.Function(
            domain="test.domain",
            name="OverloadFunc",
            overload="int32_version",
            graph=graph,
            attributes=[],
        )

        cloned_function = function.clone()

        # Verify overload is preserved
        self.assertEqual(cloned_function.overload, function.overload)

    def test_clone_function_with_complex_graph(self):
        """Test cloning a function with a complex graph."""
        v0 = _core.Value(name="x")
        v1 = _core.Value(name="y")
        node1 = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1)
        node2 = _core.Node("", "Mul", inputs=(node1.outputs[0], v0), num_outputs=1)
        node3 = _core.Node("", "Sub", inputs=(node2.outputs[0], v1), num_outputs=1)

        graph = _core.Graph(
            inputs=(v0, v1),
            outputs=(node3.outputs[0],),
            nodes=(node1, node2, node3),
        )

        function = _core.Function(
            domain="complex.domain",
            name="ComplexFunc",
            graph=graph,
            attributes=[],
        )

        cloned_function = function.clone()

        # Verify all nodes are cloned
        cloned_nodes = list(cloned_function.graph)
        original_nodes = list(function.graph)

        self.assertEqual(len(cloned_nodes), len(original_nodes))
        for cloned_node, original_node in zip(cloned_nodes, original_nodes):
            self.assertIsNot(cloned_node, original_node)
            self.assertEqual(cloned_node.op_type, original_node.op_type)

        # Verify topology is preserved
        self.assertIs(cloned_nodes[1].inputs[0], cloned_nodes[0].outputs[0])
        self.assertIs(cloned_nodes[2].inputs[0], cloned_nodes[1].outputs[0])

    def test_clone_function_with_subgraphs(self):
        """Test cloning a function containing nodes with subgraphs."""
        v0 = _core.Value(name="cond")
        v1 = _core.Value(name="x")

        # Create subgraph
        neg_node = _core.Node("", "Neg", inputs=(v1,), num_outputs=1)
        then_graph = _core.Graph(
            inputs=(),
            outputs=(neg_node.outputs[0],),
            nodes=(neg_node,),
        )

        # Create function with If node
        if_node = _core.Node(
            "",
            "If",
            inputs=(v0,),
            num_outputs=1,
            attributes=[ir.AttrGraph("then_branch", then_graph)],
        )
        func_graph = _core.Graph(
            inputs=(v0, v1),
            outputs=(if_node.outputs[0],),
            nodes=(if_node,),
        )

        function = _core.Function(
            domain="test.domain",
            name="IfFunc",
            graph=func_graph,
            attributes=[],
        )

        cloned_function = function.clone()

        # Verify subgraph is cloned
        cloned_if_node = cloned_function[0]
        cloned_then = cloned_if_node.attributes["then_branch"].value
        original_then = if_node.attributes["then_branch"].value

        self.assertIsNot(cloned_then, original_then)
        self.assertEqual(len(cloned_then), len(original_then))


class GraphViewCloneTest(unittest.TestCase):
    """Test the GraphView.clone() method."""

    def test_clone_simple_graph_view(self):
        """Test cloning a simple GraphView with basic nodes."""
        v0 = ir.Value(name="input1")
        v1 = ir.Value(name="input2")
        node = ir.Node("", "Add", inputs=(v0, v1), num_outputs=1, name="add_node")
        graph_view = ir.GraphView(
            inputs=(v0, v1),
            outputs=(node.outputs[0],),
            nodes=(node,),
            name="simple_graph_view",
            doc_string="A simple graph view",
        )

        cloned_graph = graph_view.clone()

        # Verify the clone returns a Graph, not a GraphView
        self.assertIsInstance(cloned_graph, ir.Graph)
        self.assertNotIsInstance(cloned_graph, ir.GraphView)

        # Verify graph properties are copied
        self.assertEqual(cloned_graph.name, graph_view.name)
        self.assertEqual(cloned_graph.doc_string, graph_view.doc_string)
        self.assertIsNot(cloned_graph, graph_view)

        # Verify nodes are different objects
        self.assertEqual(len(cloned_graph), len(graph_view))
        cloned_node = cloned_graph[0]
        self.assertIsNot(cloned_node, node)
        self.assertEqual(cloned_node.op_type, node.op_type)
        self.assertEqual(cloned_node.name, node.name)

        # Verify inputs are different objects
        self.assertEqual(len(cloned_graph.inputs), len(graph_view.inputs))
        self.assertIsNot(cloned_graph.inputs[0], v0)
        self.assertIsNot(cloned_graph.inputs[1], v1)
        self.assertEqual(cloned_graph.inputs[0].name, v0.name)
        self.assertEqual(cloned_graph.inputs[1].name, v1.name)

        # Verify outputs are different objects
        self.assertEqual(len(cloned_graph.outputs), len(graph_view.outputs))
        self.assertIsNot(cloned_graph.outputs[0], node.outputs[0])

    def test_clone_graph_view_with_initializers(self):
        """Test cloning a GraphView with initializers."""
        v0 = ir.Value(name="input1")
        initializer_tensor = ir.tensor([1.0, 2.0, 3.0], name="weights")
        v_init = ir.Value(name="weights", const_value=initializer_tensor)
        node = ir.Node("", "Mul", inputs=(v0, v_init), num_outputs=1)
        graph_view = ir.GraphView(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
            initializers=(v_init,),
        )

        cloned_graph = graph_view.clone()

        # Verify initializers are cloned
        self.assertEqual(len(cloned_graph.initializers), len(graph_view.initializers))
        self.assertIn("weights", cloned_graph.initializers)
        cloned_init = cloned_graph.initializers["weights"]
        self.assertIsNot(cloned_init, v_init)
        self.assertEqual(cloned_init.name, v_init.name)

        # Verify tensor is shared (not deep copied)
        self.assertIsNotNone(cloned_init.const_value)
        self.assertIs(cloned_init.const_value, initializer_tensor)

    def test_clone_graph_view_with_subgraphs(self):
        """Test cloning a GraphView with subgraphs (e.g., If node)."""
        v0 = ir.Value(name="condition")
        v1 = ir.Value(name="x")
        v2 = ir.Value(name="y")

        # Create then branch
        add_node = ir.Node("", "Add", inputs=(v1, v2), num_outputs=1)
        then_graph = ir.Graph(
            inputs=(),
            outputs=(add_node.outputs[0],),
            nodes=(add_node,),
            name="then_branch",
        )

        # Create else branch
        sub_node = ir.Node("", "Sub", inputs=(v1, v2), num_outputs=1)
        else_graph = ir.Graph(
            inputs=(),
            outputs=(sub_node.outputs[0],),
            nodes=(sub_node,),
            name="else_branch",
        )

        # Create If node
        if_node = ir.Node(
            "",
            "If",
            inputs=(v0,),
            num_outputs=1,
            attributes=[
                ir.AttrGraph("then_branch", then_graph),
                ir.AttrGraph("else_branch", else_graph),
            ],
        )
        graph_view = ir.GraphView(
            inputs=(v0, v1, v2),
            outputs=(if_node.outputs[0],),
            nodes=(if_node,),
            name="main_graph_view",
        )

        cloned_graph = graph_view.clone()

        # Verify subgraphs are cloned
        cloned_if_node = cloned_graph[0]
        self.assertIsNot(cloned_if_node, if_node)

        cloned_then = cloned_if_node.attributes["then_branch"].value
        cloned_else = cloned_if_node.attributes["else_branch"].value

        self.assertIsNot(cloned_then, then_graph)
        self.assertIsNot(cloned_else, else_graph)
        self.assertEqual(cloned_then.name, then_graph.name)
        self.assertEqual(cloned_else.name, else_graph.name)

        # Verify nodes in subgraphs are cloned
        self.assertEqual(len(cloned_then), len(then_graph))
        cloned_then_node = cloned_then[0]
        self.assertIsNot(cloned_then_node, add_node)
        self.assertEqual(cloned_then_node.op_type, add_node.op_type)

    def test_clone_graph_view_with_metadata(self):
        """Test cloning a GraphView with metadata."""
        v0 = ir.Value(name="input")
        node = ir.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph_view = ir.GraphView(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
            metadata_props={"prop_key": "prop_value"},
        )

        cloned_graph = graph_view.clone()

        # Verify metadata_props are copied
        self.assertEqual(
            cloned_graph.metadata_props["prop_key"], graph_view.metadata_props["prop_key"]
        )
        self.assertIsNot(cloned_graph.metadata_props, graph_view.metadata_props)

    def test_clone_graph_view_preserves_value_types_and_shapes(self):
        """Test that cloning a GraphView preserves value types and shapes."""
        v0 = ir.Value(
            name="input",
            shape=ir.Shape([1, 3, 224, 224]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        node = ir.Node("", "Identity", inputs=(v0,), num_outputs=1)

        graph_view = ir.GraphView(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )

        cloned_graph = graph_view.clone()
        cloned_input = cloned_graph.inputs[0]

        # Verify input shapes and types are preserved
        self.assertEqual(cloned_input.shape, v0.shape)
        self.assertEqual(cloned_input.dtype, v0.dtype)

        # Verify the cloned graph has the same structure
        self.assertEqual(len(cloned_graph.inputs), len(graph_view.inputs))
        self.assertEqual(len(cloned_graph.outputs), len(graph_view.outputs))

    def test_clone_empty_graph_view(self):
        """Test cloning an empty GraphView."""
        graph_view = ir.GraphView(inputs=(), outputs=(), nodes=())
        cloned_graph = graph_view.clone()

        self.assertIsInstance(cloned_graph, ir.Graph)
        self.assertIsNot(cloned_graph, graph_view)
        self.assertEqual(len(cloned_graph.inputs), 0)
        self.assertEqual(len(cloned_graph.outputs), 0)
        self.assertEqual(len(list(cloned_graph)), 0)

    def test_clone_graph_view_maintains_topology(self):
        """Test that cloning a GraphView preserves the graph topology."""
        v0 = ir.Value(name="input")
        node1 = ir.Node("", "Add", inputs=(v0, v0), num_outputs=1, name="add1")
        node2 = ir.Node("", "Mul", inputs=(node1.outputs[0], v0), num_outputs=1, name="mul")
        node3 = ir.Node(
            "", "Sub", inputs=(node2.outputs[0], node1.outputs[0]), num_outputs=1, name="sub"
        )

        graph_view = ir.GraphView(
            inputs=(v0,),
            outputs=(node3.outputs[0],),
            nodes=(node1, node2, node3),
        )

        cloned_graph = graph_view.clone()
        cloned_nodes = list(cloned_graph)

        # Verify nodes are in the same order
        self.assertEqual(len(cloned_nodes), 3)
        self.assertEqual(cloned_nodes[0].name, "add1")
        self.assertEqual(cloned_nodes[1].name, "mul")
        self.assertEqual(cloned_nodes[2].name, "sub")

        # Verify connections are preserved
        # node2 should take input from node1's output
        self.assertIs(cloned_nodes[1].inputs[0], cloned_nodes[0].outputs[0])
        # node3 should take input from node2's output and node1's output
        self.assertIs(cloned_nodes[2].inputs[0], cloned_nodes[1].outputs[0])
        self.assertIs(cloned_nodes[2].inputs[1], cloned_nodes[0].outputs[0])

    def test_clone_graph_view_with_opset_imports(self):
        """Test that cloning a GraphView preserves opset imports."""
        v0 = ir.Value(name="input")
        node = ir.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph_view = ir.GraphView(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
            opset_imports={"": 18, "com.microsoft": 1},
        )

        cloned_graph = graph_view.clone()

        # Verify opset imports are copied
        self.assertEqual(cloned_graph.opset_imports, graph_view.opset_imports)
        self.assertIsNot(cloned_graph.opset_imports, graph_view.opset_imports)

    def test_clone_graph_view_is_mutable(self):
        """Test that the cloned Graph from a GraphView is mutable."""
        v0 = ir.Value(name="input")
        node = ir.Node("", "Identity", inputs=(v0,), num_outputs=1)
        graph_view = ir.GraphView(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )

        cloned_graph = graph_view.clone()

        # Verify we can mutate the cloned graph (should be a Graph, not GraphView)
        new_node = ir.Node("", "Relu", inputs=(cloned_graph.inputs[0],), num_outputs=1)
        cloned_graph.append(new_node)

        self.assertEqual(len(cloned_graph), 2)
        self.assertEqual(len(graph_view), 1)

    def test_clone_graph_view_preserves_node_attributes(self):
        """Test that cloning a GraphView preserves node attributes."""
        v0 = ir.Value(name="input")
        node = ir.Node(
            "",
            "Clip",
            inputs=(v0,),
            num_outputs=1,
            attributes=[
                ir.AttrFloat32("min", -1.0),
                ir.AttrFloat32("max", 1.0),
            ],
        )
        graph_view = ir.GraphView(
            inputs=(v0,),
            outputs=(node.outputs[0],),
            nodes=(node,),
        )

        cloned_graph = graph_view.clone()
        cloned_node = cloned_graph[0]

        # Verify attributes are cloned
        self.assertEqual(len(cloned_node.attributes), len(node.attributes))
        self.assertEqual(cloned_node.attributes["min"].value, -1.0)
        self.assertEqual(cloned_node.attributes["max"].value, 1.0)

    def test_clone_graph_view_with_intermediate_values(self):
        """Test cloning a GraphView created from intermediate values of a larger graph."""
        # Create a larger graph: input -> relu -> add -> mul -> output
        v_input = ir.Value(name="graph_input")
        v_const = ir.Value(
            name="const_value", const_value=ir.tensor([2.0], name="const_value")
        )

        node_relu = ir.Node("", "Relu", inputs=(v_input,), num_outputs=1, name="relu")
        node_add = ir.Node(
            "", "Add", inputs=(node_relu.outputs[0], v_const), num_outputs=1, name="add"
        )
        node_mul = ir.Node(
            "", "Mul", inputs=(node_add.outputs[0], v_const), num_outputs=1, name="mul"
        )

        # Create the full graph
        full_graph = ir.Graph(
            inputs=(v_input,),
            outputs=(node_mul.outputs[0],),
            nodes=(node_relu, node_add, node_mul),
            initializers=(v_const,),
            name="full_graph",
        )

        # Create a GraphView that extracts only the middle "add" and "mul" nodes
        # The inputs to the view are intermediate values (relu output and const)
        # The output is also an intermediate value (mul output, which is the final output here)
        subgraph_inputs = [node_relu.outputs[0], v_const]
        subgraph_outputs = [node_mul.outputs[0]]
        subgraph_nodes = [node_add, node_mul]

        graph_view = ir.GraphView(
            inputs=subgraph_inputs,
            outputs=subgraph_outputs,
            nodes=subgraph_nodes,
            initializers=(v_const,),
            name="subgraph_view",
        )

        # Clone the graph view
        cloned_graph = graph_view.clone()

        # Verify the clone is a Graph instance
        self.assertIsInstance(cloned_graph, ir.Graph)
        self.assertNotIsInstance(cloned_graph, ir.GraphView)

        # Verify structure is preserved
        self.assertEqual(len(cloned_graph), 2)
        self.assertEqual(len(cloned_graph.inputs), 2)
        self.assertEqual(len(cloned_graph.outputs), 1)
        self.assertEqual(len(cloned_graph.initializers), 1)

        # Verify nodes are cloned
        cloned_add = cloned_graph[0]
        cloned_mul = cloned_graph[1]
        self.assertIsNot(cloned_add, node_add)
        self.assertIsNot(cloned_mul, node_mul)
        self.assertEqual(cloned_add.name, "add")
        self.assertEqual(cloned_mul.name, "mul")

        # Verify inputs are cloned
        cloned_inputs = list(cloned_graph.inputs)
        self.assertIsNot(cloned_inputs[0], node_relu.outputs[0])
        self.assertIsNot(cloned_inputs[1], v_const)

        # Verify the topology is preserved within the cloned subgraph
        # The mul node should take input from the add node's output
        self.assertIs(cloned_mul.inputs[0], cloned_add.outputs[0])

        # Verify initializers are cloned
        self.assertIn("const_value", cloned_graph.initializers)
        cloned_const = cloned_graph.initializers["const_value"]
        self.assertIsNot(cloned_const, v_const)

        # Verify the original full graph is unchanged
        self.assertEqual(len(full_graph), 3)
        self.assertEqual(full_graph[0].name, "relu")
        self.assertEqual(full_graph[1].name, "add")
        self.assertEqual(full_graph[2].name, "mul")


if __name__ == "__main__":
    unittest.main()
