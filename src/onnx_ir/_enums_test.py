# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=protected-access
import unittest

import ml_dtypes
import numpy as np
import onnx
import parameterized

from onnx_ir import _enums


class DataTypeTest(unittest.TestCase):
    def test_enums_are_the_same_as_spec(self):
        self.assertEqual(_enums.DataType.FLOAT, onnx.TensorProto.FLOAT)
        self.assertEqual(_enums.DataType.UINT8, onnx.TensorProto.UINT8)
        self.assertEqual(_enums.DataType.INT8, onnx.TensorProto.INT8)
        self.assertEqual(_enums.DataType.UINT16, onnx.TensorProto.UINT16)
        self.assertEqual(_enums.DataType.INT16, onnx.TensorProto.INT16)
        self.assertEqual(_enums.DataType.INT32, onnx.TensorProto.INT32)
        self.assertEqual(_enums.DataType.INT64, onnx.TensorProto.INT64)
        self.assertEqual(_enums.DataType.STRING, onnx.TensorProto.STRING)
        self.assertEqual(_enums.DataType.BOOL, onnx.TensorProto.BOOL)
        self.assertEqual(_enums.DataType.FLOAT16, onnx.TensorProto.FLOAT16)
        self.assertEqual(_enums.DataType.DOUBLE, onnx.TensorProto.DOUBLE)
        self.assertEqual(_enums.DataType.UINT32, onnx.TensorProto.UINT32)
        self.assertEqual(_enums.DataType.UINT64, onnx.TensorProto.UINT64)
        self.assertEqual(_enums.DataType.COMPLEX64, onnx.TensorProto.COMPLEX64)
        self.assertEqual(_enums.DataType.COMPLEX128, onnx.TensorProto.COMPLEX128)
        self.assertEqual(_enums.DataType.BFLOAT16, onnx.TensorProto.BFLOAT16)
        self.assertEqual(_enums.DataType.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FN)
        self.assertEqual(_enums.DataType.FLOAT8E4M3FNUZ, onnx.TensorProto.FLOAT8E4M3FNUZ)
        self.assertEqual(_enums.DataType.FLOAT8E5M2, onnx.TensorProto.FLOAT8E5M2)
        self.assertEqual(_enums.DataType.FLOAT8E5M2FNUZ, onnx.TensorProto.FLOAT8E5M2FNUZ)
        self.assertEqual(_enums.DataType.UINT4, onnx.TensorProto.UINT4)
        self.assertEqual(_enums.DataType.INT4, onnx.TensorProto.INT4)
        if hasattr(onnx.TensorProto, "FLOAT4E2M1"):
            self.assertEqual(_enums.DataType.FLOAT4E2M1, onnx.TensorProto.FLOAT4E2M1)
        if hasattr(onnx.TensorProto, "FLOAT8E8M0"):
            self.assertEqual(_enums.DataType.FLOAT8E8M0, onnx.TensorProto.FLOAT8E8M0)
        if hasattr(onnx.TensorProto, "INT2"):
            self.assertEqual(_enums.DataType.INT2, onnx.TensorProto.INT2)
        if hasattr(onnx.TensorProto, "UINT2"):
            self.assertEqual(_enums.DataType.UINT2, onnx.TensorProto.UINT2)
        self.assertEqual(_enums.DataType.UNDEFINED, onnx.TensorProto.UNDEFINED)

    @parameterized.parameterized.expand(
        [
            ("string", np.array("some_string").dtype, _enums.DataType.STRING),
            ("float64", np.dtype(np.float64), _enums.DataType.DOUBLE),
            ("float32", np.dtype(np.float32), _enums.DataType.FLOAT),
            ("float16", np.dtype(np.float16), _enums.DataType.FLOAT16),
            ("int32", np.dtype(np.int32), _enums.DataType.INT32),
            ("int16", np.dtype(np.int16), _enums.DataType.INT16),
            ("int8", np.dtype(np.int8), _enums.DataType.INT8),
            ("int64", np.dtype(np.int64), _enums.DataType.INT64),
            ("uint8", np.dtype(np.uint8), _enums.DataType.UINT8),
            ("uint16", np.dtype(np.uint16), _enums.DataType.UINT16),
            ("uint32", np.dtype(np.uint32), _enums.DataType.UINT32),
            ("uint64", np.dtype(np.uint64), _enums.DataType.UINT64),
            ("bool", np.dtype(np.bool_), _enums.DataType.BOOL),
            ("complex64", np.dtype(np.complex64), _enums.DataType.COMPLEX64),
            ("complex128", np.dtype(np.complex128), _enums.DataType.COMPLEX128),
            ("bfloat16", np.dtype(ml_dtypes.bfloat16), _enums.DataType.BFLOAT16),
            ("float8e4m3fn", np.dtype(ml_dtypes.float8_e4m3fn), _enums.DataType.FLOAT8E4M3FN),
            (
                "float8e4m3fnuz",
                np.dtype(ml_dtypes.float8_e4m3fnuz),
                _enums.DataType.FLOAT8E4M3FNUZ,
            ),
            ("float8e5m2", np.dtype(ml_dtypes.float8_e5m2), _enums.DataType.FLOAT8E5M2),
            (
                "float8e5m2fnuz",
                np.dtype(ml_dtypes.float8_e5m2fnuz),
                _enums.DataType.FLOAT8E5M2FNUZ,
            ),
            ("uint4", np.dtype(ml_dtypes.uint4), _enums.DataType.UINT4),
            ("int4", np.dtype(ml_dtypes.int4), _enums.DataType.INT4),
            ("float4e2m1", np.dtype(ml_dtypes.float4_e2m1fn), _enums.DataType.FLOAT4E2M1),
            ("float8e8m0", np.dtype(ml_dtypes.float8_e8m0fnu), _enums.DataType.FLOAT8E8M0),
            ("int2", np.dtype(ml_dtypes.int2), _enums.DataType.INT2),
            ("uint2", np.dtype(ml_dtypes.uint2), _enums.DataType.UINT2),
        ]
    )
    def test_from_numpy_takes_np_dtype_and_returns_data_type(
        self, _: str, np_dtype: np.dtype, onnx_type: _enums.DataType
    ):
        self.assertEqual(_enums.DataType.from_numpy(np_dtype), onnx_type)

    def test_numpy_returns_np_dtype(self):
        self.assertEqual(_enums.DataType.DOUBLE.numpy(), np.dtype(np.float64))

    def test_itemsize_returns_size_of_data_type_in_bytes(self):
        self.assertEqual(_enums.DataType.DOUBLE.itemsize, 8)
        self.assertEqual(_enums.DataType.INT4.itemsize, 0.5)

    def test_repr_and_str_return_name(self):
        self.assertEqual(str(_enums.DataType.DOUBLE), "DOUBLE")
        self.assertEqual(repr(_enums.DataType.DOUBLE), "DOUBLE")

    def test_short_name_conversion(self):
        for dtype in _enums.DataType:
            short_name = dtype.short_name()
            self.assertEqual(_enums.DataType.from_short_name(short_name), dtype)

    def test_access_by_name(self):
        self.assertEqual(_enums.DataType["FLOAT"], _enums.DataType.FLOAT)
        self.assertEqual(_enums.DataType["UINT8"], _enums.DataType.UINT8)
        self.assertEqual(_enums.DataType["INT8"], _enums.DataType.INT8)
        self.assertEqual(_enums.DataType["UINT16"], _enums.DataType.UINT16)
        self.assertEqual(_enums.DataType["INT16"], _enums.DataType.INT16)
        self.assertEqual(_enums.DataType["INT32"], _enums.DataType.INT32)
        self.assertEqual(_enums.DataType["INT64"], _enums.DataType.INT64)
        self.assertEqual(_enums.DataType["STRING"], _enums.DataType.STRING)
        self.assertEqual(_enums.DataType["BOOL"], _enums.DataType.BOOL)
        self.assertEqual(_enums.DataType["FLOAT16"], _enums.DataType.FLOAT16)
        self.assertEqual(_enums.DataType["DOUBLE"], _enums.DataType.DOUBLE)
        self.assertEqual(_enums.DataType["UINT32"], _enums.DataType.UINT32)
        self.assertEqual(_enums.DataType["UINT64"], _enums.DataType.UINT64)
        self.assertEqual(_enums.DataType["COMPLEX64"], _enums.DataType.COMPLEX64)
        self.assertEqual(_enums.DataType["COMPLEX128"], _enums.DataType.COMPLEX128)
        self.assertEqual(_enums.DataType["BFLOAT16"], _enums.DataType.BFLOAT16)
        self.assertEqual(_enums.DataType["FLOAT8E4M3FN"], _enums.DataType.FLOAT8E4M3FN)
        self.assertEqual(_enums.DataType["FLOAT8E4M3FNUZ"], _enums.DataType.FLOAT8E4M3FNUZ)
        self.assertEqual(_enums.DataType["FLOAT8E5M2"], _enums.DataType.FLOAT8E5M2)
        self.assertEqual(_enums.DataType["FLOAT8E5M2FNUZ"], _enums.DataType.FLOAT8E5M2FNUZ)
        self.assertEqual(_enums.DataType["UINT4"], _enums.DataType.UINT4)
        self.assertEqual(_enums.DataType["INT4"], _enums.DataType.INT4)
        self.assertEqual(_enums.DataType["INT2"], _enums.DataType.INT2)
        self.assertEqual(_enums.DataType["UINT2"], _enums.DataType.UINT2)
        self.assertEqual(_enums.DataType["FLOAT4E2M1"], _enums.DataType.FLOAT4E2M1)
        self.assertEqual(_enums.DataType["UNDEFINED"], _enums.DataType.UNDEFINED)


class AttributeTypeTest(unittest.TestCase):
    def test_enums_are_the_same_as_spec(self):
        self.assertEqual(_enums.AttributeType.FLOAT, onnx.AttributeProto.FLOAT)
        self.assertEqual(_enums.AttributeType.INT, onnx.AttributeProto.INT)
        self.assertEqual(_enums.AttributeType.STRING, onnx.AttributeProto.STRING)
        self.assertEqual(_enums.AttributeType.TENSOR, onnx.AttributeProto.TENSOR)
        self.assertEqual(_enums.AttributeType.GRAPH, onnx.AttributeProto.GRAPH)
        self.assertEqual(_enums.AttributeType.FLOATS, onnx.AttributeProto.FLOATS)
        self.assertEqual(_enums.AttributeType.INTS, onnx.AttributeProto.INTS)
        self.assertEqual(_enums.AttributeType.STRINGS, onnx.AttributeProto.STRINGS)
        self.assertEqual(_enums.AttributeType.TENSORS, onnx.AttributeProto.TENSORS)
        self.assertEqual(_enums.AttributeType.GRAPHS, onnx.AttributeProto.GRAPHS)
        self.assertEqual(_enums.AttributeType.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSOR)
        self.assertEqual(
            _enums.AttributeType.SPARSE_TENSORS, onnx.AttributeProto.SPARSE_TENSORS
        )
        self.assertEqual(_enums.AttributeType.TYPE_PROTO, onnx.AttributeProto.TYPE_PROTO)
        self.assertEqual(_enums.AttributeType.TYPE_PROTOS, onnx.AttributeProto.TYPE_PROTOS)
        self.assertEqual(_enums.AttributeType.UNDEFINED, onnx.AttributeProto.UNDEFINED)


# All floating point types that have finfo
_FLOAT_TYPES = [
    ("FLOAT", _enums.DataType.FLOAT),
    ("FLOAT16", _enums.DataType.FLOAT16),
    ("DOUBLE", _enums.DataType.DOUBLE),
    ("BFLOAT16", _enums.DataType.BFLOAT16),
    ("FLOAT8E4M3FN", _enums.DataType.FLOAT8E4M3FN),
    ("FLOAT8E4M3FNUZ", _enums.DataType.FLOAT8E4M3FNUZ),
    ("FLOAT8E5M2", _enums.DataType.FLOAT8E5M2),
    ("FLOAT8E5M2FNUZ", _enums.DataType.FLOAT8E5M2FNUZ),
    ("FLOAT4E2M1", _enums.DataType.FLOAT4E2M1),
]

_INT_TYPES = [
    ("INT8", _enums.DataType.INT8),
    ("INT16", _enums.DataType.INT16),
    ("INT32", _enums.DataType.INT32),
    ("INT64", _enums.DataType.INT64),
    ("UINT8", _enums.DataType.UINT8),
    ("UINT16", _enums.DataType.UINT16),
    ("UINT32", _enums.DataType.UINT32),
    ("UINT64", _enums.DataType.UINT64),
    ("INT4", _enums.DataType.INT4),
    ("UINT4", _enums.DataType.UINT4),
    ("INT2", _enums.DataType.INT2),
    ("UINT2", _enums.DataType.UINT2),
]


class DataTypeExponentMantissaTest(unittest.TestCase):
    @parameterized.parameterized.expand(_FLOAT_TYPES)
    def test_exponent_bitwidth_for_float_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.finfo(dtype.numpy()).nexp
        self.assertEqual(dtype.exponent_bitwidth, expected)

    @parameterized.parameterized.expand(_FLOAT_TYPES)
    def test_mantissa_bitwidth_for_float_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.finfo(dtype.numpy()).nmant
        self.assertEqual(dtype.mantissa_bitwidth, expected)

    def test_exponent_bitwidth_raises_for_integer(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.INT32.exponent_bitwidth

    def test_mantissa_bitwidth_raises_for_integer(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.INT32.mantissa_bitwidth


class DataTypeEpsTest(unittest.TestCase):
    @parameterized.parameterized.expand(_FLOAT_TYPES)
    def test_eps_for_float_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.finfo(dtype.numpy()).eps
        self.assertEqual(dtype.eps, expected)

    @parameterized.parameterized.expand(_INT_TYPES)
    def test_eps_returns_1_for_integer_types(self, _: str, dtype: _enums.DataType):
        self.assertEqual(dtype.eps, 1)

    def test_eps_raises_for_string(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.STRING.eps


class DataTypeTinyTest(unittest.TestCase):
    @parameterized.parameterized.expand(_FLOAT_TYPES)
    def test_tiny_for_float_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.finfo(dtype.numpy()).tiny
        self.assertEqual(dtype.tiny, expected)

    @parameterized.parameterized.expand(_INT_TYPES)
    def test_tiny_returns_1_for_integer_types(self, _: str, dtype: _enums.DataType):
        self.assertEqual(dtype.tiny, 1)

    def test_tiny_raises_for_string(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.STRING.tiny


class DataTypeMinMaxTest(unittest.TestCase):
    @parameterized.parameterized.expand(_FLOAT_TYPES)
    def test_min_for_float_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.finfo(dtype.numpy()).min
        self.assertEqual(dtype.min, expected)

    @parameterized.parameterized.expand(_FLOAT_TYPES)
    def test_max_for_float_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.finfo(dtype.numpy()).max
        self.assertEqual(dtype.max, expected)

    @parameterized.parameterized.expand(_INT_TYPES)
    def test_min_for_integer_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.iinfo(dtype.numpy()).min
        self.assertEqual(dtype.min, expected)

    @parameterized.parameterized.expand(_INT_TYPES)
    def test_max_for_integer_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.iinfo(dtype.numpy()).max
        self.assertEqual(dtype.max, expected)

    def test_min_raises_for_string(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.STRING.min

    def test_max_raises_for_string(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.STRING.max


class DataTypePrecisionResolutionTest(unittest.TestCase):
    @parameterized.parameterized.expand(_FLOAT_TYPES)
    def test_precision_for_float_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.finfo(dtype.numpy()).precision
        self.assertEqual(dtype.precision, expected)

    @parameterized.parameterized.expand(_FLOAT_TYPES)
    def test_resolution_for_float_types(self, _: str, dtype: _enums.DataType):
        expected = ml_dtypes.finfo(dtype.numpy()).resolution
        self.assertEqual(dtype.resolution, expected)

    @parameterized.parameterized.expand(_INT_TYPES)
    def test_precision_returns_0_for_integer_types(self, _: str, dtype: _enums.DataType):
        self.assertEqual(dtype.precision, 0)

    @parameterized.parameterized.expand(_INT_TYPES)
    def test_resolution_returns_1_for_integer_types(self, _: str, dtype: _enums.DataType):
        self.assertEqual(dtype.resolution, 1)

    def test_precision_raises_for_string(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.STRING.precision

    def test_resolution_raises_for_string(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.STRING.resolution


class DataTypeClassificationTest(unittest.TestCase):
    def test_is_floating_point(self):
        self.assertTrue(_enums.DataType.FLOAT.is_floating_point())
        self.assertTrue(_enums.DataType.DOUBLE.is_floating_point())
        self.assertTrue(_enums.DataType.BFLOAT16.is_floating_point())
        self.assertFalse(_enums.DataType.INT32.is_floating_point())
        self.assertFalse(_enums.DataType.STRING.is_floating_point())

    def test_is_integer(self):
        self.assertTrue(_enums.DataType.INT32.is_integer())
        self.assertTrue(_enums.DataType.UINT4.is_integer())
        self.assertTrue(_enums.DataType.INT2.is_integer())
        self.assertFalse(_enums.DataType.FLOAT.is_integer())
        self.assertFalse(_enums.DataType.STRING.is_integer())

    def test_is_signed(self):
        self.assertTrue(_enums.DataType.INT32.is_signed())
        self.assertTrue(_enums.DataType.FLOAT.is_signed())
        self.assertFalse(_enums.DataType.UINT8.is_signed())
        self.assertFalse(_enums.DataType.BOOL.is_signed())

    def test_is_string(self):
        self.assertTrue(_enums.DataType.STRING.is_string())
        self.assertFalse(_enums.DataType.FLOAT.is_string())

    def test_from_short_name_raises_for_unknown(self):
        with self.assertRaises(TypeError):
            _enums.DataType.from_short_name("nonexistent")

    def test_bitwidth_raises_for_undefined(self):
        with self.assertRaises(TypeError):
            _ = _enums.DataType.UNDEFINED.bitwidth

    def test_numpy_raises_for_unsupported(self):
        with self.assertRaises(TypeError):
            _enums.DataType.UNDEFINED.numpy()

    def test_short_name_returns_correct_name(self):
        self.assertEqual(_enums.DataType.STRING.short_name(), "s")

    def test_from_numpy_raises_for_unsupported_dtype(self):
        with self.assertRaises(TypeError):
            _enums.DataType.from_numpy(np.dtype("datetime64"))


if __name__ == "__main__":
    unittest.main()
