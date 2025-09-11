# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""ONNX IR enums that matches the ONNX spec."""

from __future__ import annotations

import enum
from typing import Any

import ml_dtypes
import numpy as np


class AttributeType(enum.IntEnum):
    """Enum for the types of ONNX attributes."""

    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSOR = 11
    SPARSE_TENSORS = 12
    TYPE_PROTO = 13
    TYPE_PROTOS = 14

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class DataType(enum.IntEnum):
    """Enum for the data types of ONNX tensors, defined in ``onnx.TensorProto``."""

    # NOTE: Naming: It is tempting to use shorter and more modern names like f32, i64,
    # but we should stick to the names used in the ONNX spec for consistency.
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20
    UINT4 = 21
    INT4 = 22
    FLOAT4E2M1 = 23
    FLOAT8E8M0 = 24

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> DataType:
        """Returns the ONNX data type for the numpy dtype.

        Raises:
            TypeError: If the data type is not supported by ONNX.
        """
        if dtype in _NP_TYPE_TO_DATA_TYPE:
            return cls(_NP_TYPE_TO_DATA_TYPE[dtype])

        if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_):
            return DataType.STRING

        # Special cases for handling custom dtypes defined in ONNX (as of onnx 1.18)
        # Ref: https://github.com/onnx/onnx/blob/2d42b6a60a52e925e57c422593e88cc51890f58a/onnx/_custom_element_types.py
        # TODO(#137): Remove this when ONNX 1.19 is the minimum requirement
        if hasattr(dtype, "names"):
            if dtype.names == ("bfloat16",):
                return DataType.BFLOAT16
            if dtype.names == ("e4m3fn",):
                return DataType.FLOAT8E4M3FN
            if dtype.names == ("e4m3fnuz",):
                return DataType.FLOAT8E4M3FNUZ
            if dtype.names == ("e5m2",):
                return DataType.FLOAT8E5M2
            if dtype.names == ("e5m2fnuz",):
                return DataType.FLOAT8E5M2FNUZ
            if dtype.names == ("uint4",):
                return DataType.UINT4
            if dtype.names == ("int4",):
                return DataType.INT4
            if dtype.names == ("float4e2m1",):
                return DataType.FLOAT4E2M1
        raise TypeError(f"Unsupported numpy data type: {dtype}")

    @classmethod
    def from_short_name(cls, short_name: str) -> DataType:
        """Returns the ONNX data type for the short name.

        Raises:
            TypeError: If the short name is not available for the data type.
        """
        if short_name not in _SHORT_NAME_TO_DATA_TYPE:
            raise TypeError(f"Unknown short name: {short_name}")
        return cls(_SHORT_NAME_TO_DATA_TYPE[short_name])

    @property
    def itemsize(self) -> float:
        """Returns the size of the data type in bytes."""
        return self.bitwidth / 8

    @property
    def bitwidth(self) -> int:
        """Returns the bit width of the data type.

        .. versionadded:: 0.1.2

        Raises:
            TypeError: If the data type is not supported.
        """
        if self not in _BITWIDTH_MAP:
            raise TypeError(f"Bitwidth not available for ONNX data type: {self}")
        return _BITWIDTH_MAP[self]

    @property
    def exponent_bitwidth(self) -> int:
        """Returns the bit width of the exponent for floating-point types.

        .. versionadded:: 0.1.8

        Raises:
            TypeError: If the data type is not supported.
        """
        if self.is_floating_point():
            return ml_dtypes.finfo(self.numpy()).nexp

        raise TypeError(f"Exponent not available for ONNX data type: {self}")

    @property
    def mantissa_bitwidth(self) -> int:
        """Returns the bit width of the mantissa for floating-point types.

        .. versionadded:: 0.1.8

        Raises:
            TypeError: If the data type is not supported.
        """
        if self.is_floating_point():
            return ml_dtypes.finfo(self.numpy()).nmant

        raise TypeError(f"Mantissa not available for ONNX data type: {self}")

    @property
    def eps(self) -> int | np.floating[Any]:
        """Returns the difference between 1.0 and the next smallest representable float larger than 1.0 for the ONNX data type.

        Returns 1 for integers.

        .. versionadded:: 0.1.8

        Raises:
            TypeError: If the data type is not a numeric data type.
        """
        if self.is_integer():
            return 1

        if self.is_floating_point():
            return ml_dtypes.finfo(self.numpy()).eps

        raise TypeError(f"Eps not available for ONNX data type: {self}")

    @property
    def tiny(self) -> int | np.floating[Any]:
        """Returns the smallest positive non-zero value for the ONNX data type.

        Returns 1 for integers.

        .. versionadded:: 0.1.8

        Raises:
            TypeError: If the data type is not a numeric data type.
        """
        if self.is_integer():
            return 1

        if self.is_floating_point():
            return ml_dtypes.finfo(self.numpy()).tiny

        raise TypeError(f"Tiny not available for ONNX data type: {self}")

    @property
    def min(self) -> int | np.floating[Any]:
        """Returns the minimum representable value for the ONNX data type.

        .. versionadded:: 0.1.8

        Raises:
            TypeError: If the data type is not a numeric data type.
        """
        if self.is_integer():
            return ml_dtypes.iinfo(self.numpy()).min

        if self.is_floating_point():
            return ml_dtypes.finfo(self.numpy()).min

        raise TypeError(f"Minimum not available for ONNX data type: {self}")

    @property
    def max(self) -> int | np.floating[Any]:
        """Returns the maximum representable value for the ONNX data type.

        .. versionadded:: 0.1.8

        Raises:
            TypeError: If the data type is not a numeric data type.
        """
        if self.is_integer():
            return ml_dtypes.iinfo(self.numpy()).max

        if self.is_floating_point():
            return ml_dtypes.finfo(self.numpy()).max

        raise TypeError(f"Maximum not available for ONNX data type: {self}")

    @property
    def precision(self) -> int:
        """Returns the precision for the ONNX dtype if supported.

        For floats returns the approximate number of decimal digits to which
        this kind of float is precise. Returns 0 for integers.

        .. versionadded:: 0.1.8

        Raises:
            TypeError: If the data type is not a numeric data type.
        """
        if self.is_integer():
            return 0

        if self.is_floating_point():
            return ml_dtypes.finfo(self.numpy()).precision

        raise TypeError(f"Precision not available for ONNX data type: {self}")

    @property
    def resolution(self) -> int | np.floating[Any]:
        """Returns the resolution for the ONNX dtype if supported.

        Returns the approximate decimal resolution of this type, i.e.,
         10**-precision. Returns 1 for integers.

        .. versionadded:: 0.1.8

        Raises:
            TypeError: If the data type is not a numeric data type.
        """
        if self.is_integer():
            return 1

        if self.is_floating_point():
            return ml_dtypes.finfo(self.numpy()).resolution

        raise TypeError(f"Resolution not available for ONNX data type: {self}")

    def numpy(self) -> np.dtype:
        """Returns the numpy dtype for the ONNX data type.

        Raises:
            TypeError: If the data type is not supported by numpy.
        """
        if self not in _DATA_TYPE_TO_NP_TYPE:
            raise TypeError(f"Numpy does not support ONNX data type: {self}")
        return _DATA_TYPE_TO_NP_TYPE[self]

    def short_name(self) -> str:
        """Returns the short name of the data type.

        The short name is a string that is used to represent the data type in a more
        compact form. For example, the short name for `DataType.FLOAT` is "f32".
        To get the corresponding data type back, call ``from_short_name`` on a string.

        Naming reference: https://github.com/pytorch/pytorch/blob/4bead7b85ea4160243c74109e0ce9bb80686d016/torch/utils/_dtype_abbrs.py

        Raises:
            TypeError: If the short name is not available for the data type.
        """
        if self not in _DATA_TYPE_TO_SHORT_NAME:
            raise TypeError(f"Short name not available for ONNX data type: {self}")
        return _DATA_TYPE_TO_SHORT_NAME[self]

    def is_floating_point(self) -> bool:
        """Returns True if the data type is a floating point type."""
        return self in {
            DataType.FLOAT,
            DataType.FLOAT16,
            DataType.DOUBLE,
            DataType.BFLOAT16,
            DataType.FLOAT8E4M3FN,
            DataType.FLOAT8E4M3FNUZ,
            DataType.FLOAT8E5M2,
            DataType.FLOAT8E5M2FNUZ,
            DataType.FLOAT4E2M1,
            DataType.FLOAT8E8M0,
        }

    def is_integer(self) -> bool:
        """Returns True if the data type is an integer.

        .. versionadded:: 0.1.4
        """
        return self in {
            DataType.UINT8,
            DataType.INT8,
            DataType.UINT16,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.UINT32,
            DataType.UINT64,
            DataType.UINT4,
            DataType.INT4,
        }

    def is_signed(self) -> bool:
        """Returns True if the data type is a signed type.

        .. versionadded:: 0.1.4
        """
        return self in {
            DataType.FLOAT,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.FLOAT16,
            DataType.DOUBLE,
            DataType.COMPLEX64,
            DataType.COMPLEX128,
            DataType.BFLOAT16,
            DataType.FLOAT8E4M3FN,
            DataType.FLOAT8E4M3FNUZ,
            DataType.FLOAT8E5M2,
            DataType.FLOAT8E5M2FNUZ,
            DataType.INT4,
            DataType.FLOAT4E2M1,
            DataType.FLOAT8E8M0,
        }

    def is_string(self) -> bool:
        """Returns True if the data type is a string type.

        .. versionadded:: 0.1.8
        """
        return self == DataType.STRING

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


_BITWIDTH_MAP = {
    DataType.FLOAT: 32,
    DataType.UINT8: 8,
    DataType.INT8: 8,
    DataType.UINT16: 16,
    DataType.INT16: 16,
    DataType.INT32: 32,
    DataType.INT64: 64,
    DataType.BOOL: 8,
    DataType.FLOAT16: 16,
    DataType.DOUBLE: 64,
    DataType.UINT32: 32,
    DataType.UINT64: 64,
    DataType.COMPLEX64: 64,  # 2 * 32
    DataType.COMPLEX128: 128,  # 2 * 64
    DataType.BFLOAT16: 16,
    DataType.FLOAT8E4M3FN: 8,
    DataType.FLOAT8E4M3FNUZ: 8,
    DataType.FLOAT8E5M2: 8,
    DataType.FLOAT8E5M2FNUZ: 8,
    DataType.UINT4: 4,
    DataType.INT4: 4,
    DataType.FLOAT4E2M1: 4,
    DataType.FLOAT8E8M0: 8,
}


# We use ml_dtypes to support dtypes that are not in numpy.
_NP_TYPE_TO_DATA_TYPE = {
    np.dtype("bool"): DataType.BOOL,
    np.dtype("complex128"): DataType.COMPLEX128,
    np.dtype("complex64"): DataType.COMPLEX64,
    np.dtype("float16"): DataType.FLOAT16,
    np.dtype("float32"): DataType.FLOAT,
    np.dtype("float64"): DataType.DOUBLE,
    np.dtype("int16"): DataType.INT16,
    np.dtype("int32"): DataType.INT32,
    np.dtype("int64"): DataType.INT64,
    np.dtype("int8"): DataType.INT8,
    np.dtype("object"): DataType.STRING,
    np.dtype("uint16"): DataType.UINT16,
    np.dtype("uint32"): DataType.UINT32,
    np.dtype("uint64"): DataType.UINT64,
    np.dtype("uint8"): DataType.UINT8,
    np.dtype(ml_dtypes.bfloat16): DataType.BFLOAT16,
    np.dtype(ml_dtypes.float8_e4m3fn): DataType.FLOAT8E4M3FN,
    np.dtype(ml_dtypes.float8_e4m3fnuz): DataType.FLOAT8E4M3FNUZ,
    np.dtype(ml_dtypes.float8_e5m2): DataType.FLOAT8E5M2,
    np.dtype(ml_dtypes.float8_e5m2fnuz): DataType.FLOAT8E5M2FNUZ,
    np.dtype(ml_dtypes.float8_e8m0fnu): DataType.FLOAT8E8M0,
    np.dtype(ml_dtypes.int4): DataType.INT4,
    np.dtype(ml_dtypes.uint4): DataType.UINT4,
    np.dtype(ml_dtypes.float4_e2m1fn): DataType.FLOAT4E2M1,
}

# ONNX DataType to Numpy dtype.
_DATA_TYPE_TO_NP_TYPE = {v: k for k, v in _NP_TYPE_TO_DATA_TYPE.items()}

_DATA_TYPE_TO_SHORT_NAME = {
    DataType.UNDEFINED: "undefined",
    DataType.BFLOAT16: "bf16",
    DataType.DOUBLE: "f64",
    DataType.FLOAT: "f32",
    DataType.FLOAT16: "f16",
    DataType.FLOAT8E4M3FN: "f8e4m3fn",
    DataType.FLOAT8E5M2: "f8e5m2",
    DataType.FLOAT8E4M3FNUZ: "f8e4m3fnuz",
    DataType.FLOAT8E5M2FNUZ: "f8e5m2fnuz",
    DataType.FLOAT8E8M0: "f8e8m0",
    DataType.FLOAT4E2M1: "f4e2m1",
    DataType.COMPLEX64: "c64",
    DataType.COMPLEX128: "c128",
    DataType.INT4: "i4",
    DataType.INT8: "i8",
    DataType.INT16: "i16",
    DataType.INT32: "i32",
    DataType.INT64: "i64",
    DataType.BOOL: "b8",
    DataType.UINT4: "u4",
    DataType.UINT8: "u8",
    DataType.UINT16: "u16",
    DataType.UINT32: "u32",
    DataType.UINT64: "u64",
    DataType.STRING: "s",
}

_SHORT_NAME_TO_DATA_TYPE = {v: k for k, v in _DATA_TYPE_TO_SHORT_NAME.items()}
