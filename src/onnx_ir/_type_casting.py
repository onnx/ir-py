# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Numpy utilities for non-native type operation."""

from __future__ import annotations

import math
import typing
from collections.abc import Sequence

import numpy as np

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def pack_4bitx2(array: np.ndarray) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to flatten, packed int4/uint4. Elements must be in the correct range."""
    # Create a 1D copy
    array_flat = array.ravel().view(np.uint8).copy()
    size = array.size
    odd_sized = size % 2 == 1
    if odd_sized:
        array_flat.resize([size + 1], refcheck=False)
    array_flat &= 0x0F
    array_flat[1::2] <<= 4
    return array_flat[0::2] | array_flat[1::2]  # type: ignore[return-type]


def unpack_4bitx2(data: npt.NDArray[np.uint8], dims: Sequence[int]) -> npt.NDArray[np.uint8]:
    """Convert a packed uint4 array to unpacked uint4 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    assert data.dtype == np.uint8, "Input data must be of type uint8"
    result = np.empty([data.size * 2], dtype=data.dtype)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    if result.size == np.prod(dims) + 1:
        # handle single-element padding due to odd number of elements
        result = result[:-1]
    result.resize(dims, refcheck=False)
    return result


def pack_2bitx4(array: np.ndarray) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to flatten, packed int2/uint2. Elements must be in the correct range."""
    # Create a 1D copy
    array_flat = array.ravel().view(np.uint8).copy()
    size = array.size
    padding = (4 - (size % 4)) % 4
    if padding > 0:
        array_flat.resize([size + padding], refcheck=False)
    array_flat &= 0x03
    array_flat[1::4] <<= 2
    array_flat[2::4] <<= 4
    array_flat[3::4] <<= 6
    return array_flat[0::4] | array_flat[1::4] | array_flat[2::4] | array_flat[3::4]  # type: ignore[return-type]


def unpack_2bitx4(data: npt.NDArray[np.uint8], dims: Sequence[int]) -> npt.NDArray[np.uint8]:
    """Convert a packed uint2 array to unpacked uint2 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    assert data.dtype == np.uint8, "Input data must be of type uint8"
    result = np.empty([data.size * 4], dtype=data.dtype)
    result[0::4] = data & np.uint8(0x03)
    result[1::4] = (data & np.uint8(0x0C)) >> np.uint8(2)
    result[2::4] = (data & np.uint8(0x30)) >> np.uint8(4)
    result[3::4] = (data & np.uint8(0xC0)) >> np.uint8(6)
    total_elements = int(np.prod(dims))
    if result.size > total_elements:
        # handle padding due to element count not being a multiple of 4
        result = result[:total_elements]
    result.resize(dims, refcheck=False)
    return result


def _validate_6bit_values(array: np.ndarray) -> None:
    """Validate that every element is a canonical unpacked 6-bit value."""
    if array.dtype.itemsize == 1:
        invalid = array.ravel().view(np.uint8) > 0x3F
    else:
        invalid = (array < 0) | (array > 0x3F)
    if np.any(invalid):
        raise ValueError("Unpacked 6-bit values must be in the range [0, 63].")


def _validate_packed_6bit(data: npt.NDArray[np.uint8], dims: Sequence[int]) -> None:
    """Validate the size and padding bits of a packed 6-bit array."""
    if data.dtype != np.uint8:
        raise TypeError(f"Packed 6-bit data must have dtype uint8, not {data.dtype}.")
    size = math.prod(dims)
    expected_nbytes = (size * 6 + 7) // 8
    if data.size < expected_nbytes:
        raise ValueError(
            f"Packed 6-bit data ({data.size} bytes) is too small for the declared "
            f"shape {list(dims)} ({expected_nbytes} bytes required)."
        )
    if data.size > expected_nbytes:
        raise ValueError(
            f"Packed 6-bit data ({data.size} bytes) is too large for the declared "
            f"shape {list(dims)} ({expected_nbytes} bytes required)."
        )
    padding_masks = {1: 0xC0, 2: 0xF0, 3: 0xFC}
    remainder = size % 4
    if remainder and data.size and data[-1] & padding_masks[remainder]:
        raise ValueError("Packed 6-bit data has nonzero padding bits.")


def pack_6bit(array: np.ndarray) -> npt.NDArray[np.uint8]:
    """Convert canonical unpacked 6-bit values to a flattened, packed array."""
    _validate_6bit_values(array)
    if array.dtype.itemsize == 1:
        array_flat = array.ravel().view(np.uint8).copy()
    else:
        array_flat = array.astype(np.uint8).ravel().copy()
    size = array.size
    padding = (4 - (size % 4)) % 4
    if padding:
        array_flat.resize([size + padding], refcheck=False)
    values = array_flat.reshape(-1, 4)
    packed = np.empty((values.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = values[:, 0] | ((values[:, 1] & 0x03) << 6)
    packed[:, 1] = (values[:, 1] >> 2) | ((values[:, 2] & 0x0F) << 4)
    packed[:, 2] = (values[:, 2] >> 4) | (values[:, 3] << 2)
    return packed.ravel()[: (size * 6 + 7) // 8]


def unpack_6bit(data: npt.NDArray[np.uint8], dims: Sequence[int]) -> npt.NDArray[np.uint8]:
    """Convert a packed 6-bit array to an unpacked uint8 array."""
    _validate_packed_6bit(data, dims)
    size = math.prod(dims)
    expected_nbytes = (size * 6 + 7) // 8
    padded = np.zeros(((size + 3) // 4) * 3, dtype=np.uint8)
    padded[:expected_nbytes] = data
    packed = padded.reshape(-1, 3)
    result = np.empty((packed.shape[0], 4), dtype=np.uint8)
    result[:, 0] = packed[:, 0] & 0x3F
    result[:, 1] = ((packed[:, 0] >> 6) & 0x03) | ((packed[:, 1] & 0x0F) << 2)
    result[:, 2] = ((packed[:, 1] >> 4) & 0x0F) | ((packed[:, 2] & 0x03) << 4)
    result[:, 3] = (packed[:, 2] >> 2) & 0x3F
    return result.ravel()[:size].reshape(dims)
