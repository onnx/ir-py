# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for using safetensors as an external data format."""

from __future__ import annotations

__all__ = ["save_safetensors"]

import functools
import io
import json
import os
import struct
from collections.abc import Callable, Sequence
from typing import Any

import packaging.version

import onnx_ir as ir

_HEADER_SIZE_NUMBER_SIZE = 8
# https://github.com/huggingface/safetensors/blob/806426784adb43631e9a1102d4621126bb589347/safetensors/src/tensor.rs#L811
_SAFETENSORS_DTYPE_TO_IR_DTYPE = {
    "BOOL": ir.DataType.BOOL,
    "F4": ir.DataType.FLOAT4E2M1,
    "F8_E5M2": ir.DataType.FLOAT8E5M2,
    "F8_E4M3": ir.DataType.FLOAT8E4M3FN,
    "F8_E8M0": ir.DataType.FLOAT8E8M0,
    "BF16": ir.DataType.BFLOAT16,
    "F16": ir.DataType.FLOAT16,
    "F32": ir.DataType.FLOAT,
    "F64": ir.DataType.DOUBLE,
    "I8": ir.DataType.INT8,
    "I16": ir.DataType.INT16,
    "I32": ir.DataType.INT32,
    "I64": ir.DataType.INT64,
    "U8": ir.DataType.UINT8,
    "U16": ir.DataType.UINT16,
    "U32": ir.DataType.UINT32,
    "U64": ir.DataType.UINT64,
    "C64": ir.DataType.COMPLEX64,
}
# https://github.com/huggingface/safetensors/blob/806426784adb43631e9a1102d4621126bb589347/bindings/python/src/view.rs#L77
_IR_DTYPE_TO_SAFETENSORS_DTYPE = {
    ir.DataType.BOOL: "bool",
    ir.DataType.FLOAT4E2M1: "float4_e2m1fn_x2",
    ir.DataType.FLOAT8E5M2: "float8_e5m2",
    ir.DataType.FLOAT8E4M3FN: "float8_e4m3fn",
    ir.DataType.FLOAT8E8M0: "float8_e8m0",
    ir.DataType.FLOAT8E4M3FNUZ: "uint8",
    ir.DataType.FLOAT8E5M2FNUZ: "uint8",
    ir.DataType.BFLOAT16: "bfloat16",
    ir.DataType.FLOAT16: "float16",
    ir.DataType.FLOAT: "float32",
    ir.DataType.DOUBLE: "float64",
    ir.DataType.INT2: "uint8",
    ir.DataType.INT4: "uint8",
    ir.DataType.INT8: "int8",
    ir.DataType.INT16: "int16",
    ir.DataType.INT32: "int32",
    ir.DataType.INT64: "int64",
    ir.DataType.UINT2: "uint8",
    ir.DataType.UINT4: "uint8",
    ir.DataType.UINT8: "uint8",
    ir.DataType.UINT16: "uint16",
    ir.DataType.UINT32: "uint32",
    ir.DataType.UINT64: "uint64",
    ir.DataType.COMPLEX64: "complex64",
}


@functools.lru_cache(maxsize=1)
def _import_safetensors():
    """Raise an error if safetensors is not installed."""
    try:
        import safetensors
    except ImportError as e:
        raise ImportError(
            "safetensors is required for using safetensors external data format. "
            "Please install it with 'pip install --upgrade safetensors'."
        ) from e

    min_required_version = packaging.version.parse("0.7.0")
    version = getattr(safetensors, "__version__", None)
    if version is None or packaging.version.parse(version) < min_required_version:
        raise ImportError(
            f"safetensors version 0.7.0 or higher is required, but version {version} is installed. "
            "Please upgrade it with 'pip install --upgrade safetensors'."
        )

    return safetensors


def _get_shard_filename(base_name: str, shard_idx: int, total_shards: int) -> str:
    """Generate a filename for a shard.

    Args:
        base_name: The base filename (e.g., 'model.safetensors').
        shard_idx: The index of this shard (1-indexed).
        total_shards: The total number of shards.

    Returns:
        The shard filename (e.g., 'model-00001-of-00003.safetensors').
    """
    if total_shards == 1:
        return base_name

    # Extract extension
    if "." in base_name:
        name, ext = base_name.rsplit(".", 1)
        ext = f".{ext}"
    else:
        name = base_name
        ext = ""

    # Always use 5 digits to follow transformers convention
    return f"{name}-{shard_idx:05d}-of-{total_shards:05d}{ext}"


def _shard_tensors(
    tensors: Sequence[ir.TensorProtocol], max_shard_size_bytes: int | None
) -> list[list[ir.TensorProtocol]]:
    """Shard tensors into multiple files based on max_shard_size_bytes.

    Args:
        tensors: The tensors to shard.
        max_shard_size_bytes: Maximum size for each shard in bytes. When None,
            no sharding is performed.

    Returns:
        A list of tensor lists for each shard.
    """
    if max_shard_size_bytes is None:
        # No sharding
        return [list(tensors)]

    # Shard the tensors by current order
    shards: list[list[ir.TensorProtocol]] = [[]]
    current_shard_size = 0

    for tensor in tensors:
        tensor_size = tensor.nbytes
        # Check if adding this tensor would exceed max_shard_size_bytes
        if current_shard_size + tensor_size > max_shard_size_bytes and current_shard_size > 0:
            # Start a new shard
            shards.append([])
            current_shard_size = 0

        shards[-1].append(tensor)
        current_shard_size += tensor_size

    return shards


def _replace_tensors(
    values: Sequence[ir.Value],
    /,
    location: str | os.PathLike,
    base_dir: str | os.PathLike,
) -> None:
    """Replace all tensors in an ONNX model with external data from a safetensors file.

    Args:
        values: List of initialized values to replace constant values from.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.
    """
    tensors: dict[str, ir.ExternalTensor] = _read_safetensors(location, base_dir=base_dir)
    value_map: dict[str, ir.Value] = {value.name: value for value in values}  # type: ignore[misc]
    for name, tensor in tensors.items():
        assert name in value_map, f"Bug: Tensor '{name}' not found in model initializers."
        value = value_map[name]
        model_tensor = value.const_value
        assert model_tensor is not None
        updated_tensor = _migrate_tensor_shape_dtype(model_tensor, tensor)
        value.const_value = updated_tensor


def _get_tensor_storage_shape(tensor: ir.TensorProtocol) -> Sequence[int]:
    """Get the storage shape of a tensor for safetensors."""
    # Handle sub-byte dtypes
    if tensor.dtype.bitwidth < 8:
        return [tensor.nbytes]
    return tensor.shape.numpy()


def _save_file(
    initialized_values: Sequence[ir.Value],
    /,
    location: str | os.PathLike,
    base_dir: str | os.PathLike = "",
    *,
    size_threshold_bytes: int,
    max_shard_size_bytes: int | None,
    callback: Callable[[ir.TensorProtocol, ir.external_data.CallbackInfo], None] | None = None,
) -> None:
    """Save all tensors in an ONNX model to a safetensors file.

    Args:
        initialized_values: List of initialized values to consider for saving.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.
        size_threshold_bytes: Save to external data if the tensor size in bytes
            is not smaller than this threshold.
        max_shard_size_bytes: Maximum size in bytes (as int) a safetensors file
            before being sharded. If None, no sharding is performed.
        callback: A callback function that is called after each tensor is saved.
    """
    safetensors = _import_safetensors()

    # Ensure that external_data ends with .safetensors
    if not str(location).endswith(".safetensors"):
        raise ValueError(
            f'The path to safetensors file must have a .safetensors extension, got: "{location}"'
        )

    # First, collect metadata without loading tensor data
    tensors_to_save: list[ir.TensorProtocol] = []
    values_to_save: list[ir.Value] = []
    for value in initialized_values:
        tensor = value.const_value
        assert tensor is not None
        if tensor.nbytes < size_threshold_bytes:
            continue
        tensors_to_save.append(tensor)
        values_to_save.append(value)

    total_size = sum(tensor.nbytes for tensor in tensors_to_save)

    if tensors_to_save:
        # Determine sharding based on max_shard_size_bytes. When max_shard_size_bytes is None,
        # It is the same as one shard (which is the same as no sharding).
        tensor_shards = _shard_tensors(tensors_to_save, max_shard_size_bytes)
        total_shards = len(tensor_shards)

        # Save each shard, loading only necessary tensor data
        all_filenames = []
        weight_map: dict[str, str] = {}  # Maps tensor name to shard filename
        current_offset = 0
        current_index = 0
        for shard_idx, tensor_shard in enumerate(tensor_shards, start=1):
            shard_filename = _get_shard_filename(str(location), shard_idx, total_shards)

            shard_path = os.path.join(base_dir, shard_filename)
            all_filenames.append(shard_filename)

            # Build tensor_dict for this shard only
            shard_dict: dict[str, Any] = {}
            for tensor in tensor_shard:
                if callback is not None:
                    callback(
                        tensor,
                        ir.external_data.CallbackInfo(
                            total=len(tensors_to_save),
                            index=current_index,
                            offset=current_offset,
                            filename=shard_filename,
                        ),
                    )
                assert tensor.name is not None
                shard_dict[tensor.name] = {
                    "dtype": _IR_DTYPE_TO_SAFETENSORS_DTYPE[tensor.dtype],
                    "shape": _get_tensor_storage_shape(tensor),
                    "data": tensor.tobytes(),
                }
                # Update weight_map with shard filename
                weight_map[tensor.name] = shard_filename
                current_offset += tensor.nbytes
                current_index += 1

            safetensors.serialize_file(shard_dict, shard_path)

        # Save index file if sharding occurred
        if total_shards > 1:
            location_str = str(location)
            if location_str.endswith(".safetensors"):
                index_filename = (
                    location_str.rsplit(".safetensors", 1)[0] + ".safetensors.index.json"
                )
            else:
                index_filename = location_str + ".index.json"
            index_path = os.path.join(base_dir, index_filename)
            index_data = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)

        # Replace tensors from each shard file
        for filename in all_filenames:
            _replace_tensors(values_to_save, filename, base_dir)


def save_safetensors(
    model: ir.Model,
    path: str | os.PathLike,
    /,
    *,
    format: str | None = None,
    size_threshold_bytes: int = 256,
    max_shard_size_bytes: int | None = None,
    callback: Callable[[ir.TensorProtocol, ir.external_data.CallbackInfo], None] | None = None,
) -> None:
    """Save an ONNX model to a file with external data in a safetensors file.

    The model object is unmodified after this operation.

    When sharding is enabled, multiple safetensors files will be created
    with names like "model-00001-of-00003.safetensors", and an index
    file "model.safetensors.index.json" will be created to map tensors
    to their respective shard files. The shards will be created only if
    the total size of tensors exceeds the specified max_shard_size_bytes.

    .. note::
        Because the safetensors data format uses key-value mapping to store tensors,
        all initializer names in the model (across subgraphs) must be unique.
        Externalizing tensor attributes in nodes to safetensors files is currently not
        supported. If you have tensors from Constant nodes that you want to externalize,
        consider converting them to initializers first with
        :class:`~onnx_ir.passes.common.LiftConstantsToInitializersPass`.

    Example::

        import onnx_ir as ir

        model = ir.load("model.onnx")

        # Save model with tensors larger than 100 bytes to safetensors external data,
        # sharding files larger than 5GB.
        ir.save_safetensors(
            model,
            "model.onnx",
            size_threshold_bytes=100,
            max_shard_size_bytes=int(5 * 1000**3),  # Shard safetensors files larger than 5GB
        )

    .. tip::

        A simple progress bar can be implemented by passing a callback function as the following::

            import onnx_ir as ir
            import tqdm

            with tqdm.tqdm() as pbar:
                total_set = False

                def callback(tensor: ir.TensorProtocol, metadata: ir.external_data.CallbackInfo) -> None:
                    nonlocal total_set
                    if not total_set:
                        pbar.total = metadata.total
                        total_set = True

                    pbar.update()
                    pbar.set_description(f"Saving {metadata.filename}: {tensor.name} ({tensor.dtype}, {tensor.shape})")

                ir.save_safetensors(
                    ...,
                    callback=callback,
                )

    .. versionadded:: 0.1.15

    Args:
        model: ONNX model to save.
        path: Path to the ONNX model file. E.g. "model.onnx".
        format: The format of the file (e.g. ``protobuf``, ``textproto``, ``json``, etc.).
            If None, the format is inferred from the file extension.
        size_threshold_bytes: Save to external data if the tensor size in bytes
            is not smaller than this threshold.
        max_shard_size_bytes: Maximum size in bytes (as int) a safetensors file
            before being sharded. If None, no sharding is performed.
        callback: A callback function that is called after each tensor is saved.
            The callback must have signature ``Callable[[ir.TensorProtocol, ir.external_data.CallbackInfo], None]``,
            where the first argument is the tensor being saved and the second contains metadata such as filename and progress.

    Raises:
        ValueError: If duplicate initializer names are found in the model.
    """
    # Derive external_data from path if not provided
    path_str = str(path)
    # Get the base name without extension
    if "." in os.path.basename(path_str):
        base_name = os.path.splitext(os.path.basename(path_str))[0]
    else:
        base_name = os.path.basename(path_str)
    external_data = f"{base_name}.safetensors"

    # Store the original initializer values so they can be restored if modify_model=False
    value_tensor_pairs: list[tuple[ir.Value, ir.TensorProtocol]] = []
    initializer_names: set[str] = set()
    for graph in model.graphs():
        for value in graph.initializers.values():
            tensor = value.const_value
            # The value.name should be the same as tensor.name. However,
            # in case there is a conflict, we do not care and will prefer value.name.
            name = value.name
            if name is None:
                raise ValueError(
                    f"Initializer value '{value!r}' has no name (in graph {graph.name!r}). "
                    "All initializers must have names."
                )
            if tensor is None:
                continue
            if name in initializer_names:
                raise ValueError(
                    f"Duplicate initializer name found: {name} (in graph {graph.name!r})."
                    " Rename the initializers to have unique names before saving to safetensors."
                )
            initializer_names.add(name)
            value_tensor_pairs.append((value, tensor))

    try:
        _save_file(
            [value for value, _ in value_tensor_pairs],
            external_data,
            os.path.dirname(path),
            size_threshold_bytes=size_threshold_bytes,
            max_shard_size_bytes=max_shard_size_bytes,
            callback=callback,
        )
        ir.save(model, path, format=format)
    finally:
        # Restore original initializers to avoid side effects
        for value, tensor in value_tensor_pairs:
            value.const_value = tensor


def _read_safetensors_header(file: io.IOBase) -> tuple[dict[str, dict[str, Any]], int]:
    """Read the header of a safetensors file.

    Args:
        file: The safetensors file to read.

    Returns:
        The header of the safetensors file.
    """
    file.seek(0)
    header_size = struct.unpack_from("<Q", file.read(_HEADER_SIZE_NUMBER_SIZE))[0]
    header = file.read(header_size)
    return json.loads(header.decode("utf-8")), header_size


def _read_safetensors(
    location: str | os.PathLike, base_dir: str | os.PathLike
) -> dict[str, ir.ExternalTensor]:
    """Read a safetensors file.

    Args:
        location: The safetensors file to read.
        base_dir: Directory where the ONNX model file is stored.

    Returns:
        The contents of the safetensors file.
    """
    path = os.path.join(base_dir, location)
    with open(path, "rb") as file:
        header, header_size = _read_safetensors_header(file)
    tensors = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        offset = metadata["data_offsets"][0] + header_size + _HEADER_SIZE_NUMBER_SIZE
        length = metadata["data_offsets"][1] - metadata["data_offsets"][0]
        tensors[name] = ir.ExternalTensor(
            location=location,
            offset=offset,
            length=length,
            dtype=_SAFETENSORS_DTYPE_TO_IR_DTYPE[metadata["dtype"]],
            shape=ir.Shape(metadata["shape"]),
            name=name,
            base_dir=base_dir,
        )
    return tensors


def _migrate_tensor_shape_dtype(
    model_tensor: ir.TensorProtocol, safe_tensor: ir.ExternalTensor
) -> ir.ExternalTensor:
    """Migrate the shape and dtype of a tensor.

    This is needed because we store 4bit and 2bit tensors as UINT8 in safetensors.

    Args:
        model_tensor: The tensor to migrate.
        safe_tensor: The tensor to migrate to.

    Returns:
        The migrated tensor.
    """
    if model_tensor.dtype in {
        # Types that safetensors does not support directly
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2FNUZ,
        ir.DataType.FLOAT4E2M1,  # Still need to migrate shape
        ir.DataType.INT4,
        ir.DataType.INT2,
        ir.DataType.UINT4,
        ir.DataType.UINT2,
    }:
        return ir.ExternalTensor(
            location=safe_tensor.location,
            offset=safe_tensor.offset,
            length=safe_tensor.length,
            dtype=model_tensor.dtype,
            shape=model_tensor.shape,  # type: ignore[arg-type]
            name=safe_tensor.name,
            base_dir=safe_tensor.base_dir,
        )
    return safe_tensor
