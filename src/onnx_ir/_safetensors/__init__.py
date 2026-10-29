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
from onnx_ir import _core
from onnx_ir._polyfill import zip
from onnx_ir._shard_filename import get_shard_filename

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
_IR_DTYPE_TO_SAFETENSORS_DTYPE = {
    dtype: name for name, dtype in _SAFETENSORS_DTYPE_TO_IR_DTYPE.items()
}
_IR_DTYPE_TO_SAFETENSORS_DTYPE.update(
    {
        ir.DataType.FLOAT8E4M3FNUZ: "U8",
        ir.DataType.FLOAT8E5M2FNUZ: "U8",
        ir.DataType.FLOAT6E2M3: "U8",
        ir.DataType.FLOAT6E3M2: "U8",
        ir.DataType.INT2: "U8",
        ir.DataType.INT4: "U8",
        ir.DataType.UINT2: "U8",
        ir.DataType.UINT4: "U8",
    }
)


def _get_shard_filename(base_name: str, shard_idx: int, total_shards: int) -> str:
    # ``.safetensors`` is the complete format suffix. Preserve dotted model
    # variants such as ``model.fp16`` as part of the stem.
    return get_shard_filename(base_name, shard_idx, total_shards, suffix_count=1)


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


def _get_tensor_storage_shape(tensor: ir.TensorProtocol) -> Sequence[int]:
    """Get the storage shape of a tensor for safetensors."""
    if tensor.dtype == ir.DataType.FLOAT4E2M1:
        # Safetensors counts F4 elements, including the final padding nibble.
        return [tensor.nbytes * 2]
    # Handle sub-byte dtypes
    if tensor.dtype.bitwidth < 8:
        return [tensor.nbytes]
    return tensor.shape.numpy()


def _validate_file(path: str) -> None:
    # Validate the header and byte ranges without materializing any tensors.
    with _import_safetensors().safe_open(path, framework="numpy"):
        pass


def _save_file(
    tensors: dict[str, ir.TensorProtocol],
    /,
    location: str | os.PathLike,
    base_dir: str | os.PathLike = "",
    *,
    max_shard_size_bytes: int | None,
    callback: Callable[[ir.TensorProtocol, ir.external_data.CallbackInfo], None] | None = None,
) -> dict[str, ir.ExternalTensor]:
    """Stream named tensors to safetensors files and return their external references.

    Args:
        tensors: Tensors indexed by unique safetensors storage keys.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.
        max_shard_size_bytes: Maximum size in bytes (as int) a safetensors file
            before being sharded. If None, no sharding is performed.
        callback: A callback function invoked before each tensor is written.
    """
    _import_safetensors()

    # Ensure that external_data ends with .safetensors
    if not str(location).endswith(".safetensors"):
        raise ValueError(
            f'The path to safetensors file must have a .safetensors extension, got: "{location}"'
        )

    if not tensors:
        return {}
    names = list(tensors)
    tensor_shards = _shard_tensors(list(tensors.values()), max_shard_size_bytes)
    filenames = [
        _get_shard_filename(str(location), index, len(tensor_shards))
        for index in range(1, len(tensor_shards) + 1)
    ]
    index_path = (
        os.path.join(
            base_dir, str(location).removesuffix(".safetensors") + ".safetensors.index.json"
        )
        if len(tensor_shards) > 1
        else None
    )
    if max_shard_size_bytes is not None:
        destinations = [os.path.join(base_dir, name) for name in filenames]
        if index_path is not None:
            destinations.append(index_path)
        ir.external_data._check_no_existing_shard_files(destinations)

    weight_map: dict[str, str] = {}
    replacements: dict[str, ir.ExternalTensor] = {}
    current_index = 0
    for tensor_shard, filename in zip(tensor_shards, filenames, strict=True):
        shard_names = names[current_index : current_index + len(tensor_shard)]
        metadata = {}
        infos = []
        offset = 0
        for name, tensor in zip(shard_names, tensor_shard, strict=True):
            metadata[name] = {
                "dtype": _IR_DTYPE_TO_SAFETENSORS_DTYPE[tensor.dtype],
                "shape": _get_tensor_storage_shape(tensor),
                "data_offsets": [offset, offset + tensor.nbytes],
            }
            infos.append(ir.external_data._ExternalDataInfo(name, offset, tensor.nbytes))
            offset += tensor.nbytes
            weight_map[name] = filename
        header = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        header += b" " * (-len(header) % 8)
        header = struct.pack("<Q", len(header)) + header
        for info in infos:
            info.offset += len(header)
        shard_callback = (
            ir.external_data._make_shard_callback(callback, len(tensors), current_index)
            if callback is not None
            else None
        )
        ir.external_data._write_external_data(
            tensor_shard,
            infos,
            os.path.join(base_dir, filename),
            callback=shard_callback,
            header=header,
            validate=_validate_file,
        )
        for name, tensor in _read_safetensors(filename, base_dir).items():
            replacements[name] = _migrate_tensor_shape_dtype(tensors[name], tensor)
        current_index += len(tensor_shard)

    if index_path is not None:
        index_data = {
            "metadata": {"total_size": sum(tensor.nbytes for tensor in tensors.values())},
            "weight_map": weight_map,
        }
        ir.external_data._write_external_data(
            [], [], index_path, header=json.dumps(index_data, indent=2).encode("utf-8")
        )
    return replacements


def save_safetensors(
    model: ir.Model,
    path: str | os.PathLike,
    /,
    *,
    format: str | None = None,
    size_threshold_bytes: int = 256,
    max_shard_size_bytes: int | None = None,
    callback: Callable[[ir.TensorProtocol, ir.external_data.CallbackInfo], None] | None = None,
    convert_attributes: bool = False,
) -> None:
    """Save an ONNX model to a file with external data in a safetensors file.

    The model object is unmodified after this operation, except that existing
    external tensors backed by overwritten files are invalidated.

    When sharding is enabled, multiple safetensors files will be created
    with names like "model-00001-of-00003.safetensors", and an index
    file "model.safetensors.index.json" will be created to map tensors
    to their respective shard files. The shards will be created only if
    the total size of tensors exceeds the specified max_shard_size_bytes.

    .. note::
        Because the safetensors data format uses key-value mapping to store tensors,
        all initializer names in the model (across subgraphs) must be unique.
        Tensor attributes use generated unique safetensors keys, independent of
        their ONNX tensor names. Initializer names are reserved before generating
        attribute keys, and original tensor names are preserved in the ONNX file.

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

    .. versionadded:: 1.2.0
        Added ``convert_attributes`` to optionally store tensors in node attributes
        as safetensors external data.

    .. versionchanged:: 1.2.0
        Existing external attributes are preserved or streamed when relocation or
        an overwrite requires it. Tensor names, docstrings, and metadata properties
        are preserved. External initializers below the size threshold are embedded
        in the model. Single-file writes replace destinations atomically; sharded
        writes refuse to overwrite existing files. Callback offsets identify the
        tensor's physical byte offset within its shard, including the header.
        External source paths are validated before writing. Preserved attribute
        references must also pass the containment policy in the destination directory.

    Args:
        model: ONNX model to save.
        path: Path to the ONNX model file. E.g. "model.onnx".
        format: The format of the file (e.g. ``protobuf``, ``textproto``, ``json``, etc.).
            If None, the format is inferred from the file extension.
        size_threshold_bytes: Save to external data if the tensor size in bytes
            is not smaller than this threshold.
        max_shard_size_bytes: Maximum size in bytes (as int) a safetensors file
            before being sharded. If None, no sharding is performed.
        callback: A callback function invoked before each tensor is written.
            The callback must have signature ``Callable[[ir.TensorProtocol, ir.external_data.CallbackInfo], None]``,
            where the first argument is the tensor being saved and the second contains
            metadata such as filename, progress, and the byte offset within the shard.
        convert_attributes: Whether to store tensors in node attributes as external data
            according to ``size_threshold_bytes``, in addition to initializers.
            When true, non-string attribute tensors at or above the threshold are
            stored as external tensors; smaller tensors are embedded in the model.
            When false, existing external attributes remain external and are streamed
            to the destination if relocation or an overwrite requires it. In-memory
            attributes remain embedded.

    Raises:
        ValueError: If an external source violates the path containment policy.
        ValueError: If initializer names are missing, duplicated, or reserved by safetensors.
        ValueError: If ``max_shard_size_bytes`` is not positive.
        ValueError: If the ONNX model and safetensors data paths are the same.
        FileExistsError: If a destination shard or index file already exists when
            ``max_shard_size_bytes`` is set.
    """
    if max_shard_size_bytes is not None and max_shard_size_bytes <= 0:
        raise ValueError(
            f"max_shard_size_bytes must be greater than 0, got {max_shard_size_bytes}."
        )
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
    tensors_to_save: dict[str, ir.TensorProtocol] = {}
    initializers_to_externalize: dict[str, ir.Value] = {}
    initializers_to_load: list[ir.Value] = []
    check_source_path = functools.cache(_core._check_external_data_path)
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
            if name in initializer_names:
                raise ValueError(
                    f"Duplicate initializer name found: {name} (in graph {graph.name!r})."
                    " Rename the initializers to have unique names before saving to safetensors."
                )
            initializer_names.add(name)
            if tensor is None:
                continue
            if isinstance(tensor, ir.ExternalTensor):
                check_source_path(os.fspath(tensor.location), os.fspath(tensor.base_dir))
            value_tensor_pairs.append((value, tensor))
            if tensor.dtype == ir.DataType.STRING:
                continue
            if tensor.nbytes >= size_threshold_bytes:
                if name == "__metadata__":
                    raise ValueError(
                        "Initializer name '__metadata__' is reserved by safetensors. "
                        "Rename the initializer before saving to safetensors."
                    )
                tensors_to_save[name] = tensor
                initializers_to_externalize[name] = value
            elif isinstance(tensor, ir.ExternalTensor):
                initializers_to_load.append(value)

    references = list(ir.external_data._attribute_tensor_references(model))
    original_attributes = {
        (node, name): node.attributes[name] for node, name, _, _ in references
    }
    attributes_to_externalize = {}
    attributes_to_load = []
    selected_attributes = []
    overlapping_attributes = []
    file_identity = functools.cache(ir.external_data._file_identity)
    needs_relocation = functools.cache(
        functools.partial(
            ir.external_data._attribute_needs_relocation,
            base_dir=os.path.dirname(path),
            file_identity=file_identity,
        )
    )
    destination_path = os.path.join(os.path.dirname(path), external_data)
    if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
        os.path.abspath(destination_path)
    ):
        raise ValueError("The ONNX model and safetensors data paths must be different.")
    destination_identity = (
        file_identity(destination_path) if max_shard_size_bytes is None else None
    )
    if (
        destination_identity is not None
        and file_identity(os.fspath(path)) == destination_identity
    ):
        raise ValueError("The ONNX model and safetensors data paths must be different.")
    for reference in references:
        tensor = reference[3]
        if isinstance(tensor, ir.ExternalTensor):
            check_source_path(os.fspath(tensor.location), os.fspath(tensor.base_dir))
        if tensor.dtype == ir.DataType.STRING:
            continue
        if convert_attributes:
            externalize = tensor.nbytes >= size_threshold_bytes
            if not externalize and isinstance(tensor, ir.ExternalTensor):
                attributes_to_load.append(reference)
        elif isinstance(tensor, ir.ExternalTensor):
            externalize = needs_relocation(
                os.fspath(tensor.location), os.fspath(tensor.base_dir)
            )
            if (
                not externalize
                and destination_identity is not None
                and file_identity(tensor.path) == destination_identity
            ):
                overlapping_attributes.append(reference)
        else:
            externalize = False
        if externalize:
            selected_attributes.append(reference)
    if tensors_to_save or selected_attributes:
        selected_attributes.extend(overlapping_attributes)
    next_attribute_key = 0
    for reference in selected_attributes:
        key = f"__onnx_ir_attribute_{next_attribute_key}"
        while key in initializer_names:
            next_attribute_key += 1
            key = f"__onnx_ir_attribute_{next_attribute_key}"
        next_attribute_key += 1
        tensors_to_save[key] = reference[3]
        attributes_to_externalize[key] = reference

    materialized_sources = [value.const_value for value in initializers_to_load]
    materialized_sources.extend(reference[3] for reference in attributes_to_load)
    overwritten_materialized_tensors = [
        tensor
        for tensor in materialized_sources
        if isinstance(tensor, ir.ExternalTensor)
        and destination_identity is not None
        and file_identity(tensor.path) == destination_identity
    ]

    try:
        loaded_initializers = ir.external_data.convert_tensors_from_external(
            [value.const_value for value in initializers_to_load]  # type: ignore[misc]
        )
        loaded_attributes = ir.external_data.convert_tensors_from_external(
            [reference[3] for reference in attributes_to_load]
        )
        replacements = _save_file(
            tensors_to_save,
            external_data,
            os.path.dirname(path),
            max_shard_size_bytes=max_shard_size_bytes,
            callback=callback,
        )
        for name, value in initializers_to_externalize.items():
            value.const_value = replacements[name]
        for value, tensor in zip(initializers_to_load, loaded_initializers, strict=True):
            value.const_value = tensor
        ir.external_data._replace_attribute_tensors(
            [
                *(
                    (reference, replacements[key])
                    for key, reference in attributes_to_externalize.items()
                ),
                *zip(attributes_to_load, loaded_attributes, strict=True),
            ]
        )
        ir.save(model, path, format=format)
    finally:
        # Restore original initializers to avoid side effects
        for value, tensor in value_tensor_pairs:
            value.const_value = tensor
        for (node, name), attr in original_attributes.items():
            node.attributes[name] = attr
        if overwritten_materialized_tensors and (
            ir.external_data._file_identity(destination_path) != destination_identity
        ):
            ir.external_data._invalidate_external_tensors(overwritten_materialized_tensors)


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
    """Restore the original ONNX tensor description after reading storage metadata.

    Packed tensors may use a different storage shape or dtype in safetensors.
    Storage keys must not replace the original tensor names or annotations.

    Args:
        model_tensor: The tensor to migrate.
        safe_tensor: The tensor to migrate to.

    Returns:
        The migrated tensor.
    """
    assert safe_tensor.offset is not None
    assert safe_tensor.length is not None
    return ir.external_data._create_external_tensor(
        model_tensor,
        ir.external_data._ExternalDataInfo(
            model_tensor.name, safe_tensor.offset, safe_tensor.length
        ),
        safe_tensor.base_dir,
        safe_tensor.location,
    )
