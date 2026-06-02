# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""External data related utilities."""

from __future__ import annotations

from typing import Callable

__all__ = [
    "set_base_dir",
    "unload_from_model",
    "load_to_model",
    "convert_tensors_to_external",
    "convert_tensors_from_external",
    "CallbackInfo",
]

import dataclasses
import logging
import os
import re
from collections.abc import Iterator, Sequence

from onnx_ir import _core, _enums, _protocols
from onnx_ir import traversal as _traversal
from onnx_ir._polyfill import zip

# Note: If needed in future, add these as parameters to the function calls
# align_offset: Offset will always be page aligned and alloction granularity aligned for mmap support. This is done by padding previous tensor data with zeros keeping same length. Tensor data will be aligned if > align_threshold
_ALIGN_OFFSET = True
# align_threshold: Alignment threshold for size of data. Having a low threshold will waste file space for small initializers. Only when tensor's data is > the page_align_threshold it will be force aligned.
_ALIGN_THRESHOLD = 1048576  # 1MB
# allocation_granularity: The allocation Granularity for mmap() support. Typically 64KB for Windows & 4KB for other OSes.
_ALLOCATION_GRANULARITY = 65536  # 64KB


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ExternalDataInfo:
    """A class that stores information about a tensor that is to be stored as external data.

    Attributes:
        name: The name of the tensor that is to be stored as external data.
        offset: The offset is used to determine where exactly in the file the external data is written to.
        length: Stores the size of the tensor.
    """

    name: str | None
    offset: int
    length: int


@dataclasses.dataclass
class CallbackInfo:
    """A class that shares information about a tensor that is to be saved as external data for callback functions.

    Attributes:
        total: The total number of tensors to save.
        index: The index of the tensor being saved.
        offset: The offset of the tensor in the external data file.
        filename: The filename of the external data file.
    """

    total: int
    index: int
    offset: int
    filename: str


def _all_tensors(
    graph: _core.Graph, include_attributes: bool = False
) -> Iterator[_protocols.TensorProtocol]:
    """Iterate over all tensors in the graph.

    Args:
        graph: The graph to traverse tensors on.
        include_attributes: Whether to include tensors in attributes.

    Yields:
        Tensors in the graph.
    """
    # Yield all tensors in initializers
    for value in graph.initializers.values():
        if (tensor := value.const_value) is not None:
            yield tensor
    if not include_attributes:
        return
    # Look at constant attributes in nodes
    for node in _traversal.RecursiveGraphIterator(graph):
        for attr in node.attributes.values():
            if attr.is_ref():
                continue
            if attr.type == _enums.AttributeType.TENSOR and attr.value is not None:
                yield attr.value
            elif attr.type == _enums.AttributeType.TENSORS and attr.value is not None:
                yield from attr.value
            elif attr.type == _enums.AttributeType.GRAPH and attr.value is not None:
                for value in attr.value.initializers.values():
                    if (tensor := value.const_value) is not None:
                        yield tensor
            elif attr.type == _enums.AttributeType.GRAPHS and attr.value is not None:
                for g in attr.value:
                    for value in g.initializers.values():
                        if (tensor := value.const_value) is not None:
                            yield tensor


def set_base_dir(graph: _core.Graph, base_dir: str | os.PathLike) -> None:
    """Set the base directory for external data in a graph (including all of its subgraphs).

    Args:
        graph: The graph to traverse tensors on.
        base_dir: The base directory. This is the directory where the ONNX file is.
    """
    for tensor in _all_tensors(graph, include_attributes=True):
        if isinstance(tensor, _core.ExternalTensor):
            tensor.base_dir = base_dir


def _get_shard_filename(base_name: str, shard_idx: int, total_shards: int) -> str:
    """Generate a filename for a shard of external data.

    Args:
        base_name: The base filename (e.g., 'model.data').
        shard_idx: The index of this shard (1-indexed).
        total_shards: The total number of shards.

    Returns:
        The shard filename (e.g., 'model-00001-of-00003.data').
    """
    if total_shards == 1:
        return base_name

    dir_name, filename = os.path.split(base_name)
    name, ext = os.path.splitext(filename)

    # Always use 5 digits to follow transformers convention
    shard_filename = f"{name}-{shard_idx:05d}-of-{total_shards:05d}{ext}"
    return os.path.join(dir_name, shard_filename) if dir_name else shard_filename


def _make_shard_callback(
    callback: Callable[[_protocols.TensorProtocol, CallbackInfo], None],
    total: int,
    index_offset: int,
) -> Callable[[_protocols.TensorProtocol, CallbackInfo], None]:
    def _shard_callback(
        tensor: _protocols.TensorProtocol,
        info: CallbackInfo,
    ) -> None:
        callback(
            tensor,
            CallbackInfo(
                total=total,
                index=index_offset + info.index,
                offset=info.offset,
                filename=info.filename,
            ),
        )

    return _shard_callback


@dataclasses.dataclass
class _ShardSizeAccumulator:
    """Incrementally track the on-disk size of a shard in O(1) per tensor.

    The size matches :func:`_estimate_shard_size_bytes` (tensors written in
    size-sorted order with offset alignment) but is maintained incrementally so
    that sharding does not need to re-sort the shard on every tensor append.

    Sorting tensors ascending by size places every unaligned tensor
    (``nbytes <= _ALIGN_THRESHOLD``) before every aligned one, so the final size
    only depends on a few running aggregates rather than the full ordering.
    """

    sum_unaligned: int = 0
    sum_aligned: int = 0
    largest_aligned_raw: int = 0
    largest_aligned_padded: int = 0
    has_aligned: bool = False

    def add(self, tensor: _protocols.TensorProtocol) -> None:
        size = tensor.nbytes
        if _ALIGN_OFFSET and size > _ALIGN_THRESHOLD:
            alignment_factor = max(4096, _ALLOCATION_GRANULARITY)
            padded = (size + alignment_factor - 1) // alignment_factor * alignment_factor
            self.sum_aligned += padded
            if size >= self.largest_aligned_raw:
                self.largest_aligned_raw = size
                self.largest_aligned_padded = padded
            self.has_aligned = True
        else:
            self.sum_unaligned += size

    @property
    def size(self) -> int:
        if not self.has_aligned:
            return self.sum_unaligned
        alignment_factor = max(4096, _ALLOCATION_GRANULARITY)
        # The first aligned tensor pads the unaligned prefix up to an alignment
        # boundary; the largest aligned tensor is written last and is not padded.
        leading = (
            (self.sum_unaligned + alignment_factor - 1) // alignment_factor * alignment_factor
        )
        return (
            leading + self.sum_aligned - self.largest_aligned_padded + self.largest_aligned_raw
        )


def _shard_tensors(
    tensors: Sequence[_protocols.TensorProtocol],
    max_shard_size_bytes: int,
) -> list[list[_protocols.TensorProtocol]]:
    """Shard tensors into multiple groups based on max_shard_size_bytes.

    Each tensor is always placed in exactly one shard. A new shard is started when
    adding the next tensor would exceed the limit once the shard is written using
    the same layout as :func:`convert_tensors_to_external` (size-sorted write order
    plus offset alignment).

    Args:
        tensors: The tensors to shard.
        max_shard_size_bytes: Maximum cumulative size in bytes for each shard.

    Returns:
        A list of tensor groups, one per shard.
    """
    shards: list[list[_protocols.TensorProtocol]] = [[]]
    accumulator = _ShardSizeAccumulator()

    for tensor in tensors:
        if tensor.nbytes > max_shard_size_bytes:
            logger.warning(
                "Tensor %s (%d bytes) exceeds max_shard_size_bytes=%d and will be written in an oversized shard.",
                tensor.name,
                tensor.nbytes,
                max_shard_size_bytes,
            )
        candidate = dataclasses.replace(accumulator)
        candidate.add(tensor)
        # Start a new shard when the current one would be exceeded
        # (but never leave a shard empty).
        if candidate.size > max_shard_size_bytes and len(shards[-1]) > 0:
            shards.append([])
            accumulator = _ShardSizeAccumulator()
            accumulator.add(tensor)
        else:
            accumulator = candidate

        shards[-1].append(tensor)

    return shards


def _external_tensor_to_memory_tensor(
    tensor: _protocols.TensorProtocol,
) -> _protocols.TensorProtocol:
    """Convert an external tensor to an in memory tensor.

    Args:
        tensor: An external tensor to load.
        base_dir: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.

    Returns:
        An ir.Tensor object with the data loaded into memory.
    """
    if not isinstance(tensor, _core.ExternalTensor):
        raise TypeError(f"Expected ExternalTensor, got {type(tensor)}")
    # Copy the data as the .numpy() call references data from a file whose data is eventually modified
    tensor_data = tensor.numpy().copy()
    tensor.release()
    return _core.Tensor(tensor_data, name=tensor.name, dtype=tensor.dtype)


def _compute_new_offset(
    current_offset: int,
    tensor_size: int,
    align_offset: bool = _ALIGN_OFFSET,
    align_threshold: int = _ALIGN_THRESHOLD,
    allocation_granularity: int = _ALLOCATION_GRANULARITY,
) -> int:
    """Compute the offset to align the tensor data based on the current offset.

    Args:
        current_offset: Current location in the file at which tensor data will be written to.
        tensor_size: Size of the tensor data to be written to file.
        align_offset: Offset will always be page aligned and alloction granularity aligned for mmap support. This is done by padding previous tensor data with zeros keeping same length. Tensor data will be aligned if > align_threshold
        align_threshold: Alignment threshold for size of data. Having a low threshold will waste file space for small initializers. Only when tensor's data is > the page_align_threshold it will be force aligned.
        allocation_granularity: The allocation Granularity for mmap() support. Typically 64KB for Windows & 4KB for other OSes.

    Returns:
        The updated offset value.
    """
    if align_offset and tensor_size > align_threshold:
        alignment_factor = max(4096, allocation_granularity)
        # Align to the next page or alloc granularity
        return (current_offset + alignment_factor - 1) // alignment_factor * alignment_factor
    return current_offset


def _estimate_shard_size_bytes(tensors: Sequence[_protocols.TensorProtocol]) -> int:
    """Estimate the shard file size in bytes for tensors written to one file."""
    current_offset = 0
    sorted_tensors = sorted(tensors, key=lambda tensor: tensor.nbytes)
    for tensor in sorted_tensors:
        current_offset = _compute_new_offset(current_offset, tensor.nbytes)
        current_offset += tensor.nbytes
    return current_offset


def _paths_refer_to_same_file(path1: str | os.PathLike, path2: str | os.PathLike) -> bool:
    """Return True if both paths exist and refer to the same file.

    Uses :func:`os.path.samefile` so that hard links and symlinks pointing to the
    same underlying file are correctly detected.
    """
    try:
        return os.path.samefile(path1, path2)
    except OSError:
        # One of the paths does not exist (or cannot be stat'd).
        return False


def _materialize_external_tensors_for_destination_paths(
    tensors: Sequence[_protocols.TensorProtocol],
    destination_paths: Sequence[str | os.PathLike],
) -> list[_protocols.TensorProtocol]:
    existing_destination_paths = [path for path in destination_paths if os.path.exists(path)]
    if not existing_destination_paths:
        return list(tensors)

    converted_tensors: list[_protocols.TensorProtocol] = []
    for tensor in tensors:
        if isinstance(tensor, _core.ExternalTensor) and any(
            _paths_refer_to_same_file(tensor.path, destination_path)
            for destination_path in existing_destination_paths
        ):
            # FIXME(shubhambhokare1): If there is a non-initializer tensor that
            # is referring to this file, that tensor is now invalid.
            # This is a special case we are ok not handling right now.
            converted_tensors.append(_external_tensor_to_memory_tensor(tensor))
            # Mark the original external tensor as invalid because it is now pointing
            # to a file that is going to be overwritten.
            tensor.invalidate()
            logger.warning(
                "External tensor %s is referring to the destination path. "
                "It has been invalidated because the data file is changed. To avoid this, "
                "save the external data to a different path or load the newly saved model back "
                "with ir.load().",
                tensor,
            )
        else:
            converted_tensors.append(tensor)

    return converted_tensors


def _cleanup_stale_shard_files(
    base_dir: str | os.PathLike,
    relative_path: str | os.PathLike,
    total_shards: int,
) -> None:
    path_str = str(relative_path)
    dir_name, file_name = os.path.split(path_str)
    stem, ext = os.path.splitext(file_name)
    shard_dir = os.path.join(base_dir, dir_name) if dir_name else str(base_dir)
    if not os.path.isdir(shard_dir):
        return

    shard_pattern = re.compile(rf"^{re.escape(stem)}-(\d{{5}})-of-(\d{{5}}){re.escape(ext)}$")
    for filename in os.listdir(shard_dir):
        match = shard_pattern.match(filename)
        if match is None:
            continue
        shard_index = int(match.group(1))
        shard_total = int(match.group(2))
        # Remove shards from a different layout, out-of-range indices, and any
        # invalid 0-indexed shard files (shard indices are 1-indexed).
        if shard_total != total_shards or shard_index < 1 or shard_index > total_shards:
            os.remove(os.path.join(shard_dir, filename))


def _compute_external_data_info(
    tensor: _protocols.TensorProtocol,
    current_offset: int,
) -> _ExternalDataInfo:
    """Capture information about a tensor that is to be stored as external data."""
    tensor_size = tensor.nbytes
    # Calculate updated offset and align tensors
    current_offset = _compute_new_offset(current_offset, tensor_size)
    # Store offset and tensor size as ExternalDataInfo
    external_data_info = _ExternalDataInfo(
        tensor.name,
        current_offset,
        tensor_size,
    )
    return external_data_info


def _write_external_data(
    tensors: Sequence[_protocols.TensorProtocol],
    external_data_infos: Sequence[_ExternalDataInfo],
    file_path: str | os.PathLike,
    callback: Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None = None,
) -> None:
    """Write tensor data to an external file according to information stored in ExternalDataInfo objects.

    Args:
        tensors: Tensors to be written as external data.
        external_data_infos: External data information stored for each tensor to be written as external data.
        file_path: Location to which external data is to be stored.
        callback: A callback function that is called for each tensor that is saved to external data
            for debugging or logging purposes.
    """
    tensors_count = len(tensors)
    assert tensors_count == len(external_data_infos), (
        "Number of tensors and external data infos should match"
    )
    with open(file_path, "wb") as data_file:
        for i, (tensor, tensor_info) in enumerate(
            zip(tensors, external_data_infos, strict=True)
        ):
            if callback is not None:
                callback(
                    tensor,
                    CallbackInfo(
                        total=tensors_count,
                        index=i,
                        offset=tensor_info.offset,
                        filename=os.path.basename(file_path),
                    ),
                )
            current_offset = tensor_info.offset
            assert tensor is not None
            # Pad file to required offset if needed
            file_size = data_file.tell()
            if current_offset > file_size:
                data_file.write(b"\0" * (current_offset - file_size))

            if hasattr(tensor, "tofile"):
                # Some existing implementation of TensorProtocol
                # may not have tofile() as it was introduced in v0.1.11
                tensor.tofile(data_file)
            else:
                raw_data = tensor.tobytes()
                if isinstance(tensor, _core.ExternalTensor):
                    tensor.release()
                data_file.write(raw_data)


def _create_external_tensor(
    tensor: _protocols.TensorProtocol,
    external_data_info: _ExternalDataInfo,
    base_dir: str | os.PathLike,
    relative_path: str | os.PathLike,
) -> _core.ExternalTensor:
    """Create external tensors from external data information.

    Args:
        tensor: Tensor to be converted to external tensor.
        external_data_info: External data information stored for the tensor to be written as external data.
        base_dir: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.

    Returns:
        External tensor created from the information.
    """
    return _core.ExternalTensor(
        os.path.normpath(relative_path),
        external_data_info.offset,
        external_data_info.length,
        tensor.dtype,  # type: ignore[arg-type]
        shape=tensor.shape,  # type: ignore[arg-type]
        name=tensor.name,  # type: ignore[arg-type]
        base_dir=os.path.normpath(base_dir),
    )


def convert_tensors_from_external(
    tensors: Sequence[_protocols.TensorProtocol],
) -> list[_protocols.TensorProtocol]:
    """Convert a sequence of external tensors to in-memory tensors.

    Args:
        tensors: External tensors to be converted to in-memory tensors.

    Returns:
        A list of in-memory tensors derived from a list of external tensors.
    """
    return [_external_tensor_to_memory_tensor(tensor) for tensor in tensors]


def convert_tensors_to_external(
    tensors: Sequence[_protocols.TensorProtocol],
    base_dir: str | os.PathLike,
    relative_path: str | os.PathLike,
    callback: Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None = None,
) -> list[_core.ExternalTensor]:
    """Convert a sequence of any TensorProtocol tensors to external tensors.

    Existing external tensors are loaded to memory if they are referring to the
    same file path as the destination path.

    Args:
        tensors: Tensors to be converted to external tensors. They can be external tensors themselves.
        base_dir: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.
        callback: A callback function that is called for each tensor that is saved to external data
            for debugging or logging purposes.

    Returns:
        A list of external tensors derived from a list of input tensors. The order
        should match the input tensor order.
    """
    path = os.path.join(base_dir, relative_path)
    tensors = _materialize_external_tensors_for_destination_paths(tensors, [path])

    external_data_infos: list[_ExternalDataInfo] = []
    # Sort all tensors based on tensor sizes, in order to avoid unnecessary alignment.
    # All the smaller tensors are written earlier and alignment is performed for the larger tensors.
    sorted_indices = sorted(range(len(tensors)), key=lambda i: tensors[i].nbytes)
    sorted_tensors = [tensors[i] for i in sorted_indices]

    # Compute external data information for each tensor and write to disk
    current_offset = 0
    for tensor in sorted_tensors:
        external_info = _compute_external_data_info(tensor, current_offset)
        external_data_infos.append(external_info)
        current_offset = external_info.offset + external_info.length
    _write_external_data(sorted_tensors, external_data_infos, path, callback=callback)

    # Create external tensor objects
    external_tensors: list[_core.ExternalTensor] = [
        _create_external_tensor(tensor, external_info, base_dir, relative_path)
        for tensor, external_info in zip(sorted_tensors, external_data_infos, strict=True)
    ]

    # Sort external_tensors based on original key order. So that it can match the input tensor order
    external_tensors = [
        external_tensors[i]
        for i in sorted(range(len(external_tensors)), key=lambda i: sorted_indices[i])
    ]

    return external_tensors


def load_to_model(model: _core.Model) -> _core.Model:
    """Convert all external model initializers to memory tensors in-place.

    All initializers in the main graph and subgraphs are handled.

    Args:
        model: Model to process.
    """
    # TODO(justinchuby): Load tensor attributes in subgraphs
    values_to_convert = []
    for graph in model.graphs():
        for value in graph.initializers.values():
            if value.const_value is None:
                # Filter out the uninitialized initializer values
                continue
            if isinstance(value.const_value, _core.ExternalTensor):
                values_to_convert.append(value)
    loaded_tensors = convert_tensors_from_external(
        [v.const_value for v in values_to_convert]  # type: ignore[misc]
    )
    for value, tensor in zip(values_to_convert, loaded_tensors, strict=True):
        value.const_value = tensor

    # Return the model because we may change the implementation to an out of place one
    # to keep the input unchanged
    return model


def unload_from_model(
    model: _core.Model,
    base_dir: str | os.PathLike,
    relative_path: str | os.PathLike,
    *,
    size_threshold_bytes: int = 0,
    max_shard_size_bytes: int | None = None,
    callback: Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None = None,
) -> _core.Model:
    """Convert all initializers equal or above size_threshold_bytes to external tensors in-place and save data to one or more data files.

    It should only replace the initializers in the model with external tensors
    and not make any other modifications to the model.

    If any existing external tensor
    references the provided ``external_data`` path, it will be invalidated
    after the external data is overwritten. To obtain a valid model, use :func:`load`
    to load the newly saved model, or provide a different external data path that
    is not currently referenced by any tensors in the model.

    All initializers in the main graph and subgraphs are handled.

    When ``max_shard_size_bytes`` is set, tensors are distributed across multiple
    shard files named like ``model-00001-of-00003.data``. Because each ONNX tensor
    already carries its own ``location``, ``offset``, and ``length`` fields, no
    separate index file is required — the ONNX proto itself encodes the routing.

    Args:
        model: Model to process.
        base_dir: Path the directory where the ONNX model file is.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.
            E.g. "model.data". When sharding is enabled this becomes the base name used to
            generate shard filenames such as "model-00001-of-00003.data".
        size_threshold_bytes: Save to external data if the tensor size in bytes is larger than this threshold.
        max_shard_size_bytes: Maximum cumulative size in bytes for a single shard file.
            When ``None`` (the default) all tensors are written to a single file given by
            ``relative_path``.  When set, tensors are written to multiple numbered shard
            files. If a single tensor is larger than this value, that tensor is written
            in its own oversized shard.
        callback: A callback function that is called for each tensor that is saved to external data
            for debugging or logging purposes. Under sharding the callback index reflects
            each shard's size-sorted write order while remaining globally contiguous.

    Returns:
        An ir.Model with all initializer data equal or above ``size_threshold_bytes``
        converted to external tensors.

    Raises:
        ValueError: If ``max_shard_size_bytes`` is not greater than 0.
    """
    if max_shard_size_bytes is not None and max_shard_size_bytes <= 0:
        raise ValueError(
            f"max_shard_size_bytes must be greater than 0, got {max_shard_size_bytes}."
        )

    # In-memory or external tensors, if equal to or above the threshold, should be converted to or re-saved as external tensors
    initializers_to_become_external = []
    # Existing external tensors, if below the threshold, should be loaded to memory
    initializers_to_load_to_memory = []
    for graph in model.graphs():
        for value in graph.initializers.values():
            if value.const_value is None:
                # Filter out the uninitialized initializer values
                continue
            if value.const_value.nbytes > size_threshold_bytes:
                initializers_to_become_external.append(value)
            elif isinstance(value.const_value, _core.ExternalTensor):
                initializers_to_load_to_memory.append(value)

    # Load to memory first, then convert to external tensors, because
    # the existing external tensors may be overwritten by the new external data
    memory_tensors = convert_tensors_from_external(
        [v.const_value for v in initializers_to_load_to_memory]  # type: ignore[misc]
    )

    external_tensors: list[_core.ExternalTensor]
    if max_shard_size_bytes is None:
        # No sharding: write all tensors to the single destination file
        external_tensors = convert_tensors_to_external(
            [v.const_value for v in initializers_to_become_external],  # type: ignore[misc]
            base_dir=base_dir,
            relative_path=relative_path,
            callback=callback,
        )
    else:
        # Sharding: distribute tensors across multiple numbered shard files
        tensors_to_externalize: list[_protocols.TensorProtocol] = [
            v.const_value  # type: ignore[misc]
            for v in initializers_to_become_external
        ]
        tensor_shards = _shard_tensors(tensors_to_externalize, max_shard_size_bytes)
        total_shards = len(tensor_shards)
        total_tensors = len(tensors_to_externalize)
        shard_relative_paths = [
            _get_shard_filename(str(relative_path), shard_idx, total_shards)
            for shard_idx in range(1, total_shards + 1)
        ]
        destination_paths = [
            os.path.join(base_dir, shard_relative_path)
            for shard_relative_path in shard_relative_paths
        ]
        tensors_to_externalize = _materialize_external_tensors_for_destination_paths(
            tensors_to_externalize, destination_paths
        )
        _cleanup_stale_shard_files(base_dir, relative_path, total_shards)

        external_tensors = []
        global_index = 0

        for shard_relative_path, shard_tensor_count in zip(
            shard_relative_paths,
            [len(shard) for shard in tensor_shards],
            strict=True,
        ):
            shard_tensors = tensors_to_externalize[
                global_index : global_index + shard_tensor_count
            ]
            # Wrap the callback so that index/total reflect the global position across shards
            shard_callback: (
                Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None
            ) = None
            if callback is not None:
                shard_callback = _make_shard_callback(callback, total_tensors, global_index)

            shard_external_tensors = convert_tensors_to_external(
                shard_tensors,
                base_dir=base_dir,
                relative_path=shard_relative_path,
                callback=shard_callback,
            )

            external_tensors.extend(shard_external_tensors)
            global_index += shard_tensor_count

    # Replace the initializer values with external tensors and save the model
    for value, external_tensor in zip(
        initializers_to_become_external, external_tensors, strict=True
    ):
        value.const_value = external_tensor
    for value, memory_tensor in zip(
        initializers_to_load_to_memory, memory_tensors, strict=True
    ):
        value.const_value = memory_tensor

    # Return the model because we may change the implementation to an out of place one
    # to keep the input unchanged
    return model
