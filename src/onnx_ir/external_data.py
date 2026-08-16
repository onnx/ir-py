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

import concurrent.futures
import dataclasses
import logging
import os
import threading
from collections.abc import Iterator, Sequence

from onnx_ir import _core, _enums, _protocols
from onnx_ir import traversal as _traversal
from onnx_ir._polyfill import zip

# Default alignment threshold used when alignment is enabled: only tensors larger
# than this get their offset aligned, so small initializers don't waste file space.
_DEFAULT_ALIGN_THRESHOLD = 1048576  # 1MB
# Default allocation granularity for mmap() support. Typically 64KB on Windows
# and 4KB elsewhere; 64KB is the safe cross-platform choice.
_DEFAULT_ALLOCATION_GRANULARITY = 65536  # 64KB
# Default upper bound on materialized tensor bytes held in memory while writing
# external data concurrently. Peak memory is this plus the largest single tensor.
_DEFAULT_MAX_IN_FLIGHT_BYTES = 1 << 29  # 512MB


logger = logging.getLogger(__name__)


def _align_offset(
    current_offset: int, tensor_size: int, alignment: int | None, align_threshold: int
) -> int:
    """Return the offset at which a tensor of ``tensor_size`` bytes should start.

    Alignment used to be applied unconditionally because ONNX Runtime refused to
    memory-map a tensor whose offset was not a multiple of the allocation
    granularity. ORT now rounds the offset down to the containing page or
    allocation block itself, so alignment is no longer required for correctness.
    Dense packing produces smaller files and matches what safetensors does, so
    ``alignment=None`` is the default.
    """
    if alignment is None:
        return current_offset
    if tensor_size <= align_threshold:
        # Aligning small initializers wastes file space for no benefit.
        return current_offset
    factor = max(4096, alignment)
    return (current_offset + factor - 1) // factor * factor


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

    .. note::
        When saving with ``max_workers`` greater than 1, callbacks are invoked
        from worker threads. Calls are serialized with a lock, so the callback
        itself does not need to be thread-safe, but they are **not** delivered in
        ``index`` order. Progress reporting should count invocations rather than
        rely on ``index`` increasing monotonically.

    Attributes:
        total: The total number of tensors to save.
        index: The index of the tensor being saved. Not necessarily the order in
            which callbacks are invoked when saving concurrently.
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


def _shard_tensors(
    tensors: Sequence[_protocols.TensorProtocol],
    max_shard_size_bytes: int,
    alignment: int | None = None,
    align_threshold: int = _DEFAULT_ALIGN_THRESHOLD,
) -> list[list[_protocols.TensorProtocol]]:
    """Shard tensors into multiple groups based on max_shard_size_bytes.

    Each tensor is always placed in exactly one shard, in declaration order. A new
    shard is started when adding the next tensor would exceed the limit. Without
    alignment a shard's on-disk size is simply the sum of its tensor sizes; with
    alignment the padding inserted before large tensors is accounted for too.

    Args:
        tensors: The tensors to shard.
        max_shard_size_bytes: Maximum cumulative size in bytes for each shard.
        alignment: Alignment in bytes for the offsets of large tensors, or ``None``
            for dense packing.
        align_threshold: Only tensors strictly larger than this many bytes are
            aligned. Ignored when ``alignment`` is ``None``.

    Returns:
        A list of tensor groups, one per shard.
    """
    shards: list[list[_protocols.TensorProtocol]] = [[]]
    shard_size = 0

    for tensor in tensors:
        if tensor.nbytes > max_shard_size_bytes:
            logger.warning(
                "Tensor %s (%d bytes) exceeds max_shard_size_bytes=%d and will be written in an oversized shard.",
                tensor.name,
                tensor.nbytes,
                max_shard_size_bytes,
            )
        offset = shard_size
        offset = _align_offset(offset, tensor.nbytes, alignment, align_threshold)
        # Start a new shard when the current one would be exceeded
        # (but never leave a shard empty).
        if offset + tensor.nbytes > max_shard_size_bytes and shards[-1]:
            shards.append([])
            offset = 0

        shards[-1].append(tensor)
        shard_size = offset + tensor.nbytes

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


def _estimate_shard_size_bytes(
    tensors: Sequence[_protocols.TensorProtocol],
    alignment: int | None = None,
    align_threshold: int = _DEFAULT_ALIGN_THRESHOLD,
) -> int:
    """Estimate the shard file size in bytes for tensors written to one file."""
    current_offset = 0
    for tensor in tensors:
        current_offset = _align_offset(
            current_offset, tensor.nbytes, alignment, align_threshold
        )
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
    """Load into memory any external tensor whose backing file is about to be overwritten or deleted.

    Safety-critical: this is what allows ``unload_from_model`` to re-save a
    loaded sharded model — even when the new shard layout differs from the
    old one — without reading from a file that has already been clobbered or
    cleaned up. Callers must pass *every* path that will be written, renamed,
    or deleted by the upcoming save, not just the immediate destination.
    """
    existing_destination_paths = [path for path in destination_paths if os.path.exists(path)]
    if not existing_destination_paths:
        return list(tensors)

    converted_tensors: list[_protocols.TensorProtocol] = []
    for tensor in tensors:
        if isinstance(tensor, _core.ExternalTensor) and any(
            _paths_refer_to_same_file(tensor.path, destination_path)
            for destination_path in existing_destination_paths
        ):
            # TODO(justinchuby): If there is a non-initializer tensor that
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


def _check_no_existing_shard_files(
    destination_paths: Sequence[str | os.PathLike],
) -> None:
    """Raise if any destination shard file already exists on disk.

    The sharded write path never overwrites a pre-existing file. Different
    shard counts produce different filenames (``-of-00002`` vs ``-of-00004``),
    so silently overwriting is ambiguous and dangerous: it can both clobber a
    foreign model's shards and leave our own stale shards behind as orphans.
    By refusing to touch any existing destination we keep the contract simple
    and avoid having to materialize tensors or stage temporary files: because
    no destination is ever overwritten, no in-memory tensor can be reading
    from a file we are about to write.

    To re-save a model whose shards already exist on disk, the caller must
    either delete the existing files first or choose a different external data
    path/directory.
    """
    existing = [os.fspath(path) for path in destination_paths if os.path.exists(path)]
    if existing:
        listing = ", ".join(repr(p) for p in existing)
        raise FileExistsError(
            "Refusing to overwrite existing external data shard file(s): "
            f"{listing}. The sharded write path never overwrites existing "
            "files. Delete the conflicting files or save into a different "
            "directory or under a different external data path."
        )


def _compute_external_data_info(
    tensor: _protocols.TensorProtocol,
    current_offset: int,
    alignment: int | None = None,
    align_threshold: int = _DEFAULT_ALIGN_THRESHOLD,
) -> _ExternalDataInfo:
    """Capture information about a tensor that is to be stored as external data."""
    tensor_size = tensor.nbytes
    current_offset = _align_offset(current_offset, tensor_size, alignment, align_threshold)
    # Store offset and tensor size as ExternalDataInfo
    external_data_info = _ExternalDataInfo(
        tensor.name,
        current_offset,
        tensor_size,
    )
    return external_data_info


class _ByteBudget:
    """Bound the number of materialized bytes held in memory at any one time.

    Limiting the number of *items* in flight is not enough because a single
    tensor can be many gigabytes. Limiting bytes gives a hard bound on peak
    memory of ``capacity + max(tensor.nbytes)``: a tensor larger than the whole
    budget is admitted alone (otherwise it could never be admitted at all).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(capacity, 1)
        self._in_flight = 0
        self._condition = threading.Condition()

    def _clamp(self, nbytes: int) -> int:
        return min(max(nbytes, 0), self._capacity)

    def acquire(self, nbytes: int) -> int:
        """Reserve ``nbytes`` of budget, blocking until it fits.

        Returns the clamped amount that must be passed back to :meth:`release`.
        """
        amount = self._clamp(nbytes)
        with self._condition:
            # ``self._in_flight == 0`` lets an oversized tensor through instead
            # of deadlocking forever waiting for a budget it can never fit in.
            self._condition.wait_for(
                lambda: self._in_flight + amount <= self._capacity or self._in_flight == 0
            )
            self._in_flight += amount
        return amount

    def release(self, amount: int) -> None:
        with self._condition:
            self._in_flight -= amount
            self._condition.notify_all()


def _write_tensor_at(
    tensor: _protocols.TensorProtocol,
    file,
    offset: int,
) -> None:
    """Write one tensor's bytes at an absolute ``offset`` in an open binary file."""
    file.seek(offset)
    if hasattr(tensor, "tofile"):
        # Some existing implementation of TensorProtocol
        # may not have tofile() as it was introduced in v0.1.11
        tensor.tofile(file)
    else:
        file.write(tensor.tobytes())
    if isinstance(tensor, _core.ExternalTensor):
        tensor.release()


def _write_external_data(
    tensors: Sequence[_protocols.TensorProtocol],
    external_data_infos: Sequence[_ExternalDataInfo],
    file_path: str | os.PathLike,
    callback: Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None = None,
    max_workers: int | None = None,
    max_in_flight_bytes: int = _DEFAULT_MAX_IN_FLIGHT_BYTES,
    budget: _ByteBudget | None = None,
) -> None:
    """Write tensor data to an external file according to information stored in ExternalDataInfo objects.

    Args:
        tensors: Tensors to be written as external data.
        external_data_infos: External data information stored for each tensor to be written as external data.
        file_path: Location to which external data is to be stored.
        callback: A callback function that is called for each tensor that is saved to external data
            for debugging or logging purposes.
        max_workers: Number of threads used to materialize and write tensors. ``None``
            or ``1`` writes serially.
        max_in_flight_bytes: Upper bound on materialized tensor bytes held in memory
            when writing concurrently. Ignored when ``budget`` is given.
        budget: An existing byte budget to share across concurrent shard writes so that
            peak memory does not scale with the number of shards.
    """
    tensors_count = len(tensors)
    assert tensors_count == len(external_data_infos), (
        "Number of tensors and external data infos should match"
    )

    filename = os.path.basename(file_path)

    def _invoke_callback(index: int, tensor: _protocols.TensorProtocol, offset: int) -> None:
        if callback is None:
            return
        callback(
            tensor,
            CallbackInfo(
                total=tensors_count,
                index=index,
                offset=offset,
                filename=filename,
            ),
        )

    if max_workers is not None and max_workers > 1 and tensors_count > 1:
        _write_external_data_parallel(
            tensors,
            external_data_infos,
            file_path,
            _invoke_callback,
            max_workers=max_workers,
            max_in_flight_bytes=max_in_flight_bytes,
            budget=budget,
        )
        return

    with open(file_path, "wb") as data_file:
        for i, (tensor, tensor_info) in enumerate(
            zip(tensors, external_data_infos, strict=True)
        ):
            assert tensor is not None
            _invoke_callback(i, tensor, tensor_info.offset)
            # Pad the file up to the target offset if needed
            file_size = data_file.tell()
            if tensor_info.offset > file_size:
                data_file.write(b"\0" * (tensor_info.offset - file_size))
            _write_tensor_at(tensor, data_file, tensor_info.offset)


def _write_external_data_parallel(
    tensors: Sequence[_protocols.TensorProtocol],
    external_data_infos: Sequence[_ExternalDataInfo],
    file_path: str | os.PathLike,
    invoke_callback: Callable[[int, _protocols.TensorProtocol, int], None],
    *,
    max_workers: int,
    max_in_flight_bytes: int,
    budget: _ByteBudget | None = None,
) -> None:
    """Write tensors concurrently to preassigned offsets in ``file_path``.

    Every tensor's byte range is known up front, so writes have no ordering
    dependency on one another. The file is preallocated to its final size and
    each worker opens its own file object, giving every thread an independent
    file position and removing the need for a write lock. Both ``write()`` and
    the numpy/torch work behind materialization release the GIL, so this
    overlaps computation with I/O *and* parallelizes each of them.

    Peak memory is bounded by a byte budget rather than a queue length: a worker
    reserves a tensor's size before materializing it and releases the
    reservation once the bytes have reached the file.

    The bytes written are identical to the serial path.
    """
    total_size = max(
        (info.offset + info.length for info in external_data_infos),
        default=0,
    )
    # Preallocate so that every worker can seek to its own offset. Holes read
    # back as zeros, matching the explicit zero padding written serially.
    with open(file_path, "wb") as data_file:
        data_file.truncate(total_size)

    budget = budget if budget is not None else _ByteBudget(max_in_flight_bytes)
    callback_lock = threading.Lock()
    thread_local = threading.local()
    files: list = []
    files_lock = threading.Lock()

    def _thread_file():
        data_file = getattr(thread_local, "data_file", None)
        if data_file is None:
            # Closed in the caller's finally block, after all workers are done.
            data_file = open(file_path, "r+b")  # ruff: ignore[open-file-with-context-handler]
            thread_local.data_file = data_file
            with files_lock:
                files.append(data_file)
        return data_file

    def _write_one(index: int) -> None:
        tensor = tensors[index]
        info = external_data_infos[index]
        assert tensor is not None
        with callback_lock:
            invoke_callback(index, tensor, info.offset)
        reserved = budget.acquire(info.length)
        try:
            _write_tensor_at(tensor, _thread_file(), info.offset)
        finally:
            budget.release(reserved)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_write_one, i) for i in range(len(tensors))]
            # Surface the first failure while letting the pool unwind cleanly.
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        for data_file in files:
            data_file.close()


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
    max_workers: int | None = None,
    max_in_flight_bytes: int = _DEFAULT_MAX_IN_FLIGHT_BYTES,
    alignment: int | None = None,
    align_threshold: int = _DEFAULT_ALIGN_THRESHOLD,
    _budget: _ByteBudget | None = None,
) -> list[_core.ExternalTensor]:
    """Convert a sequence of any TensorProtocol tensors to external tensors.

    Existing external tensors are loaded to memory if they are referring to the
    same file path as the destination path.

    Tensors are written in the order they are given, packed densely unless
    ``alignment`` is set. Preserving the input order matters: initializers are
    declared in topological order by every mainstream exporter, so the weights of
    one layer stay adjacent on disk. Runtimes memory-map each tensor separately
    and do not prefetch, so page faults are served in execution order and
    adjacency turns that into a sequential read.

    Args:
        tensors: Tensors to be converted to external tensors. They can be external tensors themselves.
        base_dir: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.
        callback: A callback function that is called for each tensor that is saved to external data
            for debugging or logging purposes. When ``max_workers`` enables concurrency the
            callback is serialized with a lock but is no longer invoked in index order.
        max_workers: Number of threads used to materialize and write tensors. ``None``
            or ``1`` writes serially.
        max_in_flight_bytes: Upper bound on materialized tensor bytes held in memory
            when writing concurrently.
        alignment: Alignment to apply to the offsets of large tensors, in bytes.
            ``None`` (the default) packs tensors densely with no padding. When set,
            offsets are aligned to ``max(4096, alignment)``; 65536 matches the
            Windows allocation granularity used for memory mapping.
        align_threshold: Only tensors strictly larger than this many bytes are
            aligned. Ignored when ``alignment`` is ``None``.

    Returns:
        A list of external tensors derived from a list of input tensors. The order
        matches the input tensor order.
    """
    path = os.path.join(base_dir, relative_path)
    tensors = _materialize_external_tensors_for_destination_paths(tensors, [path])

    # Compute external data information for each tensor and write to disk
    external_data_infos: list[_ExternalDataInfo] = []
    current_offset = 0
    for tensor in tensors:
        external_info = _compute_external_data_info(
            tensor, current_offset, alignment, align_threshold
        )
        external_data_infos.append(external_info)
        current_offset = external_info.offset + external_info.length
    _write_external_data(
        tensors,
        external_data_infos,
        path,
        callback=callback,
        max_workers=max_workers,
        max_in_flight_bytes=max_in_flight_bytes,
        budget=_budget,
    )

    # Create external tensor objects
    return [
        _create_external_tensor(tensor, external_info, base_dir, relative_path)
        for tensor, external_info in zip(tensors, external_data_infos, strict=True)
    ]


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
    max_workers: int | None = None,
    max_in_flight_bytes: int = _DEFAULT_MAX_IN_FLIGHT_BYTES,
    alignment: int | None = None,
    align_threshold: int = _DEFAULT_ALIGN_THRESHOLD,
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
            each shard's write order while remaining globally contiguous. When
            ``max_workers`` enables concurrency the callback is serialized with a lock
            but is no longer invoked in index order.
        max_workers: Number of threads used to materialize and write tensors. ``None``
            (the default) or ``1`` writes serially, preserving the previous behavior.
            Values above 1 overlap tensor materialization (dtype conversion, lazy tensor
            evaluation) with disk writes and parallelize both. Shards are also written
            concurrently.
        max_in_flight_bytes: Upper bound on materialized tensor bytes held in memory
            when writing concurrently. Peak memory is bounded by this value plus the
            size of the largest single tensor. Shared across shards so that peak memory
            does not grow with the shard count.
        alignment: Alignment to apply to the offsets of large tensors, in bytes.
            ``None`` (the default) packs tensors densely with no padding. When set,
            offsets are aligned to ``max(4096, alignment)``; 65536 matches the
            Windows allocation granularity used for memory mapping.
        align_threshold: Only tensors strictly larger than this many bytes are
            aligned. Ignored when ``alignment`` is ``None``.

    Returns:
        An ir.Model with all initializer data equal or above ``size_threshold_bytes``
        converted to external tensors.

    Raises:
        ValueError: If ``max_shard_size_bytes`` is not greater than 0.
        FileExistsError: When ``max_shard_size_bytes`` is set and any
            destination shard file already exists on disk. The sharded write
            path never overwrites existing files (re-saving a model whose
            shards already exist therefore requires deleting them first or
            choosing a different external data path). The single-file path
            (``max_shard_size_bytes is None``) instead overwrites
            ``relative_path`` unconditionally.

    Notes:
        Stale shards from a previous save (when the new layout produces
        fewer or differently named shard files) are the caller's
        responsibility to clean up. This function will neither delete nor
        rename pre-existing files that are not in the new destination set.
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
        # No sharding: write all tensors to the single destination file. The
        # single-file write path keeps its long-standing permissive semantics
        # of overwriting whatever happens to be at ``relative_path``; the
        # collision check below is sharded-only because shard layouts are
        # ambiguous (different shard counts produce different filenames) and
        # so silent overwrite is much more dangerous there.
        # TODO(justinchuby): the single-file and sharded paths should
        # eventually share the same overwrite policy.
        tensors_to_externalize: list[_protocols.TensorProtocol] = [
            v.const_value  # type: ignore[misc]
            for v in initializers_to_become_external
        ]
        external_tensors = convert_tensors_to_external(
            tensors_to_externalize,
            base_dir=base_dir,
            relative_path=relative_path,
            callback=callback,
            max_workers=max_workers,
            max_in_flight_bytes=max_in_flight_bytes,
            alignment=alignment,
            align_threshold=align_threshold,
        )
    else:
        # Sharding: distribute tensors across multiple numbered shard files
        tensors_to_externalize = [
            v.const_value  # type: ignore[misc]
            for v in initializers_to_become_external
        ]
        tensor_shards = _shard_tensors(
            tensors_to_externalize, max_shard_size_bytes, alignment, align_threshold
        )
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
        # Contract: the sharded write path never overwrites a pre-existing
        # file. If any destination shard already exists on disk we raise
        # FileExistsError so the caller cannot silently clobber another
        # model's shards or leave stale shards behind. Because no destination
        # is ever overwritten, no input ExternalTensor can be backed by a file
        # we are about to write — so there is no need to pre-materialize
        # tensors or stage temporary files before writing the shards.
        _check_no_existing_shard_files(destination_paths)

        external_tensors = []
        shard_jobs: list[tuple[Sequence[_protocols.TensorProtocol], str, Callable | None]] = []
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
            shard_jobs.append((shard_tensors, shard_relative_path, shard_callback))
            global_index += shard_tensor_count

        if max_workers is not None and max_workers > 1 and len(shard_jobs) > 1:
            # Shards are distinct files with no ordering dependency, so they are
            # written concurrently. They share one byte budget so that peak memory
            # does not grow with the number of shards.
            #
            # Split ``max_workers`` across the two levels instead of using it at
            # both: nesting a pool of that size inside each of that many shard
            # threads would spawn up to ``max_workers ** 2`` threads, breaking
            # the contract that ``max_workers`` bounds the thread count.
            # The shard driver threads count against the budget too: each one
            # occupies a thread while its inner pool runs. Reserve them first so
            # that drivers + inner workers stay within max_workers.
            shard_workers = min(max_workers, len(shard_jobs))
            workers_per_shard = max(1, (max_workers - shard_workers) // shard_workers)
            shared_budget = _ByteBudget(max_in_flight_bytes)
            shard_lock = threading.Lock()

            def _locked_callback(
                inner: Callable[[_protocols.TensorProtocol, CallbackInfo], None],
            ) -> Callable[[_protocols.TensorProtocol, CallbackInfo], None]:
                def _wrapped(tensor: _protocols.TensorProtocol, info: CallbackInfo) -> None:
                    with shard_lock:
                        inner(tensor, info)

                return _wrapped

            with concurrent.futures.ThreadPoolExecutor(max_workers=shard_workers) as executor:
                shard_futures = [
                    executor.submit(
                        convert_tensors_to_external,
                        job_tensors,
                        base_dir=base_dir,
                        relative_path=job_path,
                        callback=(
                            _locked_callback(job_callback)
                            if job_callback is not None
                            else None
                        ),
                        max_workers=workers_per_shard,
                        alignment=alignment,
                        align_threshold=align_threshold,
                        _budget=shared_budget,
                    )
                    for job_tensors, job_path, job_callback in shard_jobs
                ]
                for shard_future in shard_futures:
                    external_tensors.extend(shard_future.result())
        else:
            for job_tensors, job_path, job_callback in shard_jobs:
                external_tensors.extend(
                    convert_tensors_to_external(
                        job_tensors,
                        base_dir=base_dir,
                        relative_path=job_path,
                        callback=job_callback,
                        max_workers=max_workers,
                        max_in_flight_bytes=max_in_flight_bytes,
                        alignment=alignment,
                        align_threshold=align_threshold,
                    )
                )

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
