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
import contextlib
import dataclasses
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator, Sequence

from onnx_ir import _core, _enums, _protocols
from onnx_ir import traversal as _traversal
from onnx_ir._polyfill import zip

# Default alignment threshold used when alignment is enabled: only tensors larger
# than this get their offset aligned, so small initializers don't waste file space.
_DEFAULT_ALIGN_THRESHOLD = 1048576  # 1MB
# Default upper bound on materialized tensor bytes held in memory while writing
# external data concurrently. Peak memory is this plus the largest single tensor.
_DEFAULT_MAX_IN_FLIGHT_BYTES = 1 << 30  # 1GB


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
        shard_total: The number of tensors in this external data file. ``None``
            when the writer does not provide per-file progress information.
        shard_index: The index of this tensor within its external data file.
            ``None`` when the writer does not provide per-file progress
            information.
    """

    total: int
    index: int
    offset: int
    filename: str
    shard_total: int | None = None
    shard_index: int | None = None


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
    # Preserve the full suffix chain so the shard marker stays attached to the
    # filename stem: ``model.onnx.data`` becomes
    # ``model-00001-of-00003.onnx.data``. Repeated splitext calls implement the
    # same general rule for any compound suffix without importing pathlib.
    name = filename
    suffixes = []
    while True:
        name, suffix = os.path.splitext(name)
        if not suffix:
            break
        suffixes.append(suffix)
    ext = "".join(reversed(suffixes))

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
                shard_total=info.total,
                shard_index=info.index,
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
    memory of ``capacity + max(tensor.nbytes)``. At most one tensor larger than
    the whole budget is admitted at a time, alongside regular reservations up
    to ``capacity``. This keeps the pipeline moving while the oversized tensor
    is written without allowing multiple oversized tensors to accumulate.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(capacity, 1)
        self._in_flight = 0
        self._oversized_active = False
        self._condition = threading.Condition()

    def acquire(self, nbytes: int) -> int:
        """Reserve ``nbytes`` of budget, blocking until it fits.

        Returns a reservation token that must be passed back to
        :meth:`release`. ``-1`` represents the single oversized reservation.
        """
        amount = max(nbytes, 0)
        with self._condition:
            if amount > self._capacity:
                self._condition.wait_for(lambda: not self._oversized_active)
                self._oversized_active = True
                return -1
            self._condition.wait_for(lambda: self._in_flight + amount <= self._capacity)
            self._in_flight += amount
        return amount

    def release(self, reservation: int) -> None:
        with self._condition:
            if reservation == -1:
                self._oversized_active = False
            else:
                self._in_flight -= reservation
            self._condition.notify_all()


def _reservation_bytes(tensor: _protocols.TensorProtocol, tensor_length: int) -> int:
    """Return the maximum userspace memory needed while writing a tensor."""
    if isinstance(tensor, _core.ExternalTensor):
        return min(tensor_length, _core._EXTERNAL_TENSOR_COPY_CHUNK_SIZE)
    return tensor_length


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


def _write_tensor_with_budget_at(
    tensor: _protocols.TensorProtocol,
    file,
    offset: int,
    length: int,
    budget: _ByteBudget | None,
) -> None:
    """Write a tensor while accounting for its userspace memory."""
    if budget is None:
        _write_tensor_at(tensor, file, offset)
        return
    reservation = budget.acquire(_reservation_bytes(tensor, length))
    try:
        _write_tensor_at(tensor, file, offset)
    finally:
        # Release the budget if writing fails; the exception still propagates.
        budget.release(reservation)


def _create_tensor_write_locks(
    tensors: Sequence[_protocols.TensorProtocol],
) -> dict[int, threading.Lock]:
    """Create locks that serialize writes of the same tensor object."""
    return {id(tensor): threading.Lock() for tensor in tensors}


def _write_external_data(
    tensors: Sequence[_protocols.TensorProtocol],
    external_data_infos: Sequence[_ExternalDataInfo],
    file_path: str | os.PathLike,
    callback: Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None = None,
    max_workers: int | None = None,
    max_in_flight_bytes: int = _DEFAULT_MAX_IN_FLIGHT_BYTES,
    budget: _ByteBudget | None = None,
    tensor_write_locks: dict[int, threading.Lock] | None = None,
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
        tensor_write_locks: Locks shared by shard writers to prevent concurrent
            evaluation of the same tensor object.
    """
    requested_path = os.fspath(file_path)
    destination_path = (
        os.path.realpath(requested_path) if os.path.islink(requested_path) else requested_path
    )
    destination_dir = os.path.dirname(destination_path) or "."
    temporary_dir = tempfile.mkdtemp(
        dir=destination_dir,
        prefix=f".{os.path.basename(destination_path)}.",
    )
    temporary_path = os.path.join(temporary_dir, os.path.basename(destination_path))

    overwritten_tensors = [
        tensor
        for tensor in tensors
        if isinstance(tensor, _core.ExternalTensor)
        and _paths_refer_to_same_file(tensor.path, destination_path)
    ]
    try:
        writer = _ExternalDataWriter(
            tensors,
            external_data_infos,
            temporary_path,
            callback,
            callback_filename=os.path.basename(requested_path),
            budget=budget,
            max_workers=max_workers,
            max_in_flight_bytes=max_in_flight_bytes,
            tensor_write_locks=(
                tensor_write_locks
                if tensor_write_locks is not None
                else _create_tensor_write_locks(tensors)
            ),
        )
        writer.write()
        # Windows cannot atomically replace a file while one of its mmap handles
        # is open. Other ExternalTensors are left untouched.
        for tensor in overwritten_tensors:
            tensor.release()
        if os.path.exists(destination_path):
            shutil.copymode(destination_path, temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temporary_path)
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(temporary_dir)

    for tensor in overwritten_tensors:
        tensor.invalidate()
        logger.warning(
            "External tensor %s referred to the overwritten destination and has "
            "been invalidated. Load the newly saved model to obtain a valid tensor.",
            tensor,
        )


class _ExternalDataWriter:
    """Coordinate serial or parallel writes to one external data file."""

    def __init__(
        self,
        tensors: Sequence[_protocols.TensorProtocol],
        external_data_infos: Sequence[_ExternalDataInfo],
        file_path: str | os.PathLike,
        callback: Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None,
        *,
        callback_filename: str,
        budget: _ByteBudget | None,
        max_workers: int | None,
        max_in_flight_bytes: int,
        tensor_write_locks: dict[int, threading.Lock],
    ) -> None:
        assert len(tensors) == len(external_data_infos), (
            "Number of tensors and external data infos should match"
        )
        self._tensors = tensors
        self._external_data_infos = external_data_infos
        self._file_path = file_path
        self._filename = callback_filename
        self._callback = callback
        self._budget = budget
        self._max_workers = max_workers
        self._max_in_flight_bytes = max_in_flight_bytes
        self._tensor_write_locks = tensor_write_locks

    def write(self) -> None:
        """Select the cheapest writer that provides the requested concurrency."""
        if self._max_workers is not None and self._max_workers > 1 and len(self._tensors) > 1:
            self._write_parallel(self._max_workers)
        else:
            self._write_serial()

    def _invoke_callback(
        self, index: int, tensor: _protocols.TensorProtocol, offset: int
    ) -> None:
        if self._callback is None:
            return
        self._callback(
            tensor,
            CallbackInfo(
                total=len(self._tensors),
                index=index,
                offset=offset,
                filename=self._filename,
                shard_total=len(self._tensors),
                shard_index=index,
            ),
        )

    def _write_tensor(self, tensor, file, info: _ExternalDataInfo, budget) -> None:
        """Write one tensor, serializing repeated references to the same object."""
        with self._tensor_write_locks[id(tensor)]:
            _write_tensor_with_budget_at(
                tensor,
                file,
                info.offset,
                info.length,
                budget,
            )

    def _write_serial(self) -> None:
        """Write sequentially through one file descriptor."""
        with open(self._file_path, "wb") as data_file:
            # Seeking past EOF creates a file hole that reads back as zeros, so
            # aligned offsets need no explicit padding allocation or write.
            for i, (tensor, tensor_info) in enumerate(
                zip(self._tensors, self._external_data_infos, strict=True)
            ):
                assert tensor is not None
                self._invoke_callback(i, tensor, tensor_info.offset)
                # A shard may use a serial writer while other shards are written
                # concurrently. Honor their shared budget here too; otherwise one
                # tensor per shard can be materialized at once with no byte bound.
                self._write_tensor(tensor, data_file, tensor_info, self._budget)

    def _write_parallel(self, max_workers: int) -> None:
        """Write concurrently through one file descriptor per worker.

        Every tensor's byte range is known up front, so writes have no ordering
        dependency. The file is preallocated to its final size and each worker
        opens its own descriptor. A shared byte budget bounds materialized tensor
        memory while overlapping computation and I/O. Both file writes and tensor
        materialization release the GIL, so threads can parallelize this work.

        The bytes written are identical to the serial path.
        """
        total_size = max(
            (info.offset + info.length for info in self._external_data_infos),
            default=0,
        )
        # Preallocate so that every worker can seek to its own offset. Gaps between
        # tensors read back as zeros, matching the sparse holes created serially.
        with open(self._file_path, "wb") as data_file:
            data_file.truncate(total_size)

        budget = (
            self._budget
            if self._budget is not None
            else _ByteBudget(self._max_in_flight_bytes)
        )
        callback_lock = threading.Lock()
        thread_local = threading.local()
        files: list = []
        files_lock = threading.Lock()

        def _thread_file():
            data_file = getattr(thread_local, "data_file", None)
            if data_file is None:
                # Closed below after all workers are done.
                data_file = open(  # ruff: ignore[open-file-with-context-handler]
                    self._file_path, "r+b"
                )
                thread_local.data_file = data_file
                with files_lock:
                    files.append(data_file)
            return data_file

        def _write_one(index: int) -> None:
            tensor = self._tensors[index]
            info = self._external_data_infos[index]
            assert tensor is not None
            with callback_lock:
                self._invoke_callback(index, tensor, info.offset)
            self._write_tensor(tensor, _thread_file(), info, budget)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_write_one, i) for i in range(len(self._tensors))]
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
    _tensor_write_locks: dict[int, threading.Lock] | None = None,
) -> list[_core.ExternalTensor]:
    """Convert a sequence of any TensorProtocol tensors to external tensors.

    Data is written to a temporary file in the destination directory and atomically
    replaces the destination only after the write succeeds. Existing external
    tensors backed by the destination stream directly into the temporary file and
    are invalidated after replacement.

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
        tensor_write_locks=_tensor_write_locks,
    )

    # Create external tensor objects
    return [
        _create_external_tensor(tensor, external_info, base_dir, relative_path)
        for tensor, external_info in zip(tensors, external_data_infos, strict=True)
    ]


def _write_external_tensors(
    tensors: Sequence[_protocols.TensorProtocol],
    base_dir: str | os.PathLike,
    relative_path: str | os.PathLike,
    *,
    max_shard_size_bytes: int | None,
    callback: Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None,
    max_workers: int | None,
    max_in_flight_bytes: int,
    alignment: int | None,
    align_threshold: int,
) -> list[_core.ExternalTensor]:
    """Write tensors to one file or coordinate writes across shard files."""
    # Write strategy:
    #
    # tensors
    # |-- one destination file
    # |   `-- _ExternalDataWriter -> serial or parallel tensor writes
    # `-- multiple shard files
    #     `-- shard thread pool
    #         `-- one _ExternalDataWriter per shard -> serial or parallel
    tensor_write_locks = _create_tensor_write_locks(tensors)
    if max_shard_size_bytes is None:
        # Single-file writes atomically replace an existing destination.
        # Sharded writes reject collisions because layouts with different
        # shard counts have different, potentially ambiguous names.
        return convert_tensors_to_external(
            tensors,
            base_dir=base_dir,
            relative_path=relative_path,
            callback=callback,
            max_workers=max_workers,
            max_in_flight_bytes=max_in_flight_bytes,
            alignment=alignment,
            align_threshold=align_threshold,
            _tensor_write_locks=tensor_write_locks,
        )

    tensor_shards = _shard_tensors(tensors, max_shard_size_bytes, alignment, align_threshold)
    total_shards = len(tensor_shards)
    shard_relative_paths = [
        _get_shard_filename(str(relative_path), shard_idx, total_shards)
        for shard_idx in range(1, total_shards + 1)
    ]
    destination_paths = [
        os.path.join(base_dir, shard_relative_path)
        for shard_relative_path in shard_relative_paths
    ]
    # No shard may overwrite an existing file. This also guarantees that no
    # input ExternalTensor is backed by a destination about to be overwritten,
    # so shard inputs do not need to be staged in memory.
    _check_no_existing_shard_files(destination_paths)

    shard_jobs: list[
        tuple[
            Sequence[_protocols.TensorProtocol],
            str,
            Callable[[_protocols.TensorProtocol, CallbackInfo], None] | None,
        ]
    ] = []
    global_index = 0
    for shard_tensors, shard_relative_path in zip(
        tensor_shards, shard_relative_paths, strict=True
    ):
        shard_callback = (
            _make_shard_callback(callback, len(tensors), global_index)
            if callback is not None
            else None
        )
        shard_jobs.append((shard_tensors, shard_relative_path, shard_callback))
        global_index += len(shard_tensors)

    external_tensors: list[_core.ExternalTensor] = []
    if max_workers is not None and max_workers > 1 and len(shard_jobs) > 1:
        # Shard driver threads count toward max_workers. Split the remaining
        # workers among inner writers instead of nesting max_workers-sized pools.
        shard_workers = min(max_workers, len(shard_jobs))
        workers_per_shard = max(1, (max_workers - shard_workers) // shard_workers)
        shared_budget = _ByteBudget(max_in_flight_bytes)
        callback_lock = threading.Lock()

        def _locked_callback(
            inner: Callable[[_protocols.TensorProtocol, CallbackInfo], None],
        ) -> Callable[[_protocols.TensorProtocol, CallbackInfo], None]:
            def _wrapped(tensor: _protocols.TensorProtocol, info: CallbackInfo) -> None:
                with callback_lock:
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
                        _locked_callback(job_callback) if job_callback is not None else None
                    ),
                    max_workers=workers_per_shard,
                    alignment=alignment,
                    align_threshold=align_threshold,
                    _budget=shared_budget,
                    _tensor_write_locks=tensor_write_locks,
                )
                for job_tensors, job_path, job_callback in shard_jobs
            ]
            for shard_future in shard_futures:
                external_tensors.extend(shard_future.result())
        return external_tensors

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
                _tensor_write_locks=tensor_write_locks,
            )
        )
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
            (``max_shard_size_bytes is None``) atomically replaces
            ``relative_path`` only after the new file is complete.

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

    tensors_to_externalize: list[_protocols.TensorProtocol] = [
        v.const_value  # type: ignore[misc]
        for v in initializers_to_become_external
    ]
    external_tensors = _write_external_tensors(
        tensors_to_externalize,
        base_dir,
        relative_path,
        max_shard_size_bytes=max_shard_size_bytes,
        callback=callback,
        max_workers=max_workers,
        max_in_flight_bytes=max_in_flight_bytes,
        alignment=alignment,
        align_threshold=align_threshold,
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
