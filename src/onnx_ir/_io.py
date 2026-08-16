# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load and save ONNX models."""

from __future__ import annotations

__all__ = ["load", "save"]

import os
from typing import Callable

import onnx  # ruff: ignore[banned-api]

from onnx_ir import _core, _protocols, serde
from onnx_ir import external_data as _external_data
from onnx_ir._polyfill import zip


def load(path: str | os.PathLike, format: str | None = None) -> _core.Model:
    """Load an ONNX model from a file.

    Args:
        path: The path to the ONNX file.
        format: The format of the file (e.g. protobuf, textproto, json, etc.).
            If None, the format is inferred from the file extension.

    Returns:
        The loaded model.
    """
    # Do not use ONNX to load external data because the IR handles external data
    # by doing memory mapping directly.
    proto = onnx.load(path, format=format, load_external_data=False)
    model = serde.deserialize_model(proto)
    base_dir = os.path.dirname(path)
    # Set the base directory for external data to the directory of the ONNX file
    # so that relative paths are resolved correctly.
    _external_data.set_base_dir(model.graph, base_dir)
    return model


def save(
    model: _core.Model,
    path: str | os.PathLike,
    format: str | None = None,
    external_data: str | os.PathLike | None = None,
    size_threshold_bytes: int = 256,
    max_shard_size_bytes: int | None = None,
    callback: Callable[[_protocols.TensorProtocol, _external_data.CallbackInfo], None]
    | None = None,
    max_workers: int | None = None,
    max_in_flight_bytes: int = _external_data._DEFAULT_MAX_IN_FLIGHT_BYTES,
    alignment: int | None = None,
    align_threshold: int = 1048576,
) -> None:
    """Save an ONNX model to a file.

    The model remains unchanged after the call. If any existing external tensor
    references the provided ``external_data`` path, it will be invalidated
    after the external data is overwritten. To obtain a valid model, use :func:`load`
    to load the newly saved model, or provide a different external data path that
    is not currently referenced by any tensors in the model.

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
                    pbar.set_description(f"Saving {tensor.name} ({tensor.dtype}, {tensor.shape}) at offset {metadata.offset}")

                ir.save(
                    ...,
                    callback=callback,
                )

    Args:
        model: The model to save.
        path: The path to save the model to. E.g. "model.onnx".
        format: The format of the file (e.g. ``protobuf``, ``textproto``, ``json``, etc.).
            If None, the format is inferred from the file extension.
        external_data: The relative path to save external data to. When specified,
            all initializers in the model will be converted to external data and
            saved to the specified directory. If None, all tensors will be saved unmodified.
            That is, if a tensor in the model is already external, it will be saved
            with the same external information; if the tensor is not external,
            it will be serialized in the ONNX Proto message.
        size_threshold_bytes: Save to external data if the tensor size in bytes is larger than this threshold.
            Effective only when ``external_data`` is set.
        max_shard_size_bytes: Maximum cumulative size in bytes for a single external data shard file.
            When ``None`` (the default) all external tensors are written to the single file
            given by ``external_data``. When set, tensors are distributed across numbered shard
            files (e.g. ``model-00001-of-00003.data``). Because the ONNX format stores
            ``location``, ``offset``, and ``length`` per tensor, no separate index file is
            created — the saved ONNX proto itself encodes which shard each tensor lives in.
            If a single tensor is larger than this value, it is written in its own oversized
            shard file.
            Effective only when ``external_data`` is set.
        callback: A callback function that is called for each tensor that is saved to external data
            for debugging or logging purposes. When ``max_workers`` enables concurrency the
            callback is serialized with a lock but is no longer invoked in index order.
        max_workers: Number of threads used to write external data. ``None`` (the default)
            or ``1`` writes serially. Values above 1 overlap tensor materialization
            (lazy tensor evaluation, dtype conversion) with disk writes and parallelize
            both, which is significantly faster for large models. Peak memory stays
            bounded regardless of the worker count.
            Effective only when ``external_data`` is set.
        max_in_flight_bytes: Upper bound, in bytes, on the total size of tensors held in
            memory at once while writing. This caps peak memory use. A tensor larger
            than this budget is still admitted on its own, so the effective peak is
            roughly this value plus the size of the largest tensor.
            Effective only when ``external_data`` and ``max_workers`` are set.
        alignment: Alignment to apply to the offsets of large tensors, in bytes.
            ``None`` (the default) packs tensors densely with no padding, producing
            smaller files. When set, offsets are aligned to ``max(4096, alignment)``;
            65536 matches the Windows allocation granularity used for memory mapping.
            Effective only when ``external_data`` is set.
        align_threshold: Only tensors strictly larger than this many bytes are aligned.
            Ignored when ``alignment`` is ``None``.

    Raises:
        ValueError: If the external data path is an absolute path.
        ValueError: If ``max_shard_size_bytes`` is not greater than 0.
        ValueError: If ``max_shard_size_bytes`` is set without ``external_data``.
        FileExistsError: When ``max_shard_size_bytes`` is set and any
            destination shard file already exists on disk. The sharded write
            path never overwrites existing files; delete the conflicting
            files or choose a different external data path to re-save. The
            single-file path (``max_shard_size_bytes is None``) instead
            overwrites ``external_data`` unconditionally and never raises here.
    """
    if max_shard_size_bytes is not None and external_data is None:
        raise ValueError(
            "max_shard_size_bytes can only be used together with external_data; "
            "set external_data to the relative path where shards should be written."
        )
    if external_data is not None:
        if os.path.isabs(external_data):
            raise ValueError(
                f"The external data path must be relative to the ONNX file path, not '{external_data}'."
            )
        if max_shard_size_bytes is not None and max_shard_size_bytes <= 0:
            raise ValueError(
                f"max_shard_size_bytes must be greater than 0, got {max_shard_size_bytes}."
            )
        base_dir = os.path.dirname(path)

        # Store the original initializer values so they can be restored if modify_model=False
        initialized_values: list[_core.Value] = []
        for graph in model.graphs():
            # Collect from all subgraphs as well
            initialized_values.extend(graph.initializers.values())
        tensors = [v.const_value for v in initialized_values]

        try:
            model = _external_data.unload_from_model(
                model,
                base_dir,
                external_data,
                size_threshold_bytes=size_threshold_bytes,
                max_shard_size_bytes=max_shard_size_bytes,
                callback=callback,
                max_workers=max_workers,
                max_in_flight_bytes=max_in_flight_bytes,
                alignment=alignment,
                align_threshold=align_threshold,
            )
            proto = serde.serialize_model(model)
            onnx.save(proto, path, format=format)

        finally:
            # Restore the original initializer values so the model is unchanged
            for initializer, tensor in zip(initialized_values, tensors, strict=True):
                initializer.const_value = tensor

    else:
        proto = serde.serialize_model(model)
        onnx.save(proto, path, format=format)
