# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load and save ONNX models."""

from __future__ import annotations

__all__ = ["load", "save"]

import os
from typing import Callable

import onnx  # noqa: TID251

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
    callback: Callable[[_protocols.TensorProtocol, _external_data.CallbackInfo], None]
    | None = None,
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
        callback: A callback function that is called for each tensor that is saved to external data
            for debugging or logging purposes.

    Raises:
        ValueError: If the external data path is an absolute path.
    """
    if external_data is not None:
        if os.path.isabs(external_data):
            raise ValueError(
                f"The external data path must be relative to the ONNX file path, not '{external_data}'."
            )
        base_dir = os.path.dirname(path)

        # Store the original initializer values so they can be restored if modify_model=False
        initializer_values: list[_core.Value] = []
        for graph in model.graphs():
            # Collect from all subgraphs as well
            initializer_values.extend(graph.initializers.values())
        tensors = [v.const_value for v in initializer_values]

        try:
            model = _external_data.unload_from_model(
                model,
                base_dir,
                external_data,
                size_threshold_bytes=size_threshold_bytes,
                callback=callback,
            )
            proto = serde.serialize_model(model)
            onnx.save(proto, path, format=format)

        finally:
            # Restore the original initializer values so the model is unchanged
            for initializer, tensor in zip(initializer_values, tensors, strict=True):
                initializer.const_value = tensor

    else:
        proto = serde.serialize_model(model)
        onnx.save(proto, path, format=format)
