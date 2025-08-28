# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass for removing duplicated initializer tensors from a graph."""

from __future__ import annotations

__all__ = ["DeduplicateInitializersPass", "DeduplicateHashedInitializersPass"]


import hashlib
import logging

import numpy as np

import onnx_ir as ir

logger = logging.getLogger(__name__)


def _should_skip_initializer(initializer: ir.Value, size_limit: int) -> bool:
    """Check if the initializer should be skipped for deduplication."""
    if initializer.is_graph_input() or initializer.is_graph_output():
        # Skip graph inputs and outputs
        logger.warning(
            "Skipped deduplication of initializer '%s' as it is a graph input or output",
            initializer.name,
        )
        return True

    const_val = initializer.const_value
    if const_val is None:
        # Skip if initializer has no constant value
        logger.warning(
            "Skipped deduplication of initializer '%s' as it has no constant value. The model may contain invalid initializers",
            initializer.name,
        )
        return True

    if const_val.size > size_limit:
        # Skip if the initializer is larger than the size limit
        logger.debug(
            "Skipped initializer '%s' as it exceeds the size limit of %d elements",
            initializer.name,
            size_limit,
        )
        return True
    return False


def _tobytes(val):
    """StringTensor does not support tobytes. Use 'string_data' instead.

    However, 'string_data' yields a list of bytes which cannot be hashed, i.e.,
    cannot be used to index into a dict. To generate keys for identifying
    tensors in initializer deduplication the following converts the list of
    bytes to an array of fixed-length strings which can be flattened into a
    bytes-string. This, together with the tensor shape, is sufficient for
    identifying tensors for deduplication, but it differs from the
    representation used for serializing tensors (that is string_data) by adding
    padding bytes so that each string occupies the same number of consecutive
    bytes in the flattened .tobytes representation.
    """
    if val.dtype.is_string():
        return np.array(val.string_data()).tobytes()
    return val.tobytes()


class DeduplicateInitializersPass(ir.passes.InPlacePass):
    """Remove duplicated initializer tensors from the main graph and all subgraphs.

    This pass detects initializers with identical shape, dtype, and content,
    and replaces all duplicate references with a canonical one.

    Initializers are deduplicated within each graph. To deduplicate initializers
    in the model globally (across graphs), use :class:`~onnx_ir.passes.common.LiftSubgraphInitializersToMainGraphPass`
    to lift the initializers to the main graph first before running pass.

    .. versionadded:: 0.1.3
    .. versionchanged:: 0.1.7
        This pass now deduplicates initializers in subgraphs as well.
    """

    def __init__(self, size_limit: int = 1024):
        super().__init__()
        self.size_limit = size_limit

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for graph in model.graphs():
            initializers: dict[tuple[ir.DataType, tuple[int, ...], bytes], ir.Value] = {}
            for initializer in tuple(graph.initializers.values()):
                if _should_skip_initializer(initializer, self.size_limit):
                    continue

                const_val = initializer.const_value
                assert const_val is not None

                key = (const_val.dtype, tuple(const_val.shape), _tobytes(const_val))
                if key in initializers:
                    modified = True
                    initializer_to_keep = initializers[key]  # type: ignore[index]
                    ir.convenience.replace_all_uses_with(initializer, initializer_to_keep)
                    assert initializer.name is not None
                    graph.initializers.pop(initializer.name)
                    logger.info(
                        "Replaced initializer '%s' with existing initializer '%s'",
                        initializer.name,
                        initializer_to_keep.name,
                    )
                else:
                    initializers[key] = initializer  # type: ignore[index]

        return ir.passes.PassResult(model=model, modified=modified)


class DeduplicateHashedInitializersPass(ir.passes.InPlacePass):
    """Remove duplicated initializer tensors (using a hashed method) from the graph.

    This pass detects initializers with identical shape, dtype, and hashed content,
    and replaces all duplicate references with a canonical one.

    This pass should have a lower peak memory usage than :class:`DeduplicateInitializersPass`
    as it does not store the full tensor data in memory, but instead uses a hash of the tensor data.

    .. versionadded:: 0.1.7
    """

    def __init__(self, size_limit: int = 4 * 1024 * 1024 * 1024):
        super().__init__()
        # 4 GB default size limit for deduplication
        self.size_limit = size_limit

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for graph in model.graphs():
            initializers: dict[tuple[ir.DataType, tuple[int, ...], str], ir.Value] = {}

            for initializer in tuple(graph.initializers.values()):
                if _should_skip_initializer(initializer, self.size_limit):
                    continue

                const_val = initializer.const_value
                assert const_val is not None

                # Hash tensor data to avoid storing large amounts of data in memory
                hashed = hashlib.sha512()
                tensor_data = const_val.numpy()
                hashed.update(tensor_data)
                tensor_digest = hashed.hexdigest()

                tensor_dims = tuple(const_val.shape.numpy())

                key = (const_val.dtype, tensor_dims, tensor_digest)

                if key in initializers:
                    if _tobytes(initializers[key].const_value) != _tobytes(const_val):
                        logger.warning(
                            "Initializer deduplication failed: "
                            "hashes match but values differ with values %s and %s",
                            initializers[key],
                            initializer,
                        )
                        continue
                    modified = True
                    initializer_to_keep = initializers[key]  # type: ignore[index]
                    ir.convenience.replace_all_uses_with(initializer, initializer_to_keep)
                    assert initializer.name is not None
                    graph.initializers.pop(initializer.name)
                    logger.info(
                        "Replaced initializer '%s' with existing initializer '%s'",
                        initializer.name,
                        initializer_to_keep.name,
                    )
                else:
                    initializers[key] = initializer  # type: ignore[index]

        return ir.passes.PassResult(model=model, modified=modified)
