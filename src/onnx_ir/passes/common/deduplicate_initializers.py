# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass for removing duplicated initializer tensors from a graph."""

from __future__ import annotations

__all__ = [
    "DeduplicateInitializersPass",
]


import hashlib

import onnx_ir as ir


class DeduplicateInitializersPass(ir.passes.InPlacePass):
    """Remove duplicated initializer tensors from the graph.

    This pass detects initializers with identical shape, dtype, and content,
    and replaces all duplicate references with a canonical one.

    For efficiency, it uses a hash of tensor content to group candidates,
    then confirms exact match using the full byte content (to avoid collisions).
    Subgraphs are handled via RecursiveGraphIterator.
    """

    def __init__(self, max_elements_to_compare: int = 1024):
        super().__init__()
        self.max_elements_to_compare = max_elements_to_compare

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        initializer_groups: dict[tuple[str, tuple[int, ...]], dict[str, list[ir.Value]]] = {}

        duplicate_to_canonical = {}

        for initializer in list(graph.initializers.values()):
            const_val = initializer.const_value
            if const_val is None:
                continue  # Skip if initializer has no constant value

            # Compare shape and dtype first
            dtype = const_val.dtype.name
            shape = tuple(int(dim) if isinstance(dim, int) else -1 for dim in const_val.shape)
            num_elements = const_val.size
            if num_elements is None:
                continue  # Defensive: malformed tensor

            key = (dtype, shape)

            # Skip large tensors if over threshold
            if num_elements > self.max_elements_to_compare:
                continue

            # Use raw_data if available, else fallback to tobytes
            if hasattr(const_val, "raw_data") and const_val.raw_data:
                content = const_val.raw_data
            else:
                content = const_val.tobytes()

            content_hash = hashlib.sha256(content).hexdigest()

            if key not in initializer_groups:
                initializer_groups[key] = {}

            group = initializer_groups[key]
            if content_hash in group:
                for existing_val in group[content_hash]:
                    other_val = existing_val.const_value
                    if other_val is not None:
                        other_content = (
                            other_val.raw_data
                            if hasattr(other_val, "raw_data") and other_val.raw_data
                            else other_val.tobytes()
                        )
                        if other_content == content:
                            assert initializer.name is not None
                            duplicate_to_canonical[initializer.name] = existing_val.name
                            graph.initializers.pop(initializer.name)
                            break
                else:
                    group[content_hash].append(initializer)
            else:
                group[content_hash] = [initializer]

        for node in ir.traversal.RecursiveGraphIterator(graph):
            for i, input_val in enumerate(node.inputs):
                if input_val and input_val.name in duplicate_to_canonical:
                    canonical_name = duplicate_to_canonical[input_val.name]
                    assert canonical_name is not None
                    replacement = graph.initializers[canonical_name]
                    node.replace_input_with(i, replacement)

        return ir.passes.PassResult(model=model, modified=bool(duplicate_to_canonical))
