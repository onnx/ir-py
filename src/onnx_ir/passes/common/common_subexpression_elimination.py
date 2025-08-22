# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Eliminate common subexpression in ONNX graphs."""

from __future__ import annotations

__all__ = [
    "CommonSubexpressionEliminationPass",
]

import logging
from collections.abc import Sequence

import onnx_ir as ir

logger = logging.getLogger(__name__)


class CommonSubexpressionEliminationPass(ir.passes.InPlacePass):
    """Eliminate common subexpression in ONNX graphs.

    .. versionadded:: 0.1.1

    .. versionchanged:: 0.1.3
        Constant nodes with values smaller than ``size_limit`` will be CSE'd.

    Attributes:
        size_limit: The maximum size of the tensor to be csed. If the tensor contains
            number of elements larger than size_limit, it will not be cse'd. Default is 10.

    """

    def __init__(self, size_limit: int = 10):
        """Initialize the CommonSubexpressionEliminationPass."""
        super().__init__()
        self.size_limit = size_limit

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Return the same ir.Model but with CSE applied to the graph."""
        graph = model.graph
        modified = self._eliminate_common_subexpression(graph)

        return ir.passes.PassResult(
            model,
            modified=modified,
        )

    def _eliminate_common_subexpression(self, graph: ir.Graph) -> bool:
        """Eliminate common subexpression in ONNX graphs."""
        modified: bool = False
        # node to node identifier, length of outputs, inputs, and attributes
        existing_node_info_to_the_node: dict[
            tuple[
                ir.OperatorIdentifier,
                int,  # len(outputs)
                tuple[int, ...],  # input ids
                tuple[tuple[str, object], ...],  # attributes
            ],
            ir.Node,
        ] = {}

        for node in graph:
            # Skip control flow ops like Loop and If.
            control_flow_op: bool = False
            # Skip large tensors to avoid cse weights and bias.
            large_tensor: bool = False
            # Use equality to check if the node is a common subexpression.
            attributes = {}
            for k, v in node.attributes.items():
                # TODO(exporter team): CSE subgraphs.
                # NOTE: control flow ops like Loop and If won't be CSEd
                # because attribute: graph won't match.
                if v.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS):
                    control_flow_op = True
                    break
                # The attribute value could be directly taken from the original
                # protobuf, so we need to make a copy of it.
                value = v.value
                if v.type in (
                    ir.AttributeType.INTS,
                    ir.AttributeType.FLOATS,
                    ir.AttributeType.STRINGS,
                ):
                    # For INT, FLOAT and STRING attributes, we convert them to tuples
                    # to ensure they are hashable.
                    value = tuple(value)
                elif v.type is ir.AttributeType.TENSOR:
                    if value.size > self.size_limit:
                        # If the tensor is larger than the size limit, we skip it.
                        large_tensor = True
                        break
                    np_value = value.numpy()

                    value = (np_value.shape, str(np_value.dtype), np_value.tobytes())
                attributes[k] = value

            if control_flow_op:
                # If the node is a control flow op, we skip it.
                logger.debug("Skipping control flow op %s", node)
                continue

            if large_tensor:
                # If the node has a large tensor, we skip it.
                logger.debug("Skipping large tensor in node %s", node)
                continue

            if _is_non_deterministic_op(node):
                # If the node is a non-deterministic op, we skip it.
                logger.debug("Skipping non-deterministic op %s", node)
                continue

            node_info = (
                node.op_identifier(),
                len(node.outputs),
                tuple(id(input) for input in node.inputs),
                tuple(sorted(attributes.items())),
            )
            # Check if the node is a common subexpression.
            if node_info in existing_node_info_to_the_node:
                # If it is, this node has an existing node with the same
                # operator, number of outputs, inputs, and attributes.
                # We replace the node with the existing node.
                modified = True
                existing_node = existing_node_info_to_the_node[node_info]
                _remove_node_and_replace_values(
                    graph,
                    remove_node=node,
                    remove_values=node.outputs,
                    new_values=existing_node.outputs,
                )
                logger.debug("Reusing node %s", existing_node)
            else:
                # If it is not, add to the mapping.
                existing_node_info_to_the_node[node_info] = node
        return modified


def _remove_node_and_replace_values(
    graph: ir.Graph,
    /,
    remove_node: ir.Node,
    remove_values: Sequence[ir.Value],
    new_values: Sequence[ir.Value],
) -> None:
    """Replaces nodes and values in the graph or function.

    Args:
        graph: The graph to replace nodes and values in.
        remove_node: The node to remove.
        remove_values: The values to replace.
        new_values: The values to replace with.
    """
    # Reconnect the users of the deleted values to use the new values
    ir.convenience.replace_all_uses_with(remove_values, new_values)
    # Update graph/function outputs if the node generates output
    if any(remove_value.is_graph_output() for remove_value in remove_values):
        replacement_mapping = dict(zip(remove_values, new_values))
        for idx, graph_output in enumerate(graph.outputs):
            if graph_output in replacement_mapping:
                new_value = replacement_mapping[graph_output]
                if new_value.is_graph_output() or new_value.is_graph_input():
                    # If the new value is also a graph input/output, we need to
                    # create a Identity node to preserve the remove_value and
                    # prevent from changing new_value name.
                    identity_node = ir.node(
                        "Identity",
                        inputs=[new_value],
                        outputs=[
                            ir.Value(
                                name=graph_output.name,
                                type=graph_output.type,
                                shape=graph_output.shape,
                            )
                        ],
                    )
                    # reuse the name of the graph output
                    graph.outputs[idx] = identity_node.outputs[0]
                    graph.insert_before(
                        remove_node,
                        identity_node,
                    )
                else:
                    # if new_value is not graph output, we just
                    # update it to use old_value name.
                    new_value.name = graph_output.name
                    graph.outputs[idx] = new_value

    graph.remove(remove_node, safe=True)


def _is_non_deterministic_op(node: ir.Node) -> bool:
    non_deterministic_ops = frozenset(
        {
            "RandomUniform",
            "RandomNormal",
            "RandomUniformLike",
            "RandomNormalLike",
            "Multinomial",
        }
    )
    return node.op_type in non_deterministic_ops and _is_onnx_domain(node.domain)


def _is_onnx_domain(d: str) -> bool:
    """Check if the domain is the ONNX domain."""
    return d == ""
