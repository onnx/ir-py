"""Symbolic inference engine for ONNX IR models."""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable, Sequence

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common

logger = logging.getLogger(__name__)


class ReconciliationPolicy(enum.Enum):
    """Policy for reconciling inferred shapes/types with existing values."""

    OVERWRITE = "overwrite"  # Always use inferred values
    IGNORE = "ignore"  # Keep existing values if they exist
    RECONCILE = "reconcile"  # Try to merge/validate inferred vs existing
    STRICT = "strict"  # Fail if inferred doesn't match existing


class InferenceError(RuntimeError):
    """Error during shape inference."""


class SymbolicInferenceEngine:
    """Engine for performing symbolic shape and type inference on ONNX IR models."""

    def __init__(
        self,
        node_inferrers: Iterable[_common.NodeInferrer],
        reconciliation_policy: str = "reconcile",
    ) -> None:
        """Initialize the symbolic inference engine.

        Args:
            node_inferrers: List of node inferrers to use for shape inference.
            reconciliation_policy: Policy for handling conflicts between inferred and existing values.
        """
        self.reconciliation_policy = ReconciliationPolicy(reconciliation_policy)
        self._inferrer_registry: dict[ir.OperatorIdentifier, list[_common.NodeInferrer]] = {}

        # Register inferrers by (op_type, domain)
        for inferrer in node_inferrers:
            key = (inferrer.domain, inferrer.op_type, inferrer.overload)
            self._inferrer_registry.setdefault(key, []).append(inferrer)

    def infer_model(self, model: ir.Model) -> None:
        """Perform shape and type inference on an entire model.

        Args:
            model: The ONNX IR model to perform inference on.

        Raises:
            InferenceError: If inference fails for any node.
        """
        logger.info("Starting inference on model with %s nodes", len(model.graph))

        # Process nodes in topological order
        for i, node in enumerate(model.graph):
            try:
                self._infer_node(node, model)
                logger.debug("Successfully inferred node %s: %s", i, node.op_type)
            except Exception as e:
                error_msg = f"Failed to infer node {i} ({node.op_type}): {e}"
                logger.exception(error_msg)
                raise InferenceError(error_msg) from e

        logger.info("Model inference completed successfully")

    def _infer_node(self, node: ir.Node, model: ir.Model) -> None:
        """Perform inference on a single node.

        Args:
            node: The node to perform inference on.
            model: The model containing the node (for context).

        Raises:
            InferenceError: If no suitable inferrer is found or inference fails.
        """
        # Find suitable inferrer
        inferrer = self._find_inferrer(node, model)
        if inferrer is None:
            raise InferenceError(
                f"No inferrer found for op_type '{node.op_type}' domain '{node.domain}'"
            )

        # Perform inference
        result = inferrer.infer(node)

        if result.status == _common.InferenceStatus.INVALID_NODE:
            # TODO: Print the node information
            raise InferenceError(f"Invalid node: {result.msg}")

        if result.status == _common.InferenceStatus.MISSING_INFO:
            logger.warning("Missing info for node %s: %s", node.op_type, result.msg)
            # Continue with partial inference or skip
            if result.values is None:
                return  # Skip this node

        if result.status == _common.InferenceStatus.PARTIAL:
            logger.info("Partial inference for node %s: %s", node.op_type, result.msg)
            # Continue with partial results

        if result.values is None:
            raise InferenceError("Inference returned no values")

        # Apply reconciliation policy
        self._reconcile_outputs(node, result.values)

    def _find_inferrer(self, node: ir.Node, model: ir.Model) -> _common.NodeInferrer | None:
        """Find a suitable inferrer for the given node.

        Args:
            node: The node to find an inferrer for.
            model: The model containing the node.

        Returns:
            The best matching inferrer, or None if no suitable inferrer is found.
        """
        key = (node.domain, node.op_type, node.overload)
        inferrers = self._inferrer_registry.get(key, [])

        if not inferrers:
            return None

        # Get model opset version for this domain
        if node.version is not None:
            opset_version = node.version
        elif node.graph is not None and node.domain in node.graph.opset_imports:
            opset_version = node.graph.opset_imports[node.domain]
        else:
            # Fallback to model-level opset import
            if node.domain not in model.opset_imports:
                raise InferenceError(
                    f"No opset import found for domain '{node.domain}' in model"
                )
            opset_version = model.opset_imports[node.domain]

        # Find inferrers that support this opset version
        suitable_inferrers = [
            inferrer for inferrer in inferrers if opset_version in inferrer.opsets
        ]

        if not suitable_inferrers:
            logger.warning(
                "No inferrer supports opset %s for %s (domain: %s)",
                opset_version,
                node.op_type,
                node.domain,
            )
            return None

        # Return the first suitable inferrer (could be enhanced with priority logic)
        return suitable_inferrers[0]

    def _reconcile_outputs(self, node: ir.Node, inferred_values: Sequence[ir.Value]) -> None:
        """Reconcile inferred output values with existing node outputs.

        Args:
            node: The node whose outputs to reconcile.
            inferred_values: The inferred output values.

        Raises:
            InferenceError: If reconciliation fails under strict policy.
        """
        if len(inferred_values) != len(node.outputs):
            raise InferenceError(
                f"Inference returned {len(inferred_values)} values but node has "
                f"{len(node.outputs)} outputs"
            )

        for i, (existing_output, inferred_value) in enumerate(
            zip(node.outputs, inferred_values)
        ):
            if existing_output is None:
                # No existing output - create new one
                node.outputs[i] = inferred_value
                continue

            # Reconcile based on policy
            if self.reconciliation_policy == ReconciliationPolicy.OVERWRITE:
                node.outputs[i] = inferred_value

            elif self.reconciliation_policy == ReconciliationPolicy.IGNORE:
                # Keep existing output if it has shape/type info
                if existing_output.shape is None and existing_output.type is None:
                    node.outputs[i] = inferred_value
                # Otherwise keep existing

            elif self.reconciliation_policy == ReconciliationPolicy.RECONCILE:
                reconciled_output = self._reconcile_value(existing_output, inferred_value)
                node.outputs[i] = reconciled_output

            elif self.reconciliation_policy == ReconciliationPolicy.STRICT:
                if not self._values_compatible(existing_output, inferred_value):
                    raise InferenceError(
                        f"Output {i} mismatch: existing {existing_output} vs "
                        f"inferred {inferred_value}"
                    )
                # Keep existing in strict mode if compatible

    def _reconcile_value(self, existing: ir.Value, inferred: ir.Value) -> ir.Value:
        """Reconcile an existing value with an inferred value.

        Args:
            existing: The existing value.
            inferred: The inferred value.

        Returns:
            The reconciled value.
        """
        # Start with existing value
        result_shape = existing.shape
        result_type = existing.type

        # Use inferred shape if existing is None or less specific
        if inferred.shape is not None:
            if result_shape is None:
                result_shape = inferred.shape
            else:
                # Try to merge shapes (prefer more specific)
                result_shape = self._reconcile_shapes(result_shape, inferred.shape)

        # Use inferred type if existing is None
        if inferred.type is not None and result_type is None:
            result_type = inferred.type

        return ir.Value(shape=result_shape, type=result_type)

    def _reconcile_shapes(self, shape1: ir.Shape, shape2: ir.Shape) -> ir.Shape:
        """Reconcile two shapes by preferring more specific dimensions.

        Args:
            shape1: First shape.
            shape2: Second shape.

        Returns:
            The reconciled shape.
        """
        if len(shape1) != len(shape2):
            logger.warning(
                "Shape rank mismatch: %s vs %s. Using first shape.", len(shape1), len(shape2)
            )
            return shape1

        reconciled_dims = []
        for dim1, dim2 in zip(shape1.dims, shape2.dims):
            # Prefer concrete dimensions over None/symbolic
            if isinstance(dim1, int) and dim1 > 0:
                reconciled_dims.append(dim1)
            elif isinstance(dim2, int) and dim2 > 0:
                reconciled_dims.append(dim2)
            elif dim1 is not None:
                reconciled_dims.append(dim1)
            elif dim2 is not None:
                reconciled_dims.append(dim2)
            else:
                reconciled_dims.append(None)

        return ir.Shape(reconciled_dims)

    def _values_compatible(self, value1: ir.Value, value2: ir.Value) -> bool:
        """Check if two values are compatible (for strict mode).

        Args:
            value1: First value.
            value2: Second value.

        Returns:
            True if the values are compatible.
        """
        # Check shape compatibility
        if value1.shape is not None and value2.shape is not None:
            if not self._shapes_compatible(value1.shape, value2.shape):
                return False

        # Check type compatibility
        if value1.type is not None and value2.type is not None:
            if value1.type != value2.type:
                return False

        return True

    def _shapes_compatible(self, shape1: ir.Shape, shape2: ir.Shape) -> bool:
        """Check if two shapes are compatible.

        Args:
            shape1: First shape.
            shape2: Second shape.

        Returns:
            True if the shapes are compatible.
        """
        if len(shape1) != len(shape2):
            return False

        for dim1, dim2 in zip(shape1.dims, shape2.dims):
            # None/symbolic dimensions are compatible with anything
            if dim1 is None or dim2 is None:
                continue

            # Both concrete - must match
            if isinstance(dim1, int) and isinstance(dim2, int):
                if dim1 != dim2:
                    return False

            # Symbolic dimensions - for now assume compatible
            # Could be enhanced with symbolic expression comparison

        return True

    def get_inferrer_info(self) -> dict[str, int]:
        """Get information about registered inferrers.

        Returns:
            Dictionary mapping operation types to inferrer counts.
        """
        info = {}
        for (op_type, domain), inferrers in self._inferrer_registry.items():
            key = f"{op_type}:{domain}" if domain else op_type
            info[key] = len(inferrers)
        return info
