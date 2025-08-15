"""Symbolic shape inference for ONNX IR."""

from __future__ import annotations

import abc
import enum
import functools
from collections.abc import Collection, Sequence
from typing import Any, Callable

import sympy

import onnx_ir as ir

MAX_SUPPORTED_OPSET = 23


def get_expr(shape: ir.Shape, index: int) -> sympy.Expr:
    """Get the expression or value at a specific index in the shape.

    Args:
        shape: The shape to get the expression from.
        index: The index of the dimension to get.

    Returns:
        The expression or value at the specified index.
    """
    dim = shape[index]
    if isinstance(dim, ir.SymbolicDim):
        if dim.expr is not None:
            return dim.expr
        if dim.value is None:
            return sympy.Symbol("__unknown__")
        return sympy.Symbol(dim.value)
    return sympy.Integer(dim)


@enum.unique
class InferenceStatus(enum.Enum):
    """Status of shape inference operation."""

    SUCCESS = "success"  # Complete inference successful
    PARTIAL = "partial"  # Partial information available (e.g., type only, rank only)
    MISSING_INFO = "missing_info"  # Missing required input information
    INVALID_NODE = "invalid_node"  # Node is invalid or malformed


class InferenceResult:
    """Container for inference results with status and optional message."""

    def __init__(
        self,
        values: Sequence[ir.Value] | None = None,
        status: str | InferenceStatus = "success",
        msg: str | None = None,
    ) -> None:
        """Initialize inference result.

        Args:
            values: Sequence of inferred values.
            status: Status of inference operation (string or enum).
            msg: Optional message for context.
        """
        self.values = values
        self.status = InferenceStatus(status)
        self.msg = msg

    def __repr__(self) -> str:
        """Return string representation of the result."""
        return f"InferenceResult(values={self.values}, status={self.status.value}, msg={self.msg!r})"


class NodeInferrer(abc.ABC):
    """Base class for node inferrers.

    This class provides a common interface for all node inferrers.
    """

    def __init__(
        self, op_type: str, opsets: Collection[int], domain: str = "", overload: str = ""
    ) -> None:
        """Initialize the node inferrer.

        Args:
            op_type: The type of the operation.
            opsets: A collection of ONNX opset versions supported by this inferrer.
            domain: The domain of the operation, default is an empty string.
            overload: The overload identifier for the operation, default is an empty string.
        """
        self.op_type = op_type
        self.opsets = opsets
        self.domain = domain
        self.overload = overload

    def __repr__(self) -> str:
        """Return a string representation of the node inferrer."""
        return f"{self.__class__.__name__}(op_type={self.op_type}, opsets={self.opsets}, domain={self.domain})"

    @abc.abstractmethod
    def infer(self, node: ir.Node) -> InferenceResult:
        """Infer the shape for the node.

        Args:
            node: The ONNX node to infer the type and shape for.

        Returns:
            A sequence of ONNX values containing the inferred shapes.
        """
        raise NotImplementedError


def requires_non_none_inputs(
    count: int, /
) -> Callable[
    [Callable[[Any, ir.Node], InferenceResult]], Callable[[Any, ir.Node], InferenceResult]
]:
    """Ensure that the node has a specific number of non-None inputs.

    Args:
        count: The exact number of non-None inputs required for the node.

    Returns:
        A decorator that checks the number of inputs and their non-None status.
    """

    def decorator(
        func: Callable[[Any, ir.Node], InferenceResult],
    ) -> Callable[[Any, ir.Node], InferenceResult]:
        @functools.wraps(func)
        def wrapper(self, node: ir.Node) -> InferenceResult:
            if len(node.inputs) != count:
                return InferenceResult(
                    status="invalid_node",
                    msg=f"{node.op_type} must have {count} inputs, got {len(node.inputs)}.",
                )
            for i, inp in enumerate(node.inputs):
                if inp is None:
                    return InferenceResult(
                        status="missing_info", msg=f"{node.op_type} input {i} cannot be None."
                    )
            return func(self, node)

        return wrapper

    return decorator


def requires_outputs(
    count: int, /
) -> Callable[
    [Callable[[Any, ir.Node], InferenceResult]], Callable[[Any, ir.Node], InferenceResult]
]:
    """Ensure that the node has a specific number of outputs.

    Args:
        count: The exact number of outputs required for the node.

    Returns:
        A decorator that checks the number of outputs.
    """

    def decorator(
        func: Callable[[Any, ir.Node], InferenceResult],
    ) -> Callable[[Any, ir.Node], InferenceResult]:
        @functools.wraps(func)
        def wrapper(self, node: ir.Node) -> InferenceResult:
            if len(node.outputs) != count:
                return InferenceResult(
                    status="invalid_node",
                    msg=f"{node.op_type} must have {count} outputs, got {len(node.outputs)}.",
                )
            return func(self, node)

        return wrapper

    return decorator


def inclusive_range(start_or_end: int = 0, end: int | None = None) -> range:
    """Create an inclusive range from start to end with a given step.

    Args:
        start_or_end: The starting value of the range.
        end: The ending value of the range (inclusive).

    Returns:
        A range object that includes both start and end.
    """
    if end is None:
        end = start_or_end
        start = 0
    else:
        start = start_or_end

    return range(start, end + 1)
