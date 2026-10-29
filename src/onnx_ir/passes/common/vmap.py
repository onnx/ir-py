# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lower independent applications of a tensor graph to ordinary batched ONNX nodes."""

from __future__ import annotations

__all__ = ["VmapPass"]

from collections.abc import Sequence

import onnx_ir as ir

_UNARY = frozenset(
    {
        "Abs",
        "Cast",
        "Ceil",
        "Cos",
        "Erf",
        "Exp",
        "Floor",
        "Identity",
        "IsInf",
        "IsNaN",
        "Log",
        "Neg",
        "Not",
        "Reciprocal",
        "Relu",
        "Round",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sqrt",
        "Tan",
        "Tanh",
    }
)
_BINARY = frozenset(
    {
        "Add",
        "And",
        "Div",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Mul",
        "Or",
        "Pow",
        "Sub",
        "Xor",
    }
)
_VARIADIC = frozenset({"Max", "Mean", "Min", "Sum"})
_REDUCTIONS = frozenset(
    {
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    }
)


def _axis(axis: int, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int) or not -rank <= axis < rank:
        raise ir.passes.PreconditionError(f"Axis {axis!r} is invalid for rank {rank}.")
    return axis % rank


class VmapPass(ir.passes.FunctionalPass):
    """Batch independent applications of a single-example model.

    The source model describes *one* example. This functional pass inserts an
    batch axis into each mapped input and every output. It
    emits ordinary ONNX tensor operations, not a Loop or a custom Map operator.
    Input names and node-output names are preserved, and the source model is not
    modified. A directly returned input or initializer may need a new output name.

    Args:
        in_axes: One entry per graph input, in graph-input order. An integer
            selects where to insert the batch axis; ``None`` leaves an input
            shared between examples. Negative axes index the resulting input.
            At least one input must be mapped.
        axis_size: Optional fixed, positive number of independent examples.
            When omitted, the batch length is inferred at runtime from the first
            mapped input. All mapped inputs must have the same batch length;
            this is an input contract, not a runtime assertion.
        out_axes: Where to insert the batch axis in each output. An integer
            applies to all outputs; a sequence specifies one axis per output.
            Negative axes index the resulting output. Outputs independent of
            mapped inputs are repeated along this axis.

    Supported operations are Constant (tensor-valued), elementwise unary and
    broadcasting operations, MatMul (including vectors), Transpose, Concat,
    Softmax, LogSoftmax, and standard Reduce operations with constant axes.
    See the common-pass documentation for the exact whitelist.

    Requires standard-domain opset >= 18, tensor inputs with known ranks, and
    uniform per-example shapes. Symbolic dimension sizes are allowed. Dynamic
    reduction axes, control flow, custom/function calls, randomness, sequences,
    and operators without a batching rule raise
    :class:`onnx_ir.passes.PreconditionError`.
    Initializers are shared; a mapped input cannot have an initializer default.
    The pass does not infer general shapes or check the validity of the source
    model, which must already be a valid, topologically sorted ONNX graph.
    """

    def __init__(
        self,
        in_axes: Sequence[int | None],
        *,
        axis_size: int | None = None,
        out_axes: int | Sequence[int] = 0,
    ) -> None:
        if axis_size is not None and (
            isinstance(axis_size, bool) or not isinstance(axis_size, int) or axis_size <= 0
        ):
            raise ValueError("axis_size must be a positive integer or None.")
        self.in_axes = tuple(in_axes)
        self.axis_size = axis_size
        self.out_axes = out_axes if isinstance(out_axes, int) else tuple(out_axes)

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        if model.graph.opset_imports.get("", 0) < 18:
            raise ir.passes.PreconditionError("VmapPass requires standard-domain opset >= 18.")
        if len(self.in_axes) != len(model.graph.inputs):
            raise ir.passes.PreconditionError("in_axes must have one entry per graph input.")
        if all(axis is None for axis in self.in_axes):
            raise ir.passes.PreconditionError("At least one input must be mapped.")
        if isinstance(self.out_axes, int):
            out_axes = (self.out_axes,) * len(model.graph.outputs)
        else:
            out_axes = self.out_axes
        if len(out_axes) != len(model.graph.outputs):
            raise ir.passes.PreconditionError("out_axes must have one entry per graph output.")
        available = set(model.graph.inputs) | set(model.graph.initializers.values())
        for node in model.graph:
            if node.op_identifier() in model.functions:
                raise ir.passes.PreconditionError(
                    "VmapPass does not support local function calls. Inline functions first."
                )
            if any(value is not None and value not in available for value in node.inputs):
                raise ir.passes.PreconditionError(
                    "VmapPass requires a topologically sorted graph without outer-scope values."
                )
            available.update(node.outputs)
        if any(value not in available for value in model.graph.outputs):
            raise ir.passes.PreconditionError("Graph output is not defined in the graph.")
        result = model.clone()
        _Batcher(result.graph, self.axis_size).run(self.in_axes, out_axes)
        return ir.passes.PassResult(result, modified=True)


class _Batcher:
    def __init__(self, graph: ir.Graph, axis_size: int | None) -> None:
        self.graph = graph
        self.axis_size = axis_size
        self.ranks: dict[ir.Value, int] = {}
        self.mapped: dict[ir.Value, bool] = {}
        self.values: dict[ir.Value, ir.Value] = {}
        self.before: ir.Node | None = None
        self.batch_source: ir.Value | None = None
        self.batch_shape: ir.Value | None = None

    def _emit(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: dict[str, int | Sequence[int] | ir.TensorProtocol] | None = None,
    ) -> ir.Value:
        node = ir.node(op_type, inputs, attributes)
        if self.before is None:
            self.graph.append(node)
        else:
            self.graph.insert_before(self.before, node)
        return node.outputs[0]

    def _ints(self, values: Sequence[int]) -> ir.Value:
        return self._emit(
            "Constant", [], {"value": ir.tensor(values, dtype=ir.DataType.INT64)}
        )

    def _unsqueeze(self, value: ir.Value, axes: Sequence[int]) -> ir.Value:
        if not axes:
            return value
        result = self._emit("Unsqueeze", [value, self._ints(axes)])
        result.type = value.type
        if value.shape is not None:
            dims = list(value.shape)
            for axis in sorted(axes):
                dims.insert(axis, 1)
            result.shape = ir.Shape(dims)
        return result

    def _transpose(self, value: ir.Value, perm: Sequence[int]) -> ir.Value:
        if list(perm) == list(range(len(perm))):
            return value
        result = self._emit("Transpose", [value], {"perm": perm})
        result.type = value.type
        if value.shape is not None:
            result.shape = ir.Shape(value.shape[i] for i in perm)
        return result

    def _repeat(self, value: ir.Value) -> ir.Value:
        if self.batch_shape is None:
            if self.axis_size is not None:
                self.batch_shape = self._ints([self.axis_size])
            else:
                assert self.batch_source is not None
                batch_source_shape = self._emit("Shape", [self.batch_source])
                self.batch_shape = self._emit(
                    "Gather", [batch_source_shape, self._ints([0])], {"axis": 0}
                )
        shape = self._emit("Shape", [value])
        target = self._emit("Concat", [self.batch_shape, shape], {"axis": 0})
        result = self._emit("Expand", [self._unsqueeze(value, [0]), target])
        result.type = value.type
        if value.shape is not None:
            result.shape = ir.Shape([self.axis_size, *value.shape])
        return result

    def _align(self, value: ir.Value, rank: int, mapped: bool, target: int) -> ir.Value:
        # Padding belongs *after* a mapped batch axis, never before it.
        start = 1 if mapped else 0
        return self._unsqueeze(value, list(range(start, target - rank + 1)))

    def _register(self, value: ir.Value, rank: int, mapped: bool) -> None:
        if value.type is not None and not isinstance(value.type, ir.TensorType):
            raise ir.passes.PreconditionError(f"Expected a tensor value, got {value!r}.")
        if value.shape is not None and len(value.shape) != rank:
            raise ir.passes.PreconditionError(f"Inconsistent rank annotation for {value!r}.")
        self.ranks[value] = rank
        self.mapped[value] = mapped
        self.values[value] = value
        if mapped:
            dims = list(value.shape) if value.shape is not None else [None] * rank
            value.shape = ir.Shape([self.axis_size, *dims])
            value.const_value = None
        elif value.shape is None:
            value.shape = ir.Shape([None] * rank)

    def _constant_axes(self, value: ir.Value) -> list[int]:
        tensor = None
        if not value.is_graph_input():
            if value.is_initializer():
                tensor = value.const_value
            else:
                producer = value.producer()
                if producer is not None and producer.op_type == "Constant":
                    attr = producer.attributes.get("value")
                    if attr is not None and not attr.is_ref():
                        tensor = attr.as_tensor()
        if (
            tensor is None
            or tensor.dtype != ir.DataType.INT64
            or len(tensor.shape) != 1
            or self.mapped.get(value, False)
        ):
            raise ir.passes.PreconditionError(
                "Reduction axes must be a shared constant 1-D INT64 tensor, not a runtime input."
            )
        return [int(axis) for axis in tensor.numpy().tolist()]

    def _reduce(self, node: ir.Node, inputs: list[ir.Value], mapped: bool) -> int:
        rank = self.ranks[inputs[0]]
        axes = self._constant_axes(inputs[1]) if len(inputs) == 2 else []
        noop = node.attributes.get_int("noop_with_empty_axes", 0)
        if not axes and noop:
            return rank
        axes = [_axis(axis, rank) for axis in axes] if axes else list(range(rank))
        if len(set(axes)) != len(axes):
            raise ir.passes.PreconditionError("Reduction axes must not contain duplicates.")
        if mapped:
            if not axes:
                # A singleton example dimension preserves scalar ReduceL1,
                # ReduceLogSum, etc., without reducing the batch dimension.
                node.replace_input_with(0, self._unsqueeze(self.values[inputs[0]], [1]))
                shifted = self._ints([1])
                node.attributes["keepdims"] = ir.AttrInt64("keepdims", 0)
            else:
                shifted = self._ints([axis + 1 for axis in axes])
            node.resize_inputs(2)
            node.replace_input_with(1, shifted)
        keepdims = node.attributes.get_int("keepdims", 1)
        return rank if keepdims else rank - len(axes)

    def _matmul(self, node: ir.Node, inputs: list[ir.Value], mapped: bool) -> int:
        left, right = inputs
        left_rank, right_rank = self.ranks[left], self.ranks[right]
        if min(left_rank, right_rank) < 1:
            raise ir.passes.PreconditionError("MatMul operands must have rank >= 1.")
        target = max(left_rank, right_rank, 2)
        rank = target - int(left_rank == 1) - int(right_rank == 1)
        if not mapped:
            return rank
        left_value, right_value = self.values[left], self.values[right]
        if left_rank == 1:
            left_value = self._unsqueeze(left_value, [int(self.mapped[left])])
        if right_rank == 1:
            right_value = self._unsqueeze(right_value, [right_rank + int(self.mapped[right])])
        node.replace_input_with(
            0, self._align(left_value, max(left_rank, 2), self.mapped[left], target)
        )
        node.replace_input_with(
            1, self._align(right_value, max(right_rank, 2), self.mapped[right], target)
        )
        if left_rank == 1 or right_rank == 1:
            squeeze_axes = []
            if left_rank == 1:
                squeeze_axes.append(target - 1)
            if right_rank == 1:
                squeeze_axes.append(target)
            product = self._emit("MatMul", list(node.inputs))
            product.type = node.outputs[0].type
            axes = self._ints(squeeze_axes)
            node.op_type = "Squeeze"
            node.attributes.clear()
            node.replace_input_with(0, product)
            node.replace_input_with(1, axes)
        return rank

    def _node(self, node: ir.Node) -> None:
        if (
            node.domain != ""
            or node.overload
            or any(
                attr.is_ref() or attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for attr in node.attributes.values()
            )
        ):
            raise ir.passes.PreconditionError(
                f"VmapPass does not support custom/function calls or subgraphs: {node.op_type}."
            )
        if len(node.outputs) != 1:
            raise ir.passes.PreconditionError(
                f"VmapPass requires single-output operators: {node.op_type}."
            )
        inputs = [value for value in node.inputs if value is not None]
        mapped = any(self.mapped[value] for value in inputs)
        for i, value in enumerate(node.inputs):
            if value is not None:
                node.replace_input_with(i, self.values[value])
        op = node.op_type
        if op == "Constant":
            attr = node.attributes.get("value")
            if inputs or attr is None or attr.type != ir.AttributeType.TENSOR:
                raise ir.passes.PreconditionError("Only tensor-valued Constant is supported.")
            tensor = attr.as_tensor()
            rank = len(tensor.shape)
            node.outputs[0].type = ir.TensorType(tensor.dtype)
            if node.outputs[0].shape is None:
                node.outputs[0].shape = ir.Shape(
                    d if isinstance(d, int) else d.value for d in tensor.shape.dims
                )
        elif op in _UNARY and len(inputs) == 1:
            rank = self.ranks[inputs[0]]
        elif (
            (op in _BINARY and len(inputs) == 2)
            or (op == "Where" and len(inputs) == 3)
            or (op in _VARIADIC and inputs)
        ):
            rank = max(self.ranks[value] for value in inputs)
            if mapped:
                for i, value in enumerate(inputs):
                    node.replace_input_with(
                        i,
                        self._align(
                            self.values[value], self.ranks[value], self.mapped[value], rank
                        ),
                    )
        elif op == "MatMul" and len(inputs) == 2:
            rank = self._matmul(node, inputs, mapped)
        elif op == "Transpose" and len(inputs) == 1:
            rank = self.ranks[inputs[0]]
            perm = list(node.attributes.get_ints("perm", range(rank - 1, -1, -1)))
            if sorted(perm) != list(range(rank)):
                raise ir.passes.PreconditionError("Transpose perm must be a rank permutation.")
            if mapped:
                node.attributes["perm"] = ir.AttrInt64s("perm", [0, *(i + 1 for i in perm)])
        elif op in ("Softmax", "LogSoftmax") and len(inputs) == 1:
            rank = self.ranks[inputs[0]]
            axis = _axis(node.attributes.get_int("axis", -1), rank)
            if mapped:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        elif op in _REDUCTIONS and 1 <= len(inputs) <= 2:
            rank = self._reduce(node, inputs, mapped)
        elif op == "Concat" and inputs:
            rank = self.ranks[inputs[0]]
            if any(self.ranks[value] != rank for value in inputs):
                raise ir.passes.PreconditionError("Concat inputs must have equal ranks.")
            axis = _axis(node.attributes.get_int("axis"), rank)
            if mapped:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
                for i, value in enumerate(inputs):
                    if not self.mapped[value]:
                        node.replace_input_with(i, self._repeat(self.values[value]))
        else:
            raise ir.passes.PreconditionError(
                f"VmapPass has no batching rule for {op!r} with {len(inputs)} inputs."
            )
        output = node.outputs[0]
        self._register(output, rank, mapped)

    def run(self, in_axes: Sequence[int | None], out_axes: Sequence[int]) -> None:
        nodes = list(self.graph)
        self.before = nodes[0] if nodes else None
        for value in self.graph.initializers.values():
            if value.shape is None:
                raise ir.passes.PreconditionError("Initializers must have known ranks.")
            self._register(value, len(value.shape), False)
        for value, axis in zip(self.graph.inputs, in_axes):
            if not isinstance(value.type, ir.TensorType) or value.shape is None:
                raise ir.passes.PreconditionError(
                    "Graph inputs must be tensors with known ranks."
                )
            rank = len(value.shape)
            if axis is None:
                self._register(value, rank, False)
                continue
            if value.is_initializer():
                raise ir.passes.PreconditionError(
                    "Mapped inputs cannot have initializer defaults."
                )
            axis = _axis(axis, rank + 1)
            self._register(value, rank, True)
            dims = list(value.shape)
            dims.insert(axis, dims.pop(0))
            value.shape = ir.Shape(dims)
            self.values[value] = self._transpose(
                value, [axis, *(i for i in range(rank + 1) if i != axis)]
            )
            if self.batch_source is None:
                self.batch_source = self.values[value]
        for node in nodes:
            self.before = node
            self._node(node)
        self.before = None
        for i, (output, axis) in enumerate(zip(tuple(self.graph.outputs), out_axes)):
            rank = self.ranks[output]
            axis = _axis(axis, rank + 1)
            value = self.values[output]
            if not self.mapped[output]:
                value = self._repeat(value)
            perm = list(range(1, rank + 1))
            perm.insert(axis, 0)
            value = self._transpose(value, perm)
            if (
                value is not output
                and not output.is_graph_input()
                and not output.is_initializer()
            ):
                output.name, value.name = value.name, output.name
            self.graph.outputs[i] = value
