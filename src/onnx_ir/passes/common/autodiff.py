# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""First-order automatic differentiation as ordinary ONNX graph transformations.

The functional passes in this module implement forward linearization (JVP) and
reverse transposition (VJP), rather than inserting training-domain gradient
operators or depending on a training runtime. See :class:`JvpPass` for the
shared support contract and :class:`GradPass` for a scalar-loss example.
"""

from __future__ import annotations

__all__ = ["JvpPass", "VjpPass", "GradPass"]

import math
from collections.abc import Sequence

import numpy as np

import onnx_ir as ir
from onnx_ir._convenience import SupportedAttrTypes
from onnx_ir.passes.common.shape_inference import ShapeInferencePass

_FLOATS = frozenset((ir.DataType.FLOAT, ir.DataType.DOUBLE))
_MARKER = "onnx_ir.autodiff"
_BINARY = frozenset(("Add", "Sub", "Mul", "Div", "Pow"))
_ELEMENTWISE = frozenset(
    (
        "Identity",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Neg",
        "Abs",
        "Exp",
        "Log",
        "Sqrt",
        "Reciprocal",
        "Sin",
        "Cos",
        "Tan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Sigmoid",
        "Relu",
        "LeakyRelu",
        "Softplus",
        "Softsign",
        "Erf",
    )
)
_STRUCTURAL = frozenset(("Reshape", "Flatten", "Squeeze", "Unsqueeze", "Expand"))
_SUPPORTED = (
    _ELEMENTWISE
    | _STRUCTURAL
    | frozenset(
        (
            "MatMul",
            "Gemm",
            "ReduceSum",
            "ReduceMean",
            "Transpose",
            "Cast",
            "Where",
            "Concat",
            "Split",
            "Gather",
            "Softmax",
            "LogSoftmax",
        )
    )
)
_NO_DATA = frozenset(
    (
        "Constant",
        "ConstantOfShape",
        "Shape",
        "Size",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "And",
        "Or",
        "Xor",
        "Not",
        "IsInf",
        "IsNaN",
    )
)


def _data_indices(node: ir.Node) -> Sequence[int]:
    if node.domain:
        return range(len(node.inputs))
    if node.op_type in _NO_DATA:
        return ()
    if node.op_type == "Where":
        return (1, 2)
    if node.op_type in _STRUCTURAL | {"Gather", "Split", "ReduceSum", "ReduceMean"}:
        return (0,)
    return range(len(node.inputs))


def _input(node: ir.Node, index: int = 0) -> ir.Value:
    if index >= len(node.inputs) or node.inputs[index] is None:
        raise ValueError(f"{node.op_type} requires input {index}.")
    value = node.inputs[index]
    assert value is not None
    return value


def _int(node: ir.Node, name: str, default: int) -> int:
    attr = node.attributes.get(name)
    return default if attr is None else attr.as_int()


def _float(node: ir.Node, name: str, default: float) -> float:
    attr = node.attributes.get(name)
    return default if attr is None else attr.as_float()


def _rank(value: ir.Value) -> int:
    if value.shape is None:
        raise ValueError(
            f"Autodiff requires a known rank for {value.name!r}; "
            "provide shape metadata or run shape inference first."
        )
    return len(value.shape)


def _select(
    values: Sequence[ir.Value], names: Sequence[str] | None, what: str
) -> list[ir.Value]:
    if isinstance(names, str):
        raise TypeError(f"{what} must be a sequence of names, not a string.")
    if names is None:
        selected = [value for value in values if value.dtype in _FLOATS]
    else:
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate names in {what}: {names!r}.")
        by_name = {value.name: value for value in values}
        selected = []
        for name in names:
            if name not in by_name:
                raise ValueError(f"Unknown {what} name {name!r}.")
            selected.append(by_name[name])
    if not selected:
        raise ValueError(f"Select at least one FLOAT or DOUBLE tensor in {what}.")
    for value in selected:
        if not isinstance(value.type, ir.TensorType) or value.dtype not in _FLOATS:
            raise TypeError(
                f"Autodiff {what} {value.name!r} must be a FLOAT or DOUBLE tensor, "
                f"got {value.type}."
            )
        if value.shape is None:
            raise ValueError(
                f"Selected {what} {value.name!r} needs known-rank shape metadata; "
                "dimension sizes may be symbolic."
            )
    return selected


class _Builder:
    def __init__(self, graph: ir.Graph) -> None:
        self.graph = graph
        self.tape = ir.tape.Tape(graph)
        self.names = set(ir.convenience.create_value_mapping(graph, include_subgraphs=False))

    def name(self, stem: str) -> str:
        name = stem
        suffix = 1
        while name in self.names:
            name = f"{stem}_{suffix}"
            suffix += 1
        self.names.add(name)
        return name

    def op(self, op: str, *inputs: ir.Value | None, **attrs: SupportedAttrTypes) -> ir.Value:
        return self.tape.op(op, inputs, attrs)

    def constant(
        self, value: float | int | Sequence[int], dtype: ir.DataType = ir.DataType.INT64
    ) -> ir.Value:
        return self.op("Constant", value=ir.tensor(np.asarray(value, dtype=dtype.numpy())))

    def scalar(self, reference: ir.Value, value: float) -> ir.Value:
        assert reference.dtype in _FLOATS
        return self.constant(value, reference.dtype)

    def like(self, reference: ir.Value, fill: float = 0.0) -> ir.Value:
        assert reference.dtype in _FLOATS
        return self.op(
            "ConstantOfShape",
            self.op("Shape", reference),
            value=ir.tensor([fill], dtype=reference.dtype),
        )

    def reshape(self, value: ir.Value, shape: ir.Value) -> ir.Value:
        return self.op("Reshape", value, shape, allowzero=1)

    def sum(self, values: Sequence[ir.Value]) -> ir.Value:
        result = values[0]
        for value in values[1:]:
            result = self.op("Add", result, value)
        return result

    def seed(self, value: ir.Value, suffix: str) -> ir.Value:
        assert value.dtype is not None
        result = ir.Value(
            name=self.name(f"{value.name}_{suffix}"),
            type=ir.TensorType(value.dtype),
            shape=ir.Shape(value.shape) if value.shape is not None else None,
        )
        self.graph.inputs.append(result)
        return result

    def output(self, derivative: ir.Value, primal: ir.Value, suffix: str) -> None:
        assert primal.dtype is not None
        output = self.op("Identity", derivative)
        output.name = self.name(f"{primal.name}_{suffix}")
        output.type = ir.TensorType(primal.dtype)
        output.shape = ir.Shape(primal.shape) if primal.shape is not None else None
        self.graph.outputs.append(output)

    def unbroadcast(self, gradient: ir.Value, primal: ir.Value) -> ir.Value:
        """Sum runtime singleton and leading broadcast dimensions, including zero extents."""
        shape = self.op("Shape", primal)
        grad_shape = self.op("Shape", gradient)
        leading = self.op("Sub", self.op("Size", grad_shape), self.op("Size", shape))
        ones = self.op(
            "ConstantOfShape",
            self.op("Unsqueeze", leading, self.constant([0])),
            value=ir.tensor([1], dtype=ir.DataType.INT64),
        )
        padded = self.op("Concat", ones, shape, axis=0)
        axes = self.op(
            "Squeeze",
            self.op("NonZero", self.op("Equal", padded, self.constant(1))),
            self.constant([0]),
        )
        # A dummy singleton axis avoids runtime bugs in empty-axis reductions of empty tensors.
        axes = self.op("Concat", axes, self.constant([-1]), axis=0)
        expanded = self.op("Unsqueeze", gradient, self.constant([-1]))
        reduced = self.op("ReduceSum", expanded, axes, keepdims=1)
        return self.reshape(reduced, shape)

    def transpose(self, value: ir.Value, rank: int) -> ir.Value:
        perm = list(range(rank))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return self.op("Transpose", value, perm=perm)

    def matmul_vjp(
        self, node: ir.Node, gradient: ir.Value, indices: Sequence[int]
    ) -> dict[int, ir.Value]:
        a, b = _input(node), _input(node, 1)
        ra, rb = _rank(a), _rank(b)
        if ra == 0 or rb == 0:
            raise ValueError("MatMul inputs must have rank at least one.")
        av = self.op("Unsqueeze", a, self.constant([0])) if ra == 1 else a
        bv = self.op("Unsqueeze", b, self.constant([-1])) if rb == 1 else b
        g = gradient
        if rb == 1:
            g = self.op("Unsqueeze", g, self.constant([-1]))
        if ra == 1:
            g = self.op("Unsqueeze", g, self.constant([-2]))
        result = {}
        if 0 in indices:
            ga = self.op("MatMul", g, self.transpose(bv, max(rb, 2)))
            if ra == 1:
                ga = self.op("Squeeze", ga, self.constant([-2]))
            result[0] = self.unbroadcast(ga, a)
        if 1 in indices:
            gb = self.op("MatMul", self.transpose(av, max(ra, 2)), g)
            if rb == 1:
                gb = self.op("Squeeze", gb, self.constant([-1]))
            result[1] = self.unbroadcast(gb, b)
        return result

    def reduction_vjp(self, node: ir.Node, g: ir.Value) -> ir.Value:
        x = _input(node)
        shape = self.op("Shape", x)
        rank = self.op("Size", shape)
        zero, one = self.constant(0), self.constant(1)
        axes = node.inputs[1] if len(node.inputs) > 1 else None
        noop = _int(node, "noop_with_empty_axes", 0)
        if axes is None:
            axes = self.constant([]) if noop else self.op("Range", zero, rank, one)
        elif not noop:
            stop = self.op("Where", self.op("Equal", self.op("Size", axes), zero), rank, zero)
            axes = self.op("Concat", axes, self.op("Range", zero, stop, one), axis=0)
        if not _int(node, "keepdims", 1):
            ones = self.op(
                "ConstantOfShape",
                self.op("Shape", axes),
                value=ir.tensor([1], dtype=ir.DataType.INT64),
            )
            keep_shape = self.op("ScatterElements", shape, axes, ones, axis=0)
            g = self.reshape(g, keep_shape)
        result = self.op("Expand", g, shape)
        if node.op_type == "ReduceMean":
            count = self.op(
                "Div",
                self.op("Cast", self.op("Size", x), to=x.dtype),
                self.op("Cast", self.op("Size", node.outputs[0]), to=x.dtype),
            )
            result = self.op("Div", result, count)
        return result

    def gather_vjp(self, node: ir.Node, g: ir.Value) -> ir.Value:
        x, indices = _input(node), _input(node, 1)
        rank, index_rank = _rank(x), _rank(indices)
        axis = _int(node, "axis", 0) % rank
        perm = [axis, *range(axis), *range(axis + 1, rank)]
        updates_perm = [
            *range(axis, axis + index_rank),
            *range(axis),
            *range(axis + index_rank, rank - 1 + index_rank),
        ]
        updates = self.op("Transpose", g, perm=updates_perm)
        zeros = self.op("Transpose", self.like(x), perm=perm)
        if indices.dtype != ir.DataType.INT64:
            indices = self.op("Cast", indices, to=ir.DataType.INT64)
        result = self.op(
            "ScatterND",
            zeros,
            self.op("Unsqueeze", indices, self.constant([-1])),
            updates,
            reduction="add",
        )
        return self.op("Transpose", result, perm=np.argsort(perm).tolist())

    def partials(self, node: ir.Node, indices: Sequence[int]) -> dict[int, ir.Value]:
        """Elementwise local derivatives; do not evaluate unused partials (notably Pow)."""
        op = node.op_type
        x, y = _input(node), node.outputs[0]
        one = self.scalar(x, 1)
        result = {}
        for index in indices:
            if op in ("Identity", "Add"):
                p = one
            elif op == "Sub":
                p = one if index == 0 else self.scalar(x, -1)
            elif op == "Mul":
                p = _input(node, 1 - index)
            elif op == "Div":
                b = _input(node, 1)
                p = (
                    self.op("Div", one, b)
                    if index == 0
                    else self.op("Neg", self.op("Div", y, b))
                )
            elif op == "Pow":
                b = _input(node, 1)
                if b.dtype != x.dtype:
                    b = self.op("Cast", b, to=x.dtype)
                if index == 0:
                    # x**0 has zero base derivative even at x=0: avoid 0 * x**-1.
                    safe_b = self.op("Where", self.op("Equal", b, self.scalar(x, 0)), one, b)
                    p = self.op("Mul", b, self.op("Pow", x, self.op("Sub", safe_b, one)))
                else:
                    p = self.op("Mul", y, self.op("Log", x))
            elif op == "Neg":
                p = self.scalar(x, -1)
            elif op == "Abs":
                p = self.op("Sign", x)
            elif op == "Exp":
                p = y
            elif op == "Log":
                p = self.op("Div", one, x)
            elif op == "Sqrt":
                p = self.op("Div", self.scalar(x, 0.5), y)
            elif op == "Reciprocal":
                p = self.op("Neg", self.op("Mul", y, y))
            elif op == "Sin":
                p = self.op("Cos", x)
            elif op == "Cos":
                p = self.op("Neg", self.op("Sin", x))
            elif op == "Tan":
                p = self.op("Add", one, self.op("Mul", y, y))
            elif op == "Sinh":
                p = self.op("Cosh", x)
            elif op == "Cosh":
                p = self.op("Sinh", x)
            elif op == "Tanh":
                p = self.op("Sub", one, self.op("Mul", y, y))
            elif op == "Sigmoid":
                p = self.op("Mul", y, self.op("Sub", one, y))
            elif op in ("Relu", "LeakyRelu"):
                negative = 0.0 if op == "Relu" else _float(node, "alpha", 0.01)
                p = self.op(
                    "Where",
                    self.op("Greater", x, self.scalar(x, 0)),
                    one,
                    self.scalar(x, negative),
                )
            elif op == "Softplus":
                p = self.op("Sigmoid", x)
            elif op == "Softsign":
                denom = self.op("Add", one, self.op("Abs", x))
                p = self.op("Div", one, self.op("Mul", denom, denom))
            elif op == "Erf":
                p = self.op(
                    "Mul",
                    self.scalar(x, 2 / math.sqrt(math.pi)),
                    self.op("Exp", self.op("Neg", self.op("Mul", x, x))),
                )
            else:
                raise NotImplementedError(f"No elementwise derivative for {op}.")
            result[index] = p
        return result

    def gemm(
        self, node: ir.Node, derivatives: dict[int, ir.Value], reverse: bool
    ) -> dict[int, ir.Value]:
        a, b = _input(node), _input(node, 1)
        ta, tb = _int(node, "transA", 0), _int(node, "transB", 0)
        av = self.transpose(a, 2) if ta else a
        bv = self.transpose(b, 2) if tb else b
        alpha = self.scalar(a, _float(node, "alpha", 1))
        beta = self.scalar(a, _float(node, "beta", 1))
        if reverse:
            g = derivatives[0]
            ga = self.op("MatMul", g, self.transpose(bv, 2))
            gb = self.op("MatMul", self.transpose(av, 2), g)
            result = {
                0: self.op("Mul", self.transpose(ga, 2) if ta else ga, alpha),
                1: self.op("Mul", self.transpose(gb, 2) if tb else gb, alpha),
            }
            if len(node.inputs) > 2 and node.inputs[2] is not None:
                result[2] = self.unbroadcast(self.op("Mul", g, beta), _input(node, 2))
            return result
        terms = []
        if 0 in derivatives:
            da = self.transpose(derivatives[0], 2) if ta else derivatives[0]
            terms.append(self.op("Mul", self.op("MatMul", da, bv), alpha))
        if 1 in derivatives:
            db = self.transpose(derivatives[1], 2) if tb else derivatives[1]
            terms.append(self.op("Mul", self.op("MatMul", av, db), alpha))
        if 2 in derivatives:
            terms.append(self.op("Mul", derivatives[2], beta))
        # A bias-only direction still has the full Gemm output shape.
        return {0: self.op("Expand", self.sum(terms), self.op("Shape", node.outputs[0]))}

    def jvp(self, node: ir.Node, d: dict[int, ir.Value]) -> list[ir.Value]:
        op = node.op_type
        if op in _ELEMENTWISE:
            partials = self.partials(node, tuple(d))
            result = self.sum([self.op("Mul", d[i], partials[i]) for i in d])
            return [
                self.op("Expand", result, self.op("Shape", node.outputs[0]))
                if op in _BINARY
                else result
            ]
        if op in _STRUCTURAL | {"Transpose", "Cast", "Gather", "ReduceSum", "ReduceMean"}:
            return [
                self.tape.op(op, [d[0], *node.inputs[1:]], attributes=dict(node.attributes))
            ]
        if op == "MatMul":
            terms = []
            if 0 in d:
                terms.append(self.op("MatMul", d[0], _input(node, 1)))
            if 1 in d:
                terms.append(self.op("MatMul", _input(node), d[1]))
            return [self.sum(terms)]
        if op == "Gemm":
            return [self.gemm(node, d, reverse=False)[0]]
        if op == "Where":
            return [
                self.op(
                    "Where",
                    _input(node),
                    d[1] if 1 in d else self.like(_input(node, 1)),
                    d[2] if 2 in d else self.like(_input(node, 2)),
                )
            ]
        if op == "Concat":
            return [
                self.op(
                    "Concat",
                    *(
                        d[i] if i in d else self.like(_input(node, i))
                        for i in range(len(node.inputs))
                    ),
                    axis=_int(node, "axis", 0),
                )
            ]
        if op == "Split":
            return list(
                self.tape.op_multi_out(
                    "Split",
                    [d[0], *node.inputs[1:]],
                    dict(node.attributes),
                    num_outputs=len(node.outputs),
                )
            )
        if op in ("Softmax", "LogSoftmax"):
            y = node.outputs[0]
            axis = self.constant([_int(node, "axis", -1)])
            probability = y if op == "Softmax" else self.op("Exp", y)
            weighted = self.op("Mul", d[0], probability)
            mean = self.op("ReduceSum", weighted, axis, keepdims=1)
            centered = self.op("Sub", d[0], mean)
            return [self.op("Mul", y, centered) if op == "Softmax" else centered]
        raise NotImplementedError(f"No JVP rule for {op}.")

    def vjp(
        self, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
    ) -> dict[int, ir.Value]:
        op, g = node.op_type, gs[0]
        if op in _ELEMENTWISE:
            partials = self.partials(node, indices)
            result = {}
            for i in indices:
                contribution = self.op("Mul", g, partials[i])
                result[i] = (
                    self.unbroadcast(contribution, _input(node, i))
                    if op in _BINARY
                    else contribution
                )
            return result
        if op == "MatMul":
            return self.matmul_vjp(node, g, indices)
        if op == "Gemm":
            return self.gemm(node, {0: g}, reverse=True)
        if op in ("ReduceSum", "ReduceMean"):
            return {0: self.reduction_vjp(node, g)}
        if op in ("Reshape", "Flatten", "Squeeze", "Unsqueeze"):
            return {0: self.reshape(g, self.op("Shape", _input(node)))}
        if op == "Expand":
            return {0: self.unbroadcast(g, _input(node))}
        if op == "Transpose":
            attr = node.attributes.get("perm")
            perm = (
                attr.as_ints()
                if attr is not None
                else tuple(reversed(range(_rank(_input(node)))))
            )
            return {0: self.op("Transpose", g, perm=np.argsort(perm).tolist())}
        if op == "Cast":
            return {0: self.op("Cast", g, to=_input(node).dtype)}
        if op == "Where":
            zero = self.scalar(node.outputs[0], 0)
            return {
                i: self.unbroadcast(
                    self.op(
                        "Where", _input(node), g if i == 1 else zero, zero if i == 1 else g
                    ),
                    _input(node, i),
                )
                for i in indices
            }
        if op == "Concat":
            axis = _int(node, "axis", 0)
            sizes = [
                self.op(
                    "Gather", self.op("Shape", _input(node, i)), self.constant([axis]), axis=0
                )
                for i in range(len(node.inputs))
            ]
            parts = self.tape.op_multi_out(
                "Split",
                [g, self.op("Concat", *sizes, axis=0)],
                {"axis": axis},
                num_outputs=len(node.inputs),
            )
            return {i: parts[i] for i in indices}
        if op == "Split":
            return {0: self.op("Concat", *gs, axis=_int(node, "axis", 0))}
        if op == "Gather":
            return {0: self.gather_vjp(node, g)}
        if op in ("Softmax", "LogSoftmax"):
            y = node.outputs[0]
            axis_tensor = self.constant([_int(node, "axis", -1)])
            if op == "Softmax":
                weighted = self.op("Mul", g, y)
                mean = self.op("ReduceSum", weighted, axis_tensor, keepdims=1)
                return {0: self.op("Mul", y, self.op("Sub", g, mean))}
            total = self.op("ReduceSum", g, axis_tensor, keepdims=1)
            return {0: self.op("Sub", g, self.op("Mul", self.op("Exp", y), total))}
        raise NotImplementedError(f"No VJP rule for {op}.")


def _prepare(
    model: ir.Model, inputs: Sequence[str] | None, outputs: Sequence[str] | None
) -> tuple[ir.Model, list[ir.Value], list[ir.Value], list[ir.Node], set[ir.Value]]:
    if _MARKER in model.graph.metadata_props:
        raise NotImplementedError("Higher-order differentiation is not supported.")
    if model.graph.opset_imports.get("", 0) < 18:
        raise ValueError("Autodiff requires the standard ONNX opset >= 18; convert first.")
    result = model.clone()
    graph = result.graph
    graph.sort()
    for graph_like in (graph, *result.functions.values()):
        for node in graph_like:
            if any(
                a.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for a in node.attributes.values()
            ):
                raise NotImplementedError(
                    "Autodiff does not support control flow or subgraphs."
                )
            if node.domain == "" and (
                node.op_type.startswith("Random")
                or node.op_type in ("Dropout", "Multinomial", "Bernoulli")
            ):
                raise NotImplementedError(
                    f"Autodiff does not support randomness: {node.op_type}."
                )
    ShapeInferencePass()(result)
    candidates = list(graph.inputs)
    if inputs is None:
        candidates = [v for v in candidates if not v.is_initializer()]
    else:
        candidates.extend(v for v in graph.initializers.values() if v not in candidates)
    xs = _select(candidates, inputs, "inputs")
    ys = _select(list(graph.outputs), outputs, "outputs")
    nodes = list(graph)
    needed = set(ys)
    for node in reversed(nodes):
        if any(y in needed for y in node.outputs):
            needed.update(
                _input(node, i)
                for i in _data_indices(node)
                if i < len(node.inputs) and node.inputs[i] is not None
            )
    active = set(xs)
    relevant_nodes = []
    for node in nodes:
        indices = [
            i for i in _data_indices(node) if i < len(node.inputs) and node.inputs[i] in active
        ]
        if not indices or not any(y in needed for y in node.outputs):
            continue
        if (
            node.domain
            or node.overload
            or node.op_type not in _SUPPORTED
            or node.op_identifier() in result.functions
        ):
            raise NotImplementedError(
                f"No autodiff rule for active operator {node.domain!r}::{node.op_type} "
                f"(node {node.name!r}). Inline local functions before differentiation."
            )
        if node.op_type == "Pow":
            base, exponent = _input(node), _input(node, 1)
            if exponent.dtype in _FLOATS and exponent.dtype != base.dtype:
                raise TypeError("Differentiating Pow requires matching floating-point dtypes.")
        for value in [*(_input(node, i) for i in indices), *node.outputs]:
            if not isinstance(value.type, ir.TensorType) or value.dtype not in _FLOATS:
                raise TypeError(
                    f"Active value {value.name!r} at {node.op_type} must be FLOAT or DOUBLE. "
                    "Integer/boolean casts and complex differentiation are not supported."
                )
        active.update(y for y in node.outputs if y in needed)
        relevant_nodes.append(node)
    return result, xs, ys, relevant_nodes, active


class JvpPass(ir.passes.FunctionalPass):
    """Linearize a model in forward mode, preserving the source model.

    Given selected inputs ``x`` and outputs ``y = f(x)``, produce a new model
    computing ``(all_primal_outputs, J_f(x) @ tangent)``. All original initializers
    and functions are retained; original inputs and primal outputs keep their
    order. Append one tangent input per selected input and one tangent output per
    selected output, in selection order. Seeds must have exactly the corresponding
    primal tensor's runtime shape and dtype; broadcasting seeds is not supported.
    Names use ``<primal>_tangent`` with collision-avoiding numeric suffixes.

    Args:
        inputs: Ordered, unique names of graph inputs or initializers to
            differentiate. By default, select FLOAT and DOUBLE graph inputs that
            are not initializers, in graph-input order. Initializers are held
            constant unless explicitly selected; selecting a parameter does not
            change its stored primal value or promote it to a graph input.
        outputs: Ordered, unique names of graph outputs to differentiate. By
            default, select all FLOAT and DOUBLE outputs in graph-output order.
            Each selection must contain at least one tensor.

    Support contract (also applies to :class:`VjpPass` and :class:`GradPass`):
        * Standard ONNX opset 18 or later; real FLOAT/DOUBLE tensor derivatives
          only. Original opset imports are preserved, not silently upgraded.
          Shape inference runs on a clone using :class:`ShapeInferencePass`.
          Existing symbolic shape metadata is reused; no additional inference
          package or training runtime is required.
          Selected graph inputs/outputs need known ranks so their derivative
          interfaces are valid ONNX; all dimension sizes may be symbolic.
        * Elementwise: Identity, Add, Sub, Mul, Div, Pow, Neg, Abs, Exp, Log,
          Sqrt, Reciprocal, Sin, Cos, Tan, Sinh, Cosh, Tanh, Sigmoid, Relu,
          LeakyRelu, Softplus, Softsign, Erf.
        * Linear algebra: MatMul (vectors, matrices, broadcast batches), Gemm
          (including transposes and broadcast bias).
        * Reductions: ReduceSum and ReduceMean, including negative/runtime axes,
          keepdims, absent/empty axes, and noop_with_empty_axes.
        * Layout/indexing: Reshape, Flatten, Squeeze, Unsqueeze, Transpose,
          Expand, Concat, Split, Gather (including repeated/negative indices).
          Also floating-point Cast, Where, Softmax, and LogSoftmax.
          Pow accepts an inactive integer exponent; floating operands must have
          matching dtypes.
        * Reverse-mode broadcast reduction uses runtime shapes, including
          singleton/leading axes and zero-length dimensions. MatMul, Gather, and
          default Transpose need known ranks, but dimension sizes may be symbolic.
        * Shape, size, axes, indices, split sizes, and boolean conditions are
          structural, not differentiable tensor values. Comparisons are treated
          as fixed branch predicates. Where differentiates only its chosen data
          branch; it does not differentiate the condition.
        * Every unsupported operator on an active path from selected inputs to
          selected outputs raises NotImplementedError, rather than silently
          returning zero. Inactive paths can contain unsupported/custom operators.
          Control flow/subgraphs and random operators (including Dropout) are
          rejected. Inline active local-function calls with InlinePass first.
        * Only first derivatives are supported. Applying an autodiff pass to its
          generated model is rejected. Complex numbers, sequences, sparse tensors,
          integer/boolean casts on active data paths, and half-precision
          differentiation are not supported.
        * Derivatives assume finite inputs within the operators' differentiable
          real domains: in particular, positive bases for Pow, positive inputs
          for Log/Sqrt, nonzero divisors, and nonempty ReduceMean groups.
          Inactive integral Pow exponents also support negative bases and, for
          nonnegative exponents, zero bases (including the ONNX ``0**0 == 1``).
          No domain checks are inserted. At zero, Abs and Relu use derivative
          zero and LeakyRelu uses alpha. Where predicates are fixed, including
          at branch boundaries; no derivative of a discontinuous predicate is
          claimed. Floating Cast uses the usual real-valued identity derivative,
          not the derivative of rounding/overflow in finite-precision arithmetic.

    Example::

        import onnx_ir as ir
        from onnx_ir.passes.common import JvpPass

        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
        tape = ir.tape.Tape()
        y = tape.op("Mul", [x, x])
        y.name = "squared"
        model = ir.Model(ir.Graph([x], [y], nodes=tape.nodes,
                                 opset_imports={"": 18}, name="square"), ir_version=9)
        linearized = JvpPass(inputs=["x"])(model).model
        # Inputs: x, x_tangent; outputs: squared, squared_tangent.
        # At x=[1, 2, 3], tangent=[1, 1, 1], the latter is [2, 4, 6].
        print(model)       # Original graph, unchanged.
        print(linearized)  # Original Mul plus the executable derivative graph.

    See Also:
        ``python tools/examples/autodiff.py`` prints original/derivative graphs
        and executes parameter gradients and prediction sensitivities.
    """

    def __init__(
        self, inputs: Sequence[str] | None = None, outputs: Sequence[str] | None = None
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        result, xs, ys, nodes, _ = _prepare(model, self.inputs, self.outputs)
        b = _Builder(result.graph)
        tangents = {x: b.seed(x, "tangent") for x in xs}
        for node in nodes:
            d = {
                i: tangents[_input(node, i)]
                for i in _data_indices(node)
                if node.inputs[i] in tangents
            }
            values = b.jvp(node, d)
            tangents.update(zip(node.outputs, values))
        for y in ys:
            b.output(tangents[y] if y in tangents else b.like(y), y, "tangent")
        result.graph.metadata_props[_MARKER] = "jvp"
        return ir.passes.PassResult(result, modified=True)


class VjpPass(ir.passes.FunctionalPass):
    """Transpose a model's linearization in reverse mode.

    Compute ``(all_primal_outputs, J_f(x).T @ cotangent)`` in a cloned model.
    Append one cotangent input per selected output, then append one cotangent
    output per selected input. Selection, dtype, shape, operator, and support
    semantics are shared with :class:`JvpPass`. Input/output names use
    ``<primal>_cotangent`` with collision-avoiding suffixes; selection order is
    authoritative. Original inputs/outputs retain their original order.

    Contributions from fan-out, repeated operands, and multiple selected outputs
    are summed. Broadcast contributions are reduced to each input's runtime
    shape. Disconnected selected inputs produce correctly shaped zeros, not
    missing outputs. For example, selecting ``outputs=["prediction", "loss"]``
    appends those two seed inputs in that order, and selecting
    ``inputs=["bias", "weight"]`` appends those two gradients in that order.

    Args:
        inputs: Ordered input/initializer names; see :class:`JvpPass`.
        outputs: Ordered output names; see :class:`JvpPass`.

    Example::

        from onnx_ir.passes.common import VjpPass

        pullback = VjpPass(inputs=["x"], outputs=["squared"])(model).model
        # For the model in JvpPass's example:
        # Inputs: x, squared_cotangent; outputs: squared, x_cotangent.
        # At x=[1, 2, 3], seed=[1, 2, 3], x_cotangent=[2, 8, 18].
    """

    def __init__(
        self, inputs: Sequence[str] | None = None, outputs: Sequence[str] | None = None
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def _transform(self, model: ir.Model, scalar: bool = False) -> ir.passes.PassResult:
        result, xs, ys, nodes, active = _prepare(model, self.inputs, self.outputs)
        if scalar and (len(ys) != 1 or ys[0].shape is None or len(ys[0].shape) != 0):
            raise ValueError("GradPass requires one selected rank-zero (scalar) output.")
        b = _Builder(result.graph)
        cotangents = {y: b.scalar(y, 1) if scalar else b.seed(y, "cotangent") for y in ys}
        for node in reversed(nodes):
            if not any(y in cotangents for y in node.outputs):
                continue
            gs = [cotangents[y] if y in cotangents else b.like(y) for y in node.outputs]
            indices = [i for i in _data_indices(node) if node.inputs[i] in active]
            contributions = b.vjp(node, gs, indices)
            for i in indices:
                x, contribution = _input(node, i), contributions[i]
                cotangents[x] = (
                    b.op("Add", cotangents[x], contribution)
                    if x in cotangents
                    else contribution
                )
        for x in xs:
            b.output(cotangents[x] if x in cotangents else b.like(x), x, "cotangent")
        result.graph.metadata_props[_MARKER] = "grad" if scalar else "vjp"
        return ir.passes.PassResult(result, modified=True)

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return self._transform(model)


class GradPass(VjpPass):
    """Differentiate a scalar loss using a VJP with an implicit unit seed.

    Retain all original inputs and primal outputs, append gradients in selected
    input order, and add no seed inputs. ``output`` must name a rank-zero
    FLOAT/DOUBLE graph output, not a length-one vector. If omitted, the model
    must have exactly one floating-point output, and it must be rank zero.
    Gradient outputs use ``<input>_cotangent`` names as in :class:`VjpPass`.
    See :class:`JvpPass` for the shared support and domain contract.

    Args:
        inputs: Ordered input/initializer names; see :class:`JvpPass`.
        output: Scalar graph output to differentiate.

    Example::

        import onnx_ir as ir
        from onnx_ir.passes.common import GradPass

        prediction = ir.val("prediction", dtype=ir.DataType.DOUBLE, shape=["batch"])
        target = ir.val("target", dtype=ir.DataType.DOUBLE, shape=["batch"])
        tape = ir.tape.Tape()
        error = tape.op("Sub", [prediction, target])
        squared = tape.op("Mul", [error, error])
        loss = tape.op("ReduceMean", [squared], {"keepdims": 0})
        loss.name = "mse"
        model = ir.Model(ir.Graph([prediction, target], [loss], nodes=tape.nodes,
                                 opset_imports={"": 18}, name="mse"), ir_version=9)
        differentiated = GradPass(inputs=["prediction"])(model).model
        # Outputs: mse, prediction_cotangent.
        # The gradient is 2 * (prediction - target) / batch.
        print(model)
        print(differentiated)
    """

    def __init__(self, inputs: Sequence[str] | None = None, output: str | None = None) -> None:
        super().__init__(inputs=inputs, outputs=None if output is None else [output])

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return self._transform(model, scalar=True)
