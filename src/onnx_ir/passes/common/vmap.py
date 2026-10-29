# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lower independent applications of a tensor graph to ordinary batched ONNX nodes."""

from __future__ import annotations

__all__ = ["VmapPass"]

import collections
import string
from collections.abc import Mapping, Sequence

import onnx  # ruff: ignore[banned-api]

import onnx_ir as ir

_UNARY = frozenset(
    {
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitwiseNot",
        "Cast",
        "Ceil",
        "Celu",
        "Cos",
        "Cosh",
        "Elu",
        "Erf",
        "Exp",
        "Floor",
        "Gelu",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "IsInf",
        "IsNaN",
        "Log",
        "LeakyRelu",
        "Mish",
        "Neg",
        "Not",
        "Reciprocal",
        "Relu",
        "Round",
        "Selu",
        "Shrink",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
    }
)
_BINARY = frozenset(
    {
        "Add",
        "And",
        "BitShift",
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
        "Div",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Mul",
        "Mod",
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
_NATIVE_BATCH = frozenset(
    {
        "Conv",
        "ConvTranspose",
        "AveragePool",
        "MaxPool",
        "LpPool",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "GlobalLpPool",
        "InstanceNormalization",
        "BatchNormalization",
        "LRN",
        "GridSample",
        "ConvInteger",
        "QLinearConv",
        "DepthToSpace",
        "SpaceToDepth",
    }
)
_RANDOM = frozenset(
    {
        "Bernoulli",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)
_MULTI_OUTPUT = frozenset({"Split", "TopK", "LayerNormalization", "Dropout"})


def _axis(axis: int, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int) or not -rank <= axis < rank:
        raise ir.passes.PreconditionError(f"Axis {axis!r} is invalid for rank {rank}.")
    return axis % rank


class VmapPass(ir.passes.FunctionalPass):
    """Batch independent applications of a single-example model.

    The source model describes *one* example. This functional pass inserts a
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

    Example:
        Map vectors while sharing a projection matrix::

            import onnx_ir as ir
            from onnx_ir.passes.common import VmapPass

            # Source inputs: x: [D], weight: [D, H]; output: y: [H].
            model = ir.load("single_example.onnx")
            batched = VmapPass(in_axes=(0, None), out_axes=-1)(model).model
            # Batched inputs: x: [B, D], weight: [D, H]; output: y: [H, B].
            # The same model accepts different positive runtime values of B.
            ir.save(batched, "batched.onnx")

        See :doc:`/vmap_examples` for runnable image-classification and
        recommendation examples that save and print the before/after graphs,
        and compare batched execution with independent examples.

    .. rubric:: Semantics

    Internally, mapped values carry a leading batch dimension. Broadcasting
    pads singleton dimensions *after* that dimension, while reductions and
    transpose permutations refer only to example dimensions. Shape and Size
    describe one example and produce shared values internally. Shared outputs
    are repeated when adapting graph outputs.

    Operators with an existing native N dimension, such as Conv and pooling,
    temporarily merge B and N, run the ordinary kernel, and restore both
    dimensions using runtime shapes. Normalization rules must not mix statistics
    across B: BatchNormalization requires inference mode when mapped, while
    InstanceNormalization and LayerNormalization preserve example statistics.
    MatMul, MatMulInteger, and QLinearMatMul promote/squeeze vector operands;
    Gemm lowers to MatMul and arithmetic. Einsum introduces a distinct batch
    label. Indexing and scattering pair matching examples, rather than forming
    cross-example products.

    .. rubric:: Supported operators

    Mapped tensor paths have rules for the following standard-domain families.
    An operator must also exist in the source model's imported opset; the pass
    does not upgrade the opset or add runtime extensions.

    * Unary math and predicates: Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh,
      Ceil, Cos, Cosh, Erf, Exp, Floor, Identity, IsInf, IsNaN, Log, Neg, Not,
      Reciprocal, Round, Sign, Sin, Sinh, Sqrt, Tan, and Cast/CastLike.
    * Activations: Celu, Clip, Elu, Gelu (opset >= 20), HardSigmoid, HardSwish,
      LeakyRelu, Mish, PRelu, Relu, Selu, Shrink, Sigmoid, Softplus, Softsign,
      Tanh, and ThresholdedRelu.
    * Broadcasting arithmetic/logical: Add, And, Div, Equal, Greater,
      GreaterOrEqual, Less, LessOrEqual, Max, Mean, Min, Mul, Or, Pow, Sub,
      Sum, Where, Xor; BitShift, BitwiseAnd, BitwiseNot, BitwiseOr,
      BitwiseXor, and Mod.
    * Linear algebra: MatMul, Gemm, Einsum, Det, MatMulInteger, QLinearMatMul.
    * Image/native-N kernels: Conv, ConvTranspose, ConvInteger, QLinearConv,
      AveragePool, MaxPool (without indices), LpPool, GlobalAveragePool,
      GlobalMaxPool, GlobalLpPool, GridSample, DepthToSpace, SpaceToDepth.
    * Normalization: BatchNormalization (inference), InstanceNormalization,
      LayerNormalization (including statistics outputs),
      MeanVarianceNormalization, LRN, LpNormalization, Softmax, LogSoftmax,
      Hardmax; Dropout in inference mode, including its optional mask.
    * Shapes/layout: Reshape, Expand, Flatten, Squeeze, Unsqueeze, Shape, Size,
      Transpose, Concat, Split, Tile, Slice, Pad, Resize, Trilu, EyeLike,
      ConstantOfShape, tensor-valued Constant and numeric literal Constants.
    * Indexing: Gather, GatherElements, GatherND, ScatterElements, ScatterND,
      OneHot, TopK, ArgMax, ArgMin, CumSum.
    * Reductions: ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax,
      ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare.
    * Quantization: QuantizeLinear and DequantizeLinear with shared scalar or
      per-axis scale/zero-point parameters (not blocked quantization).

    Deterministic operators whose inputs are all shared need no mapped rule:
    they remain shared if their outputs have tensor types and known ranks,
    and their standard ONNX schema is available. This permits, for example,
    shared NonZero results, even though mapped NonZero can produce ragged
    outputs and is rejected. Random operators and subgraphs are still rejected.

    .. rubric:: Controls and restrictions

    * Requires a valid, topologically sorted source graph, standard-domain
      opset >= 18, tensor graph inputs with known ranks, and uniform example
      shapes. Dimensions may be symbolic, unknown, or zero. Empty *batches*
      are outside the input contract. The pass does not perform general shape
      inference or validate the source model.
    * Initializers are shared. Mapped inputs cannot have initializer defaults.
      All mapped inputs must have equal positive batch lengths, even if a
      singleton batch would happen to broadcast. Equality is not asserted
      inside the generated graph.
    * Reduction and Squeeze/Unsqueeze axes must be omitted where allowed, or
      shared constant 1-D INT64 tensors. CumSum's axis must be a shared constant
      integer scalar. Negative axes retain example semantics, as do
      ``keepdims`` and ``noop_with_empty_axes``. Omitted Squeeze axes require
      a static example shape; B=1 never causes the batch axis to be squeezed.
    * Reshape/Expand targets are shared 1-D INT64 tensors of known length;
      values may be runtime inputs. Reshape preserves per-example ``0``,
      ``-1``, and ``allowzero`` semantics. Flatten never folds B into an
      example dimension. ConstantOfShape also needs a shared shape vector
      of known length.
    * Slice controls and Pad controls are shared; explicit axes must be
      constants. Slice without axes needs a known starts-vector length.
      Resize controls, Tile repeats, Split sizes, TopK K, OneHot depth/values,
      and Trilu's diagonal are shared. GatherND needs a known index depth
      or annotated output rank.
    * Conv/native-N parameters, normalization parameters, and quantization
      scales/zero points must be shared; GridSample's data and grid may each
      be mapped or shared. Clip bounds must be shared scalars. Dropout's
      training mode must be omitted or a constant false scalar.
    * Control flow, custom/local function calls, random/training Dropout,
      mapped sequences/optionals, ragged outputs, and operators without a
      mapped rule raise :class:`onnx_ir.passes.PreconditionError`. Inline
      supported local functions first; unused function definitions remain
      unchanged. General nested mapping is not guaranteed.

    Shape annotations are updated using known ranks. Names and graph/model
    metadata are preserved, except directly returned inputs/initializers can
    need renamed output adapters. Failure never modifies the source model.
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
        attributes: Mapping[
            str, ir.Attr | int | float | str | Sequence[int] | ir.TensorProtocol
        ]
        | None = None,
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

    def _get_batch_shape(self) -> ir.Value:
        if self.batch_shape is None:
            if self.axis_size is not None:
                self.batch_shape = self._ints([self.axis_size])
            else:
                assert self.batch_source is not None
                batch_source_shape = self._emit("Shape", [self.batch_source])
                self.batch_shape = self._emit(
                    "Gather", [batch_source_shape, self._ints([0])], {"axis": 0}
                )
        return self.batch_shape

    def _repeat(self, value: ir.Value) -> ir.Value:
        batch_shape = self._get_batch_shape()
        shape = self._emit("Shape", [value])
        target = self._emit("Concat", [batch_shape, shape], {"axis": 0})
        result = self._emit("Expand", [self._unsqueeze(value, [0]), target])
        result.type = value.type
        if value.shape is not None:
            result.shape = ir.Shape([self.axis_size, *value.shape])
        return result

    def _align(self, value: ir.Value, rank: int, mapped: bool, target: int) -> ir.Value:
        # Padding belongs *after* a mapped batch axis, never before it.
        start = 1 if mapped else 0
        return self._unsqueeze(value, list(range(start, target - rank + 1)))

    def _example_shape(self, value: ir.Value) -> ir.Shape:
        shape = self.values[value].shape
        assert shape is not None
        return ir.Shape(list(shape)[int(self.mapped[value]) :])

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

    def _constant_tensor(self, value: ir.Value) -> ir.TensorProtocol | None:
        tensor = None
        if not value.is_graph_input() and not self.mapped.get(value, False):
            if value.is_initializer():
                tensor = value.const_value
            else:
                producer = value.producer()
                if producer is not None and producer.op_type == "Constant":
                    attr = producer.attributes.get("value")
                    if attr is not None and not attr.is_ref():
                        tensor = attr.as_tensor()
        return tensor

    def _constant_ints(self, value: ir.Value, purpose: str) -> list[int]:
        tensor = self._constant_tensor(value)
        if (
            tensor is None
            or tensor.dtype != ir.DataType.INT64
            or len(tensor.shape) != 1
            or self.mapped.get(value, False)
        ):
            raise ir.passes.PreconditionError(
                f"{purpose} must be a shared constant 1-D INT64 tensor, not a runtime input."
            )
        return [int(axis) for axis in tensor.numpy().tolist()]

    def _require_shared(self, values: Sequence[ir.Value], op: str) -> None:
        if any(self.mapped[value] for value in values):
            raise ir.passes.PreconditionError(
                f"{op} requires shared control/parameter inputs."
            )

    def _constant_axis(self, value: ir.Value, rank: int) -> int:
        tensor = self._constant_tensor(value)
        if (
            tensor is None
            or tensor.dtype not in (ir.DataType.INT32, ir.DataType.INT64)
            or len(tensor.shape) != 0
        ):
            raise ir.passes.PreconditionError("Axis must be a shared constant integer scalar.")
        return _axis(int(tensor.numpy().item()), rank)

    def _scalar_int(self, value: int) -> ir.Value:
        return self._emit("Constant", [], {"value": ir.tensor(value, dtype=ir.DataType.INT64)})

    def _cast_int64(self, value: ir.Value) -> ir.Value:
        return self._emit("Cast", [value], {"to": int(ir.DataType.INT64)})

    def _index_aligned(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        rank = self.ranks[inputs[0]]
        if any(self.ranks[value] != rank for value in inputs):
            raise ir.passes.PreconditionError(
                f"{node.op_type} operands must have equal ranks."
            )
        axis = _axis(node.attributes.get_int("axis", 0), rank)
        if any(self.mapped[value] for value in inputs):
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
            for i, value in enumerate(inputs):
                if not self.mapped[value]:
                    node.replace_input_with(i, self._repeat(self.values[value]))
        return rank

    def _gather_nd(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        data, indices = inputs
        batch_dims = node.attributes.get_int("batch_dims", 0)
        index_shape = self._example_shape(indices)
        if not len(index_shape):
            raise ir.passes.PreconditionError("GatherND indices must have rank >= 1.")
        depth = index_shape[-1]
        if isinstance(depth, int):
            rank = self.ranks[data] + self.ranks[indices] - depth - batch_dims - 1
        elif node.outputs[0].shape is not None:
            rank = len(node.outputs[0].shape)
        else:
            raise ir.passes.PreconditionError(
                "GatherND needs a known index depth or output rank."
            )
        if any(self.mapped[value] for value in inputs):
            node.attributes["batch_dims"] = ir.AttrInt64("batch_dims", batch_dims + 1)
            for i, value in enumerate(inputs):
                if not self.mapped[value]:
                    node.replace_input_with(i, self._repeat(self.values[value]))
        return rank

    def _scatter_nd(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        rank = self.ranks[inputs[0]]
        if not any(self.mapped[value] for value in inputs):
            return rank
        values = [
            self.values[value] if self.mapped[value] else self._repeat(self.values[value])
            for value in inputs
        ]
        indices = values[1]
        index_rank = self.ranks[inputs[1]]
        if index_rank < 1:
            raise ir.passes.PreconditionError("ScatterND indices must have rank >= 1.")
        batch = self._emit("Squeeze", [self._get_batch_shape(), self._ints([0])])
        ids = self._emit("Range", [self._scalar_int(0), batch, self._scalar_int(1)])
        id_shape = self._emit(
            "Concat", [self._get_batch_shape(), self._ints([1] * index_rank)], {"axis": 0}
        )
        ids = self._emit("Reshape", [ids, id_shape], {"allowzero": 1})
        prefix = self._emit("Shape", [indices], {"end": index_rank})
        target = self._emit("Concat", [prefix, self._ints([1])], {"axis": 0})
        ids = self._emit("Expand", [ids, target])
        values[1] = self._emit("Concat", [ids, indices], {"axis": -1})
        for i, value in enumerate(values):
            node.replace_input_with(i, value)
        return rank

    def _slice(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        data = inputs[0]
        rank = self.ranks[data]
        self._require_shared(inputs[1:], "Slice")
        if not self.mapped[data]:
            return rank
        axes_value = node.inputs[3] if len(node.inputs) > 3 else None
        if axes_value is not None:
            axes = self._constant_ints(axes_value, "Slice axes")
        else:
            starts_shape = self.values[inputs[1]].shape
            if (
                starts_shape is None
                or len(starts_shape) != 1
                or not isinstance(starts_shape[0], int)
            ):
                raise ir.passes.PreconditionError(
                    "Slice without axes needs a known starts length."
                )
            axes = list(range(starts_shape[0]))
        axes = [_axis(axis, rank) + 1 for axis in axes]
        if len(set(axes)) != len(axes):
            raise ir.passes.PreconditionError("Slice axes must not contain duplicates.")
        for i in (1, 2, 4):
            if i < len(node.inputs):
                control = node.inputs[i]
                if control is not None:
                    node.replace_input_with(i, self._cast_int64(control))
        if len(node.inputs) < 4:
            node.resize_inputs(4)
        node.replace_input_with(3, self._ints(axes))
        return rank

    def _pad(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        data = inputs[0]
        rank = self.ranks[data]
        self._require_shared(inputs[1:], "Pad")
        if not self.mapped[data]:
            return rank
        axes = node.inputs[3] if len(node.inputs) > 3 else None
        if axes is not None:
            shifted = [_axis(axis, rank) + 1 for axis in self._constant_ints(axes, "Pad axes")]
            node.replace_input_with(3, self._ints(shifted))
        else:
            pads = node.inputs[1]
            assert pads is not None
            before = self._emit("Slice", [pads, self._ints([0]), self._ints([rank])])
            after = self._emit("Slice", [pads, self._ints([rank]), self._ints([2 * rank])])
            target = self._emit(
                "Concat", [self._ints([0]), before, self._ints([0]), after], {"axis": 0}
            )
            node.replace_input_with(1, target)
        return rank

    def _einsum(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        equation = node.attributes.get_string("equation").replace(" ", "")
        left, separator, right = equation.partition("->")
        terms = left.split(",")
        if len(terms) != len(inputs):
            raise ir.passes.PreconditionError("Einsum equation must match the input count.")
        if not separator:
            counts = collections.Counter(char for char in left if char in string.ascii_letters)
            right = ("..." if "..." in left else "") + "".join(
                sorted(char for char, count in counts.items() if count == 1)
            )
        ellipsis = max(
            (
                self.ranks[value] - len(term.replace("...", ""))
                for value, term in zip(inputs, terms)
                if "..." in term
            ),
            default=0,
        )
        rank = len(right.replace("...", "")) + (ellipsis if "..." in right else 0)
        if any(self.mapped[value] for value in inputs):
            label = next((char for char in string.ascii_letters if char not in equation), None)
            if label is None:
                raise ir.passes.PreconditionError(
                    "Einsum needs an unused label for the batch axis."
                )
            terms = [
                label + term if self.mapped[value] else term
                for value, term in zip(inputs, terms)
            ]
            node.attributes["equation"] = ir.AttrString(
                "equation", ",".join(terms) + "->" + label + right
            )
        return rank

    def _gemm(self, node: ir.Node, inputs: list[ir.Value]) -> None:
        if any(self.ranks[value] != 2 for value in inputs[:2]):
            raise ir.passes.PreconditionError("Gemm matrix operands must have example rank 2.")
        left, right = [self.values[value] for value in inputs[:2]]
        for i, key in enumerate(("transA", "transB")):
            if node.attributes.get_int(key, 0):
                perm = [0, 2, 1] if self.mapped[inputs[i]] else [1, 0]
                if i == 0:
                    left = self._transpose(left, perm)
                else:
                    right = self._transpose(right, perm)
        product = self._emit("MatMul", [left, right])
        for key, value in (
            ("alpha", product),
            ("beta", self.values[inputs[2]] if len(inputs) == 3 else None),
        ):
            coefficient = node.attributes.get_float(key, 1.0)
            if value is not None and coefficient != 1:
                scalar = self._emit("Constant", [], {"value": ir.tensor(coefficient)})
                scalar = self._emit("CastLike", [scalar, left])
                scaled = self._emit("Mul", [value, scalar])
                if key == "alpha":
                    product = scaled
                else:
                    node.replace_input_with(2, scaled)
        if len(inputs) == 3:
            bias = node.inputs[2]
            assert bias is not None
            bias = self._align(bias, self.ranks[inputs[2]], self.mapped[inputs[2]], 2)
            node.op_type = "Add"
            node.resize_inputs(2)
            node.replace_input_with(1, bias)
        else:
            node.op_type = "Identity"
            node.resize_inputs(1)
        node.attributes.clear()
        node.replace_input_with(0, product)

    def _native_batch(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        data = inputs[0]
        rank = self.ranks[data]
        if rank < 3:
            raise ir.passes.PreconditionError(f"{node.op_type} requires example rank >= 3.")
        if node.op_type in ("DepthToSpace", "SpaceToDepth") and rank != 4:
            raise ir.passes.PreconditionError(f"{node.op_type} requires example rank 4.")
        native_positions = (0, 1) if node.op_type == "GridSample" else (0,)
        self._require_shared(
            [value for i, value in enumerate(inputs) if i not in native_positions],
            node.op_type,
        )
        training_mode = node.attributes.get_int("training_mode", 0)
        if node.op_type == "BatchNormalization" and training_mode:
            raise ir.passes.PreconditionError(
                "Mapped BatchNormalization requires inference mode."
            )
        if not any(self.mapped[value] for value in inputs):
            return rank
        native_size = None
        for i in native_positions:
            value = inputs[i]
            batched = (
                self.values[value] if self.mapped[value] else self._repeat(self.values[value])
            )
            shape = self._emit("Shape", [batched])
            size = self._emit("Gather", [shape, self._ints([1])], {"axis": 0})
            if native_size is None:
                native_size = size
            merged = self._emit("Mul", [self._get_batch_shape(), size])
            tail = self._emit("Shape", [batched], {"start": 2})
            target = self._emit("Concat", [merged, tail], {"axis": 0})
            node.replace_input_with(
                i, self._emit("Reshape", [batched, target], {"allowzero": 1})
            )
        result = self._emit(node.op_type, list(node.inputs), node.attributes)
        tail = self._emit("Shape", [result], {"start": 1})
        assert native_size is not None
        target = self._emit(
            "Concat", [self._get_batch_shape(), native_size, tail], {"axis": 0}
        )
        node.op_type = "Reshape"
        node.attributes.clear()
        node.attributes["allowzero"] = ir.AttrInt64("allowzero", 1)
        node.resize_inputs(2)
        node.replace_input_with(0, result)
        node.replace_input_with(1, target)
        return rank

    def _reduce(self, node: ir.Node, inputs: list[ir.Value], mapped: bool) -> int:
        rank = self.ranks[inputs[0]]
        axes = self._constant_ints(inputs[1], "Reduction axes") if len(inputs) == 2 else []
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

    def _matmul(
        self,
        node: ir.Node,
        inputs: list[ir.Value],
        mapped: bool,
        positions: tuple[int, int] = (0, 1),
    ) -> int:
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
            positions[0], self._align(left_value, max(left_rank, 2), self.mapped[left], target)
        )
        node.replace_input_with(
            positions[1],
            self._align(right_value, max(right_rank, 2), self.mapped[right], target),
        )
        if left_rank == 1 or right_rank == 1:
            squeeze_axes = []
            if left_rank == 1:
                squeeze_axes.append(target - 1)
            if right_rank == 1:
                squeeze_axes.append(target)
            product = self._emit(node.op_type, list(node.inputs), node.attributes)
            product.type = node.outputs[0].type
            product.shape = ir.Shape([self.axis_size, *([None] * target)])
            axes = self._ints(squeeze_axes)
            node.op_type = "Squeeze"
            node.attributes.clear()
            node.resize_inputs(2)
            node.replace_input_with(0, product)
            node.replace_input_with(1, axes)
        return rank

    def _shape_vector_length(self, value: ir.Value) -> int:
        shape = self.values[value].shape
        if (
            self.mapped[value]
            or self.ranks[value] != 1
            or shape is None
            or not isinstance(shape[0], int)
            or value.dtype not in (None, ir.DataType.INT64)
        ):
            raise ir.passes.PreconditionError(
                "Shape targets must be shared 1-D INT64 tensors with statically known lengths."
            )
        return shape[0]

    def _reshape_or_expand(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        data, shape = inputs
        length = self._shape_vector_length(shape)
        rank = length if node.op_type == "Reshape" else max(self.ranks[data], length)
        if self.mapped[data]:
            parts = [self._get_batch_shape()]
            if node.op_type == "Expand":
                if rank > length:
                    parts.append(self._ints([1] * (rank - length)))
                node.replace_input_with(
                    0, self._align(self.values[data], self.ranks[data], True, rank)
                )
            parts.append(self.values[shape])
            node.replace_input_with(1, self._emit("Concat", parts, {"axis": 0}))
        return rank

    def _flatten(self, node: ir.Node, value: ir.Value) -> None:
        rank = self.ranks[value]
        axis = node.attributes.get_int("axis", 1)
        if not -rank <= axis <= rank:
            raise ir.passes.PreconditionError(
                f"Flatten axis {axis} is invalid for rank {rank}."
            )
        if axis < 0:
            axis += rank
        if not self.mapped[value]:
            return
        sizes = [self._get_batch_shape()]
        for start, end in ((1, axis + 1), (axis + 1, rank + 1)):
            part = self._emit("Shape", [self.values[value]], {"start": start, "end": end})
            sizes.append(self._emit("ReduceProd", [part, self._ints([0])], {"keepdims": 1}))
        target = self._emit("Concat", sizes, {"axis": 0})
        node.op_type = "Reshape"
        node.attributes.clear()
        node.attributes["allowzero"] = ir.AttrInt64("allowzero", 1)
        node.resize_inputs(2)
        node.replace_input_with(1, target)

    def _squeeze_or_unsqueeze(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        value = inputs[0]
        rank = self.ranks[value]
        if len(inputs) == 2:
            axes = self._constant_ints(inputs[1], f"{node.op_type} axes")
        else:
            shape = self._example_shape(value)
            if not shape.is_static():
                raise ir.passes.PreconditionError(
                    "Squeeze without axes requires a fully static example shape."
                )
            axes = [i for i, dim in enumerate(shape) if dim == 1]
        output_rank = rank + len(axes) if node.op_type == "Unsqueeze" else rank - len(axes)
        axis_rank = output_rank if node.op_type == "Unsqueeze" else rank
        axes = [_axis(axis, axis_rank) for axis in axes]
        if len(set(axes)) != len(axes):
            raise ir.passes.PreconditionError(
                f"{node.op_type} axes must not contain duplicates."
            )
        if node.outputs[0].shape is None:
            dims = list(self._example_shape(value))
            if node.op_type == "Unsqueeze":
                for axis in sorted(axes):
                    dims.insert(axis, 1)
            else:
                dims = [dim for i, dim in enumerate(dims) if i not in axes]
            node.outputs[0].shape = ir.Shape(dims)
        if self.mapped[value]:
            node.resize_inputs(2)
            node.replace_input_with(1, self._ints([axis + 1 for axis in axes]))
        return output_rank

    def _shape(self, node: ir.Node, value: ir.Value) -> None:
        rank = self.ranks[value]
        bounds = [node.attributes.get_int("start", 0), node.attributes.get_int("end", rank)]
        start, end = [
            min(max(bound + rank if bound < 0 else bound, 0), rank) for bound in bounds
        ]
        if self.mapped[value]:
            node.attributes["start"] = ir.AttrInt64("start", start + 1)
            node.attributes["end"] = ir.AttrInt64("end", end + 1)
        node.outputs[0].dtype = ir.DataType.INT64
        if node.outputs[0].shape is None:
            node.outputs[0].shape = ir.Shape([max(end - start, 0)])

    def _gather(self, node: ir.Node, inputs: list[ir.Value]) -> int:
        data, indices = inputs
        data_rank, index_rank = self.ranks[data], self.ranks[indices]
        axis = _axis(node.attributes.get_int("axis", 0), data_rank)
        rank = data_rank + index_rank - 1
        if node.outputs[0].shape is None:
            dims = list(self._example_shape(data))
            node.outputs[0].shape = ir.Shape(
                [*dims[:axis], *self._example_shape(indices), *dims[axis + 1 :]]
            )
        if not self.mapped[indices]:
            if self.mapped[data]:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
            return rank
        if self.mapped[data]:
            data_perm = [0, axis + 1, *range(1, axis + 1), *range(axis + 2, data_rank + 1)]
            data_value = self._transpose(self.values[data], data_perm)
            indices_value = self._emit(
                "Cast", [self.values[indices]], {"to": int(ir.DataType.INT64)}
            )
            indices_value = self._unsqueeze(indices_value, [index_rank + 1])
            gathered = self._emit("GatherND", [data_value, indices_value], {"batch_dims": 1})
            perm = [
                0,
                *range(index_rank + 1, index_rank + axis + 1),
                *range(1, index_rank + 1),
                *range(index_rank + axis + 1, rank + 1),
            ]
        else:
            gathered = self._emit(
                "Gather", [self.values[data], self.values[indices]], {"axis": axis}
            )
            perm = [axis, *range(axis), *range(axis + 1, rank + 1)]
        node.op_type = "Transpose"
        node.attributes.clear()
        node.attributes["perm"] = ir.AttrInt64s("perm", perm)
        node.resize_inputs(1)
        node.replace_input_with(0, gathered)
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
        inputs = [value for value in node.inputs if value is not None]
        mapped = any(self.mapped[value] for value in inputs)
        op = node.op_type
        if op in _RANDOM:
            raise ir.passes.PreconditionError(
                f"VmapPass has no batching rule for random operator {op}."
            )
        if (
            not mapped
            and op not in ("Constant", "Dropout")
            and all(
                isinstance(value.type, ir.TensorType) and value.shape is not None
                for value in node.outputs
            )
        ):
            try:
                onnx.defs.get_schema(op, self.graph.opset_imports[""], "")
            except onnx.defs.SchemaError as e:
                raise ir.passes.PreconditionError(
                    f"No standard ONNX schema for shared operator {op}."
                ) from e
            for output in node.outputs:
                assert output.shape is not None
                self._register(output, len(output.shape), False)
            return
        if len(node.outputs) != 1 and op not in _MULTI_OUTPUT:
            raise ir.passes.PreconditionError(
                f"VmapPass requires single-output operators for {op}."
            )
        for i, value in enumerate(node.inputs):
            if value is not None:
                node.replace_input_with(i, self.values[value])
        if op == "Constant":
            attr = node.attributes.get("value")
            if attr is None:
                for key in ("value_int", "value_ints", "value_float", "value_floats"):
                    literal = node.attributes.get(key)
                    if literal is None:
                        continue
                    if key == "value_int":
                        tensor = ir.tensor(literal.as_int(), dtype=ir.DataType.INT64)
                    elif key == "value_ints":
                        tensor = ir.tensor(literal.as_ints(), dtype=ir.DataType.INT64)
                    elif key == "value_float":
                        tensor = ir.tensor(literal.as_float(), dtype=ir.DataType.FLOAT)
                    else:
                        tensor = ir.tensor(literal.as_floats(), dtype=ir.DataType.FLOAT)
                    node.attributes.clear()
                    attr = ir.AttrTensor("value", tensor)
                    node.attributes["value"] = attr
                    break
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
            if op == "Gelu" and self.graph.opset_imports[""] < 20:
                raise ir.passes.PreconditionError("Gelu requires standard-domain opset >= 20.")
            rank = self.ranks[inputs[0]]
            if node.outputs[0].shape is None:
                node.outputs[0].shape = self._example_shape(inputs[0])
        elif op == "CastLike" and len(inputs) == 2:
            rank = self.ranks[inputs[0]]
            mapped = self.mapped[inputs[0]]
            if node.outputs[0].shape is None:
                node.outputs[0].shape = self._example_shape(inputs[0])
        elif op == "Clip" and 1 <= len(inputs) <= 3:
            if any(self.mapped[value] or self.ranks[value] != 0 for value in inputs[1:]):
                raise ir.passes.PreconditionError("Clip bounds must be shared scalars.")
            rank = self.ranks[inputs[0]]
        elif op == "PRelu" and len(inputs) == 2:
            data, slope = inputs
            rank = self.ranks[data]
            if self.ranks[slope] > rank:
                raise ir.passes.PreconditionError(
                    "PRelu slope rank must not exceed data rank."
                )
            if mapped:
                if not self.mapped[data]:
                    node.replace_input_with(0, self._repeat(self.values[data]))
                node.replace_input_with(
                    1,
                    self._align(
                        self.values[slope], self.ranks[slope], self.mapped[slope], rank
                    ),
                )
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
        elif op == "MatMulInteger" and 2 <= len(inputs) <= 4:
            self._require_shared(inputs[2:], op)
            rank = self._matmul(node, inputs[:2], mapped)
        elif op == "QLinearMatMul" and len(inputs) == 8:
            self._require_shared(
                [value for i, value in enumerate(inputs) if i not in (0, 3)], op
            )
            rank = self._matmul(node, [inputs[0], inputs[3]], mapped, positions=(0, 3))
        elif op in ("QuantizeLinear", "DequantizeLinear") and 2 <= len(inputs) <= 3:
            self._require_shared(inputs[1:], op)
            rank = self.ranks[inputs[0]]
            block_size = node.attributes.get_int("block_size", 0)
            if self.ranks[inputs[1]] > 1 or block_size:
                raise ir.passes.PreconditionError(
                    f"{op} supports shared scalar or per-axis parameters, not blocked quantization."
                )
            if mapped and self.ranks[inputs[1]] == 1:
                axis = _axis(node.attributes.get_int("axis", 1), rank)
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        elif op == "Gemm" and 2 <= len(inputs) <= 3:
            self._gemm(node, inputs)
            rank = 2
        elif op == "Einsum" and inputs:
            rank = self._einsum(node, inputs)
        elif op in _NATIVE_BATCH and inputs:
            rank = self._native_batch(node, inputs)
        elif op in ("Resize", "Trilu", "LpNormalization") and inputs:
            self._require_shared(inputs[1:], op)
            rank = self.ranks[inputs[0]]
            if op == "Resize":
                axes = node.attributes.get_ints("axes", range(rank))
                axes = [_axis(axis, rank) for axis in axes]
                if mapped:
                    node.attributes["axes"] = ir.AttrInt64s(
                        "axes", [axis + 1 for axis in axes]
                    )
            elif op == "LpNormalization":
                axis = _axis(node.attributes.get_int("axis", -1), rank)
                if mapped:
                    node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
            elif rank < 2:
                raise ir.passes.PreconditionError("Trilu requires example rank >= 2.")
        elif op == "LayerNormalization" and inputs:
            self._require_shared(inputs[1:], op)
            rank = self.ranks[inputs[0]]
            axis = _axis(node.attributes.get_int("axis", -1), rank)
            if mapped:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        elif op == "MeanVarianceNormalization" and len(inputs) == 1:
            rank = self.ranks[inputs[0]]
            axes = [_axis(axis, rank) for axis in node.attributes.get_ints("axes", (0, 2, 3))]
            if mapped:
                node.attributes["axes"] = ir.AttrInt64s("axes", [axis + 1 for axis in axes])
        elif op == "Det" and len(inputs) == 1:
            rank = self.ranks[inputs[0]] - 2
            if rank < 0:
                raise ir.passes.PreconditionError("Det requires example rank >= 2.")
        elif op in ("Split", "TopK") and inputs:
            self._require_shared(inputs[1:], op)
            rank = self.ranks[inputs[0]]
            axis = _axis(node.attributes.get_int("axis", -1 if op == "TopK" else 0), rank)
            if mapped:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        elif op == "Dropout" and inputs:
            training = node.inputs[2] if len(node.inputs) > 2 else None
            if training is not None:
                training_tensor = self._constant_tensor(training)
                if (
                    training_tensor is None
                    or training_tensor.dtype != ir.DataType.BOOL
                    or len(training_tensor.shape) != 0
                    or bool(training_tensor.numpy().item())
                ):
                    raise ir.passes.PreconditionError(
                        "Dropout requires a constant false or omitted training_mode."
                    )
            rank = self.ranks[inputs[0]]
            mapped = self.mapped[inputs[0]]
            node.resize_inputs(1)
        elif op == "CumSum" and len(inputs) == 2:
            rank = self.ranks[inputs[0]]
            axis = self._constant_axis(inputs[1], rank)
            if mapped:
                node.replace_input_with(1, self._scalar_int(axis + 1))
        elif op == "Tile" and len(inputs) == 2:
            self._require_shared(inputs[1:], op)
            rank = self.ranks[inputs[0]]
            if mapped:
                target = self._emit(
                    "Concat", [self._ints([1]), self.values[inputs[1]]], {"axis": 0}
                )
                node.replace_input_with(1, target)
        elif op == "Slice" and 3 <= len(inputs) <= 5:
            rank = self._slice(node, inputs)
        elif op == "Pad" and 2 <= len(inputs) <= 4:
            rank = self._pad(node, inputs)
        elif op == "OneHot" and len(inputs) == 3:
            self._require_shared(inputs[1:], op)
            rank = self.ranks[inputs[0]] + 1
            axis = _axis(node.attributes.get_int("axis", -1), rank)
            if mapped:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        elif op == "ConstantOfShape" and len(inputs) == 1:
            rank = self._shape_vector_length(inputs[0])
            mapped = False
        elif op == "EyeLike" and len(inputs) == 1:
            rank = self.ranks[inputs[0]]
            if rank != 2:
                raise ir.passes.PreconditionError("EyeLike requires example rank 2.")
            if mapped:
                shape_value = self._emit("Shape", [self.values[inputs[0]]], {"start": 1})
                stub = self._emit("ConstantOfShape", [shape_value])
                stub = self._emit("CastLike", [stub, self.values[inputs[0]]])
                node.replace_input_with(0, stub)
            mapped = False
        elif op == "GatherElements" and len(inputs) == 2:
            rank = self._index_aligned(node, inputs)
        elif op == "ScatterElements" and len(inputs) == 3:
            rank = self._index_aligned(node, inputs)
        elif op == "GatherND" and len(inputs) == 2:
            rank = self._gather_nd(node, inputs)
        elif op == "ScatterND" and len(inputs) == 3:
            rank = self._scatter_nd(node, inputs)
        elif op in ("Reshape", "Expand") and len(inputs) == 2:
            rank = self._reshape_or_expand(node, inputs)
        elif op == "Flatten" and len(inputs) == 1:
            self._flatten(node, inputs[0])
            rank = 2
        elif (op == "Squeeze" and 1 <= len(inputs) <= 2) or (
            op == "Unsqueeze" and len(inputs) == 2
        ):
            rank = self._squeeze_or_unsqueeze(node, inputs)
        elif op == "Shape" and len(inputs) == 1:
            self._shape(node, inputs[0])
            rank, mapped = 1, False
        elif op == "Size" and len(inputs) == 1:
            if mapped:
                shape = self._emit("Shape", [self.values[inputs[0]]], {"start": 1})
                node.op_type = "ReduceProd"
                node.attributes.clear()
                node.attributes["keepdims"] = ir.AttrInt64("keepdims", 0)
                node.resize_inputs(2)
                node.replace_input_with(0, shape)
                node.replace_input_with(1, self._ints([0]))
            rank, mapped = 0, False
        elif op == "Gather" and len(inputs) == 2:
            rank = self._gather(node, inputs)
        elif op in ("ArgMax", "ArgMin") and len(inputs) == 1:
            rank = self.ranks[inputs[0]]
            axis = _axis(node.attributes.get_int("axis", 0), rank)
            if mapped:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
            keepdims = node.attributes.get_int("keepdims", 1)
            rank -= int(not keepdims)
        elif op == "Transpose" and len(inputs) == 1:
            rank = self.ranks[inputs[0]]
            perm = list(node.attributes.get_ints("perm", range(rank - 1, -1, -1)))
            if sorted(perm) != list(range(rank)):
                raise ir.passes.PreconditionError("Transpose perm must be a rank permutation.")
            if mapped:
                node.attributes["perm"] = ir.AttrInt64s("perm", [0, *(i + 1 for i in perm)])
        elif op in ("Softmax", "LogSoftmax", "Hardmax") and len(inputs) == 1:
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
            if node.outputs[0].shape is None:
                output_shape = self._example_shape(inputs[0])
                sizes = [self._example_shape(value)[axis] for value in inputs]
                output_shape[axis] = (
                    sum(size for size in sizes if isinstance(size, int))
                    if all(isinstance(size, int) for size in sizes)
                    else None
                )
                node.outputs[0].shape = output_shape
            if mapped:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
                for i, value in enumerate(inputs):
                    if not self.mapped[value]:
                        node.replace_input_with(i, self._repeat(self.values[value]))
        else:
            raise ir.passes.PreconditionError(
                f"VmapPass has no batching rule for {op!r} with {len(inputs)} inputs."
            )
        for output in node.outputs:
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
