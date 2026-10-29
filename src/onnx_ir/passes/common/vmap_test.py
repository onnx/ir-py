# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Semantic tests for graph-level vectorization."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

import onnx_ir as ir
from onnx_ir.passes.common import VmapPass


def _model(op, shapes, attributes=None, *, dtype=ir.DataType.FLOAT, opset=18):
    inputs = [
        ir.val(f"x{i}", dtype=ir.DataType.FLOAT, shape=shape) for i, shape in enumerate(shapes)
    ]
    node = ir.node(op, inputs, attributes)
    node.outputs[0].name = "y"
    node.outputs[0].dtype = dtype
    model = ir.Model(
        ir.Graph(
            inputs, node.outputs, nodes=[node], opset_imports={"": opset}, name="example"
        ),
        ir_version=10,
    )
    return ir.from_proto(onnx.shape_inference.infer_shapes(ir.to_proto(model)))


def _data(shape, offset=0):
    return np.asarray(
        (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1 + offset) / 10,
        dtype=np.float32,
    )


def _equivalent(model, in_axes, feeds, *, out_axes=0, size=3):
    before = ir.to_proto(model).SerializeToString()
    result = VmapPass(in_axes, axis_size=size, out_axes=out_axes)(model)
    assert result.modified
    assert result.model is not model
    assert ir.to_proto(model).SerializeToString() == before
    assert [value.name for value in result.model.graph.inputs] == [
        value.name for value in model.graph.inputs
    ]
    for old, new in zip(model.graph.outputs, result.model.graph.outputs):
        if not old.is_graph_input() and not old.is_initializer():
            assert new.name == old.name
    proto = ir.to_proto(result.model)
    onnx.checker.check_model(proto, full_check=True)
    roundtrip = ir.to_proto(ir.from_proto(onnx.load_from_string(proto.SerializeToString())))
    actual = ReferenceEvaluator(roundtrip).run(None, feeds)
    evaluator = ReferenceEvaluator(ir.to_proto(model))
    examples = [
        evaluator.run(
            None,
            {
                value.name: (
                    feeds[value.name]
                    if axis is None
                    else np.take(feeds[value.name], i, axis=axis)
                )
                for value, axis in zip(model.graph.inputs, in_axes)
            },
        )
        for i in range(size)
    ]
    axes = [out_axes] * len(actual) if isinstance(out_axes, int) else out_axes
    for j, (output, axis) in enumerate(zip(actual, axes)):
        expected = np.stack([example[j] for example in examples], axis=axis)
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=1e-6)
    assert not any(node.op_type in ("Loop", "Scan") for node in result.model.graph)
    return result.model


@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Sum"])
@pytest.mark.parametrize("in_axes", [(0, None), (None, -1), (-1, 1)])
def test_elementwise_broadcasting(op, in_axes):
    model = _model(op, [[], [2, 4]])
    feeds = {}
    for i, (shape, axis) in enumerate(zip([[], [2, 4]], in_axes)):
        data = _data([3, *shape], i) if axis is not None else _data(shape, i)
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(model, in_axes, feeds, out_axes=-1)


def test_mean():
    # ONNX's reference Mean uses in-place addition and cannot broadcast shapes.
    _equivalent(
        _model("Mean", [[2, 4], [2, 4]]),
        [0, 0],
        {"x0": _data([3, 2, 4]), "x1": _data([3, 2, 4], 1)},
    )


@pytest.mark.parametrize(
    "op",
    [
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
        "Reciprocal",
        "Relu",
        "Round",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sqrt",
        "Tan",
        "Tanh",
    ],
)
def test_unary(op):
    attrs = {"to": int(ir.DataType.FLOAT)} if op == "Cast" else None
    dtype = ir.DataType.BOOL if op in ("IsInf", "IsNaN") else ir.DataType.FLOAT
    model = _model(op, [[2, 4]], attrs, dtype=dtype)
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, out_axes=2)


def test_where_and_boolean_operators():
    model = _model("Where", [[2, 4], [], [4]])
    model.graph.inputs[0].dtype = ir.DataType.BOOL
    _equivalent(
        model,
        [1, -1, None],
        {"x0": _data([2, 3, 4]) > 1, "x1": _data([3]), "x2": _data([4])},
    )
    for op in ("And", "Or", "Xor"):
        model = _model(op, [[2, 4], [4]], dtype=ir.DataType.BOOL)
        for value in model.graph.inputs:
            value.dtype = ir.DataType.BOOL
        _equivalent(model, [0, None], {"x0": _data([3, 2, 4]) > 1, "x1": _data([4]) > 0.2})
    model = _model("Not", [[4]], dtype=ir.DataType.BOOL)
    model.graph.inputs[0].dtype = ir.DataType.BOOL
    _equivalent(model, [-1], {"x0": _data([4, 3]) > 0.5})


@pytest.mark.parametrize("op", ["Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"])
def test_comparisons(op):
    _equivalent(
        _model(op, [[2, 4], []], dtype=ir.DataType.BOOL),
        [0, None],
        {"x0": _data([3, 2, 4]), "x1": np.array(0.5, dtype=np.float32)},
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ([4], [4]),
        ([4], [4, 2]),
        ([2, 4], [4]),
        ([2, 4], [4, 5]),
        ([2, 3, 4], [4]),
        ([4], [2, 4, 5]),
        ([2, 3, 4], [4, 5]),
        ([2, 1, 3, 4], [2, 4, 5]),
        ([3, 4], [2, 4, 5]),
    ],
)
@pytest.mark.parametrize("in_axes", [(0, None), (None, -1), (-1, 1)])
def test_matmul(left, right, in_axes):
    model = _model("MatMul", [left, right])
    feeds = {}
    for i, (shape, axis) in enumerate(zip([left, right], in_axes)):
        data = _data([3, *shape], i) if axis is not None else _data(shape, i)
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(model, in_axes, feeds, out_axes=-1)


@pytest.mark.parametrize(
    "op",
    [
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
    ],
)
@pytest.mark.parametrize("axes", [None, [], [0], [-1], [0, 1]])
@pytest.mark.parametrize("keepdims", [0, 1])
def test_reductions(op, axes, keepdims):
    model = _model(op, [[2, 4]], {"keepdims": keepdims})
    node = model.graph[0]
    if axes is not None:
        value = ir.val("axes", const_value=ir.tensor(axes, dtype=ir.DataType.INT64))
        value.type = ir.TensorType(ir.DataType.INT64)
        value.shape = ir.Shape([len(axes)])
        model.graph.register_initializer(value)
        node.resize_inputs(2)
        node.replace_input_with(1, value)
    model.graph.outputs[0].shape = None
    model = ir.from_proto(onnx.shape_inference.infer_shapes(ir.to_proto(model)))
    _equivalent(model, [-1], {"x0": _data([2, 4, 3])})


@pytest.mark.parametrize(
    "op",
    [
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    ],
)
def test_scalar_reductions(op):
    _equivalent(_model(op, [[]]), [0], {"x0": _data([3])})


def test_scalar_reduce_log_sum_exp():
    # The reference evaluator's scalar ReduceLogSumExp raises TypeError.
    model = _model("ReduceLogSumExp", [[]])
    result = VmapPass([0], axis_size=3)(model).model
    proto = ir.to_proto(result)
    onnx.checker.check_model(proto, full_check=True)
    data = _data([3])
    actual = ReferenceEvaluator(proto).run(None, {"x0": data})[0]
    np.testing.assert_allclose(actual, data, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("axes", [None, []])
def test_noop_reduction(axes):
    model = _model("ReduceSum", [[2, 4]], {"noop_with_empty_axes": 1})
    if axes is not None:
        node = model.graph[0]
        const = ir.node("Constant", [], {"value": ir.tensor(axes, dtype=ir.DataType.INT64)})
        model.graph.insert_before(node, const)
        node.resize_inputs(2)
        node.replace_input_with(1, const.outputs[0])
    _equivalent(model, [0], {"x0": _data([3, 2, 4])})


@pytest.mark.parametrize("perm", [None, [1, 2, 0], [0, 2, 1]])
def test_transpose(perm):
    attrs = {"perm": perm} if perm is not None else None
    _equivalent(
        _model("Transpose", [[2, 4, 5]], attrs), [-2], {"x0": _data([2, 4, 3, 5])}, out_axes=1
    )


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax(op, axis):
    _equivalent(_model(op, [[2, 4]], {"axis": axis}), [1], {"x0": _data([2, 3, 4])})


@pytest.mark.parametrize("axis", [0, -1])
def test_concat(axis):
    shapes = [[2, 4], [1, 4]] if axis == 0 else [[2, 4], [2, 1]]
    _equivalent(
        _model("Concat", shapes, {"axis": axis}),
        [-1, None],
        {"x0": _data([2, 4, 3]), "x1": _data(shapes[1])},
    )


def test_composed_graph_with_missing_intermediate_shapes():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["length", 4])
    tape = ir.tape.Tape()
    weight = tape.initializer(ir.tensor(_data([4, 5]), name="weight"))
    y = tape.op("MatMul", [x, weight])
    y = tape.op(
        "Relu", [tape.op("Add", [y, tape.op("Constant", [], {"value": ir.tensor(0.1)})])]
    )
    y = tape.op("Transpose", [y])
    y = tape.op("ReduceMean", [y], {"keepdims": 0})
    y.name = "result"
    y.dtype = ir.DataType.FLOAT
    y.shape = ir.Shape([])
    model = ir.Model(
        ir.Graph(
            [x],
            [y],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="composed",
        ),
        ir_version=10,
    )
    _equivalent(model, [0], {"x": _data([3, 2, 4])})


def test_shared_outputs_and_per_output_axes():
    model = _model("Add", [[2, 4], [4]])
    shared = ir.node("Neg", [model.graph.inputs[1]])
    shared.outputs[0].name = "shared"
    shared.outputs[0].type = model.graph.inputs[1].type
    shared.outputs[0].shape = ir.Shape([4])
    model.graph.append(shared)
    model.graph.outputs.append(shared.outputs[0])
    constant = ir.node("Constant", [], {"value": ir.tensor([2.0, 3.0])})
    constant.outputs[0].name = "constant"
    constant.outputs[0].dtype = ir.DataType.FLOAT
    constant.outputs[0].shape = ir.Shape([2])
    model.graph.append(constant)
    model.graph.outputs.append(constant.outputs[0])
    _equivalent(
        model, [1, None], {"x0": _data([2, 3, 4]), "x1": _data([4])}, out_axes=[2, -1, 0]
    )


def test_shared_matmul_and_reduction():
    model = _model("MatMul", [[2, 4], [4, 5]])
    # The third input is mapped but unused, so the whole computation is shared.
    model.graph.inputs.append(ir.val("x2", dtype=ir.DataType.FLOAT, shape=[]))
    _equivalent(
        model,
        [None, None, 0],
        {"x0": _data([2, 4]), "x1": _data([4, 5]), "x2": _data([3])},
    )
    model = _model("ReduceMean", [[2, 4]])
    model.graph.inputs.append(ir.val("x1", dtype=ir.DataType.FLOAT, shape=[]))
    _equivalent(model, [None, 0], {"x0": _data([2, 4]), "x1": _data([3])})


@pytest.mark.parametrize("in_axis,out_axis", [(0, 0), (1, 0), (1, 1)])
def test_direct_input_output(in_axis, out_axis):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])
    model = ir.Model(
        ir.Graph([x], [x], nodes=[], opset_imports={"": 18}, name="direct"), ir_version=10
    )
    data = np.moveaxis(_data([3, 4]), 0, in_axis)
    _equivalent(model, [in_axis], {"x": data}, out_axes=out_axis)


@pytest.mark.parametrize("size", [1, 4])
def test_axis_size(size):
    _equivalent(_model("Identity", [[2]]), [0], {"x0": _data([size, 2])}, size=size)


def test_empty_example_dimension():
    _equivalent(
        _model("Add", [[0, 4], [4]]), [0, None], {"x0": _data([3, 0, 4]), "x1": _data([4])}
    )


@pytest.mark.parametrize(
    "op", ["Reshape", "Dropout", "RandomNormalLike", "NonZero", "If", "Gather"]
)
def test_unsupported_operators_leave_source_unchanged(op):
    model = _model("Identity", [[4]])
    model.graph[0].op_type = op
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises(ir.passes.PreconditionError, match="batching rule"):
        VmapPass([0], axis_size=3)(model)
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize("axes", [[0], [0, 1]])
def test_dynamic_reduction_axes_rejected(axes):
    model = _model("ReduceSum", [[2, 4]], {"keepdims": 1})
    axes_value = ir.val("axes", dtype=ir.DataType.INT64, shape=[len(axes)])
    model.graph.inputs.append(axes_value)
    model.graph[0].resize_inputs(2)
    model.graph[0].replace_input_with(1, axes_value)
    with pytest.raises(ir.passes.PreconditionError, match="shared constant"):
        VmapPass([0, None], axis_size=3)(model)


@pytest.mark.parametrize(
    "in_axes,out_axes,match",
    [
        ([], 0, "one entry"),
        ([None], 0, "At least"),
        ([2], 0, "Axis"),
        ([True], 0, "Axis"),
        ([0], 2, "Axis"),
        ([0], [], "one entry"),
        ([0], None, ""),
    ],
)
def test_invalid_axes(in_axes, out_axes, match):
    if out_axes is None:
        with pytest.raises(TypeError):
            VmapPass(in_axes, axis_size=3, out_axes=out_axes)
        return
    with pytest.raises(ir.passes.PreconditionError, match=match):
        VmapPass(in_axes, axis_size=3, out_axes=out_axes)(_model("Identity", [[4]]))


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_axis_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        VmapPass([0], axis_size=size)


def test_invalid_model_preconditions():
    model = _model("Identity", [[4]])
    model.graph.opset_imports[""] = 13
    with pytest.raises(ir.passes.PreconditionError, match="opset"):
        VmapPass([0], axis_size=3)(model)
    model.graph.opset_imports[""] = 18
    model.graph.inputs[0].shape = None
    with pytest.raises(ir.passes.PreconditionError, match="known ranks"):
        VmapPass([0], axis_size=3)(model)
    model.graph.inputs[0].shape = ir.Shape([4])
    model.graph[0].domain = "custom"
    with pytest.raises(ir.passes.PreconditionError, match="custom/function"):
        VmapPass([0], axis_size=3)(model)


def test_mapped_initializer_default_rejected():
    model = _model("Identity", [[4]])
    model.graph.inputs[0].const_value = ir.tensor(_data([4]))
    model.graph.register_initializer(model.graph.inputs[0])
    with pytest.raises(ir.passes.PreconditionError, match="initializer defaults"):
        VmapPass([0], axis_size=3)(model)


def test_vector_matmul_consumers_and_metadata():
    model = _model("MatMul", [[4], [4]])
    dot = model.graph.outputs[0]
    bias = ir.val(
        "bias", dtype=ir.DataType.FLOAT, shape=[2, 4], const_value=ir.tensor(_data([2, 4]))
    )
    model.graph.register_initializer(bias)
    addition = ir.node("Add", [dot, bias])
    addition.outputs[0].dtype = ir.DataType.FLOAT
    addition.outputs[0].shape = ir.Shape([2, 4])
    addition.outputs[0].name = "addition"
    model.graph.append(addition)
    model.graph.outputs.append(addition.outputs[0])
    model.metadata_props["model-key"] = "model-value"
    model.graph.metadata_props["graph-key"] = "graph-value"
    result = _equivalent(
        model, [1, None], {"x0": _data([4, 3]), "x1": _data([4])}, out_axes=[0, 1]
    )
    assert result.metadata_props == model.metadata_props
    assert result.graph.metadata_props == model.graph.metadata_props


def test_shared_initializer_and_input_outputs():
    model = _model("Identity", [[4], [2]])
    model.graph[0].resize_inputs(1)
    model.graph.outputs.append(model.graph.inputs[1])
    initializer = ir.val(
        "weight", dtype=ir.DataType.FLOAT, shape=[2], const_value=ir.tensor(_data([2]))
    )
    model.graph.register_initializer(initializer)
    model.graph.outputs.append(initializer)
    _equivalent(model, [0, None], {"x0": _data([3, 4]), "x1": _data([2])}, out_axes=[0, 1, 0])


def test_shared_overridable_initializer():
    model = _model("Add", [[2, 4], [4]])
    shared = model.graph.inputs[1]
    shared.const_value = ir.tensor(_data([4]))
    model.graph.register_initializer(shared)
    _equivalent(model, [0, None], {"x0": _data([3, 2, 4]), "x1": _data([4], 1)})


@pytest.mark.parametrize("op", ["MatMul", "Concat", "Transpose", "Softmax", "ReduceSum"])
def test_shared_operator_branches(op):
    shapes = [[2, 4], [4, 2]] if op == "MatMul" else [[2, 4]]
    attrs = {"axis": 0} if op == "Concat" else None
    model = _model(op, shapes, attrs)
    model.graph.inputs.append(ir.val("unused", dtype=ir.DataType.FLOAT, shape=[]))
    feeds = {f"x{i}": _data(shape) for i, shape in enumerate(shapes)}
    feeds["unused"] = _data([3])
    _equivalent(model, [*[None] * len(shapes), 0], feeds)


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean"])
def test_constant_node_reduction_axes(op):
    model = _model(op, [[2, 4]], {"keepdims": 0})
    node = model.graph[0]
    axes = ir.node("Constant", [], {"value": ir.tensor([-1], dtype=ir.DataType.INT64)})
    model.graph.insert_before(node, axes)
    node.resize_inputs(2)
    node.replace_input_with(1, axes.outputs[0])
    node.outputs[0].shape = ir.Shape([2])
    _equivalent(model, [1], {"x0": _data([2, 3, 4])})


def test_overridable_reduction_axes_rejected():
    model = _model("ReduceSum", [[2, 4]])
    axes = ir.val("axes", dtype=ir.DataType.INT64, shape=[1], const_value=ir.tensor([0]))
    model.graph.register_initializer(axes)
    model.graph.inputs.append(axes)
    model.graph[0].resize_inputs(2)
    model.graph[0].replace_input_with(1, axes)
    with pytest.raises(ir.passes.PreconditionError, match="shared constant"):
        VmapPass([0, None], axis_size=3)(model)


def test_subgraph_and_multiple_outputs_rejected():
    model = _model("Identity", [[4]])
    model.graph[0].attributes["body"] = ir.AttrGraph("body", model.graph.clone())
    with pytest.raises(ir.passes.PreconditionError, match="subgraphs"):
        VmapPass([0], axis_size=3)(model)
    model.graph[0].attributes.clear()
    extra = ir.node("Dropout", [model.graph.inputs[0]], num_outputs=2)
    model.graph.append(extra)
    with pytest.raises(ir.passes.PreconditionError, match="single-output"):
        VmapPass([0], axis_size=3)(model)


def test_local_functions_rejected_but_unused_definitions_preserved():
    model = _model("Identity", [[4]])
    body = _model("Neg", [[4]]).graph
    function = ir.Function("local", "MyFunction", graph=body, attributes=[])
    model.functions[function.identifier()] = function
    result = _equivalent(model, [0], {"x0": _data([3, 4])})
    assert ir.serde.serialize_function(result.functions[function.identifier()]) == (
        ir.serde.serialize_function(function)
    )
    model.graph[0].domain = "local"
    model.graph[0].op_type = "MyFunction"
    with pytest.raises(ir.passes.PreconditionError, match="local function calls"):
        VmapPass([0], axis_size=3)(model)


def test_tensor_inputs_and_constant_values_required():
    model = _model("Identity", [[4]])
    model.graph.inputs[0].type = ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
    with pytest.raises(ir.passes.PreconditionError, match="tensors"):
        VmapPass([0], axis_size=3)(model)
    model.graph.inputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    node = ir.node("Constant", [], {"value_int": 1})
    model.graph.append(node)
    with pytest.raises(ir.passes.PreconditionError, match="tensor-valued Constant"):
        VmapPass([0], axis_size=3)(model)


def test_non_topological_graph_rejected_without_mutation():
    model = _model("Identity", [[4]])
    later = ir.node("Neg", [model.graph.inputs[0]])
    model.graph[0].replace_input_with(0, later.outputs[0])
    model.graph.append(later)
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises(ir.passes.PreconditionError, match="topologically sorted"):
        VmapPass([0], axis_size=3)(model)
    assert ir.to_proto(model).SerializeToString() == before
