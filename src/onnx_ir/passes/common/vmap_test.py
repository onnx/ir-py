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


def _model(
    op,
    shapes,
    attributes=None,
    *,
    dtype=ir.DataType.FLOAT,
    opset=18,
    input_dtypes=None,
    num_outputs=1,
    output_dtypes=None,
):
    input_dtypes = input_dtypes or [ir.DataType.FLOAT] * len(shapes)
    inputs = [
        ir.val(f"x{i}", dtype=input_dtype, shape=shape)
        for i, (shape, input_dtype) in enumerate(zip(shapes, input_dtypes))
    ]
    node = ir.node(op, inputs, attributes, num_outputs=num_outputs)
    for i, output in enumerate(node.outputs):
        output.name = "y" if i == 0 else f"y{i}"
        output.dtype = output_dtypes[i] if output_dtypes else dtype
    model = ir.Model(
        ir.Graph(
            inputs, node.outputs, nodes=[node], opset_imports={"": opset}, name="example"
        ),
        ir_version=10,
    )
    return ir.from_proto(onnx.shape_inference.infer_shapes(ir.to_proto(model)))


def _constant_model(
    op, shape, constants, attributes=None, *, num_outputs=1, output_dtypes=None
):
    x = ir.val("x0", dtype=ir.DataType.FLOAT, shape=shape)
    initializers = []
    for i, value in enumerate(constants):
        if value is None:
            initializers.append(None)
        else:
            tensor = ir.tensor(value)
            initializers.append(
                ir.val(
                    f"constant{i}", const_value=tensor, dtype=tensor.dtype, shape=value.shape
                )
            )
    node = ir.node(op, [x, *initializers], attributes, num_outputs=num_outputs)
    for i, output in enumerate(node.outputs):
        output.name = "y" if i == 0 else f"y{i}"
        output.dtype = output_dtypes[i] if output_dtypes else ir.DataType.FLOAT
    model = ir.Model(
        ir.Graph(
            [x],
            node.outputs,
            nodes=[node],
            initializers=[value for value in initializers if value is not None],
            opset_imports={"": 18},
            name="constant_parameters",
        ),
        ir_version=10,
    )
    return ir.from_proto(onnx.shape_inference.infer_shapes(ir.to_proto(model)))


def _data(shape, offset=0):
    return np.asarray(
        (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1 + offset) / 10,
        dtype=np.float32,
    )


def _equivalent(model, in_axes, feeds, *, out_axes=0, size=3, dynamic=False, ort=False):
    before = ir.to_proto(model).SerializeToString()
    result = VmapPass(in_axes, axis_size=None if dynamic else size, out_axes=out_axes)(model)
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
    if ort:
        onnxruntime = pytest.importorskip("onnxruntime")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1

        def evaluator_factory(proto):
            return onnxruntime.InferenceSession(
                proto.SerializeToString(),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
    else:
        evaluator_factory = ReferenceEvaluator
    actual = evaluator_factory(roundtrip).run(None, feeds)
    evaluator = evaluator_factory(ir.to_proto(model))
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
@pytest.mark.parametrize("dynamic", [False, True])
def test_elementwise_broadcasting(op, in_axes, dynamic):
    model = _model(op, [[], [2, 4]])
    feeds = {}
    for i, (shape, axis) in enumerate(zip([[], [2, 4]], in_axes)):
        data = _data([3, *shape], i) if axis is not None else _data(shape, i)
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(model, in_axes, feeds, out_axes=-1, dynamic=dynamic)


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
@pytest.mark.parametrize("dynamic", [False, True])
def test_matmul(left, right, in_axes, dynamic):
    model = _model("MatMul", [left, right])
    feeds = {}
    for i, (shape, axis) in enumerate(zip([left, right], in_axes)):
        data = _data([3, *shape], i) if axis is not None else _data(shape, i)
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(model, in_axes, feeds, out_axes=-1, dynamic=dynamic)


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
@pytest.mark.parametrize("dynamic", [False, True])
def test_reductions(op, axes, keepdims, dynamic):
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
    _equivalent(model, [-1], {"x0": _data([2, 4, 3])}, dynamic=dynamic)


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


@pytest.mark.parametrize("dynamic", [False, True])
def test_shared_outputs_and_per_output_axes(dynamic):
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
        model,
        [1, None],
        {"x0": _data([2, 3, 4]), "x1": _data([4])},
        out_axes=[2, -1, 0],
        dynamic=dynamic,
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_shared_matmul_and_reduction(dynamic):
    model = _model("MatMul", [[2, 4], [4, 5]])
    # The third input is mapped but unused, so the whole computation is shared.
    model.graph.inputs.append(ir.val("x2", dtype=ir.DataType.FLOAT, shape=[]))
    _equivalent(
        model,
        [None, None, 0],
        {"x0": _data([2, 4]), "x1": _data([4, 5]), "x2": _data([3])},
        dynamic=dynamic,
    )
    model = _model("ReduceMean", [[2, 4]])
    model.graph.inputs.append(ir.val("x1", dtype=ir.DataType.FLOAT, shape=[]))
    _equivalent(model, [None, 0], {"x0": _data([2, 4]), "x1": _data([3])}, dynamic=dynamic)


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
    "op", ["RNN", "Bernoulli", "RandomNormalLike", "NonZero", "If", "Compress"]
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
    extra = ir.node("MaxPool", [model.graph.inputs[0]], {"kernel_shape": [2]}, num_outputs=2)
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
    node = ir.node("Constant", [])
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


def test_dynamic_model_reused_at_different_batch_sizes():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 4])
    scalar = ir.val("scalar", dtype=ir.DataType.FLOAT, shape=[])
    shared = ir.val("shared", dtype=ir.DataType.FLOAT, shape=[2, 5])
    tape = ir.tape.Tape()
    weight = tape.initializer(ir.tensor(_data([4, 5]), name="weight"))
    y = tape.op("MatMul", [x, weight])
    y = tape.op("Add", [y, scalar])
    y = tape.op("Relu", [y])
    y = tape.op("Softmax", [y], {"axis": -1})
    y = tape.op("Concat", [y, shared], {"axis": 0})
    y = tape.op("Transpose", [y])
    axes = tape.op("Constant", [], {"value": ir.tensor([-1], dtype=ir.DataType.INT64)})
    y = tape.op("ReduceMean", [y, axes], {"keepdims": 0})
    y.name = "result"
    y.dtype = ir.DataType.FLOAT
    y.shape = ir.Shape([5])
    model = ir.Model(
        ir.Graph(
            [x, scalar, shared],
            [y, weight, shared],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="dynamic",
        ),
        ir_version=10,
    )
    before = ir.to_proto(model).SerializeToString()
    transformed = VmapPass([1, 0, None], out_axes=[1, 2, 0])(model).model
    assert ir.to_proto(model).SerializeToString() == before
    assert transformed.graph.inputs[0].shape == [2, None, 4]
    proto = ir.to_proto(transformed)
    onnx.checker.check_model(proto, full_check=True)
    evaluator = ReferenceEvaluator(
        ir.to_proto(ir.from_proto(onnx.load_from_string(proto.SerializeToString())))
    )
    single_example = ReferenceEvaluator(ir.to_proto(model))
    for size in (1, 3, 7):
        feeds = {"x": _data([2, size, 4]), "scalar": _data([size]), "shared": _data([2, 5])}
        actual = evaluator.run(None, feeds)
        examples = [
            single_example.run(
                None,
                {
                    "x": feeds["x"][:, i],
                    "scalar": feeds["scalar"][i],
                    "shared": feeds["shared"],
                },
            )
            for i in range(size)
        ]
        for j, axis in enumerate([1, 2, 0]):
            expected = np.stack([example[j] for example in examples], axis=axis)
            np.testing.assert_allclose(actual[j], expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize(
    "op",
    [
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Celu",
        "Cosh",
        "Elu",
        "Gelu",
        "HardSigmoid",
        "HardSwish",
        "LeakyRelu",
        "Mish",
        "Selu",
        "Shrink",
        "Sinh",
        "Softplus",
        "Softsign",
        "ThresholdedRelu",
    ],
)
def test_additional_unary_operators(op):
    data = np.linspace(-0.8, 0.8, 24, dtype=np.float32).reshape(2, 3, 4)
    if op == "Acosh":
        data = np.abs(data) + 1
    model = _model(op, [[2, 4]], opset=20 if op == "Gelu" else 18)
    _equivalent(model, [1], {"x0": data}, out_axes=-1, dynamic=True)


@pytest.mark.parametrize("op", ["BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitShift"])
def test_additional_integer_binary_operators(op):
    attrs = {"direction": "LEFT"} if op == "BitShift" else None
    model = _model(
        op,
        [[2, 4], [4]],
        attrs,
        dtype=ir.DataType.UINT32,
        input_dtypes=[ir.DataType.UINT32, ir.DataType.UINT32],
    )
    _equivalent(
        model,
        [1, None],
        {
            "x0": np.arange(24, dtype=np.uint32).reshape(2, 3, 4),
            "x1": np.arange(4, dtype=np.uint32),
        },
        dynamic=True,
    )


def test_bitwise_not_and_mod():
    model = _model(
        "BitwiseNot", [[4]], dtype=ir.DataType.INT32, input_dtypes=[ir.DataType.INT32]
    )
    _equivalent(model, [0], {"x0": np.arange(12, dtype=np.int32).reshape(3, 4)}, dynamic=True)
    model = _model("Mod", [[2, 4], []], {"fmod": 1})
    _equivalent(
        model,
        [0, None],
        {"x0": _data([3, 2, 4]), "x1": np.array(0.3, dtype=np.float32)},
        dynamic=True,
    )


@pytest.mark.parametrize("in_axes", [(0, None), (None, -1), (-1, 0)])
@pytest.mark.parametrize("slope_shape", [[], [4]])
def test_prelu(in_axes, slope_shape):
    shapes = [[2, 4], slope_shape]
    feeds = {}
    for i, (shape, axis) in enumerate(zip(shapes, in_axes)):
        data = _data([3, *shape]) - 1 if axis is not None else _data(shape) - 1
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(_model("PRelu", shapes), in_axes, feeds, dynamic=True)


@pytest.mark.parametrize("in_axes", [(0, None), (None, 0), (0, 0)])
def test_cast_like(in_axes):
    model = _model(
        "CastLike",
        [[2, 4], []],
        dtype=ir.DataType.DOUBLE,
        input_dtypes=[ir.DataType.FLOAT, ir.DataType.DOUBLE],
    )
    feeds = {
        "x0": _data([3, 2, 4]) if in_axes[0] is not None else _data([2, 4]),
        "x1": np.zeros([3] if in_axes[1] is not None else [], dtype=np.float64),
    }
    _equivalent(model, in_axes, feeds, dynamic=True)


@pytest.mark.parametrize(
    "bounds",
    [
        [np.array(-0.5, dtype=np.float32)],
        [np.array(-0.5, dtype=np.float32), np.array(0.5, dtype=np.float32)],
        [None, np.array(0.5, dtype=np.float32)],
    ],
)
def test_clip(bounds):
    model = _constant_model("Clip", [2, 4], bounds)
    _equivalent(model, [1], {"x0": _data([2, 3, 4]) - 1}, dynamic=True)


@pytest.mark.parametrize("op", ["ArgMax", "ArgMin"])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("last", [0, 1])
def test_argument_reductions(op, axis, keepdims, last):
    model = _model(
        op,
        [[2, 4]],
        {"axis": axis, "keepdims": keepdims, "select_last_index": last},
        dtype=ir.DataType.INT64,
    )
    _equivalent(model, [1], {"x0": np.ones([2, 3, 4], dtype=np.float32)}, dynamic=True)


@pytest.mark.parametrize(
    "shape,target,attrs",
    [
        ([2, 4], [-1], {}),
        ([2, 4], [4, 2], {}),
        ([2, 4], [0, -1], {}),
        ([1, 1], [], {}),
        ([0, 4], [0, 4], {"allowzero": 1}),
    ],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_reshape(shape, target, attrs, dynamic):
    model = _constant_model("Reshape", shape, [np.array(target, dtype=np.int64)], attrs)
    _equivalent(model, [-1], {"x0": _data([*shape, 3])}, dynamic=dynamic, out_axes=-1)


@pytest.mark.parametrize(
    "shape,target",
    [
        ([], [2, 4]),
        ([4], [2, 4]),
        ([2, 4], [4]),
        ([2, 4], []),
        ([2, 4], [1]),
        ([], []),
        ([0, 4], [0, 4]),
    ],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_expand(shape, target, dynamic):
    model = _constant_model("Expand", shape, [np.array(target, dtype=np.int64)])
    _equivalent(model, [-1], {"x0": _data([*shape, 3])}, dynamic=dynamic, out_axes=-1)


@pytest.mark.parametrize(
    "shape,axis",
    [
        ([2, 4], 0),
        ([2, 4], -1),
        ([2, 4], 2),
        ([0, 4], 0),
        ([], 0),
    ],
)
def test_flatten(shape, axis):
    model = _model("Flatten", [shape], {"axis": axis})
    _equivalent(model, [-1], {"x0": _data([*shape, 3])}, dynamic=True, out_axes=-1)


def test_flatten_empty_left_partition():
    # The reference evaluator's Flatten uses -1 and fails for an empty left partition.
    model = _model("Flatten", [[0, 4]], {"axis": 1})
    result = VmapPass([0])(model).model
    proto = ir.to_proto(result)
    onnx.checker.check_model(proto, full_check=True)
    actual = ReferenceEvaluator(proto).run(None, {"x0": _data([3, 0, 4])})[0]
    np.testing.assert_array_equal(actual, np.empty([3, 0, 4], dtype=np.float32))


@pytest.mark.parametrize(
    "op,shape,axes",
    [
        ("Squeeze", [1, 2, 1, 4], None),
        ("Squeeze", [1, 2, 1, 4], [0, -2]),
        ("Squeeze", [1, 2, 1, 4], []),
        ("Squeeze", [], None),
        ("Unsqueeze", [2, 4], [0, -1]),
        ("Unsqueeze", [], [0]),
        ("Unsqueeze", [2, 4], []),
    ],
)
@pytest.mark.parametrize("size", [1, 3])
def test_squeeze_unsqueeze(op, shape, axes, size):
    constants = [] if axes is None else [np.array(axes, dtype=np.int64)]
    model = _constant_model(op, shape, constants)
    _equivalent(model, [-1], {"x0": _data([*shape, size])}, dynamic=True, size=size)


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"start": 1},
        {"start": -2, "end": -1},
        {"start": -10, "end": 10},
        {"start": 2, "end": 0},
    ],
)
@pytest.mark.parametrize("shape", [[2, 4, 5], []])
def test_shape(attrs, shape):
    model = _model("Shape", [shape], attrs, dtype=ir.DataType.INT64)
    _equivalent(model, [-1], {"x0": _data([*shape, 3])}, dynamic=True)


@pytest.mark.parametrize("shape", [[2, 4], [0, 4], []])
def test_size(shape):
    model = _model("Size", [shape], dtype=ir.DataType.INT64)
    _equivalent(model, [-1], {"x0": _data([*shape, 3])}, dynamic=True)


@pytest.mark.parametrize("in_axes", [(0, None), (None, -1), (-1, 0)])
@pytest.mark.parametrize("index_shape", [[], [2], [2, 1]])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("index_dtype", [ir.DataType.INT32, ir.DataType.INT64])
def test_gather(in_axes, index_shape, axis, index_dtype):
    model = _model(
        "Gather",
        [[2, 4, 5], index_shape],
        {"axis": axis},
        input_dtypes=[ir.DataType.FLOAT, index_dtype],
    )
    data = _data([3, 2, 4, 5]) if in_axes[0] is not None else _data([2, 4, 5])
    index_dims = [3, *index_shape] if in_axes[1] is not None else index_shape
    indices = (np.arange(np.prod(index_dims)).reshape(index_dims) % 2 - 1).astype(
        index_dtype.numpy()
    )
    feeds = {
        "x0": np.moveaxis(data, 0, in_axes[0]) if in_axes[0] is not None else data,
        "x1": np.moveaxis(indices, 0, in_axes[1]) if in_axes[1] is not None else indices,
    }
    _equivalent(model, in_axes, feeds, dynamic=True, out_axes=-1)


def test_shape_gather_unsqueeze_concat_reshape_chain_without_annotations():
    x = ir.val("x0", dtype=ir.DataType.FLOAT, shape=["height", 3, 4])
    tape = ir.tape.Tape()
    shape = tape.op("Shape", [x], {"start": -2})
    index = tape.op("Constant", [], {"value": ir.tensor(np.array(0, dtype=np.int64))})
    dimension = tape.op("Gather", [shape, index])
    axes = tape.op("Constant", [], {"value": ir.tensor([0], dtype=ir.DataType.INT64)})
    dimension = tape.op("Unsqueeze", [dimension, axes])
    infer = tape.op("Constant", [], {"value": ir.tensor([-1], dtype=ir.DataType.INT64)})
    target = tape.op("Concat", [dimension, infer], {"axis": 0})
    y = tape.op("Reshape", [x, target])
    y.name = "y"
    y.dtype = ir.DataType.FLOAT
    y.shape = ir.Shape([3, None])
    model = ir.Model(
        ir.Graph([x], [y], nodes=tape.nodes, opset_imports={"": 18}, name="shape_chain"),
        ir_version=10,
    )
    _equivalent(model, [1], {"x0": _data([2, 3, 3, 4])}, dynamic=True)


@pytest.mark.parametrize("op", ["Reshape", "Expand"])
def test_runtime_shared_shape_target(op):
    model = _model(op, [[4], [2]], input_dtypes=[ir.DataType.FLOAT, ir.DataType.INT64])
    _equivalent(
        model,
        [1, None],
        {"x0": _data([4, 3]), "x1": np.array([1, 4], dtype=np.int64)},
        dynamic=True,
    )


@pytest.mark.parametrize("op", ["Reshape", "Expand"])
def test_mapped_and_unknown_length_shapes_rejected(op):
    model = _model(op, [[4], [2]], input_dtypes=[ir.DataType.FLOAT, ir.DataType.INT64])
    with pytest.raises(ir.passes.PreconditionError, match="Shape targets must be shared"):
        VmapPass([0, 0])(model)
    model.graph.inputs[1].shape = ir.Shape([None])
    with pytest.raises(ir.passes.PreconditionError, match="known lengths"):
        VmapPass([0, None])(model)


def test_ambiguous_squeeze_and_runtime_axes_rejected():
    model = _model("Squeeze", [[None, 4]])
    with pytest.raises(ir.passes.PreconditionError, match="fully static"):
        VmapPass([0])(model)
    model = _model(
        "Unsqueeze", [[4], [1]], input_dtypes=[ir.DataType.FLOAT, ir.DataType.INT64]
    )
    with pytest.raises(ir.passes.PreconditionError, match="shared constant"):
        VmapPass([0, None])(model)


def test_mapped_clip_bounds_and_old_gelu_rejected():
    model = _model("Clip", [[4], []])
    with pytest.raises(ir.passes.PreconditionError, match="shared scalars"):
        VmapPass([0, 0])(model)
    model = _model("Gelu", [[4]], opset=20)
    model.graph.opset_imports[""] = 18
    with pytest.raises(ir.passes.PreconditionError, match="opset >= 20"):
        VmapPass([0])(model)


@pytest.mark.parametrize(
    "op,shape,constants,attrs",
    [
        ("Reshape", [2, 4], [np.array([4, 2], dtype=np.int64)], None),
        ("Expand", [4], [np.array([2, 4], dtype=np.int64)], None),
        ("Flatten", [2, 4], [], {"axis": 1}),
        ("Squeeze", [1, 2, 1], [], None),
        ("Unsqueeze", [2, 4], [np.array([-1], dtype=np.int64)], None),
        ("Gather", [2, 4], [np.array([1, 0], dtype=np.int64)], {"axis": 1}),
        ("PRelu", [2, 4], [np.array(0.2, dtype=np.float32)], None),
        ("Clip", [2, 4], [np.array(0.5, dtype=np.float32)], None),
    ],
)
def test_additional_shared_branches(op, shape, constants, attrs):
    model = _constant_model(op, shape, constants, attrs)
    model.graph.inputs.append(ir.val("unused", dtype=ir.DataType.FLOAT, shape=[]))
    _equivalent(
        model,
        [None, 0],
        {"x0": _data(shape), "unused": _data([3])},
        dynamic=True,
    )


@pytest.mark.parametrize("op", ["Shape", "Size", "ArgMax", "ArgMin"])
def test_additional_shared_integer_output_branches(op):
    model = _model(op, [[2, 4]], dtype=ir.DataType.INT64)
    model.graph.inputs.append(ir.val("unused", dtype=ir.DataType.FLOAT, shape=[]))
    _equivalent(model, [None, 0], {"x0": _data([2, 4]), "unused": _data([3])}, dynamic=True)


def test_runtime_shape_targets_and_batch_size_change_without_retransforming():
    model = _model("Reshape", [[4], [2]], input_dtypes=[ir.DataType.FLOAT, ir.DataType.INT64])
    transformed = VmapPass([1, None])(model).model
    proto = ir.to_proto(transformed)
    onnx.checker.check_model(proto, full_check=True)
    evaluator = ReferenceEvaluator(proto)
    for size, target in [(1, [2, 2]), (3, [1, 4]), (5, [4, 1])]:
        data = _data([4, size])
        actual = evaluator.run(None, {"x0": data, "x1": np.array(target, dtype=np.int64)})[0]
        expected = np.stack([data[:, i].reshape(target) for i in range(size)])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("op", ["Conv", "ConvTranspose"])
@pytest.mark.parametrize("in_axis", [0, -1])
def test_native_convolution(op, in_axis):
    shapes = (
        [[2, 1, 5, 5], [2, 1, 3, 3], [2]]
        if op == "Conv"
        else [[2, 1, 3, 3], [1, 2, 2, 2], [2]]
    )
    model = _model(op, shapes, {"strides": [2, 2]})
    feeds = {
        "x0": np.moveaxis(_data([3, *shapes[0]]), 0, in_axis),
        "x1": _data(shapes[1]),
        "x2": _data(shapes[2]),
    }
    _equivalent(model, [in_axis, None, None], feeds, dynamic=True, out_axes=-1)


@pytest.mark.parametrize(
    "op",
    [
        "AveragePool",
        "MaxPool",
        "LpPool",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "GlobalLpPool",
    ],
)
def test_native_pooling(op):
    attrs = {} if op.startswith("Global") else {"kernel_shape": [2, 2], "strides": [2, 2]}
    _equivalent(
        _model(op, [[2, 2, 4, 4]], attrs),
        [1],
        {"x0": _data([2, 3, 2, 4, 4])},
        dynamic=True,
        ort=op == "GlobalLpPool",
    )


@pytest.mark.parametrize("op", ["InstanceNormalization", "BatchNormalization", "LRN"])
def test_native_normalization(op):
    shape = [2, 2, 3, 3]
    count = 2 if op == "InstanceNormalization" else 4 if op == "BatchNormalization" else 0
    attrs = {"size": 3} if op == "LRN" else None
    model = _model(op, [shape, *[[2]] * count], attrs)
    feeds = {"x0": _data([3, *shape])}
    feeds.update({f"x{i + 1}": _data([2]) + 0.5 for i in range(count)})
    _equivalent(model, [0, *[None] * count], feeds, dynamic=True)


@pytest.mark.parametrize("in_axes", [(0, None), (None, 1), (-1, 0)])
def test_grid_sample(in_axes):
    shapes = [[2, 1, 4, 4], [2, 3, 3, 2]]
    feeds = {}
    for i, (shape, axis) in enumerate(zip(shapes, in_axes)):
        data = _data([3, *shape]) if axis is not None else _data(shape)
        if i == 1:
            data = (data % 2) - 1
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(_model("GridSample", shapes), in_axes, feeds, dynamic=True)


@pytest.mark.parametrize("axis", [0, -1])
def test_layer_normalization_with_statistics(axis):
    scale_shape = [2, 4] if axis == 0 else [4]
    model = _model(
        "LayerNormalization",
        [[2, 4], scale_shape, scale_shape],
        {"axis": axis},
        num_outputs=3,
    )
    _equivalent(
        model,
        [1, None, None],
        {"x0": _data([2, 3, 4]), "x1": _data(scale_shape) + 0.5, "x2": _data(scale_shape)},
        dynamic=True,
        out_axes=[2, 0, 1],
    )


@pytest.mark.parametrize(
    "in_axes", [(0, None, None), (None, 1, None), (-1, 0, 0), (None, None, 0)]
)
@pytest.mark.parametrize("transposed", [False, True])
def test_gemm(in_axes, transposed):
    shapes = [[4, 2], [3, 4], [3]] if transposed else [[2, 4], [4, 3], [3]]
    attrs = {"alpha": 0.7, "beta": 0.3, "transA": int(transposed), "transB": int(transposed)}
    feeds = {}
    for i, (shape, axis) in enumerate(zip(shapes, in_axes)):
        data = _data([3, *shape]) if axis is not None else _data(shape)
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(_model("Gemm", shapes, attrs), in_axes, feeds, dynamic=True)


@pytest.mark.parametrize(
    "equation,shapes",
    [
        ("ij,jk", [[2, 4], [4, 5]]),
        ("...ij,...jk->...ik", [[2, 3, 4], [1, 4, 5]]),
        ("ij,ij->", [[2, 4], [2, 4]]),
        (",i->i", [[], [4]]),
    ],
)
@pytest.mark.parametrize("in_axes", [(0, None), (None, -1), (-1, 0)])
def test_einsum(equation, shapes, in_axes):
    feeds = {}
    for i, (shape, axis) in enumerate(zip(shapes, in_axes)):
        data = _data([3, *shape]) if axis is not None else _data(shape)
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(_model("Einsum", shapes, {"equation": equation}), in_axes, feeds, dynamic=True)


def test_einsum_diagonal_and_gemm_without_bias():
    _equivalent(
        _model("Einsum", [[4, 4]], {"equation": "ii"}),
        [0],
        {"x0": _data([3, 4, 4])},
        dynamic=True,
    )
    _equivalent(
        _model("Gemm", [[2, 4], [4, 5]], {"alpha": 0.5}),
        [0, None],
        {"x0": _data([3, 2, 4]), "x1": _data([4, 5])},
        dynamic=True,
    )


@pytest.mark.parametrize("axes", [None, [-1]])
def test_resize(axes):
    attrs = {"mode": "nearest"}
    if axes is not None:
        attrs["axes"] = axes
    target = np.array([4, 6] if axes is None else [6], dtype=np.int64)
    model = _constant_model("Resize", [2, 3], [None, None, target], attrs)
    _equivalent(model, [1], {"x0": _data([2, 3, 3])}, dynamic=True, ort=axes is not None)


@pytest.mark.parametrize("shape,repeats", [([2, 3], [2, 1]), ([], []), ([2, 3], [0, 2])])
def test_tile(shape, repeats):
    model = _constant_model("Tile", shape, [np.array(repeats, dtype=np.int64)])
    _equivalent(model, [-1], {"x0": _data([*shape, 3])}, dynamic=True)


@pytest.mark.parametrize(
    "parameters",
    [
        [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)],
        [
            np.array([-1], dtype=np.int64),
            np.array([-10], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            np.array([-1], dtype=np.int64),
        ],
    ],
)
def test_slice(parameters):
    model = _constant_model("Slice", [2, 4], parameters)
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, dynamic=True)


@pytest.mark.parametrize(
    "pads,axes,mode",
    [
        ([1, 0, 1, 0], None, "constant"),
        ([1, 2], [-1], "edge"),
        ([0, -1, 0, -1], None, "constant"),
        ([1, 0, 1, 0], None, "reflect"),
    ],
)
def test_pad(pads, axes, mode):
    constants = [np.array(pads, dtype=np.int64)]
    if axes is not None:
        constants += [None, np.array(axes, dtype=np.int64)]
    model = _constant_model("Pad", [2, 4], constants, {"mode": mode})
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, dynamic=True)


def test_cumsum_trilu_lp_normalization():
    model = _constant_model(
        "CumSum", [2, 4], [np.array(-1, dtype=np.int32)], {"exclusive": 1, "reverse": 1}
    )
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, dynamic=True)
    model = _constant_model("Trilu", [2, 4], [np.array(1, dtype=np.int64)], {"upper": 0})
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, dynamic=True)
    _equivalent(
        _model("LpNormalization", [[2, 4]], {"axis": 0, "p": 2}),
        [1],
        {"x0": _data([2, 3, 4])},
        dynamic=True,
    )


def test_split_topk_and_inference_dropout_multiple_outputs():
    model = _constant_model(
        "Split",
        [2, 4],
        [np.array([1, 3], dtype=np.int64)],
        {"axis": -1},
        num_outputs=2,
    )
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, dynamic=True, out_axes=[2, 0])
    model = _constant_model(
        "TopK",
        [2, 4],
        [np.array([2], dtype=np.int64)],
        {"axis": -1},
        num_outputs=2,
        output_dtypes=[ir.DataType.FLOAT, ir.DataType.INT64],
    )
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, dynamic=True)
    model = _constant_model(
        "Dropout",
        [2, 4],
        [None, np.array(False)],
        num_outputs=2,
        output_dtypes=[ir.DataType.FLOAT, ir.DataType.BOOL],
    )
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, dynamic=True)


@pytest.mark.parametrize("in_axes", [(0, None), (None, -1), (-1, 0)])
def test_gather_elements(in_axes):
    model = _model(
        "GatherElements",
        [[2, 4], [2, 3]],
        {"axis": 1},
        input_dtypes=[ir.DataType.FLOAT, ir.DataType.INT64],
    )
    data = _data([3, 2, 4]) if in_axes[0] is not None else _data([2, 4])
    shape = [3, 2, 3] if in_axes[1] is not None else [2, 3]
    indices = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) % 4
    feeds = {
        "x0": np.moveaxis(data, 0, in_axes[0]) if in_axes[0] is not None else data,
        "x1": np.moveaxis(indices, 0, in_axes[1]) if in_axes[1] is not None else indices,
    }
    _equivalent(model, in_axes, feeds, dynamic=True)


@pytest.mark.parametrize("op", ["ScatterElements", "ScatterND"])
@pytest.mark.parametrize(
    "in_axes", [(0, None, None), (None, 0, None), (None, None, 0), (0, 0, 0)]
)
def test_scatter(op, in_axes):
    shapes = [[2, 4], [2, 2], [2, 2]] if op == "ScatterElements" else [[3, 4], [2, 1], [2, 4]]
    attrs = (
        {"axis": 1, "reduction": "add"} if op == "ScatterElements" else {"reduction": "add"}
    )
    model = _model(
        op,
        shapes,
        attrs,
        input_dtypes=[ir.DataType.FLOAT, ir.DataType.INT64, ir.DataType.FLOAT],
    )
    feeds = {}
    for i, (shape, axis) in enumerate(zip(shapes, in_axes)):
        dims = [3, *shape] if axis is not None else shape
        data = (
            np.arange(np.prod(dims), dtype=np.int64).reshape(dims) % 2
            if i == 1
            else _data(dims)
        )
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(model, in_axes, feeds, dynamic=True)


@pytest.mark.parametrize("batch_dims", [0, 1])
@pytest.mark.parametrize("in_axes", [(0, None), (None, 0), (0, 0)])
def test_gather_nd(batch_dims, in_axes):
    shapes = [[2, 3, 4], [2, 2, 1] if batch_dims else [2, 1]]
    model = _model(
        "GatherND",
        shapes,
        {"batch_dims": batch_dims},
        input_dtypes=[ir.DataType.FLOAT, ir.DataType.INT64],
    )
    feeds = {}
    for i, (shape, axis) in enumerate(zip(shapes, in_axes)):
        dims = [3, *shape] if axis is not None else shape
        data = (
            np.arange(np.prod(dims), dtype=np.int64).reshape(dims) % 2
            if i == 1
            else _data(dims)
        )
        feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
    _equivalent(model, in_axes, feeds, dynamic=True)


@pytest.mark.parametrize("axis", [0, -1])
def test_one_hot(axis):
    model = _constant_model(
        "OneHot",
        [2],
        [np.array(3, dtype=np.int64), np.array([-0.5, 0.5], dtype=np.float32)],
        {"axis": axis},
    )
    model.graph.inputs[0].dtype = ir.DataType.INT64
    _equivalent(
        model, [0], {"x0": np.arange(6, dtype=np.int64).reshape(3, 2) % 3}, dynamic=True
    )


def test_constant_of_shape_eye_like_and_literal_constants():
    model = _model("EyeLike", [[2, 4]])
    _equivalent(model, [1], {"x0": _data([2, 3, 4])}, dynamic=True)
    shape = ir.val("shape", dtype=ir.DataType.INT64, shape=[2])
    unused = ir.val("unused", dtype=ir.DataType.FLOAT, shape=[])
    node = ir.node("ConstantOfShape", [shape])
    node.outputs[0].name = "y"
    node.outputs[0].dtype = ir.DataType.FLOAT
    model = ir.Model(
        ir.Graph(
            [shape, unused], node.outputs, nodes=[node], opset_imports={"": 18}, name="fill"
        ),
        ir_version=10,
    )
    _equivalent(
        model,
        [None, 0],
        {"shape": np.array([2, 4], dtype=np.int64), "unused": _data([3])},
        dynamic=True,
    )
    for key, literal, dtype, dims in (
        ("value_int", 2, ir.DataType.INT64, []),
        ("value_ints", [2, 3], ir.DataType.INT64, [2]),
        ("value_float", 0.5, ir.DataType.FLOAT, []),
        ("value_floats", [0.5, 1.0], ir.DataType.FLOAT, [2]),
    ):
        model = _model("Identity", [[4]])
        const = ir.node("Constant", [], {key: literal})
        const.outputs[0].name = "literal"
        const.outputs[0].dtype = dtype
        const.outputs[0].shape = ir.Shape(dims)
        model.graph.append(const)
        model.graph.outputs.append(const.outputs[0])
        _equivalent(model, [0], {"x0": _data([3, 4])}, dynamic=True)


@pytest.mark.parametrize(
    "op,dtype", [("Det", ir.DataType.FLOAT), ("NonZero", ir.DataType.INT64)]
)
def test_shared_deterministic_schema_fallback(op, dtype):
    model = _model(op, [[2, 2]], dtype=dtype)
    model.graph.inputs.append(ir.val("unused", dtype=ir.DataType.FLOAT, shape=[]))
    _equivalent(model, [None, 0], {"x0": _data([2, 2]), "unused": _data([3])}, dynamic=True)


def test_unsafe_native_parameters_and_training_rejected():
    model = _model("Conv", [[1, 1, 4, 4], [1, 1, 2, 2]])
    with pytest.raises(ir.passes.PreconditionError, match="shared control/parameter"):
        VmapPass([0, 0])(model)
    model = _constant_model("Dropout", [4], [None, np.array(True)])
    with pytest.raises(ir.passes.PreconditionError, match="training_mode"):
        VmapPass([0])(model)
    model = _model(
        "BatchNormalization", [[1, 1, 4, 4], [1], [1], [1], [1]], {"training_mode": 1}
    )
    with pytest.raises(ir.passes.PreconditionError, match="inference mode"):
        VmapPass([0, None, None, None, None])(model)


@pytest.mark.parametrize("op", ["Hardmax", "MeanVarianceNormalization", "Det"])
def test_additional_normalization_and_determinant(op):
    shapes = [[2, 2]] if op == "Det" else [[2, 2, 3, 4]]
    _equivalent(_model(op, shapes), [1], {"x0": _data([2, 3, *shapes[0][1:]])}, dynamic=True)


@pytest.mark.parametrize("op", ["DepthToSpace", "SpaceToDepth"])
def test_space_depth_rearrangement(op):
    shape = [2, 4, 4, 4]
    _equivalent(
        _model(op, [shape], {"blocksize": 2}),
        [1],
        {"x0": _data([2, 3, 4, 4, 4])},
        dynamic=True,
    )


@pytest.mark.parametrize("op", ["QuantizeLinear", "DequantizeLinear"])
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_quantization(op, axis):
    quantize = op == "QuantizeLinear"
    scale_shape = [] if axis is None else [2 if axis == 0 else 4]
    model = _model(
        op,
        [[2, 4], scale_shape, scale_shape],
        {} if axis is None else {"axis": axis},
        input_dtypes=[
            ir.DataType.FLOAT if quantize else ir.DataType.UINT8,
            ir.DataType.FLOAT,
            ir.DataType.UINT8,
        ],
        dtype=ir.DataType.UINT8 if quantize else ir.DataType.FLOAT,
    )
    data = _data([2, 3, 4])
    _equivalent(
        model,
        [1, None, None],
        {
            "x0": data if quantize else (data * 10).astype(np.uint8),
            "x1": np.full(scale_shape, 0.1, dtype=np.float32),
            "x2": np.full(scale_shape, 1, dtype=np.uint8),
        },
        dynamic=True,
        ort=not quantize,
    )


@pytest.mark.parametrize("op", ["MatMulInteger", "QLinearMatMul"])
@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize("mapped", [(0, None), (None, -1), (0, -1)])
def test_quantized_matmul(op, vectors, mapped):
    shapes = [[4], [4]] if vectors else [[2, 4], [4, 5]]
    if op == "MatMulInteger":
        shapes += [[], []]
        dtypes = [ir.DataType.UINT8] * 4
        axes = [*mapped, None, None]
        rhs = 1
        dtype = ir.DataType.INT32
    else:
        shapes = [shapes[0], [], [], shapes[1], [], [], [], []]
        dtypes = [ir.DataType.UINT8, ir.DataType.FLOAT, ir.DataType.UINT8] * 2 + [
            ir.DataType.FLOAT,
            ir.DataType.UINT8,
        ]
        axes = [mapped[0], None, None, mapped[1], None, None, None, None]
        rhs = 3
        dtype = ir.DataType.UINT8
    feeds = {}
    for i, (shape, axis, input_dtype) in enumerate(zip(shapes, axes, dtypes)):
        if i in (0, rhs):
            data_shape = [3, *shape] if axis is not None else shape
            data = (np.arange(np.prod(data_shape)).reshape(data_shape) % 7 + 1).astype(
                np.uint8
            )
            feeds[f"x{i}"] = np.moveaxis(data, 0, axis) if axis is not None else data
        else:
            feeds[f"x{i}"] = np.array(
                0.25 if input_dtype == ir.DataType.FLOAT else 1, dtype=input_dtype.numpy()
            )
    _equivalent(
        _model(op, shapes, input_dtypes=dtypes, dtype=dtype),
        axes,
        feeds,
        dynamic=True,
        ort=True,
    )


@pytest.mark.parametrize("op", ["ConvInteger", "QLinearConv"])
def test_quantized_convolution(op):
    shapes = [[2, 1, 4, 4], [1, 1, 2, 2]]
    if op == "ConvInteger":
        shapes += [[], []]
        dtypes = [ir.DataType.UINT8] * 4
        weight_index, dtype = 1, ir.DataType.INT32
    else:
        shapes = [shapes[0], [], [], shapes[1], [], [], [], []]
        dtypes = [ir.DataType.UINT8, ir.DataType.FLOAT, ir.DataType.UINT8] * 2 + [
            ir.DataType.FLOAT,
            ir.DataType.UINT8,
        ]
        weight_index, dtype = 3, ir.DataType.UINT8
    axes = [1, *([None] * (len(shapes) - 1))]
    feeds = {}
    for i, (shape, input_dtype) in enumerate(zip(shapes, dtypes)):
        if i in (0, weight_index):
            dims = [2, 3, 1, 4, 4] if i == 0 else shape
            feeds[f"x{i}"] = (np.arange(np.prod(dims)).reshape(dims) % 5).astype(np.uint8)
        else:
            feeds[f"x{i}"] = np.array(
                0.25 if input_dtype == ir.DataType.FLOAT else 0, dtype=input_dtype.numpy()
            )
    _equivalent(
        _model(op, shapes, input_dtypes=dtypes, dtype=dtype),
        axes,
        feeds,
        dynamic=True,
        ort=True,
    )
