# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

import onnx_ir as ir
from onnx_ir.passes.common import GradPass, InlinePass, JvpPass, VjpPass


def _value(name, shape, dtype=ir.DataType.DOUBLE):
    return ir.val(name, dtype=dtype, shape=shape)


def _model(inputs, tape, outputs, opset=18, initializers=()):
    outputs = [tape.op("Identity", [v]) if v in inputs else v for v in outputs]
    for i, output in enumerate(outputs):
        output.name = f"y{i}"
    return ir.Model(
        ir.Graph(
            inputs,
            outputs,
            nodes=tape.nodes,
            initializers=initializers,
            opset_imports={"": opset},
            name="autodiff_test",
        ),
        ir_version=9,
    )


def _evaluator(model):
    proto = onnx.shape_inference.infer_shapes(ir.to_proto(model), strict_mode=True)
    onnx.checker.check_model(proto, full_check=True)
    return ReferenceEvaluator(proto)


def _products(model, feeds, inputs=None, outputs=None, rtol=2e-5, atol=2e-6, epsilon=1e-6):
    """Compare both products with independent central differences and their duality."""
    inputs = list(feeds) if inputs is None else inputs
    before = ir.to_proto(model).SerializeToString()
    primal = _evaluator(model)
    original = primal.run(None, feeds)
    selected_outputs = (
        list(range(len(original)))
        if outputs is None
        else [[o.name for o in model.graph.outputs].index(n) for n in outputs]
    )
    rng = np.random.default_rng(42)
    tangents = [rng.normal(size=feeds[n].shape).astype(feeds[n].dtype) for n in inputs]
    seeds = [
        rng.normal(size=original[i].shape).astype(original[i].dtype) for i in selected_outputs
    ]
    jvp = JvpPass(inputs=inputs, outputs=outputs)(model).model
    vjp = VjpPass(inputs=inputs, outputs=outputs)(model).model
    jf = dict(feeds)
    vf = dict(feeds)
    for value, tangent in zip(jvp.graph.inputs[len(model.graph.inputs) :], tangents):
        jf[value.name] = tangent
    for value, seed in zip(vjp.graph.inputs[len(model.graph.inputs) :], seeds):
        vf[value.name] = seed
    jvalues = _evaluator(jvp).run(None, jf)
    vvalues = _evaluator(vjp).run(None, vf)
    n = len(original)
    for actual in (jvalues, vvalues):
        for expected, value in zip(original, actual[:n]):
            np.testing.assert_allclose(value, expected, rtol=rtol, atol=atol)
    plus, minus = dict(feeds), dict(feeds)
    for name, tangent in zip(inputs, tangents):
        plus[name] = feeds[name] + epsilon * tangent
        minus[name] = feeds[name] - epsilon * tangent
    yp, ym = primal.run(None, plus), primal.run(None, minus)
    for index, derivative in zip(selected_outputs, jvalues[n:]):
        assert derivative.shape == original[index].shape
        assert derivative.dtype == original[index].dtype
        np.testing.assert_allclose(
            derivative, (yp[index] - ym[index]) / (2 * epsilon), rtol=rtol, atol=atol
        )
    for name, gradient in zip(inputs, vvalues[n:]):
        assert gradient.shape == feeds[name].shape
        assert gradient.dtype == feeds[name].dtype
        expected = np.empty_like(feeds[name])
        for coordinate in np.ndindex(expected.shape):
            plus, minus = dict(feeds), dict(feeds)
            plus[name], minus[name] = feeds[name].copy(), feeds[name].copy()
            plus[name][coordinate] += epsilon
            minus[name][coordinate] -= epsilon
            yp, ym = primal.run(None, plus), primal.run(None, minus)
            expected[coordinate] = sum(
                np.sum(seed * (yp[index] - ym[index])) / (2 * epsilon)
                for index, seed in zip(selected_outputs, seeds)
            )
        np.testing.assert_allclose(gradient, expected, rtol=rtol, atol=atol)
    lhs = sum(np.sum(a * b) for a, b in zip(jvalues[n:], seeds))
    rhs = sum(np.sum(a * b) for a, b in zip(tangents, vvalues[n:]))
    np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)
    assert ir.to_proto(model).SerializeToString() == before
    return jvp, vjp


@pytest.mark.parametrize(
    "op",
    [
        "Identity",
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
    ],
)
def test_unary_finite_differences(op):
    x = _value("x", [2, 3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    values = np.array([[0.2, 0.4, 0.9], [0.7, 0.3, 0.8]])
    if op not in ("Log", "Sqrt"):
        values[1] *= -1
    # The ONNX reference Erf implementation rounds its intermediate result to float32.
    options = {"epsilon": 1e-3, "rtol": 2e-4, "atol": 2e-5} if op == "Erf" else {}
    _products(_model([x], tape, [y]), {"x": values}, **options)


@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div", "Pow"])
@pytest.mark.parametrize("shapes", [((), (2, 3)), ((3,), (2, 1, 3)), ((2, 1), (1, 3))])
def test_binary_broadcast_finite_differences(op, shapes):
    x, z = _value("x", shapes[0]), _value("z", shapes[1])
    tape = ir.tape.Tape()
    y = tape.op(op, [x, z])
    _products(
        _model([x, z], tape, [y]),
        {"x": np.full(shapes[0], 1.3), "z": np.full(shapes[1], 0.7)},
    )


def test_pow_integer_exponent():
    x = _value("x", [3])
    tape = ir.tape.Tape()
    exponent = tape.op("Constant", [], {"value": ir.tensor(np.array(3, dtype=np.int64))})
    y = tape.op("Pow", [x, exponent])
    _products(_model([x], tape, [y]), {"x": np.array([0.3, 0.8, 1.2])})


def test_negative_base_integer_power_does_not_evaluate_exponent_partial():
    x = _value("x", [2])
    tape = ir.tape.Tape()
    exponent = tape.op("Constant", [], {"value": ir.tensor(np.array(2.0))})
    y = tape.op("Pow", [x, exponent])
    _products(_model([x], tape, [y]), {"x": np.array([-0.8, -1.2])})


@pytest.mark.parametrize("exponent", [0, 1, 2, 3])
def test_polynomial_powers_at_zero_and_negative_bases(exponent):
    x = _value("x", [3])
    tape = ir.tape.Tape()
    p = tape.op("Constant", [], {"value": ir.tensor(np.array(exponent, dtype=np.int64))})
    y = tape.op("Pow", [x, p])
    _products(_model([x], tape, [y]), {"x": np.array([0.0, -0.8, 1.2])})


@pytest.mark.parametrize(
    "shapes",
    [
        ((3,), (3,)),
        ((3,), (3, 2)),
        ((2, 3), (3,)),
        ((2, 3), (3, 2)),
        ((3,), (2, 3, 2)),
        ((2, 2, 3), (3,)),
        ((2, 1, 2, 3), (1, 3, 3, 2)),
        ((1, 2, 3), (2, 3, 2)),
    ],
)
def test_matmul_vectors_and_broadcast_batches(shapes):
    x, z = _value("x", shapes[0]), _value("z", shapes[1])
    tape = ir.tape.Tape()
    y = tape.op("MatMul", [x, z])
    rng = np.random.default_rng(7)
    _products(
        _model([x, z], tape, [y]),
        {"x": rng.normal(size=shapes[0]), "z": rng.normal(size=shapes[1])},
    )


@pytest.mark.parametrize("trans_a, trans_b", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("bias_shape", [(), (2,), (1, 2), (2, 1)])
def test_gemm_transposes_and_bias(trans_a, trans_b, bias_shape):
    sa, sb = ((3, 2) if trans_a else (2, 3)), ((2, 3) if trans_b else (3, 2))
    a, b, c = _value("a", sa), _value("b", sb), _value("c", bias_shape)
    tape = ir.tape.Tape()
    y = tape.op(
        "Gemm", [a, b, c], {"transA": trans_a, "transB": trans_b, "alpha": 0.7, "beta": 1.3}
    )
    rng = np.random.default_rng(7)
    _products(
        _model([a, b, c], tape, [y]),
        {"a": rng.normal(size=sa), "b": rng.normal(size=sb), "c": rng.normal(size=bias_shape)},
    )


def test_gemm_bias_only_direction_is_expanded():
    a, b, c = _value("a", [2, 3]), _value("b", [3, 2]), _value("c", [])
    tape = ir.tape.Tape()
    y = tape.op("Gemm", [a, b, c])
    _products(
        _model([a, b, c], tape, [y]),
        {"a": np.ones((2, 3)), "b": np.ones((3, 2)), "c": np.array(0.7)},
        inputs=["c"],
    )


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean"])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axes,noop", [(None, 0), (None, 1), ([], 0), ([], 1), ([-1, 0], 0)])
def test_reductions(op, keepdims, axes, noop):
    x = _value("x", [2, 3, 2])
    tape = ir.tape.Tape()
    args = [x]
    if axes is not None:
        args.append(
            tape.op("Constant", [], {"value": ir.tensor(np.array(axes, dtype=np.int64))})
        )
    y = tape.op(op, args, {"keepdims": keepdims, "noop_with_empty_axes": noop})
    _products(_model([x], tape, [y]), {"x": np.arange(12.0).reshape(2, 3, 2) / 10})


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean"])
@pytest.mark.parametrize("axes", [[], [-1], [0, 2]])
@pytest.mark.parametrize("noop", [0, 1])
def test_runtime_reduction_axes(op, axes, noop):
    x, axis = _value("x", ["a", "b", "c"]), _value("axes", ["n"], ir.DataType.INT64)
    tape = ir.tape.Tape()
    y = tape.op(op, [x, axis], {"keepdims": 0, "noop_with_empty_axes": noop})
    output_rank = 3 if not axes and noop else (0 if not axes else 3 - len(axes))
    y.shape = ir.Shape([None] * output_rank)
    y.dtype = ir.DataType.DOUBLE
    _products(
        _model([x, axis], tape, [y]),
        {"x": np.arange(12.0).reshape(2, 3, 2) / 10, "axes": np.array(axes, dtype=np.int64)},
        inputs=["x"],
    )


@pytest.mark.parametrize(
    "op", ["Reshape", "Flatten", "Squeeze", "Unsqueeze", "Expand", "Transpose"]
)
def test_layout_operations(op):
    shape = [2, 1, 3]
    x = _value("x", shape)
    tape = ir.tape.Tape()
    args, attrs = [x], {}
    if op in ("Reshape", "Expand", "Squeeze", "Unsqueeze"):
        operand = {
            "Reshape": [3, 2],
            "Expand": [2, 4, 3],
            "Squeeze": [1],
            "Unsqueeze": [-1, 0],
        }[op]
        args.append(
            tape.op("Constant", [], {"value": ir.tensor(np.array(operand, dtype=np.int64))})
        )
    elif op == "Transpose":
        attrs = {"perm": [2, 0, 1]}
    elif op == "Flatten":
        attrs = {"axis": -1}
    y = tape.op(op, args, attrs)
    _products(_model([x], tape, [y]), {"x": np.arange(6.0).reshape(shape) / 10})


def test_default_transpose():
    x = _value("x", [2, 3, 1])
    tape = ir.tape.Tape()
    y = tape.op("Transpose", [x])
    _products(_model([x], tape, [y]), {"x": np.arange(6.0).reshape(2, 3, 1)})


def test_shape_path_is_structural():
    x = _value("x", ["batch", 3])
    tape = ir.tape.Tape()
    shape = tape.op("Shape", [x])
    y = tape.op("Reshape", [tape.op("Mul", [x, x]), shape])
    _products(_model([x], tape, [y]), {"x": np.arange(6.0).reshape(2, 3)})


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_concat_and_split_partial_outputs(axis):
    x, z = _value("x", [2, 2]), _value("z", [2, 2])
    tape = ir.tape.Tape()
    combined = tape.op("Concat", [x, z, x], {"axis": axis})
    split = tape.op_multi_out(
        "Split", [combined], {"axis": axis, "num_outputs": 3}, num_outputs=3
    )
    model = _model([x, z], tape, [*split, combined])
    _products(
        model,
        {"x": np.arange(4.0).reshape(2, 2), "z": np.full((2, 2), 0.7)},
        outputs=["y2", "y0"],
    )


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize(
    "indices", [np.array(1), np.array([1, 1, -1]), np.array([[1, 0], [1, -1]])]
)
def test_gather_repeated_negative_and_scalar_indices(axis, indices):
    x = _value("x", [2, 3, 2])
    index = _value("indices", list(indices.shape), ir.DataType.INT64)
    tape = ir.tape.Tape()
    y = tape.op("Gather", [x, index], {"axis": axis})
    _products(
        _model([x, index], tape, [y]),
        {"x": np.arange(12.0).reshape(2, 3, 2), "indices": indices.astype(np.int64)},
        inputs=["x"],
    )


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("axis", [0, -1])
def test_softmax(op, axis):
    x = _value("x", [2, 3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x], {"axis": axis})
    _products(_model([x], tape, [y]), {"x": np.arange(6.0).reshape(2, 3) / 10})


def test_where_branch_predicate_is_not_differentiated():
    x, z = _value("x", [2, 1]), _value("z", [3])
    tape = ir.tape.Tape()
    zero = tape.op("Constant", [], {"value": ir.tensor(np.array(0.0))})
    condition = tape.op("Greater", [x, zero])
    y = tape.op("Where", [condition, x, z])
    _products(
        _model([x, z], tape, [y]),
        {"x": np.array([[-0.7], [0.8]]), "z": np.array([0.2, 0.5, 0.9])},
    )


def test_multiple_outputs_fanout_order_and_disconnected_inputs():
    x, z, unused = _value("x", [2]), _value("z", []), _value("unused", [3])
    tape = ir.tape.Tape()
    square = tape.op("Mul", [x, x])
    y = tape.op("Add", [square, tape.op("Mul", [x, z])])
    model = _model([x, z, unused], tape, [y, square, z])
    jvp, vjp = _products(
        model,
        {"x": np.array([0.3, 0.7]), "z": np.array(1.1), "unused": np.ones(3)},
        inputs=["unused", "z", "x"],
        outputs=["y1", "y2", "y0"],
    )
    assert [v.name for v in jvp.graph.inputs[-3:]] == [
        "unused_tangent",
        "z_tangent",
        "x_tangent",
    ]
    assert [v.name for v in vjp.graph.outputs[-3:]] == [
        "unused_cotangent",
        "z_cotangent",
        "x_cotangent",
    ]


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize("opset", [18, 21, 24])
def test_grad_analytic_dynamic_mse_and_runtime_execution(dtype, opset):
    x, target = _value("x", ["batch", 3], dtype), _value("target", ["batch", 3], dtype)
    tape = ir.tape.Tape()
    error = tape.op("Sub", [x, target])
    y = tape.op("ReduceMean", [tape.op("Mul", [error, error])], {"keepdims": 0})
    model = _model([x, target], tape, [y], opset=opset)
    before = ir.to_proto(model).SerializeToString()
    gradient = GradPass(inputs=["x"])(model).model
    assert len(gradient.graph.inputs) == 2
    evaluator = _evaluator(gradient)
    for batch in (1, 4, 7):
        data = np.arange(batch * 3, dtype=dtype.numpy()).reshape(batch, 3) / 10
        feeds = {"x": data, "target": np.ones_like(data)}
        actual = evaluator.run(None, feeds)
        np.testing.assert_allclose(actual[0], np.mean((data - 1) ** 2), rtol=1e-6)
        np.testing.assert_allclose(actual[1], 2 * (data - 1) / data.size, rtol=1e-6, atol=1e-7)
    assert ir.to_proto(model).SerializeToString() == before


def test_dynamic_broadcast_sizes_and_zero_extents():
    x, z = _value("x", ["batch", "width"]), _value("z", ["bias_width"])
    tape = ir.tape.Tape()
    y = tape.op("Add", [x, z])
    model = _model([x, z], tape, [y])
    vjp = VjpPass()(model).model
    evaluator = _evaluator(vjp)
    for sx, sz in [((2, 3), (3,)), ((5, 4), (1,)), ((0, 3), (1,)), ((2, 0), (0,))]:
        feeds = {
            "x": np.ones(sx),
            "z": np.ones(sz),
            vjp.graph.inputs[-1].name: np.ones(np.broadcast_shapes(sx, sz)),
        }
        _, gx, gz = evaluator.run(None, feeds)
        np.testing.assert_array_equal(gx, np.ones(sx))
        expected = np.full(sz, np.prod(sx) / max(np.prod(sz), 1))
        np.testing.assert_array_equal(gz, expected)


def test_dynamic_matmul_runtime_batches():
    a, b = _value("a", ["batch", 2, 3]), _value("b", [3, 2])
    tape = ir.tape.Tape()
    y = tape.op("MatMul", [a, b])
    model = _model([a, b], tape, [y])
    for batch in (1, 3):
        _products(model, {"a": np.ones((batch, 2, 3)), "b": np.ones((3, 2))})


def test_initializer_parameter_selection():
    x = _value("x", [2, 3])
    weight = _value("weight", [3, 2])
    weight.const_value = ir.tensor(np.arange(6.0).reshape(3, 2))
    tape = ir.tape.Tape()
    prediction = tape.op("MatMul", [x, weight])
    loss = tape.op("ReduceSum", [prediction], {"keepdims": 0})
    model = _model([x], tape, [loss], initializers=[weight])
    default = GradPass()(model).model
    gradient = GradPass(inputs=["weight", "x"])(model).model
    actual = _evaluator(gradient).run(None, {"x": np.ones((2, 3))})
    np.testing.assert_array_equal(actual[1], np.full((3, 2), 2.0))
    np.testing.assert_array_equal(
        actual[2], np.broadcast_to(np.sum(weight.const_value.numpy(), axis=1), (2, 3))
    )
    assert len(default.graph.outputs) == 2
    assert len(gradient.graph.inputs) == 1
    assert list(gradient.graph.initializers) == ["weight"]
    jvp = JvpPass(inputs=["weight"])(model).model
    _, tangent = _evaluator(jvp).run(
        None, {"x": np.ones((2, 3)), jvp.graph.inputs[-1].name: np.ones((3, 2))}
    )
    assert tangent == 12


def test_initializer_also_input_excluded_by_default():
    x, w = _value("x", [2]), _value("w", [2])
    w.const_value = ir.tensor(np.ones(2))
    tape = ir.tape.Tape()
    y = tape.op("Mul", [x, w])
    model = _model([x, w], tape, [y], initializers=[w])
    result = VjpPass()(model).model
    assert len(result.graph.outputs) == 2
    result = VjpPass(inputs=["w"])(model).model
    _, gradient = _evaluator(result).run(
        None, {"x": np.array([2.0, 3.0]), "y0_cotangent": np.ones(2)}
    )
    np.testing.assert_array_equal(gradient, [2, 3])


def test_floating_cast():
    x = _value("x", [2], ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    y = tape.op("Cast", [x], {"to": ir.DataType.DOUBLE})
    model = _model([x], tape, [y])
    jvp = JvpPass()(model).model
    _, tangent = _evaluator(jvp).run(
        None,
        {"x": np.ones(2, dtype=np.float32), "x_tangent": np.array([2, 3], dtype=np.float32)},
    )
    assert tangent.dtype == np.float64
    np.testing.assert_array_equal(tangent, [2, 3])
    vjp = VjpPass()(model).model
    _, gradient = _evaluator(vjp).run(
        None, {"x": np.ones(2, dtype=np.float32), "y0_cotangent": np.array([2.0, 3.0])}
    )
    assert gradient.dtype == np.float32
    np.testing.assert_array_equal(gradient, [2, 3])


@pytest.mark.parametrize("op,expected", [("Abs", 0), ("Relu", 0), ("LeakyRelu", 0.01)])
def test_documented_nondifferentiable_zero_convention(op, expected):
    x = _value("x", [])
    tape = ir.tape.Tape()
    model = _model([x], tape, [tape.op(op, [x])])
    result = GradPass()(model).model
    _, gradient = _evaluator(result).run(None, {"x": np.array(0.0)})
    np.testing.assert_allclose(gradient, expected)


def test_name_collisions_and_pass_reuse():
    x, collision = _value("x", [2]), _value("x_tangent", [])
    tape = ir.tape.Tape()
    y = tape.op("Mul", [x, x])
    model = _model([x, collision], tape, [y])
    pass_ = JvpPass(inputs=["x"])
    first, second = pass_(model).model, pass_(model).model
    assert first.graph.inputs[-1].name == "x_tangent_1"
    assert ir.to_proto(first) == ir.to_proto(second)
    assert not pass_.in_place and not pass_.changes_input
    assert not VjpPass().in_place and not GradPass().changes_input


@pytest.mark.parametrize("pass_type", [JvpPass, VjpPass, GradPass])
def test_explicit_error_paths_and_source_preservation(pass_type):
    x = _value("x", [])
    tape = ir.tape.Tape()
    y = tape.op("Floor", [x])
    model = _model([x], tape, [y])
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises(NotImplementedError, match=r"active operator.*Floor"):
        pass_type()(model)
    assert ir.to_proto(model).SerializeToString() == before
    for inputs, error in [
        (["absent"], "Unknown inputs"),
        (["x", "x"], "Duplicate"),
        ([], "at least one"),
    ]:
        with pytest.raises(ValueError, match=error):
            pass_type(inputs=inputs)(model)
    with pytest.raises(TypeError, match="sequence"):
        pass_type(inputs="x")(model)
    model.graph.opset_imports[""] = 17
    with pytest.raises(ValueError, match="opset >= 18"):
        pass_type()(model)


def test_unsupported_inactive_and_shape_only_paths_are_not_differentiated():
    x, z = _value("x", [2]), _value("z", [2])
    tape = ir.tape.Tape()
    floor = tape.op("Floor", [x])
    shape = tape.op("Shape", [floor])
    y = tape.op("Reshape", [z, shape])
    _products(
        _model([x, z], tape, [y]),
        {"x": np.array([0.3, 0.7]), "z": np.array([0.9, 1.2])},
    )


def test_unsupported_active_custom_operator():
    x = _value("x", [])
    tape = ir.tape.Tape()
    y = tape.op("Custom", [x], domain="example")
    y.dtype, y.shape = ir.DataType.DOUBLE, ir.Shape([])
    model = _model([x], tape, [y])
    model.graph.opset_imports["example"] = 1
    with pytest.raises(NotImplementedError, match=r"example.*Custom"):
        VjpPass()(model)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.INT64, ir.DataType.BOOL, ir.DataType.FLOAT16, ir.DataType.COMPLEX64]
)
def test_reject_nondifferentiable_selections(dtype):
    x = _value("x", [], dtype)
    tape = ir.tape.Tape()
    model = _model([x], tape, [tape.op("Identity", [x])])
    with pytest.raises(TypeError, match="FLOAT or DOUBLE"):
        JvpPass(inputs=["x"])(model)


def test_integer_cast_on_active_data_path_rejected():
    x = _value("x", [])
    tape = ir.tape.Tape()
    integer = tape.op("Cast", [x], {"to": ir.DataType.INT64})
    y = tape.op("Cast", [integer], {"to": ir.DataType.DOUBLE})
    with pytest.raises(TypeError, match="Integer/boolean casts"):
        VjpPass()(_model([x], tape, [y]))


def test_grad_requires_scalar_and_unambiguous_output():
    x = _value("x", [1])
    tape = ir.tape.Tape()
    y = tape.op("Mul", [x, x])
    z = tape.op("ReduceSum", [y], {"keepdims": 0})
    model = _model([x], tape, [y, z])
    with pytest.raises(ValueError, match="rank-zero"):
        GradPass()(model)
    with pytest.raises(ValueError, match="rank-zero"):
        GradPass(output="y0")(model)
    result = GradPass(output="y1")(model).model
    actual = _evaluator(result).run(None, {"x": np.array([2.0])})
    np.testing.assert_array_equal(actual[-1], [4])


def test_randomness_and_control_flow_rejected():
    x = _value("x", [])
    tape = ir.tape.Tape()
    y = tape.op("RandomNormalLike", [x])
    with pytest.raises(NotImplementedError, match="randomness"):
        JvpPass()(_model([x], tape, [y]))
    body = ir.Graph([], [], nodes=[], opset_imports={"": 18})
    x = _value("x", [])
    tape = ir.tape.Tape()
    y = tape.op("UnknownControlFlow", [x], {"body": body})
    with pytest.raises(NotImplementedError, match="control flow"):
        VjpPass()(_model([x], tape, [y]))


def test_higher_order_rejected():
    x = _value("x", [])
    tape = ir.tape.Tape()
    model = _model([x], tape, [tape.op("Mul", [x, x])])
    first = GradPass()(model).model
    with pytest.raises(NotImplementedError, match="Higher-order"):
        JvpPass()(first)


def test_onnxruntime_executes_standard_gradient_graph():
    ort = pytest.importorskip("onnxruntime")
    x, bias = (
        _value("x", ["batch", 3], ir.DataType.FLOAT),
        _value("bias", [3], ir.DataType.FLOAT),
    )
    tape = ir.tape.Tape()
    y = tape.op("Tanh", [tape.op("Add", [x, bias])])
    loss = tape.op("ReduceMean", [y], {"keepdims": 0})
    gradient = GradPass()(_model([x, bias], tape, [loss])).model
    session = ort.InferenceSession(
        ir.to_proto(gradient).SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for batch in (1, 5):
        data = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3) / 10
        actual = session.run(None, {"x": data, "bias": np.ones(3, dtype=np.float32)})
        expected = (1 - np.tanh(data + 1) ** 2) / data.size
        np.testing.assert_allclose(actual[1], expected, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(actual[2], expected.sum(axis=0), rtol=1e-5, atol=1e-7)


def test_direct_input_output_identity_and_seed_name_collision():
    x = _value("x", [2])
    model = ir.Model(
        ir.Graph([x], [x], nodes=[], opset_imports={"": 18}, name="identity"),
        ir_version=9,
    )
    jvp, vjp = _products(model, {"x": np.array([0.3, 0.7])})
    assert jvp.graph.inputs[-1].name == "x_tangent"
    assert jvp.graph.outputs[-1].name == "x_tangent_1"
    assert vjp.graph.inputs[-1].name == "x_cotangent"
    assert vjp.graph.outputs[-1].name == "x_cotangent_1"


def test_constant_selected_output_gives_zero_products():
    x = _value("x", ["n"])
    tape = ir.tape.Tape()
    y = tape.op("Constant", [], {"value": ir.tensor(np.array([2.0, 3.0]))})
    model = _model([x], tape, [y])
    _products(model, {"x": np.array([0.3, 0.7, 0.9])})


def test_constant_scalar_loss_gives_zero_gradient():
    x = _value("x", ["n"])
    tape = ir.tape.Tape()
    y = tape.op("Constant", [], {"value": ir.tensor(np.array(2.0))})
    gradient = GradPass()(_model([x], tape, [y])).model
    _, g = _evaluator(gradient).run(None, {"x": np.array([np.nan, np.inf])})
    np.testing.assert_array_equal(g, [0.0, 0.0])


@pytest.mark.parametrize("op", ["Add", "Mul", "Div", "Pow"])
def test_repeated_binary_operands(op):
    x = _value("x", [3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x, x])
    _products(_model([x], tape, [y]), {"x": np.array([0.3, 0.7, 1.2])})


def test_runtime_split_sizes_and_unequal_concat_lengths():
    x = _value("x", ["batch", "width"])
    sizes = _value("sizes", [3], ir.DataType.INT64)
    tape = ir.tape.Tape()
    parts = tape.op_multi_out("Split", [x, sizes], {"axis": -1}, num_outputs=3)
    for part in parts:
        part.dtype = ir.DataType.DOUBLE
        part.shape = ir.Shape(["batch", None])
    combined = tape.op("Concat", [parts[2], parts[0]], {"axis": -1})
    model = _model([x, sizes], tape, [*parts, combined])
    _products(
        model,
        {"x": np.arange(10.0).reshape(2, 5), "sizes": np.array([1, 3, 1], dtype=np.int64)},
        inputs=["x"],
        outputs=["y3", "y1"],
    )


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean"])
@pytest.mark.parametrize("noop", [0, 1])
def test_same_graph_runtime_axes_change_with_keepdims(op, noop):
    x = _value("x", ["a", "b", "c"])
    axes = _value("axes", ["n"], ir.DataType.INT64)
    tape = ir.tape.Tape()
    y = tape.op(op, [x, axes], {"keepdims": 1, "noop_with_empty_axes": noop})
    y.dtype, y.shape = ir.DataType.DOUBLE, ir.Shape([None, None, None])
    model = _model([x, axes], tape, [y])
    transformed = VjpPass(inputs=["x"])(model).model
    evaluator = _evaluator(transformed)
    data = np.arange(24.0).reshape(2, 3, 4)
    for axis_values in ([], [0], [-1], [0, 2]):
        effective = tuple(axis_values) if axis_values or noop else None
        reduce = np.sum if op == "ReduceSum" else np.mean
        value = reduce(data, axis=effective, keepdims=True)
        _, g = evaluator.run(
            None,
            {
                "x": data,
                "axes": np.array(axis_values, dtype=np.int64),
                transformed.graph.inputs[-1].name: np.ones_like(value),
            },
        )
        expected = 1.0 if op == "ReduceSum" else value.size / data.size
        np.testing.assert_array_equal(g, np.full_like(data, expected))


def test_local_functions_require_inlining_and_are_source_preserving():
    model = ir.from_onnx_text("""
        <ir_version: 9, opset_import: [ "" : 18, "local" : 1 ]>
        square (double[2] x) => (double[2] y) {
            y = local.square (x)
        }
        <opset_import: [ "" : 18 ], domain: "local">
        square (x) => (y) { y = Mul(x, x) }
    """)
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises(NotImplementedError, match="Inline local functions"):
        JvpPass()(model)
    assert ir.to_proto(model).SerializeToString() == before
    inlined = InlinePass()(model.clone()).model
    _products(inlined, {"x": np.array([0.3, 0.7])})


def test_selection_output_errors_and_unknown_rank():
    x = _value("x", [2])
    tape = ir.tape.Tape()
    model = _model([x], tape, [tape.op("Identity", [x])])
    for outputs, match in [
        (["absent"], "Unknown outputs"),
        (["y0", "y0"], "Duplicate"),
        ([], "at least one"),
    ]:
        with pytest.raises(ValueError, match=match):
            VjpPass(outputs=outputs)(model)
    with pytest.raises(TypeError, match="sequence"):
        JvpPass(outputs="y0")(model)
    x.shape = None
    model.graph.outputs[0].shape = None
    with pytest.raises(ValueError, match="known-rank"):
        VjpPass()(model)


def test_mixed_float_pow_rejected_instead_of_emitting_invalid_onnx():
    x, p = _value("x", [2], ir.DataType.FLOAT), _value("p", [2])
    tape = ir.tape.Tape()
    model = _model([x, p], tape, [tape.op("Pow", [x, p])])
    with pytest.raises(TypeError, match="matching floating-point dtypes"):
        JvpPass()(model)


@pytest.mark.parametrize("index_dtype", [ir.DataType.INT32, ir.DataType.INT64])
@pytest.mark.parametrize("index_shape", [(), (2,), (0,)])
def test_gather_vector_data_and_integer_index_dtypes(index_dtype, index_shape):
    x = _value("x", ["n"])
    indices = _value("indices", index_shape, index_dtype)
    tape = ir.tape.Tape()
    y = tape.op("Gather", [x, indices])
    model = _model([x, indices], tape, [y])
    _products(
        model,
        {
            "x": np.array([0.3, 0.7, 1.2]),
            "indices": np.full(index_shape, -1, dtype=index_dtype.numpy()),
        },
        inputs=["x"],
    )


@pytest.mark.parametrize("pass_type", [JvpPass, VjpPass])
@pytest.mark.parametrize(
    "case",
    [
        "dot",
        "batched_vector",
        "zero_broadcast",
        "zero_last",
        "zero_scalar",
        "gather_int32",
        "runtime_axes",
    ],
)
def test_onnxruntime_derivative_edge_cases(pass_type, case):
    ort = pytest.importorskip("onnxruntime")
    tape = ir.tape.Tape()
    if case in ("dot", "batched_vector"):
        sx, sz = ((3,), (3,)) if case == "dot" else ((3,), (2, 3, 2))
        x, z = _value("x", sx), _value("z", sz)
        y = tape.op("MatMul", [x, z])
        inputs = [x, z]
        feeds = {"x": np.ones(sx), "z": np.ones(sz)}
        selected = ["x", "z"]
    elif case in ("zero_broadcast", "zero_last", "zero_scalar"):
        sx, sz = {
            "zero_broadcast": ((0, 3), (1,)),
            "zero_last": ((2, 0), (0,)),
            "zero_scalar": ((2, 0), ()),
        }[case]
        x, z = _value("x", [None, None]), _value("z", [None] * len(sz))
        y = tape.op("Add", [x, z])
        inputs = [x, z]
        feeds = {"x": np.ones(sx), "z": np.ones(sz)}
        selected = ["x", "z"]
    elif case == "gather_int32":
        x, indices = _value("x", [3]), _value("indices", [3], ir.DataType.INT32)
        y = tape.op("Gather", [x, indices])
        inputs = [x, indices]
        feeds = {
            "x": np.array([0.3, 0.7, 0.9]),
            "indices": np.array([1, 1, -1], dtype=np.int32),
        }
        selected = ["x"]
    else:
        x, axes = _value("x", ["a", "b"]), _value("axes", ["n"], ir.DataType.INT64)
        y = tape.op("ReduceMean", [x, axes], {"keepdims": 0})
        y.dtype, y.shape = ir.DataType.DOUBLE, ir.Shape([])
        inputs = [x, axes]
        feeds = {"x": np.arange(6.0).reshape(2, 3), "axes": np.array([], dtype=np.int64)}
        selected = ["x"]
    model = _model(inputs, tape, [y])
    transformed = pass_type(inputs=selected)(model).model
    original = _evaluator(model).run(None, feeds)
    seed_values = (
        [np.ones_like(feeds[n]) for n in selected]
        if pass_type is JvpPass
        else [np.ones_like(original[0])]
    )
    for value, seed in zip(transformed.graph.inputs[len(inputs) :], seed_values):
        feeds[value.name] = seed
    expected = _evaluator(transformed).run(None, feeds)
    session = ort.InferenceSession(
        ir.to_proto(transformed).SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for actual, reference in zip(session.run(None, feeds), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)


def test_bernoulli_on_inactive_path_is_still_rejected():
    x, p = _value("x", [2]), _value("p", [2])
    tape = ir.tape.Tape()
    random = tape.op("Bernoulli", [p])
    y = tape.op("Mul", [x, random])
    with pytest.raises(NotImplementedError, match="randomness"):
        JvpPass(inputs=["x"])(_model([x, p], tape, [y]))
