# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

import onnx_ir as ir
from onnx_ir.passes import _autodiff, _batching, _rule_registry


def _node(domain: str = "", op_type: str = "Scale", *, version: int | None = None) -> ir.Node:
    return ir.Node(domain, op_type, [], num_outputs=1, version=version)


def _first() -> None:
    pass


def _second() -> None:
    pass


@pytest.mark.parametrize(
    ("version", "expected"),
    [(2, None), (3, _first), (5, _first), (6, None), (7, _second), (9, _second), (10, None)],
)
def test_inclusive_ranges_and_gaps(version, expected):
    registry = _rule_registry._RuleRegistry()
    registry.register("", "Scale", _second, since_version=7, until_version=9)
    registry.register("", "Scale", _first, since_version=3, until_version=5)
    if expected is None:
        with pytest.raises(NotImplementedError, match=f"at opset {version}"):
            registry.lookup(_node(), {"": version})
    else:
        assert registry.lookup(_node(), {"": version}) is expected


@pytest.mark.parametrize("registered_domain", ["", "ai.onnx"])
@pytest.mark.parametrize("node_domain", ["", "ai.onnx"])
@pytest.mark.parametrize("import_domain", ["", "ai.onnx"])
def test_default_domain_aliases(registered_domain, node_domain, import_domain):
    registry = _rule_registry._RuleRegistry()
    registry.register(registered_domain, "Scale", _first, since_version=18, until_version=18)
    assert registry.lookup(_node(node_domain), {import_domain: 18}) is _first
    assert registry.lookup(_node(node_domain), {"": 18, "ai.onnx": 18}) is _first


def test_import_scope_domain_and_operator_isolation():
    registry = _rule_registry._RuleRegistry()
    registry.register("example", "Scale", _first, since_version=1, until_version=1)
    registry.register("example", "Scale", _second, since_version=2, until_version=2)
    node = _node("example")
    graph = ir.Graph([], [], nodes=[node], opset_imports={"": 18, "example": 1})
    assert registry.lookup(node, graph.opset_imports) is _first
    assert registry.lookup(node, {"": 18, "example": 2}) is _second
    with pytest.raises(ValueError, match="No opset import"):
        registry.lookup(node, {"": 1})
    with pytest.raises(NotImplementedError, match="Registered intervals: none"):
        registry.lookup(_node(), {"": 1})
    with pytest.raises(NotImplementedError, match="Registered intervals: none"):
        registry.lookup(_node("example", "Other"), {"example": 1})
    with pytest.raises(NotImplementedError):
        registry.lookup(_node("Example"), {"Example": 1})


def test_explicit_node_version_takes_precedence():
    registry = _rule_registry._RuleRegistry()
    registry.register("", "Scale", _first, since_version=3, until_version=3)
    assert registry.lookup(_node(version=3), {"": 18}) is _first
    assert registry.lookup(_node(version=3), {}) is _first
    with pytest.raises(NotImplementedError, match="at opset 4"):
        registry.lookup(_node(version=4), {"": 3})


@pytest.mark.parametrize("version", [None, 3])
def test_conflicting_alias_imports(version):
    with pytest.raises(ValueError, match="Conflicting"):
        _rule_registry._effective_opset(_node(version=version), {"": 3, "ai.onnx": 4})


def test_alias_import_validation_does_not_confuse_bool_with_integer():
    with pytest.raises(ValueError, match="positive integer"):
        _rule_registry._effective_opset(_node(), {"": 1, "ai.onnx": True})


@pytest.mark.parametrize("version", [0, -1, True, 1.5, "18"])
def test_invalid_versions(version):
    registry = _rule_registry._RuleRegistry()
    with pytest.raises(ValueError, match="positive integer"):
        registry.register("", "Scale", _first, since_version=version, until_version=18)
    with pytest.raises(ValueError, match="positive integer"):
        registry.register("", "Scale", _first, since_version=1, until_version=version)
    with pytest.raises(ValueError, match="positive integer"):
        registry.lookup(_node(), {"": version})
    with pytest.raises(ValueError, match="positive integer"):
        registry.lookup(_node(version=version), {"": 18})


@pytest.mark.parametrize("interval", [(3, 5), (1, 3), (5, 7), (4, 4), (1, 9)])
def test_overlap_rejected_atomically(interval):
    registry = _rule_registry._RuleRegistry()
    registry.register("", "Scale", _first, since_version=3, until_version=5)
    for rule in (_first, _second):
        with pytest.raises(ValueError, match="Overrides are not supported"):
            registry.register(
                "ai.onnx", "Scale", rule, since_version=interval[0], until_version=interval[1]
            )
    assert registry.lookup(_node(), {"": 3}) is _first
    registry.register("", "Scale", _second, since_version=6, until_version=6)
    assert registry.lookup(_node(), {"": 6}) is _second


def test_invalid_registration_and_overloaded_node():
    registry = _rule_registry._RuleRegistry()
    with pytest.raises(ValueError, match="must not exceed"):
        registry.register("", "Scale", _first, since_version=5, until_version=3)
    with pytest.raises(ValueError, match="must not be empty"):
        registry.register("", "", _first, since_version=1, until_version=1)
    with pytest.raises(TypeError, match="must be callable"):
        registry.register("", "Scale", 42, since_version=1, until_version=1)
    registry.register("", "Scale", _first, since_version=1, until_version=1)
    node = _node()
    node.overload = "local_function"
    with pytest.raises(NotImplementedError, match="overloaded"):
        registry.lookup(node, {"": 1})


def test_registries_are_instance_local_and_mode_specific():
    batching = _batching._BatchingRegistry[ir.tape.Tape]()
    jvp = _autodiff._JvpRegistry[ir.tape.Tape]()
    vjp = _autodiff._VjpRegistry[ir.tape.Tape]()

    def batch(context: ir.tape.Tape, node: ir.Node) -> None:
        context.op("Identity", node.inputs)

    batching.register("", "Scale", batch, since_version=18, until_version=18)
    assert batching.lookup(_node(), {"": 18}) is batch
    for registry, mode in [
        (_batching._BatchingRegistry(), "batching"),
        (jvp, "JVP"),
        (vjp, "VJP"),
    ]:
        with pytest.raises(NotImplementedError, match=f"No {mode} rule"):
            registry.lookup(_node(), {"": 18})

    def forward(
        context: ir.tape.Tape, node: ir.Node, tangents: Mapping[int, ir.Value]
    ) -> Sequence[ir.Value]:
        return [context.op("Identity", [tangents[0]])]

    jvp.register("", "Scale", forward, since_version=18, until_version=18)
    assert jvp.lookup(_node(), {"": 18}) is forward
    with pytest.raises(NotImplementedError, match="No VJP rule"):
        vjp.lookup(_node(), {"": 18})


def test_private_custom_domain_extension_executes_all_three_contracts():
    """Lower a custom Scale into executable ONNX batching and derivative graphs."""
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])
    node = ir.Node(
        "com.example", "Scale", [x], attributes=[ir.AttrFloat32("factor", 2.5)], num_outputs=1
    )
    source = ir.Graph(
        [x], list(node.outputs), nodes=[node], opset_imports={"": 18, "com.example": 2}
    )
    source_before = str(source)

    class Context:
        def __init__(self, value: ir.Value) -> None:
            self.value = value
            self.tape = ir.tape.Tape()
            self.outputs: list[ir.Value] = []

        def scale(self, value: ir.Value, node: ir.Node) -> ir.Value:
            factor = self.tape.op(
                "Constant",
                [],
                {"value": ir.tensor(np.float32(node.attributes["factor"].as_float()))},
            )
            result = self.tape.op("Mul", [value, factor])
            result.type = value.type
            result.shape = value.shape
            return result

    def batch(context: Context, node: ir.Node) -> None:
        context.outputs.append(context.scale(context.value, node))

    def jvp(
        context: Context, node: ir.Node, tangents: Mapping[int, ir.Value]
    ) -> Sequence[ir.Value]:
        return [context.scale(tangents[0], node)]

    def vjp(
        context: Context, node: ir.Node, cotangents: Sequence[ir.Value], indices: Sequence[int]
    ) -> Mapping[int, ir.Value]:
        return {i: context.scale(cotangents[0], node) for i in indices}

    batching = _batching._BatchingRegistry[Context]()
    forward = _autodiff._JvpRegistry[Context]()
    reverse = _autodiff._VjpRegistry[Context]()
    batching.register("com.example", "Scale", batch, since_version=2, until_version=2)
    forward.register("com.example", "Scale", jvp, since_version=2, until_version=2)
    reverse.register("com.example", "Scale", vjp, since_version=2, until_version=2)

    for mode in ("batching", "jvp", "vjp"):
        shape = [3, 2] if mode == "batching" else [2]
        value = ir.val("input", dtype=ir.DataType.FLOAT, shape=shape)
        context = Context(value)
        if mode == "batching":
            batching.lookup(node, source.opset_imports)(context, node)
            outputs = context.outputs
        elif mode == "jvp":
            outputs = list(
                forward.lookup(node, source.opset_imports)(context, node, {0: value})
            )
        else:
            contributions = reverse.lookup(node, source.opset_imports)(
                context, node, [value], [0]
            )
            outputs = [contributions[0]]
        outputs[0].name = "output"
        graph = ir.Graph(
            [value], outputs, nodes=context.tape.nodes, opset_imports={"": 18}, name=mode
        )
        proto = ir.to_proto(ir.Model(graph, ir_version=9))
        onnx.checker.check_model(proto, full_check=True)
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        actual = ReferenceEvaluator(proto).run(None, {"input": data})[0]
        np.testing.assert_array_equal(actual, 2.5 * data)
        with pytest.raises(NotImplementedError, match="at opset 3"):
            batching.lookup(node, {"com.example": 3})
    assert list(source) == [node]
    assert str(source) == source_before
