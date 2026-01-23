# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the inliner pass."""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import Callable

import onnx

import onnx_ir as ir
from onnx_ir.passes.common import inliner


def _name_checker(renameable: Sequence[str] | None) -> Callable[[str, str], bool]:
    """Construct function to check if actual value name matches expected value name.

    This is used to avoid hard-coding the expected names in the test cases.
    """
    # Default to exact match if no renaming is allowed.
    if renameable is None:
        return lambda a, b: a == b
    # If some names are allowed to be renamed, keep track of the renaming.
    # And check that the renaming is consistent across all nodes.
    renaming_map: dict[str, str] = {}

    def check(actual: str, expected: str) -> bool:
        if expected in renameable:
            # actual name can be different, as long as it is consistently used.
            if expected in renaming_map:
                return renaming_map[expected] == actual
            renaming_map[expected] = actual
            return True
        else:
            return actual == expected

    return check


class InlinerTest(unittest.TestCase):
    def _check(
        self, input_model: str, expected_model: str, renameable: Sequence[str] | None = None
    ) -> None:
        name_check = _name_checker(renameable)
        model_ir = ir.from_onnx_text(input_model)
        inliner.InlinePass()(model_ir)
        proto = ir.serde.serialize_model(model_ir)
        text = onnx.printer.to_text(proto)
        print(text)
        expected_ir = ir.from_onnx_text(expected_model)
        self.assertEqual(len(model_ir.graph), len(expected_ir.graph))
        for node, expected_node in zip(model_ir.graph, expected_ir.graph):
            # TODO: handle node renaming
            self.assertEqual(node.op_type, expected_node.op_type)
            self.assertEqual(len(node.inputs), len(expected_node.inputs))
            for input, expected_input in zip(node.inputs, expected_node.inputs):
                self.assertEqual(input is None, expected_input is None)
                if input is not None:
                    self.assertTrue(name_check(input.name, expected_input.name))
            self.assertEqual(len(node.attributes), len(expected_node.attributes))
            for key, value in node.attributes.items():
                self.assertIn(key, expected_node.attributes)
                expected_value = expected_node.attributes[key]
                self.assertTrue(isinstance(value, ir.Attr))
                self.assertTrue(isinstance(expected_value, ir.Attr))
                self.assertEqual(value.type, expected_value.type)
                if value.type not in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS):
                    self.assertEqual(value.value, expected_value.value)
                else:
                    self.fail("Graph attributes are not supported yet")
                    # TODO: handle graph attributes
            self.assertEqual(len(node.outputs), len(expected_node.outputs))
            for output, expected_output in zip(node.outputs, expected_node.outputs):
                self.assertTrue(name_check(output.name, expected_output.name))

    def test_single_call(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = Mul(temp, temp)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                temp = Add(X, X)
                Y = Mul(temp, temp)
            }
        """
        self._check(input_model, expected_model, renameable=["temp"])

    def test_two_calls(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.foo (X)
                Y = local.foo (T)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = Mul(temp, temp)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                temp1 = Add(X, X)
                T = Mul(temp1, temp1)
                temp2 = Add(T, T)
                Y = Mul(temp2, temp2)
            }
        """
        self._check(input_model, expected_model, renameable=["temp1", "temp2"])

    def test_nested_call(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = local.bar(temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul (x, x)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                temp = Add(X, X)
                Y = Mul(temp, temp)
            }
        """
        self._check(input_model, expected_model, renameable=["temp"])

    def test_attr_parameter(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo <alpha = 0.5> (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo <alpha> (x) => (y) {
                y = Selu <alpha: float = @alpha> (x)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = Selu <alpha: float = 0.5> (X)
            }
        """
        self._check(input_model, expected_model)

    def test_attr_parameter_with_default_value(self):
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.foo <alpha = 0.5> (X)
                Y = local.foo (T)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo <alpha: float=0.6> (x) => (y) {
                y = Selu <alpha: float = @alpha> (x)
            }
        """
        expected_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = Selu <alpha: float = 0.5> (X)
                Y = Selu <alpha: float = 0.6> (T)
            }
        """
        self._check(input_model, expected_model)

    def test_criteria_skips_nodes_that_do_not_match(self):
        """Test that the criteria option allows selective inlining."""
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.foo (X)
                Y = local.bar (T)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            foo (x) => (y) {
                y = Add(x, x)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul(x, x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)

        # Only inline calls to "foo", skip "bar"
        def criteria(node: ir.Node) -> bool:
            return node.op_type == "foo"

        inliner.InlinePass(criteria=criteria)(model_ir)

        # foo should be inlined (Add), bar should remain as a call
        self.assertEqual(len(model_ir.graph), 2)
        nodes = list(model_ir.graph)
        self.assertEqual(nodes[0].op_type, "Add")
        self.assertEqual(nodes[1].op_type, "bar")
        self.assertEqual(nodes[1].domain, "local")

    def test_criteria_inlines_all_when_criteria_returns_true(self):
        """Test that all calls are inlined when criteria always returns True."""
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.foo (X)
                Y = local.bar (T)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            foo (x) => (y) {
                y = Add(x, x)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul(x, x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)

        inliner.InlinePass(criteria=lambda _: True)(model_ir)

        # Both should be inlined
        self.assertEqual(len(model_ir.graph), 2)
        nodes = list(model_ir.graph)
        self.assertEqual(nodes[0].op_type, "Add")
        self.assertEqual(nodes[1].op_type, "Mul")

    def test_criteria_inlines_none_when_criteria_returns_false(self):
        """Test that no calls are inlined when criteria always returns False."""
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.foo (X)
                Y = local.bar (T)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            foo (x) => (y) {
                y = Add(x, x)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul(x, x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)

        inliner.InlinePass(criteria=lambda _: False)(model_ir)

        # Neither should be inlined
        self.assertEqual(len(model_ir.graph), 2)
        nodes = list(model_ir.graph)
        self.assertEqual(nodes[0].op_type, "foo")
        self.assertEqual(nodes[0].domain, "local")
        self.assertEqual(nodes[1].op_type, "bar")
        self.assertEqual(nodes[1].domain, "local")


class InlinerTopologicalSortTest(unittest.TestCase):
    """Tests for topological sorting of functions during inlining."""

    def test_nested_functions_are_inlined_in_dependency_order(self):
        """Test that nested function calls are inlined correctly.

        When function A calls function B, B should be inlined into A first,
        then A (with B's body) should be inlined into the main graph.
        """
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.outer (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            outer (x) => (y) {
                temp = local.inner(x)
                y = Add(temp, temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            inner (x) => (y) {
                y = Mul(x, x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)
        inliner.InlinePass()(model_ir)

        # Both functions should be fully inlined
        # inner: Mul(x, x) -> temp
        # outer: temp = inner(x), Add(temp, temp) -> y
        # Final: Mul(X, X), Add(temp, temp)
        self.assertEqual(len(model_ir.graph), 2)
        nodes = list(model_ir.graph)
        self.assertEqual(nodes[0].op_type, "Mul")
        self.assertEqual(nodes[1].op_type, "Add")

    def test_diamond_dependency_functions(self):
        """Test inlining with diamond-shaped function dependencies.

        main -> A -> C
             -> B -> C

        C should be inlined into both A and B, then A and B into main.
        """
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T1 = local.funcA (X)
                T2 = local.funcB (X)
                Y = Add(T1, T2)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            funcA (x) => (y) {
                temp = local.funcC(x)
                y = Relu(temp)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            funcB (x) => (y) {
                temp = local.funcC(x)
                y = Sigmoid(temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            funcC (x) => (y) {
                y = Mul(x, x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)
        inliner.InlinePass()(model_ir)

        # All functions should be inlined
        # Expected: Mul, Relu, Mul, Sigmoid, Add
        self.assertEqual(len(model_ir.graph), 5)
        nodes = list(model_ir.graph)
        op_types = [n.op_type for n in nodes]
        self.assertEqual(op_types, ["Mul", "Relu", "Mul", "Sigmoid", "Add"])

    def test_chain_of_nested_functions(self):
        """Test inlining with a chain of function dependencies: A -> B -> C -> D."""
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.funcA (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            funcA (x) => (y) {
                y = local.funcB(x)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            funcB (x) => (y) {
                y = local.funcC(x)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            funcC (x) => (y) {
                y = local.funcD(x)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            funcD (x) => (y) {
                y = Relu(x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)
        inliner.InlinePass()(model_ir)

        # All functions should be inlined to a single Relu
        self.assertEqual(len(model_ir.graph), 1)
        nodes = list(model_ir.graph)
        self.assertEqual(nodes[0].op_type, "Relu")

    def test_functions_defined_in_reverse_order_are_still_inlined_correctly(self):
        """Test that function definition order doesn't affect inlining.

        Even if outer is defined before inner in the model, inner should
        be inlined into outer first.
        """
        # Note: inner is defined after outer, but inner should still be inlined first
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.outer (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            outer (x) => (y) {
                temp = local.inner(x)
                y = Add(temp, temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            inner (x) => (y) {
                y = Mul(x, x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)
        inliner.InlinePass()(model_ir)

        self.assertEqual(len(model_ir.graph), 2)
        nodes = list(model_ir.graph)
        self.assertEqual(nodes[0].op_type, "Mul")
        self.assertEqual(nodes[1].op_type, "Add")

    def test_cyclic_function_dependency_raises_error(self):
        """Test that cyclic function dependencies raise an error."""
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.funcA (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            funcA (x) => (y) {
                y = local.funcB(x)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            funcB (x) => (y) {
                y = local.funcA(x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)

        with self.assertRaises(ir.passes.PreconditionError):
            inliner.InlinePass()(model_ir)

    def test_independent_functions_are_all_inlined(self):
        """Test that independent functions (no dependencies between them) are all inlined."""
        input_model = """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T1 = local.foo (X)
                T2 = local.bar (X)
                T3 = local.baz (X)
                T4 = Add(T1, T2)
                Y = Add(T4, T3)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            foo (x) => (y) {
                y = Relu(x)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Sigmoid(x)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            baz (x) => (y) {
                y = Tanh(x)
            }
        """
        model_ir = ir.from_onnx_text(input_model)
        inliner.InlinePass()(model_ir)

        # All functions should be inlined
        self.assertEqual(len(model_ir.graph), 5)
        nodes = list(model_ir.graph)
        op_types = [n.op_type for n in nodes]
        self.assertIn("Relu", op_types)
        self.assertIn("Sigmoid", op_types)
        self.assertIn("Tanh", op_types)
        self.assertEqual(op_types.count("Add"), 2)


if __name__ == "__main__":
    unittest.main()
