# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator

import onnx_ir as ir
import onnx_ir.passes.common


class RemoveUnusedTest(unittest.TestCase):
    def remove_unused_nodes(self, model: onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
        onnx_ir.passes.common.RemoveUnusedNodesPass()(model_ir)
        model = ir.serde.serialize_model(model_ir)
        return model

    def batchnorm_feeds(self):
        return {
            "x": np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32),
            "scale": np.ones(2, dtype=np.float32),
            "bias": np.zeros(2, dtype=np.float32),
            "mean": np.array([100.0, -100.0], dtype=np.float32),
            "var": np.array([4.0, 9.0], dtype=np.float32),
        }

    def batchnorm_model(self, training_mode=None, *, include_running_outputs=False):
        training_attribute = (
            "" if training_mode is None else f"<training_mode={training_mode}>"
        )
        graph_outputs = (
            "float[2, 2] y, float[2] running_mean, float[2] running_var"
            if include_running_outputs
            else "float[2, 2] y"
        )
        return onnx.parser.parse_model(
            f"""
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (
                float[2, 2] x,
                float[2] scale,
                float[2] bias,
                float[2] mean,
                float[2] var
            ) => ({graph_outputs}) {{
                y, running_mean, running_var =
                    BatchNormalization {training_attribute} (x, scale, bias, mean, var)
            }}
        """
        )

    def legacy_batchnorm_model(self):
        return onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 9]>
            agraph (
                float[2, 2] x,
                float[2] scale,
                float[2] bias,
                float[2] mean,
                float[2] var
            ) => (float[2, 2] y) {
                y, running_mean, running_var, saved_mean, saved_var =
                    BatchNormalization (x, scale, bias, mean, var)
            }
        """
        )

    def test_remove_unused_nodes(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")

    def test_remove_unused_initializers(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z)
            <float two = {2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        self.assertEqual(len(model.graph.initializer), 1)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")
        self.assertEqual(len(model.graph.initializer), 0)

    def test_unused_initialized_inputs_are_kept(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] two) => (float[N] z)
            <float two = {2.0,2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")
        self.assertEqual(len(model.graph.input), 2)
        self.assertEqual(len(model.graph.initializer), 1)

    def test_unused_inputs_are_not_removed(self):
        # preserve inputs as part of interface
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] two) => (float[N] z)
            {
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")
        self.assertEqual(len(model.graph.input), 2)

    def test_partially_used_nodes(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[M] z) {
                w1, w2, w3 = Split (x)
                z = Mul(w3, w3)
            }
        """
        )
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 2)
        self.assertEqual(model.graph.node[0].op_type, "Split")

    def test_remove_unused_optional_outputs_maxpool(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 1, 5, 5] x) => (float[1, 1, 5, 5] z) {
                z, indices = MaxPool <pads = [2, 2, 2, 2], kernel_shape = [5, 5]> (x)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(len(model.graph.node[0].output), 2)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(model.graph.node[0].output, ["z"])

    def test_remove_unused_optional_outputs_dropout_in_function(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[1, 1, 5, 5] x) => (float[1, 1, 5, 5] z)
            {
                z = pkg.custom.afunction (x)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (x) => (z)
            {
                z, indices = MaxPool <pads = [2, 2, 2, 2], kernel_shape = [5, 5]> (x)
            }
        """
        )
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.functions[0].node), 1)
        self.assertEqual(model.functions[0].node[0].op_type, "MaxPool")
        self.assertEqual(len(model.functions[0].node[0].output), 2)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.functions[0].node), 1)
        self.assertEqual(model.functions[0].node[0].op_type, "MaxPool")
        self.assertEqual(model.functions[0].node[0].output, ["z"])

    def test_remove_used_optional_outputs_maxpool(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 1, 5, 5] x) => (float[1, 1, 5, 5] y, float[1, 1, 5, 5] z) {
                y, z = MaxPool <pads = [2, 2, 2, 2], kernel_shape = [5, 5]> (x)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(len(model.graph.node[0].output), 2)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(model.graph.node[0].output, ["y", "z"])

    def test_remove_multiple_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 3, 5, 5] x) => (float[1, 3, 5, 5] z) {
                scale = Constant <value_ints=[3]> ()
                B = Constant <value_ints=[3]> ()
                z, mean, InvStdDev = LayerNormalization(x, scale, B)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 3)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(list(model.graph.node[2].output), ["z"])

    def test_remove_trailing_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 3, 5, 5] x) => (float[1, 3, 5, 5] z, float[1, 3, 5, 5] mean) {
                scale = Constant <value_ints=[3]> ()
                B = Constant <value_ints=[3]> ()
                z, mean, InvStdDev = LayerNormalization(x, scale, B)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 3)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(list(model.graph.node[2].output), ["z", "mean"])

    def test_avoid_remove_non_trailing_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 3, 5, 5] x) => (float[1, 3, 5, 5] z, float[1, 3, 5, 5] InvStdDev) {
                scale = Constant <value_ints=[3]> ()
                B = Constant <value_ints=[3]> ()
                z, mean, InvStdDev = LayerNormalization(x, scale, B)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 3)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(list(model.graph.node[2].output), ["z", "", "InvStdDev"])

    def test_preserve_unused_optional_outputs_batchnorm_in_training_mode(self):
        model = self.batchnorm_model(training_mode=1)
        feeds = self.batchnorm_feeds()
        original_output = ReferenceEvaluator(model).run(None, feeds)[0]
        expected_output = (feeds["x"] - feeds["x"].mean(axis=0)) / np.sqrt(
            feeds["x"].var(axis=0) + 1e-5
        )
        np.testing.assert_allclose(original_output, expected_output, rtol=1e-5, atol=1e-5)

        model = self.remove_unused_nodes(model)
        node = model.graph.node[0]
        self.assertEqual(list(node.output), ["y", "running_mean", "running_var"])
        self.assertEqual({attr.name: attr.i for attr in node.attribute}, {"training_mode": 1})
        onnx.checker.check_model(model)
        optimized_output = ReferenceEvaluator(model).run(None, feeds)[0]
        np.testing.assert_allclose(optimized_output, original_output, rtol=1e-5, atol=1e-5)

    def test_preserve_unused_outputs_batchnorm_opset9(self):
        model = self.legacy_batchnorm_model()
        onnx.checker.check_model(model)
        feeds = self.batchnorm_feeds()
        original_output = ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
        expected_output = (feeds["x"] - feeds["x"].mean(axis=0)) / np.sqrt(
            feeds["x"].var(axis=0) + 1e-5
        )
        np.testing.assert_allclose(original_output, expected_output, rtol=1e-5, atol=1e-5)

        model = self.remove_unused_nodes(model)
        self.assertEqual(
            list(model.graph.node[0].output),
            ["y", "running_mean", "running_var", "saved_mean", "saved_var"],
        )
        onnx.checker.check_model(model)
        optimized_output = ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
        np.testing.assert_allclose(optimized_output, original_output, rtol=1e-5, atol=1e-5)

    def test_remove_unused_optional_outputs_batchnorm_in_inference_mode(self):
        for training_mode in (None, 0):
            with self.subTest(training_mode=training_mode):
                model = self.remove_unused_nodes(self.batchnorm_model(training_mode))
                node = model.graph.node[0]
                self.assertEqual(list(node.output), ["y"])
                self.assertNotIn("training_mode", {attr.name for attr in node.attribute})
                onnx.checker.check_model(model)

    def test_preserve_batchnorm_graph_outputs(self):
        model = self.batchnorm_model(training_mode=0, include_running_outputs=True)
        model = self.remove_unused_nodes(model)
        node = model.graph.node[0]
        self.assertEqual(list(node.output), ["y", "running_mean", "running_var"])
        onnx.checker.check_model(model)

    def test_preserve_batchnorm_with_referenced_training_mode(self):
        inputs = [ir.Value(name=name) for name in ("x", "scale", "bias", "mean", "var")]
        batch_norm = ir.node(
            "BatchNormalization",
            inputs=inputs,
            attributes={
                "training_mode": ir.RefAttr(
                    "training_mode", "training_mode", ir.AttributeType.INT
                )
            },
            num_outputs=3,
        )
        for output, name in zip(batch_norm.outputs, ("y", "running_mean", "running_var")):
            output.name = name
        function = ir.Function(
            domain="test",
            name="BatchNorm",
            graph=ir.Graph(
                inputs,
                [batch_norm.outputs[0]],
                nodes=[batch_norm],
                opset_imports={"": 17},
                name="function",
            ),
            attributes=[ir.Attr("training_mode", ir.AttributeType.UNDEFINED, None)],
        )
        model = ir.Model(
            ir.Graph([], [], nodes=[], opset_imports={"": 17}, name="main"),
            ir_version=10,
            functions=[function],
        )

        onnx_ir.passes.common.RemoveUnusedNodesPass()(model)

        self.assertEqual(
            [output.name for output in batch_norm.outputs],
            ["y", "running_mean", "running_var"],
        )
        training_mode = batch_norm.attributes["training_mode"]
        self.assertTrue(training_mode.is_ref())
        self.assertEqual(training_mode.ref_attr_name, "training_mode")
        onnx.checker.check_model(ir.serde.serialize_model(model))


class RemoveUnusedFunctionsTest(unittest.TestCase):
    def test_removes_unused_function(self):
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = Relu(x)
            }
            <domain: "custom", opset_import: ["": 17]>
            unused_func (x) => (z) { z = Relu(x) }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)
        self.assertEqual(len(model.functions), 1)

        result = onnx_ir.passes.common.RemoveUnusedFunctionsPass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(len(model.functions), 0)

    def test_keeps_used_function(self):
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = custom.used_func(x)
            }
            <domain: "custom", opset_import: ["": 17]>
            used_func (x) => (z) { z = Relu(x) }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)
        self.assertEqual(len(model.functions), 1)

        result = onnx_ir.passes.common.RemoveUnusedFunctionsPass()(model)

        self.assertFalse(result.modified)
        self.assertEqual(len(model.functions), 1)

    def test_keeps_transitively_used_function(self):
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = custom.outer_func(x)
            }
            <domain: "custom", opset_import: ["": 17, "custom": 1]>
            outer_func (x) => (z) { z = custom.inner_func(x) }
            <domain: "custom", opset_import: ["": 17]>
            inner_func (x) => (z) { z = Relu(x) }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)
        self.assertEqual(len(model.functions), 2)

        result = onnx_ir.passes.common.RemoveUnusedFunctionsPass()(model)

        self.assertFalse(result.modified)
        self.assertEqual(len(model.functions), 2)

    def test_removes_only_unused_functions(self):
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = custom.used_func(x)
            }
            <domain: "custom", opset_import: ["": 17]>
            used_func (x) => (z) { z = Relu(x) }
            <domain: "custom", opset_import: ["": 17]>
            unused_func (x) => (z) { z = Sigmoid(x) }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)
        self.assertEqual(len(model.functions), 2)

        result = onnx_ir.passes.common.RemoveUnusedFunctionsPass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(len(model.functions), 1)
        remaining_names = [f.name for f in model.functions.values()]
        self.assertIn("used_func", remaining_names)


class RemoveUnusedOpsetsTest(unittest.TestCase):
    def test_removes_unused_opset(self):
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "unused_domain": 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = Relu(x)
            }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)
        self.assertIn("unused_domain", model.graph.opset_imports)

        result = onnx_ir.passes.common.RemoveUnusedOpsetsPass()(model)

        self.assertTrue(result.modified)
        self.assertNotIn("unused_domain", model.graph.opset_imports)
        self.assertIn("", model.graph.opset_imports)  # Default domain kept

    def test_keeps_used_opsets(self):
        model_text = """
            <ir_version: 10, opset_import: ["": 17]>
            agraph (float[N] x) => (float[N] z)
            {
                z = Relu(x)
            }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)

        result = onnx_ir.passes.common.RemoveUnusedOpsetsPass()(model)

        self.assertFalse(result.modified)
        self.assertIn("", model.graph.opset_imports)

    def test_process_functions_removes_unused_opset_from_function(self):
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = custom.my_func(x)
            }
            <domain: "custom", opset_import: ["": 17, "unused_in_func": 1]>
            my_func (x) => (z) { z = Relu(x) }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)

        result = onnx_ir.passes.common.RemoveUnusedOpsetsPass(process_functions=True)(model)

        self.assertTrue(result.modified)
        func = next(iter(model.functions.values()))
        self.assertNotIn("unused_in_func", func.opset_imports)

    def test_process_functions_false_skips_function_opsets(self):
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = custom.my_func(x)
            }
            <domain: "custom", opset_import: ["": 17, "unused_in_func": 1]>
            my_func (x) => (z) { z = Relu(x) }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)

        onnx_ir.passes.common.RemoveUnusedOpsetsPass(process_functions=False)(model)

        # The function's unused opset should NOT be removed
        func = next(iter(model.functions.values()))
        self.assertIn("unused_in_func", func.opset_imports)

    def test_default_domain_always_retained(self):
        """Even if no nodes use the default domain, it's retained."""
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z)
            {
                z = custom.my_func(x)
            }
            <domain: "custom", opset_import: ["": 17]>
            my_func (x) => (z) { z = Relu(x) }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)

        onnx_ir.passes.common.RemoveUnusedOpsetsPass()(model)

        # Default domain always retained
        self.assertIn("", model.graph.opset_imports)


class RemoveUnusedNodesSubgraphTest(unittest.TestCase):
    """Test that unused removal works on subgraphs in attributes."""

    def test_remove_unused_nodes_in_if_subgraph(self):
        x = ir.Value(name="x")
        cond = ir.Value(name="cond")

        # Then branch: has an unused Relu and a used Sigmoid
        then_out = ir.Value(name="then_z")
        then_unused = ir.Node("", "Relu", [x], num_outputs=1, name="unused_relu")
        then_used = ir.Node("", "Sigmoid", [x], outputs=[then_out], name="used_sigmoid")
        then_graph = ir.Graph(
            [], [then_out], nodes=[then_unused, then_used], name="then_graph"
        )

        # Else branch
        else_out = ir.Value(name="else_z")
        else_node = ir.Node("", "Tanh", [x], outputs=[else_out], name="else_tanh")
        else_graph = ir.Graph([], [else_out], nodes=[else_node], name="else_graph")

        z = ir.Value(name="z")
        if_node = ir.Node(
            "",
            "If",
            [cond],
            outputs=[z],
            attributes=[
                ir.AttrGraph("then_branch", then_graph),
                ir.AttrGraph("else_branch", else_graph),
            ],
        )
        graph = ir.Graph([cond, x], [z], nodes=[if_node], opset_imports={"": 17}, name="main")
        model = ir.Model(graph, ir_version=10)

        result = onnx_ir.passes.common.RemoveUnusedNodesPass()(model)

        self.assertTrue(result.modified)
        # Check the then_branch has unused node removed
        updated_then = if_node.attributes["then_branch"].as_graph()
        op_types = [n.op_type for n in updated_then]
        self.assertNotIn("Relu", op_types)
        self.assertIn("Sigmoid", op_types)


if __name__ == "__main__":
    unittest.main()
