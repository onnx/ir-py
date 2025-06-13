# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DeduplicateInitializersPass."""

import unittest

import onnx

import onnx_ir as ir
import onnx_ir.passes.common.deduplicate_initializers as dedup_pass


class DeduplicateInitializersTest(unittest.TestCase):
    def apply_pass(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model_ir = ir.serde.deserialize_model(model)
        dedup_pass.DeduplicateInitializersPass()(model_ir)
        return ir.serde.serialize_model(model_ir)

    def test_deduplicates_identical_initializers(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[3] w1 = {1.0, 2.0, 3.0}, float[3] w2 = {1.0, 2.0, 3.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        self.assertEqual(len(model.graph.initializer), 2)
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 1)
        add_node = new_model.graph.node[0]
        self.assertEqual(add_node.input[0], add_node.input[1])

    def test_initializers_with_different_shapes_not_deduplicated(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, float[3] w2 = {1.0, 2.0, 3.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 2)

    def test_initializers_with_different_dtypes_not_deduplicated(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, double[2] w2 = {1.0, 2.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 2)

    def test_scalar_initializer_deduplication(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float w1 = {5.0}, float w2 = {5.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 1)

    def test_multiple_duplicates(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 1.0}, float[2] w2 = {1.0, 1.0}, float[2] w3 = {1.0, 1.0}> {
                temp = Add(w1, w2)
                out = Add(temp, w3)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 1)

    def test_unique_values_not_deduplicated(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, float[2] w2 = {2.0, 1.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 2)

    def test_deduplication_within_subgraph(self):
        # Create a shared initializer tensor
        shared_init = onnx.helper.make_tensor(
            name="w_shared",
            data_type=onnx.TensorProto.FLOAT,
            dims=[3],
            vals=[1.0, 2.0, 3.0],
        )

        # Then branch: w1 and w2 use the same initializer
        w1 = onnx.helper.make_tensor_value_info("w1", onnx.TensorProto.FLOAT, [3])
        w2 = onnx.helper.make_tensor_value_info("w2", onnx.TensorProto.FLOAT, [3])
        sum_out = onnx.helper.make_tensor_value_info("sum", onnx.TensorProto.FLOAT, [3])
        then_node = onnx.helper.make_node("Add", ["w1", "w2"], ["sum"])
        then_branch = onnx.helper.make_graph(
            [then_node],
            "then_branch",
            inputs=[],
            outputs=[sum_out],
            value_info=[w1, w2],
            initializer=[shared_init, shared_init],
        )

        # Else branch: w3 uses the same initializer
        w3 = onnx.helper.make_tensor_value_info("w3", onnx.TensorProto.FLOAT, [3])
        identity_out = onnx.helper.make_tensor_value_info(
            "identity", onnx.TensorProto.FLOAT, [3]
        )
        else_node = onnx.helper.make_node("Identity", ["w3"], ["identity"])
        else_branch = onnx.helper.make_graph(
            [else_node],
            "else_branch",
            inputs=[],
            outputs=[identity_out],
            value_info=[w3],
            initializer=[shared_init],
        )

        # If node with then and else branches
        cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
        if_out = onnx.helper.make_tensor_value_info("if_out", onnx.TensorProto.FLOAT, [3])
        if_node = onnx.helper.make_node(
            "If",
            ["cond"],
            ["if_out"],
            then_branch=then_branch,
            else_branch=else_branch,
        )

        # Main graph
        main_graph = onnx.helper.make_graph(
            [if_node],
            "main_graph",
            inputs=[cond],
            outputs=[if_out],
        )
        model = onnx.helper.make_model(main_graph)

        # Run deduplication pass
        new_model = self.apply_pass(model)

        # Assert only one unique initializer remains across subgraphs
        then_inits = new_model.graph.node[0].attribute[0].g.initializer
        else_inits = new_model.graph.node[0].attribute[1].g.initializer
        initializer_names = {t.name for t in list(then_inits) + list(else_inits)}
        self.assertEqual(len(initializer_names), 1)


if __name__ == "__main__":
    unittest.main()
