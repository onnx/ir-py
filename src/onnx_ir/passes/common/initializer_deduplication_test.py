# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the initializer_deduplication passes."""

import unittest

import onnx_ir as ir
from onnx_ir.passes.common import initializer_deduplication


class DeduplicateInitializersTest(unittest.TestCase):
    def apply_pass(self, model: ir.Model) -> ir.Model:
        result = initializer_deduplication.DeduplicateInitializersPass()(model)
        return result.model

    def test_deduplicates_identical_initializers(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[3] w1 = {1.0, 2.0, 3.0}, float[3] w2 = {1.0, 2.0, 3.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        self.assertEqual(len(model.graph.initializers), 2)
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 1)
        add_node = new_model.graph[0]
        self.assertEqual(add_node.inputs[0], add_node.inputs[1])

    def test_initializers_with_different_shapes_not_deduplicated(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, float[1,2] w2 = {1.0, 2.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 2)

    def test_initializers_with_different_dtypes_not_deduplicated(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, double[2] w2 = {1.0, 2.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 2)

    def test_scalar_initializer_deduplication(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float w1 = {5.0}, float w2 = {5.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 1)

    def test_multiple_duplicates(self):
        model = ir.from_onnx_text(
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
        self.assertEqual(len(new_model.graph.initializers), 1)

    def test_unique_values_not_deduplicated(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, float[2] w2 = {2.0, 1.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 2)


if __name__ == "__main__":
    unittest.main()
