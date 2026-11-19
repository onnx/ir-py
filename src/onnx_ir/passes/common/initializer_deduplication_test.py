# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the initializer_deduplication passes."""

import unittest

import numpy as np
import onnx
import parameterized

import onnx_ir as ir
from onnx_ir.passes.common import initializer_deduplication


@parameterized.parameterized_class(
    [
        {
            "name": "DeduplicateInitializersPass",
            "pass_class": initializer_deduplication.DeduplicateInitializersPass,
        },
        {
            "name": "DeduplicateHashedInitializersPass",
            "pass_class": initializer_deduplication.DeduplicateHashedInitializersPass,
        },
    ]
)
class DeduplicateInitializersTest(unittest.TestCase):
    name: str
    pass_class: type[ir.passes.InPlacePass]

    def apply_pass(self, model: ir.Model) -> ir.Model:
        result = self.pass_class()(model)
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

    def test_deduplicates_identical_string_initializers(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <string[2] s1 = {"A", "B"}, string[2] s2 = {"A", "B"}> {
            }
            """
        )
        self.assertEqual(len(model.graph.initializers), 2)
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 1)

    def test_deduplicates_identical_multibyte_string_initializers(self):
        # Multi-byte strings are separate bytes objects in each initializer
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <string[2] s1 = {"AB", "CD"}, string[2] s2 = {"AB", "CD"}> {
            }
            """
        )
        self.assertEqual(len(model.graph.initializers), 2)
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 1)

    def test_string_initializers_differing_by_trailing_nul_not_deduplicated(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <string[2] s1 = {"a", "bc"}, string[2] s2 = {"a", "bc"}> {
            }
            """
        )
        model.graph.initializers["s2"].const_value = ir.StringTensor(
            [b"a\x00", b"bc"], shape=ir.Shape([2])
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 2)

    def test_deduplicates_numpy_backed_string_initializers(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <string[1] s1 = {"AB"}, string[1] s2 = {"AB"}> {
            }
            """
        )
        # ir.tensor() wraps a numpy string array in a plain Tensor, not a StringTensor
        model.graph.initializers["s1"].const_value = ir.tensor(np.array([b"AB"]))
        model.graph.initializers["s2"].const_value = ir.tensor(np.array([b"AB"]))
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 1)

    @parameterized.parameterized.expand(
        [
            ("equal_bytes", [b"AB", b"\xff\x00"], [b"AB", b"\xff\x00"], True),
            ("trailing_nul", [b"a"], [b"a\x00"], False),
            ("empty_string", [b""], [b"\x00"], False),
            ("element_boundaries", [b"a\x00", b"b"], [b"a", b"\x00b"], False),
            ("empty_tensor", [], [], True),
        ]
    )
    def test_proto_backed_string_initializers(self, _, left, right, should_merge):
        initializers = [
            ir.val(
                name,
                const_value=ir.serde.TensorProtoTensor(
                    onnx.TensorProto(
                        name=name,
                        data_type=onnx.TensorProto.STRING,
                        dims=[len(data)],
                        string_data=data,
                    )
                ),
            )
            for name, data in (("left", left), ("right", right))
        ]
        concat = ir.node("Concat", inputs=initializers, attributes={"axis": 0})
        model = ir.Model(
            ir.Graph(
                inputs=[],
                outputs=concat.outputs,
                nodes=[concat],
                initializers=initializers,
                opset_imports={"": 18},
            ),
            ir_version=8,
        )
        result = self.pass_class()(model)
        self.assertEqual(result.modified, should_merge)
        self.assertEqual(len(model.graph.initializers), 1 if should_merge else 2)
        self.assertEqual(concat.inputs[0] is concat.inputs[1], should_merge)
        for value, expected in zip(concat.inputs, (left, right)):
            self.assertEqual(
                list(ir.serde.serialize_tensor(value.const_value).string_data), expected
            )

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

    def test_string_initializers_with_different_shapes_not_deduplicated(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <string[2] s1 = {"A", "B"}, string[1,2] s2 = {"A", "B"}> {
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializers), 2)

    def test_string_initializers_with_same_bytes_but_different_grouping_not_deduplicated(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <string[2] s1 = {"AB", "C"}, string[2] s2 = {"A", "BC"}> {
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

    def test_deduplication_in_subgraphs(self):
        a = ir.Value(
            name="a",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(()),
            const_value=ir.tensor(1),
        )
        b = ir.Value(
            name="b",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(()),
            const_value=ir.tensor(1),
        )
        c = ir.Value(
            name="c",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(()),
            const_value=ir.tensor(1),
        )

        node_with_subgraph = ir.node(
            name="node_with_subgraph",
            op_type="SubgraphOp",
            inputs=[],
            outputs=[c],
            attributes={
                "subgraph": ir.Graph(
                    inputs=[],
                    outputs=[],
                    nodes=[],
                    initializers=[b, c],
                    name="subgraph",
                )
            },
        )

        main_graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[node_with_subgraph],
            initializers=[a],
            name="main_graph",
        )
        model = ir.Model(main_graph, ir_version=10)
        self.apply_pass(model)
        self.assertEqual(len(model.graph.initializers), 1)
        np.testing.assert_equal(
            model.graph.initializers["a"].const_value.numpy(), np.array(1, dtype=np.int64)
        )
        subgraph_initializers = (
            node_with_subgraph.attributes["subgraph"].as_graph().initializers
        )
        self.assertEqual(len(subgraph_initializers), 1)
        np.testing.assert_equal(
            subgraph_initializers["b"].const_value.numpy(), np.array(1, dtype=np.int64)
        )


if __name__ == "__main__":
    unittest.main()
