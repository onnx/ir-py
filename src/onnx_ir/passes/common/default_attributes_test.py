# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common import default_attributes


class TestAddDefaultAttributesPass(unittest.TestCase):
    """Test the AddDefaultAttributes pass."""

    def test_add_default_attributes_to_conv(self):
        """Test adding default attributes to a Conv node."""
        # Create a Conv node without optional attributes
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight_val = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 7, 7))
        )

        conv_node = ir.node(
            "Conv",
            inputs=[input_val, weight_val],
            num_outputs=1,
        )

        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, weight_val],
                outputs=conv_node.outputs,
                nodes=[conv_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Verify the node doesn't have the default attributes initially
        self.assertNotIn("group", conv_node.attributes)
        self.assertNotIn("auto_pad", conv_node.attributes)

        # Apply the pass
        pass_instance = default_attributes.AddDefaultAttributesPass()
        result = pass_instance(model)

        # Check that the pass was applied
        self.assertTrue(result.modified)

        # Check that default attributes were added
        self.assertIn("group", conv_node.attributes)
        self.assertEqual(conv_node.attributes["group"].value, 1)
        self.assertIn("auto_pad", conv_node.attributes)
        self.assertEqual(conv_node.attributes["auto_pad"].value, "NOTSET")

    def test_add_default_attributes_to_batchnorm(self):
        """Test adding default attributes to a BatchNormalization node."""
        # Create a BatchNormalization node without optional attributes
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        scale_val = ir.Value(
            name="scale", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        bias_val = ir.Value(
            name="bias", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        mean_val = ir.Value(
            name="mean", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        var_val = ir.Value(
            name="var", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )

        bn_node = ir.node(
            "BatchNormalization",
            inputs=[input_val, scale_val, bias_val, mean_val, var_val],
            num_outputs=1,
        )

        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, scale_val, bias_val, mean_val, var_val],
                outputs=bn_node.outputs,
                nodes=[bn_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Verify the node doesn't have the default attributes initially
        self.assertNotIn("epsilon", bn_node.attributes)
        self.assertNotIn("momentum", bn_node.attributes)
        self.assertNotIn("training_mode", bn_node.attributes)

        # Apply the pass
        pass_instance = default_attributes.AddDefaultAttributesPass()
        result = pass_instance(model)

        # Check that the pass was applied
        self.assertTrue(result.modified)

        # Check that default attributes were added
        self.assertIn("epsilon", bn_node.attributes)
        self.assertAlmostEqual(bn_node.attributes["epsilon"].value, 1e-5)
        self.assertIn("momentum", bn_node.attributes)
        self.assertAlmostEqual(bn_node.attributes["momentum"].value, 0.9)
        self.assertIn("training_mode", bn_node.attributes)
        self.assertEqual(bn_node.attributes["training_mode"].value, 0)

    def test_existing_attributes_not_overwritten(self):
        """Test that existing attributes are not overwritten."""
        # Create a Conv node with some attributes already set
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight_val = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 7, 7))
        )

        conv_node = ir.node(
            "Conv",
            inputs=[input_val, weight_val],
            attributes={
                "group": ir.Attr("group", ir.AttributeType.INT, 2),
            },
            num_outputs=1,
        )

        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, weight_val],
                outputs=conv_node.outputs,
                nodes=[conv_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = default_attributes.AddDefaultAttributesPass()
        result = pass_instance(model)

        # Check that the pass was applied (added auto_pad but not group)
        self.assertTrue(result.modified)

        # Check that existing attribute was not overwritten
        self.assertEqual(conv_node.attributes["group"].value, 2)

        # Check that other default attributes were added
        self.assertIn("auto_pad", conv_node.attributes)
        self.assertEqual(conv_node.attributes["auto_pad"].value, "NOTSET")

    def test_node_with_no_default_attributes(self):
        """Test a node that has no default attributes."""
        # Create a Relu node (which has no attributes at all)
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )

        relu_node = ir.node(
            "Relu",
            inputs=[input_val],
            num_outputs=1,
        )

        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val],
                outputs=relu_node.outputs,
                nodes=[relu_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = default_attributes.AddDefaultAttributesPass()
        result = pass_instance(model)

        # Check that the pass didn't modify anything
        self.assertFalse(result.modified)
        self.assertEqual(len(relu_node.attributes), 0)

    def test_add_default_attributes_to_pad(self):
        """Test adding default attributes to a Pad node."""
        # Create a Pad node without the mode attribute
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        pads_val = ir.Value(
            name="pads", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape((8,))
        )

        pad_node = ir.node(
            "Pad",
            inputs=[input_val, pads_val],
            num_outputs=1,
        )

        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, pads_val],
                outputs=pad_node.outputs,
                nodes=[pad_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Verify the node doesn't have the mode attribute initially
        self.assertNotIn("mode", pad_node.attributes)

        # Apply the pass
        pass_instance = default_attributes.AddDefaultAttributesPass()
        result = pass_instance(model)

        # Check that the pass was applied
        self.assertTrue(result.modified)

        # Check that mode attribute was added with default value
        self.assertIn("mode", pad_node.attributes)
        self.assertEqual(pad_node.attributes["mode"].value, "constant")

    def test_add_default_attributes_in_subgraph(self):
        """Test adding default attributes to nodes in a subgraph."""
        # Create a simple If node with subgraphs
        cond_val = ir.Value(
            name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(())
        )
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight_val = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 7, 7))
        )

        # Create Conv node in then branch
        then_input = ir.Value(
            name="then_input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        then_weight = ir.Value(
            name="then_weight",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((64, 3, 7, 7)),
        )
        then_conv = ir.node(
            "Conv",
            inputs=[then_input, then_weight],
            num_outputs=1,
        )
        then_branch = ir.Graph(
            inputs=[then_input, then_weight],
            outputs=then_conv.outputs,
            nodes=[then_conv],
            opset_imports={"": 20},
        )

        # Create Identity node in else branch
        else_input = ir.Value(
            name="else_input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        else_identity = ir.node(
            "Identity",
            inputs=[else_input],
            num_outputs=1,
        )
        else_branch = ir.Graph(
            inputs=[else_input],
            outputs=else_identity.outputs,
            nodes=[else_identity],
            opset_imports={"": 20},
        )

        # Create If node
        if_node = ir.node(
            "If",
            inputs=[cond_val],
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, then_branch),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, else_branch),
            },
            num_outputs=1,
        )

        model = ir.Model(
            graph=ir.Graph(
                inputs=[cond_val, input_val, weight_val],
                outputs=if_node.outputs,
                nodes=[if_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Verify the Conv node in subgraph doesn't have default attributes
        self.assertNotIn("group", then_conv.attributes)

        # Apply the pass
        pass_instance = default_attributes.AddDefaultAttributesPass()
        result = pass_instance(model)

        # Check that the pass was applied
        self.assertTrue(result.modified)

        # Check that default attributes were added to Conv in subgraph
        self.assertIn("group", then_conv.attributes)
        self.assertEqual(then_conv.attributes["group"].value, 1)


if __name__ == "__main__":
    unittest.main()
