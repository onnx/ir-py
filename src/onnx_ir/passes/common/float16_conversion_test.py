# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for float16 conversion pass."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir.passes.common.float16_conversion import (
    DEFAULT_OP_BLOCK_LIST,
    ConvertFloatToFloat16Pass,
    convert_np_to_float16,
)


class ConvertNpToFloat16Test(unittest.TestCase):
    """Test the convert_np_to_float16 utility function."""

    def test_basic_conversion(self):
        """Test basic float32 to float16 conversion."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = convert_np_to_float16(arr)
        self.assertEqual(result.dtype, np.float16)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0], dtype=np.float16))

    def test_clamping_small_positive(self):
        """Test clamping of small positive values."""
        arr = np.array([1e-8, 1e-7, 1e-6], dtype=np.float32)
        result = convert_np_to_float16(arr, min_positive_val=1e-7, max_finite_val=1e4)
        # Values less than min_positive_val should be clamped
        self.assertGreaterEqual(result[0], 1e-7)

    def test_clamping_large_positive(self):
        """Test clamping of large positive values."""
        arr = np.array([1e3, 1e4, 1e5], dtype=np.float32)
        result = convert_np_to_float16(arr, min_positive_val=1e-7, max_finite_val=1e4)
        # Values greater than max_finite_val should be clamped
        self.assertLessEqual(result[2], 1e4)

    def test_special_values(self):
        """Test that special values (NaN, inf, -inf, 0) are preserved."""
        arr = np.array([0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
        result = convert_np_to_float16(arr)
        self.assertEqual(result[0], 0.0)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isinf(result[2]) and result[2] > 0)
        self.assertTrue(np.isinf(result[3]) and result[3] < 0)


class ConvertFloatToFloat16PassTest(unittest.TestCase):
    """Test the ConvertFloatToFloat16Pass."""

    def test_simple_model_conversion(self):
        """Test conversion of a simple model with float32 initializers."""
        # Create initializer
        const_tensor = ir.tensor([1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT, name="const")
        const_val = ir.Value(
            name="const",
            const_value=const_tensor,
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        # Create input
        input_val = ir.Value(name="input", type=ir.TensorType(ir.DataType.FLOAT))

        # Create Add node
        add_node = ir.Node("", "Add", inputs=[input_val, const_val])
        add_node.outputs[0].name = "output"
        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            initializers=[const_val],
        )

        model = ir.Model(graph, ir_version=8)

        # Apply the conversion pass
        pass_ = ConvertFloatToFloat16Pass(keep_io_types=False)
        result = pass_(model)

        self.assertTrue(result.modified)

        # Check that initializer is converted to float16
        converted_const = result.model.graph.initializers["const"]
        self.assertEqual(converted_const.const_value.dtype, ir.DataType.FLOAT16)

        # Check that input and output types are converted
        self.assertEqual(result.model.graph.inputs[0].type.dtype, ir.DataType.FLOAT16)
        self.assertEqual(result.model.graph.outputs[0].type.dtype, ir.DataType.FLOAT16)

    def test_op_block_list(self):
        """Test that operators in the default block list exist."""
        # Just check that the default block list contains expected operators
        self.assertIn("Resize", DEFAULT_OP_BLOCK_LIST)
        self.assertIn("NonMaxSuppression", DEFAULT_OP_BLOCK_LIST)
        self.assertIn("TopK", DEFAULT_OP_BLOCK_LIST)

    def test_no_conversion_for_non_float32(self):
        """Test that non-float32 types are not converted."""
        # Create input with int32 type
        input_val = ir.Value(name="input", type=ir.TensorType(ir.DataType.INT32))

        # Create node
        identity_node = ir.Node("", "Identity", inputs=[input_val])
        identity_node.outputs[0].name = "output"
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.INT32)

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[identity_node.outputs[0]],
            nodes=[identity_node],
        )

        model = ir.Model(graph, ir_version=8)

        # Apply the conversion pass
        pass_ = ConvertFloatToFloat16Pass(keep_io_types=False)
        result = pass_(model)

        # The model should not be modified (no float32 types to convert)
        self.assertFalse(result.modified)

        # Check that types remain INT32
        self.assertEqual(result.model.graph.inputs[0].type.dtype, ir.DataType.INT32)
        self.assertEqual(result.model.graph.outputs[0].type.dtype, ir.DataType.INT32)

    def test_already_float16_model(self):
        """Test that already converted float16 models are handled correctly."""
        # Create input with float16 type
        input_val = ir.Value(name="input", type=ir.TensorType(ir.DataType.FLOAT16))

        # Create node
        identity_node = ir.Node("", "Identity", inputs=[input_val])
        identity_node.outputs[0].name = "output"
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT16)

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[identity_node.outputs[0]],
            nodes=[identity_node],
        )

        model = ir.Model(graph, ir_version=8)

        # Apply the conversion pass
        pass_ = ConvertFloatToFloat16Pass(keep_io_types=False)
        result = pass_(model)

        # The model should not be modified (already float16)
        self.assertFalse(result.modified)

    def test_tensor_attributes_conversion(self):
        """Test that tensor attributes in nodes are converted."""
        # Create a Constant node with a float32 tensor attribute
        const_tensor = ir.tensor([1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT)

        constant_node = ir.Node(
            "",
            "Constant",
            inputs=[],
            attributes={"value": ir.AttrTensor("value", const_tensor)},
        )
        constant_node.outputs[0].name = "output"
        constant_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[],
            outputs=[constant_node.outputs[0]],
            nodes=[constant_node],
        )

        model = ir.Model(graph, ir_version=8)

        # Apply the conversion pass
        pass_ = ConvertFloatToFloat16Pass(keep_io_types=False)
        result = pass_(model)

        self.assertTrue(result.modified)

        # Check that the tensor attribute is converted to float16
        constant_node_converted = next(iter(result.model.graph))
        value_attr = constant_node_converted.attributes["value"]
        self.assertIsInstance(value_attr, ir.Attr)
        self.assertEqual(value_attr.value.dtype, ir.DataType.FLOAT16)

    def test_custom_op_block_list(self):
        """Test using a custom op_block_list."""
        # Create a simple model with an Add node
        input_val = ir.Value(name="input", type=ir.TensorType(ir.DataType.FLOAT))

        add_node = ir.Node("", "Add", inputs=[input_val, input_val])
        add_node.outputs[0].name = "output"
        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
        )

        model = ir.Model(graph, ir_version=8)

        # Apply the conversion pass with custom block list (blocking Add)
        pass_ = ConvertFloatToFloat16Pass(op_block_list=["Add"], keep_io_types=False)
        result = pass_(model)

        # The model should be modified (input/output types changed)
        # But the Add node's outputs are blocked
        self.assertTrue(result.modified)

        # Input types should be converted
        self.assertEqual(result.model.graph.inputs[0].type.dtype, ir.DataType.FLOAT16)

    def test_node_block_list(self):
        """Test blocking specific nodes by name."""
        # Create a model with two Add nodes
        input_val = ir.Value(name="input", type=ir.TensorType(ir.DataType.FLOAT))

        # First Add node (blocked)
        add1_node = ir.Node("", "Add", inputs=[input_val, input_val], name="blocked_add")
        add1_node.outputs[0].name = "intermediate"
        add1_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Second Add node (not blocked)
        add2_node = ir.Node(
            "", "Add", inputs=[add1_node.outputs[0], add1_node.outputs[0]], name="normal_add"
        )
        add2_node.outputs[0].name = "output"
        add2_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[add2_node.outputs[0]],
            nodes=[add1_node, add2_node],
        )

        model = ir.Model(graph, ir_version=8)

        # Apply the conversion pass with node block list
        pass_ = ConvertFloatToFloat16Pass(
            op_block_list=frozenset(),  # Empty op block list
            node_block_list=["blocked_add"],
            keep_io_types=False,
        )
        result = pass_(model)

        # The model should be modified
        self.assertTrue(result.modified)

        # The normal_add node's outputs should be converted
        normal_add = list(result.model.graph)[1]
        self.assertEqual(normal_add.outputs[0].type.dtype, ir.DataType.FLOAT16)


if __name__ == "__main__":
    unittest.main()
