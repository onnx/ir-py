# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir.passes.common import remove_default_attributes


class TestRemoveDefaultAttributesPass(unittest.TestCase):
    def test_remove_default_int_attributes(self):
        """Test removal of default int attributes (0, 1, -1)."""
        # Create a ReduceSum node with keepdims=1 (default)
        input_val = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3, 4))
        )
        axes = ir.tensor(np.array([1], dtype=np.int64))
        axes_const = ir.node("Constant", inputs=[], attributes={"value": axes}, num_outputs=1)

        reduce_node = ir.node(
            "ReduceSum",
            inputs=[input_val, axes_const.outputs[0]],
            attributes={"keepdims": 1},  # Default value
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val],
                outputs=reduce_node.outputs,
                nodes=[axes_const, reduce_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that the pass was applied and the attribute was removed
        self.assertTrue(result.modified)
        self.assertNotIn("keepdims", reduce_node.attributes)

    def test_keep_non_default_int_attributes(self):
        """Test that non-default int attributes are kept."""
        # Create a ReduceSum node with keepdims=0 (non-default)
        input_val = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3, 4))
        )
        axes = ir.tensor(np.array([1], dtype=np.int64))
        axes_const = ir.node("Constant", inputs=[], attributes={"value": axes}, num_outputs=1)

        reduce_node = ir.node(
            "ReduceSum",
            inputs=[input_val, axes_const.outputs[0]],
            attributes={"keepdims": 0},  # Non-default value
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val],
                outputs=reduce_node.outputs,
                nodes=[axes_const, reduce_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that the attribute was NOT removed
        self.assertFalse(result.modified)
        self.assertIn("keepdims", reduce_node.attributes)
        self.assertEqual(reduce_node.attributes["keepdims"].value, 0)

    def test_conv_pads_all_zeros(self):
        """Test removal of pads attribute when all zeros for Conv."""
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 3, 3))
        )

        conv_node = ir.node(
            "Conv",
            inputs=[input_val, weight],
            attributes={"pads": [0, 0, 0, 0]},  # All zeros - should be removed
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, weight],
                outputs=conv_node.outputs,
                nodes=[conv_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that pads was removed
        self.assertTrue(result.modified)
        self.assertNotIn("pads", conv_node.attributes)

    def test_conv_pads_not_all_zeros(self):
        """Test that pads attribute is kept when not all zeros for Conv."""
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 3, 3))
        )

        conv_node = ir.node(
            "Conv",
            inputs=[input_val, weight],
            attributes={"pads": [1, 1, 1, 1]},  # Not all zeros - should be kept
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, weight],
                outputs=conv_node.outputs,
                nodes=[conv_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that pads was NOT removed
        self.assertFalse(result.modified)
        self.assertIn("pads", conv_node.attributes)

    def test_conv_strides_all_ones(self):
        """Test removal of strides attribute when all ones for Conv."""
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 3, 3))
        )

        conv_node = ir.node(
            "Conv",
            inputs=[input_val, weight],
            attributes={"strides": [1, 1]},  # All ones - should be removed
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, weight],
                outputs=conv_node.outputs,
                nodes=[conv_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that strides was removed
        self.assertTrue(result.modified)
        self.assertNotIn("strides", conv_node.attributes)

    def test_conv_strides_not_all_ones(self):
        """Test that strides attribute is kept when not all ones for Conv."""
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 3, 3))
        )

        conv_node = ir.node(
            "Conv",
            inputs=[input_val, weight],
            attributes={"strides": [2, 2]},  # Not all ones - should be kept
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, weight],
                outputs=conv_node.outputs,
                nodes=[conv_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that strides was NOT removed
        self.assertFalse(result.modified)
        self.assertIn("strides", conv_node.attributes)

    def test_conv_group_default(self):
        """Test removal of group attribute when set to default value 1 for Conv."""
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 3, 3))
        )

        conv_node = ir.node(
            "Conv",
            inputs=[input_val, weight],
            attributes={"group": 1},  # Default value
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, weight],
                outputs=conv_node.outputs,
                nodes=[conv_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that group was removed
        self.assertTrue(result.modified)
        self.assertNotIn("group", conv_node.attributes)

    def test_keep_float_default_attributes(self):
        """Test that float default attributes are NOT removed."""
        # BatchNormalization has epsilon with default 1e-5 and momentum with default 0.9
        # These should NOT be removed as they are not 0, 1, or -1
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        scale = ir.Value(
            name="scale", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        bias = ir.Value(
            name="bias", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        mean = ir.Value(
            name="mean", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        var = ir.Value(name="var", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,)))

        bn_node = ir.node(
            "BatchNormalization",
            inputs=[input_val, scale, bias, mean, var],
            attributes={
                "epsilon": 1e-5,
                "momentum": 0.9,
            },  # Float defaults - should NOT be removed
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, scale, bias, mean, var],
                outputs=bn_node.outputs,
                nodes=[bn_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Float defaults should be kept for clarity
        self.assertFalse(result.modified)
        self.assertIn("epsilon", bn_node.attributes)
        self.assertIn("momentum", bn_node.attributes)

    def test_remove_training_mode_default(self):
        """Test removal of training_mode=0 default for BatchNormalization."""
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        scale = ir.Value(
            name="scale", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        bias = ir.Value(
            name="bias", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        mean = ir.Value(
            name="mean", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,))
        )
        var = ir.Value(name="var", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3,)))

        bn_node = ir.node(
            "BatchNormalization",
            inputs=[input_val, scale, bias, mean, var],
            attributes={"training_mode": 0},  # Default value - should be removed
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, scale, bias, mean, var],
                outputs=bn_node.outputs,
                nodes=[bn_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that training_mode was removed
        self.assertTrue(result.modified)
        self.assertNotIn("training_mode", bn_node.attributes)

    def test_no_modification_when_no_default_attributes(self):
        """Test that pass doesn't modify when no default attributes exist."""
        input_val = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))
        )

        relu_node = ir.node("Relu", inputs=[input_val], num_outputs=1)
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
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # No modifications should be made
        self.assertFalse(result.modified)

    def test_multiple_nodes_with_mixed_attributes(self):
        """Test removal of default attributes across multiple nodes."""
        input_val = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((1, 3, 224, 224)),
        )
        weight = ir.Value(
            name="weight", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((64, 3, 3, 3))
        )

        # Conv with default pads and strides
        conv_node = ir.node(
            "Conv",
            inputs=[input_val, weight],
            attributes={"pads": [0, 0, 0, 0], "strides": [1, 1], "group": 1},
            num_outputs=1,
        )

        # Relu (no attributes)
        relu_node = ir.node("Relu", inputs=[conv_node.outputs[0]], num_outputs=1)

        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val, weight],
                outputs=relu_node.outputs,
                nodes=[conv_node, relu_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that all default attributes were removed from Conv
        self.assertTrue(result.modified)
        self.assertNotIn("pads", conv_node.attributes)
        self.assertNotIn("strides", conv_node.attributes)
        self.assertNotIn("group", conv_node.attributes)

    def test_node_version_takes_precedence(self):
        """Test that node.version takes precedence over graph opset version."""
        # Create a node with a specific version set
        input_val = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3, 4))
        )
        axes = ir.tensor(np.array([1], dtype=np.int64))
        axes_const = ir.node("Constant", inputs=[], attributes={"value": axes}, num_outputs=1)

        # Create ReduceSum node with keepdims=1 (default) and explicit version
        reduce_node = ir.node(
            "ReduceSum",
            inputs=[input_val, axes_const.outputs[0]],
            attributes={"keepdims": 1},  # keepdims=1 is the default for ReduceSum in opset 20
            num_outputs=1,
            version=20,  # Explicit version
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_val],
                outputs=reduce_node.outputs,
                nodes=[axes_const, reduce_node],
                opset_imports={"": 18},  # Different graph opset version
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # keepdims=1 should still be removed because node.version=20 is used
        self.assertTrue(result.modified)
        self.assertNotIn("keepdims", reduce_node.attributes)

    def test_remove_float_default_attributes_1_0(self):
        """Test removal of float attributes with default value 1.0."""
        # Create a Gemm node with alpha=1.0 and beta=1.0 (both defaults)
        A = ir.Value(name="A", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3)))
        B = ir.Value(name="B", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3, 4)))
        C = ir.Value(name="C", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 4)))

        gemm_node = ir.node(
            "Gemm",
            inputs=[A, B, C],
            attributes={"alpha": 1.0, "beta": 1.0},  # Both are default values
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[A, B, C],
                outputs=gemm_node.outputs,
                nodes=[gemm_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that both attributes were removed
        self.assertTrue(result.modified)
        self.assertNotIn("alpha", gemm_node.attributes)
        self.assertNotIn("beta", gemm_node.attributes)

    def test_keep_non_default_float_attributes(self):
        """Test that non-default float attributes are kept."""
        # Create a Gemm node with alpha=2.0 (non-default)
        A = ir.Value(name="A", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3)))
        B = ir.Value(name="B", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3, 4)))
        C = ir.Value(name="C", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 4)))

        gemm_node = ir.node(
            "Gemm",
            inputs=[A, B, C],
            attributes={"alpha": 2.0, "beta": 1.0},  # alpha is non-default, beta is default
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[A, B, C],
                outputs=gemm_node.outputs,
                nodes=[gemm_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that alpha was kept and beta was removed
        self.assertTrue(result.modified)
        self.assertIn("alpha", gemm_node.attributes)
        self.assertEqual(gemm_node.attributes["alpha"].value, 2.0)
        self.assertNotIn("beta", gemm_node.attributes)

    def test_remove_float_default_attributes_0_0(self):
        """Test removal of float attributes with default value 0.0."""
        # RandomNormal has mean=0.0 as default
        input_shape = ir.Value(
            name="shape", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape((3,))
        )

        random_normal_node = ir.node(
            "RandomNormal",
            inputs=[input_shape],
            attributes={
                "mean": 0.0,
                "scale": 1.0,
            },  # mean=0.0 is default, scale=1.0 is default
            num_outputs=1,
        )
        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_shape],
                outputs=random_normal_node.outputs,
                nodes=[random_normal_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Apply the pass
        pass_instance = remove_default_attributes.RemoveDefaultAttributesPass()
        result = pass_instance(model)

        # Check that both attributes were removed (both are defaults)
        self.assertTrue(result.modified)
        self.assertNotIn("mean", random_normal_node.attributes)
        self.assertNotIn("scale", random_normal_node.attributes)


if __name__ == "__main__":
    unittest.main()
