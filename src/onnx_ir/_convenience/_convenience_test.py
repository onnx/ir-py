# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import parameterized

import onnx_ir as ir
from onnx_ir import _convenience


class GetConstantTensorTest(unittest.TestCase):
    def test_direct_const_value(self):
        # Test when value has a direct const_value
        tensor = ir.Tensor(np.array([1, 2, 3], dtype=np.int64), name="test_tensor")
        value = ir.Value(name="test_value", type=ir.TensorType(ir.DataType.INT64))
        value.const_value = tensor
        self.assertIs(_convenience.get_const_tensor(value), tensor)

    def test_no_const_value(self):
        value = ir.Value(name="test_value", type=ir.TensorType(ir.DataType.FLOAT))

        self.assertIsNone(_convenience.get_const_tensor(value))

    def test_non_constant_producer_node(self):
        # Test when producer node is not a Constant
        node = ir.Node(
            name="test_node",
            domain="",
            op_type="Add",
            inputs=[],
        )

        output_value = node.outputs[0]
        self.assertIsNone(_convenience.get_const_tensor(output_value))

    @parameterized.parameterized.expand(
        [
            (
                "value_float",
                ir.AttrFloat32("value_float", 3.14),
                np.array(3.14, dtype=np.float32),
            ),
            ("value_int", ir.AttrInt64("value_int", 42), np.array(42, dtype=np.int64)),
            (
                "value_string",
                ir.AttrString("value_string", "test"),
                np.array(b"test", dtype=object),
            ),
            (
                "value_floats",
                ir.AttrFloat32s("value_floats", [1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            ),
            (
                "value_ints",
                ir.AttrInt64s("value_ints", [1, 2, 3]),
                np.array([1, 2, 3], dtype=np.int64),
            ),
            (
                "value_strings",
                ir.AttrStrings("value_strings", ["a", "b", "c"]),
                np.array([b"a", b"b", b"c"], dtype=object),
            ),
            (
                "value",
                ir.AttrTensor("value", ir.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            ),
        ]
    )
    def test_constant_value(self, _: str, attr: ir.Attr, expected: np.ndarray):
        # Test with Constant node with float value
        node = ir.Node(
            name="constant_node",
            domain="",
            op_type="Constant",
            inputs=[],
            attributes=(attr,),
        )
        node.outputs[0].name = "output"

        result = _convenience.get_const_tensor(node.outputs[0])

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "output")
        np.testing.assert_array_equal(result.numpy(), expected)

        self.assertIsNone(node.outputs[0].shape)
        self.assertIsNone(node.outputs[0].type)

        result_2 = _convenience.get_const_tensor(node.outputs[0], propagate_shape_type=True)
        self.assertIsNotNone(result_2)
        self.assertEqual(result_2.name, "output")
        np.testing.assert_array_equal(result_2.numpy(), expected)
        self.assertEqual(node.outputs[0].shape, expected.shape)
        self.assertEqual(node.outputs[0].type, ir.TensorType(result_2.dtype))


if __name__ == "__main__":
    unittest.main()
