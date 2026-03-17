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


class RenameValuesTest(unittest.TestCase):
    def test_rename_values_supports_initializer_swaps(self):
        first = ir.Value(name="const_0", const_value=ir.tensor([1], name="const_0"))
        second = ir.Value(name="const_1", const_value=ir.tensor([2], name="const_1"))
        graph = ir.Graph(
            inputs=(),
            outputs=[first, second],
            nodes=(),
            initializers=[first, second],
            name="test_graph",
        )

        _convenience.rename_values((first, second), ("const_1", "const_0"))

        self.assertEqual(first.name, "const_1")
        self.assertEqual(second.name, "const_0")
        self.assertEqual(first.const_value.name, "const_1")
        self.assertEqual(second.const_value.name, "const_0")
        self.assertEqual(set(graph.initializers), {"const_0", "const_1"})
        self.assertIs(graph.initializers["const_1"], first)
        self.assertIs(graph.initializers["const_0"], second)

    def test_rename_values_rejects_none_names(self):
        value = ir.Value(name="value")

        with self.assertRaisesRegex(TypeError, "name must be a string"):
            _convenience.rename_values(value, None)

    def test_rename_values_rejects_initializer_collisions_outside_rename_set(self):
        first = ir.Value(name="const_0", const_value=ir.tensor([1], name="const_0"))
        second = ir.Value(name="const_1", const_value=ir.tensor([2], name="const_1"))
        graph = ir.Graph(
            inputs=(),
            outputs=[first, second],
            nodes=(),
            initializers=[first, second],
            name="test_graph",
        )

        with self.assertRaisesRegex(ValueError, "an initializer with that name already exists"):
            _convenience.rename_values(first, "const_1")

        self.assertIs(graph.initializers["const_0"], first)
        self.assertIs(graph.initializers["const_1"], second)

    def test_rename_values_rejects_empty_initializer_name_without_mutating_graph(self):
        value = ir.Value(name="const_0", const_value=ir.tensor([1], name="const_0"))
        graph = ir.Graph(
            inputs=(),
            outputs=[value],
            nodes=(),
            initializers=[value],
            name="test_graph",
        )

        with self.assertRaisesRegex(ValueError, "empty string"):
            _convenience.rename_values(value, "")

        self.assertEqual(value.name, "const_0")
        self.assertEqual(value.const_value.name, "const_0")
        self.assertEqual(list(graph.initializers), ["const_0"])
        self.assertIs(graph.initializers["const_0"], value)


if __name__ == "__main__":
    unittest.main()
