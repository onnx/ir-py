# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx
import parameterized

import onnx_ir as ir
from onnx_ir import schemas


class TypeConstraintParamTest(unittest.TestCase):
    def test_init(self):
        allowed = {ir.TensorType(ir.DataType.FLOAT)}
        param = schemas.TypeConstraintParam("T", allowed, "A description")
        self.assertEqual(param.name, "T")
        self.assertEqual(param.allowed_types, allowed)
        self.assertEqual(param.description, "A description")

    def test_hash(self):
        allowed = {ir.TensorType(ir.DataType.FLOAT)}
        param1 = schemas.TypeConstraintParam("T", allowed)
        param2 = schemas.TypeConstraintParam("T", allowed)
        self.assertEqual(hash(param1), hash(param2))

    def test_str(self):
        allowed = {ir.TensorType(ir.DataType.FLOAT)}
        param = schemas.TypeConstraintParam("T", allowed)
        result = str(param)
        self.assertIn("T=", result)
        self.assertIn("FLOAT", result)

    def test_any_tensor(self):
        param = schemas.TypeConstraintParam.any_tensor("TFloat", "Float tensor")
        self.assertEqual(param.name, "TFloat")
        self.assertEqual(param.description, "Float tensor")
        # Should contain all tensor types
        for dtype in ir.DataType:
            self.assertIn(ir.TensorType(dtype), param.allowed_types)

    def test_any_value(self):
        param = schemas.TypeConstraintParam.any_value("TValue", "Any value")
        self.assertEqual(param.name, "TValue")
        self.assertEqual(param.description, "Any value")
        # Should contain tensor, sequence, and optional types
        self.assertIn(ir.TensorType(ir.DataType.FLOAT), param.allowed_types)
        self.assertIn(
            ir.SequenceType(ir.TensorType(ir.DataType.FLOAT)),
            param.allowed_types,
        )
        self.assertIn(
            ir.OptionalType(ir.TensorType(ir.DataType.FLOAT)),
            param.allowed_types,
        )


class ParameterTest(unittest.TestCase):
    def setUp(self):
        self.type_constraint = schemas.TypeConstraintParam(
            "T", {ir.TensorType(ir.DataType.FLOAT)}
        )

    def test_init(self):
        param = schemas.Parameter(
            name="X",
            type_constraint=self.type_constraint,
            required=True,
            variadic=False,
        )
        self.assertEqual(param.name, "X")
        self.assertEqual(param.type_constraint, self.type_constraint)
        self.assertTrue(param.required)
        self.assertFalse(param.variadic)
        self.assertTrue(param.homogeneous)
        self.assertEqual(param.min_arity, 1)

    def test_str_without_default(self):
        param = schemas.Parameter(
            name="X",
            type_constraint=self.type_constraint,
            required=True,
            variadic=False,
        )
        self.assertEqual(str(param), "X: T")

    def test_str_with_default(self):
        param = schemas.Parameter(
            name="X",
            type_constraint=self.type_constraint,
            required=False,
            variadic=False,
            default=1.0,
        )
        self.assertEqual(str(param), "X: T = 1.0")

    def test_has_default_false(self):
        param = schemas.Parameter(
            name="X",
            type_constraint=self.type_constraint,
            required=True,
            variadic=False,
        )
        self.assertFalse(param.has_default())

    def test_has_default_true(self):
        param = schemas.Parameter(
            name="X",
            type_constraint=self.type_constraint,
            required=False,
            variadic=False,
            default=1.0,
        )
        self.assertTrue(param.has_default())

    def test_is_input(self):
        param = schemas.Parameter(
            name="X",
            type_constraint=self.type_constraint,
            required=True,
            variadic=False,
        )
        self.assertTrue(param.is_input())

    def test_is_attribute(self):
        param = schemas.Parameter(
            name="X",
            type_constraint=self.type_constraint,
            required=True,
            variadic=False,
        )
        self.assertFalse(param.is_attribute())


class AttributeParameterTest(unittest.TestCase):
    def test_init(self):
        param = schemas.AttributeParameter(
            name="axis",
            type=ir.AttributeType.INT,
            required=True,
        )
        self.assertEqual(param.name, "axis")
        self.assertEqual(param.type, ir.AttributeType.INT)
        self.assertTrue(param.required)
        self.assertIsNone(param.default)

    def test_str_without_default(self):
        param = schemas.AttributeParameter(
            name="axis",
            type=ir.AttributeType.INT,
            required=True,
        )
        self.assertEqual(str(param), "axis: INT")

    def test_str_with_default(self):
        default_attr = ir.Attr("axis", ir.AttributeType.INT, 0)
        param = schemas.AttributeParameter(
            name="axis",
            type=ir.AttributeType.INT,
            required=False,
            default=default_attr,
        )
        result = str(param)
        self.assertIn("axis: INT = ", result)

    def test_has_default_false(self):
        param = schemas.AttributeParameter(
            name="axis",
            type=ir.AttributeType.INT,
            required=True,
        )
        self.assertFalse(param.has_default())

    def test_has_default_true(self):
        default_attr = ir.Attr("axis", ir.AttributeType.INT, 0)
        param = schemas.AttributeParameter(
            name="axis",
            type=ir.AttributeType.INT,
            required=False,
            default=default_attr,
        )
        self.assertTrue(param.has_default())

    def test_is_input(self):
        param = schemas.AttributeParameter(
            name="axis",
            type=ir.AttributeType.INT,
            required=True,
        )
        self.assertFalse(param.is_input())

    def test_is_attribute(self):
        param = schemas.AttributeParameter(
            name="axis",
            type=ir.AttributeType.INT,
            required=True,
        )
        self.assertTrue(param.is_attribute())


class GetTypeFromStrTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("tensor_float", "tensor(float)", ir.TensorType(ir.DataType.FLOAT)),
            ("tensor_int64", "tensor(int64)", ir.TensorType(ir.DataType.INT64)),
            ("tensor_double", "tensor(double)", ir.TensorType(ir.DataType.DOUBLE)),
            ("tensor_bool", "tensor(bool)", ir.TensorType(ir.DataType.BOOL)),
            ("tensor_string", "tensor(string)", ir.TensorType(ir.DataType.STRING)),
        ]
    )
    def test_tensor_types(self, _: str, type_str: str, expected):
        result = schemas._get_type_from_str(type_str)
        self.assertEqual(result, expected)

    def test_sequence_type(self):
        result = schemas._get_type_from_str("seq(tensor(float))")
        expected = ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
        self.assertEqual(result, expected)

    def test_optional_type(self):
        result = schemas._get_type_from_str("optional(tensor(int32))")
        expected = ir.OptionalType(ir.TensorType(ir.DataType.INT32))
        self.assertEqual(result, expected)

    def test_nested_sequence_optional(self):
        result = schemas._get_type_from_str("seq(optional(tensor(float)))")
        expected = ir.SequenceType(ir.OptionalType(ir.TensorType(ir.DataType.FLOAT)))
        self.assertEqual(result, expected)

    def test_unknown_type_part_raises(self):
        with self.assertRaises(ValueError) as context:
            schemas._get_type_from_str("unknown(tensor(float))")
        self.assertIn("Unknown type part: 'unknown'", str(context.exception))


class OpSignatureTest(unittest.TestCase):
    def setUp(self):
        self.type_constraint = schemas.TypeConstraintParam(
            "T", {ir.TensorType(ir.DataType.FLOAT)}
        )
        self.input_param = schemas.Parameter(
            name="X",
            type_constraint=self.type_constraint,
            required=True,
            variadic=False,
        )
        self.attr_param = schemas.AttributeParameter(
            name="axis",
            type=ir.AttributeType.INT,
            required=True,
        )
        self.output_param = schemas.Parameter(
            name="Y",
            type_constraint=self.type_constraint,
            required=True,
            variadic=False,
        )

    def test_init(self):
        sig = schemas.OpSignature(
            domain="",
            name="TestOp",
            overload="",
            params=[self.input_param, self.attr_param],
            outputs=[self.output_param],
        )
        self.assertEqual(sig.domain, "")
        self.assertEqual(sig.name, "TestOp")
        self.assertEqual(sig.overload, "")
        self.assertEqual(len(sig.params), 2)
        self.assertEqual(len(sig.outputs), 1)

    def test_params_map_created(self):
        sig = schemas.OpSignature(
            domain="",
            name="TestOp",
            overload="",
            params=[self.input_param, self.attr_param],
            outputs=[self.output_param],
        )
        self.assertIn("X", sig.params_map)
        self.assertIn("axis", sig.params_map)
        self.assertEqual(sig.params_map["X"], self.input_param)
        self.assertEqual(sig.params_map["axis"], self.attr_param)

    def test_get(self):
        sig = schemas.OpSignature(
            domain="",
            name="TestOp",
            overload="",
            params=[self.input_param, self.attr_param],
            outputs=[self.output_param],
        )
        self.assertEqual(sig.get("X"), self.input_param)
        self.assertEqual(sig.get("axis"), self.attr_param)

    def test_contains(self):
        sig = schemas.OpSignature(
            domain="",
            name="TestOp",
            overload="",
            params=[self.input_param, self.attr_param],
            outputs=[self.output_param],
        )
        self.assertIn("X", sig)
        self.assertIn("axis", sig)
        self.assertNotIn("Z", sig)

    def test_iter(self):
        sig = schemas.OpSignature(
            domain="",
            name="TestOp",
            overload="",
            params=[self.input_param, self.attr_param],
            outputs=[self.output_param],
        )
        params_list = list(sig)
        self.assertEqual(len(params_list), 2)
        self.assertEqual(params_list[0], self.input_param)
        self.assertEqual(params_list[1], self.attr_param)

    def test_str(self):
        sig = schemas.OpSignature(
            domain="",
            name="TestOp",
            overload="",
            params=[self.input_param, self.attr_param],
            outputs=[self.output_param],
        )
        result = str(sig)
        self.assertIn("''::TestOp", result)
        self.assertIn("X: T", result)
        self.assertIn("axis: INT", result)
        self.assertIn("where", result)

    def test_str_with_domain_and_overload(self):
        sig = schemas.OpSignature(
            domain="custom",
            name="TestOp",
            overload="v2",
            params=[self.input_param],
            outputs=[self.output_param],
        )
        result = str(sig)
        self.assertIn("custom::TestOp::v2", result)

    def test_inputs_property(self):
        sig = schemas.OpSignature(
            domain="",
            name="TestOp",
            overload="",
            params=[self.input_param, self.attr_param],
            outputs=[self.output_param],
        )
        inputs = sig.inputs
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0], self.input_param)

    def test_attributes_property(self):
        sig = schemas.OpSignature(
            domain="",
            name="TestOp",
            overload="",
            params=[self.input_param, self.attr_param],
            outputs=[self.output_param],
        )
        attributes = sig.attributes
        self.assertEqual(len(attributes), 1)
        self.assertEqual(attributes[0], self.attr_param)


class OpSignatureFromOpSchemaTest(unittest.TestCase):
    """Test OpSignature.from_op_schema with real ONNX op schemas."""

    def test_from_op_schema_add(self):
        """Test conversion of Add op schema."""
        op_schema = onnx.defs.get_schema("Add", 14)
        sig = schemas.OpSignature.from_op_schema(op_schema)
        self.assertEqual(sig.domain, "")
        self.assertEqual(sig.name, "Add")
        self.assertEqual(sig.since_version, 14)
        # Add has 2 inputs and 1 output
        self.assertEqual(len(sig.inputs), 2)
        self.assertEqual(len(sig.outputs), 1)
        # Check inputs
        self.assertEqual(sig.inputs[0].name, "A")
        self.assertEqual(sig.inputs[1].name, "B")
        # Check output
        self.assertEqual(sig.outputs[0].name, "C")

    def test_from_op_schema_relu(self):
        """Test conversion of Relu op schema."""
        op_schema = onnx.defs.get_schema("Relu", 14)
        sig = schemas.OpSignature.from_op_schema(op_schema)
        self.assertEqual(sig.domain, "")
        self.assertEqual(sig.name, "Relu")
        # Relu has 1 input and 1 output
        self.assertEqual(len(sig.inputs), 1)
        self.assertEqual(len(sig.outputs), 1)

    def test_from_op_schema_with_attributes(self):
        """Test conversion of op schema with attributes (Transpose)."""
        op_schema = onnx.defs.get_schema("Transpose", 13)
        sig = schemas.OpSignature.from_op_schema(op_schema)
        self.assertEqual(sig.name, "Transpose")
        # Transpose has optional 'perm' attribute
        self.assertEqual(len(sig.attributes), 1)
        self.assertEqual(sig.attributes[0].name, "perm")
        self.assertEqual(sig.attributes[0].type, ir.AttributeType.INTS)
        # perm is optional
        self.assertFalse(sig.attributes[0].required)

    def test_from_op_schema_with_required_attribute(self):
        """Test conversion of op schema with required attribute (Dropout)."""
        op_schema = onnx.defs.get_schema("Constant", 13)
        sig = schemas.OpSignature.from_op_schema(op_schema)
        # Constant has multiple optional attributes
        self.assertGreater(len(sig.attributes), 0)

    def test_from_op_schema_variadic_inputs(self):
        """Test conversion of op schema with variadic inputs (Concat)."""
        op_schema = onnx.defs.get_schema("Concat", 13)
        sig = schemas.OpSignature.from_op_schema(op_schema)
        self.assertEqual(sig.name, "Concat")
        # Concat has variadic input 'inputs'
        self.assertEqual(len(sig.inputs), 1)
        self.assertTrue(sig.inputs[0].variadic)

    def test_from_op_schema_optional_inputs(self):
        """Test conversion of op schema with optional inputs (BatchNormalization)."""
        op_schema = onnx.defs.get_schema("BatchNormalization", 15)
        sig = schemas.OpSignature.from_op_schema(op_schema)
        # BatchNormalization has required and optional inputs
        self.assertEqual(sig.name, "BatchNormalization")
        self.assertGreater(len(sig.inputs), 0)


class ConvertFormalParameterTest(unittest.TestCase):
    """Test _convert_formal_parameter helper function."""

    def test_with_type_constraint(self):
        """Test conversion when type_str matches a type constraint."""
        op_schema = onnx.defs.get_schema("Add", 14)
        type_constraints = {
            constraint.type_param_str: schemas.TypeConstraintParam(
                name=constraint.type_param_str,
                allowed_types={
                    schemas._get_type_from_str(type_str)
                    for type_str in constraint.allowed_type_strs
                },
            )
            for constraint in op_schema.type_constraints
        }
        formal_param = op_schema.inputs[0]
        param = schemas._convert_formal_parameter(formal_param, type_constraints)
        self.assertEqual(param.name, "A")
        self.assertTrue(param.required)
        self.assertFalse(param.variadic)

    def test_with_plain_type(self):
        """Test conversion when type_str is a plain type like 'int64'."""
        # Shape op has type constraint with plain type
        op_schema = onnx.defs.get_schema("Shape", 15)
        type_constraints = {
            constraint.type_param_str: schemas.TypeConstraintParam(
                name=constraint.type_param_str,
                allowed_types={
                    schemas._get_type_from_str(type_str)
                    for type_str in constraint.allowed_type_strs
                },
            )
            for constraint in op_schema.type_constraints
        }
        # The output of Shape is tensor(int64) which is a specific type
        formal_param = op_schema.outputs[0]
        param = schemas._convert_formal_parameter(formal_param, type_constraints)
        self.assertEqual(param.name, "shape")


if __name__ == "__main__":
    unittest.main()
