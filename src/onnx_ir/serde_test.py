# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import itertools
import unittest
import warnings

import google.protobuf.text_format
import ml_dtypes
import numpy as np
import onnx
import parameterized

import onnx_ir as ir
from onnx_ir import _multi_device, _version_utils, serde


class ConvenienceFunctionsTest(unittest.TestCase):
    def test_from_to_onnx_text(self):
        model_text = """\
<
   ir_version: 10,
   opset_import: ["" : 17]
>
agraph (float[1,4,512,512] input_x, float[1,4,512,64] input_y) => (float[4,512,512] reshape_x) {
   [node_name] shape_a = Constant <value: tensor = int64[3] {4,512,512}> ()
   reshape_x = Reshape (input_x, shape_a)
}"""
        self.maxDiff = None
        model = serde.from_onnx_text(model_text)
        self.assertIsInstance(model, ir.Model)
        self.assertEqual(model.ir_version, 10)
        self.assertEqual(len(model.graph.inputs), 2)
        self.assertEqual(len(model.graph.outputs), 1)
        onnx_text_roundtrip = serde.to_onnx_text(model)
        self.assertEqual(model_text.strip(), onnx_text_roundtrip.strip())

    def test_from_to_onnx_text_with_initializers(self):
        model_text = """\
<
   ir_version: 10,
   opset_import: ["" : 17]
>
agraph (float[1] input_x, float[2] input_y) => (float[2] result) {
   [node_1] add = Add (input_x, input_y)
   [node_2] result = Add (add, initializer_z)
}"""
        self.maxDiff = None
        array = np.array([1.0, 2.0], dtype=np.float32)
        init_array = np.array([3.0, 4.0], dtype=np.float32)
        model = serde.from_onnx_text(
            model_text,
            initializers=[
                ir.tensor(init_array, name="initializer_z"),
                ir.tensor(array, name="input_y"),
            ],
        )
        np.testing.assert_array_equal(model.graph.inputs[1].const_value.numpy(), array)
        np.testing.assert_array_equal(
            model.graph.initializers["initializer_z"].const_value.numpy(), init_array
        )
        expected_text = """\
<
   ir_version: 10,
   opset_import: ["" : 17]
>
agraph (float[1] input_x, float[2] input_y) => (float[2] result)
   <float[2] initializer_z =  {3,4}, float[2] input_y =  {1,2}>
{
   [node_1] add = Add (input_x, input_y)
   [node_2] result = Add (add, initializer_z)
}"""
        onnx_text_roundtrip = serde.to_onnx_text(model)
        stripped_lines = [line.rstrip() for line in onnx_text_roundtrip.splitlines()]
        result = "\n".join(stripped_lines)
        self.assertEqual(result, expected_text)

    def test_to_onnx_text_excluding_initializers(self):
        model_text = """\
<
   ir_version: 10,
   opset_import: ["" : 17]
>
agraph (float[1] input_x, float[2] input_y) => (float[2] result) {
   [node_name] result = Add (input_x, input_y)
}"""
        self.maxDiff = None
        array = np.array([1.0, 2.0], dtype=np.float32)
        model = serde.from_onnx_text(
            model_text, initializers=[ir.tensor(array, name="input_y")]
        )
        onnx_text_without_initializers = serde.to_onnx_text(model, exclude_initializers=True)
        expected_text_without_initializers = """\
<
   ir_version: 10,
   opset_import: ["" : 17]
>
agraph (float[1] input_x, float[2] input_y) => (float[2] result) {
   [node_name] result = Add (input_x, input_y)
}"""
        self.assertEqual(
            onnx_text_without_initializers.strip(), expected_text_without_initializers
        )


class TensorProtoTensorTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("FLOAT", onnx.TensorProto.FLOAT),
            ("BOOL", onnx.TensorProto.BOOL),
            ("FLOAT16", onnx.TensorProto.FLOAT16),
            ("DOUBLE", onnx.TensorProto.DOUBLE),
        ]
    )
    def test_tensor_proto_tensor(self, _: str, dtype: int):
        tensor_proto = onnx.helper.make_tensor(
            "test_tensor", dtype, [1, 9], [-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]
        )
        tensor = serde.TensorProtoTensor(tensor_proto)
        expected_array = onnx.numpy_helper.to_array(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        if dtype == onnx.TensorProto.BOOL and _version_utils.numpy_older_than("1.25"):
            self.skipTest("numpy<1.25 does not support bool dtype in from_dlpack")
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @unittest.skipIf(
        _version_utils.onnx_older_than("1.17"),
        "numpy_helper.to_array was not correctly implemented in onnx<1.17",
    )
    def test_tensor_proto_tensor_bfloat16(self):
        expected_array = np.array(
            [[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]], dtype=ml_dtypes.bfloat16
        )
        tensor_proto = onnx.helper.make_tensor(
            "test_tensor",
            onnx.TensorProto.BFLOAT16,
            [1, 9],
            np.array([[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]]),
        )
        tensor = serde.TensorProtoTensor(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(
            array_from_raw_data.view(ml_dtypes.bfloat16), expected_array
        )
        # Test dlpack
        with self.assertRaises(BufferError):
            # NumPy does not support bfloat16 in from_dlpack
            np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            (
                "FLOAT8E4M3FN",
                onnx.TensorProto.FLOAT8E4M3FN,
                ml_dtypes.float8_e4m3fn,
            ),
            (
                "FLOAT8E4M3FNUZ",
                onnx.TensorProto.FLOAT8E4M3FNUZ,
                ml_dtypes.float8_e4m3fnuz,
            ),
            (
                "FLOAT8E5M2",
                onnx.TensorProto.FLOAT8E5M2,
                ml_dtypes.float8_e5m2,
            ),
            (
                "FLOAT8E5M2FNUZ",
                onnx.TensorProto.FLOAT8E5M2FNUZ,
                ml_dtypes.float8_e5m2fnuz,
            ),
            (
                "FLOAT8E8M0",
                24,  # FLOAT8E8M0 value from the enum
                ml_dtypes.float8_e8m0fnu,
            ),
        ]
    )
    def test_tensor_proto_tensor_float8(self, _: str, dtype: int, np_dtype):
        # FLOAT8E8M0 has different precision characteristics (8 exponent bits, 0 mantissa bits)
        # It can only represent powers of 2 and special values
        if dtype == 24:  # FLOAT8E8M0
            expected_array = np.array([[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]])
        else:
            expected_array = np.array([[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 40.0, 2.0]])

        # Handle the case where ONNX doesn't support FLOAT8E8M0 yet (value 24)
        if dtype == 24:  # FLOAT8E8M0
            # Create tensor proto manually since ONNX helper might not support this type yet
            tensor_proto = onnx.TensorProto()
            tensor_proto.name = "test_tensor"
            tensor_proto.data_type = dtype
            tensor_proto.dims[:] = [1, 9]
            tensor_proto.raw_data = expected_array.astype(np_dtype).tobytes()
        else:
            tensor_proto = onnx.helper.make_tensor(
                "test_tensor", dtype, [1, 9], expected_array
            )

        tensor = serde.TensorProtoTensor(tensor_proto)
        np.testing.assert_array_equal(
            tensor.numpy().view(np_dtype).astype(np.float32), expected_array
        )
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = (
            serde.TensorProtoTensor(tensor_proto_from_raw_data)
            .numpy()
            .view(np_dtype)
            .astype(np.float32)
        )
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        with self.assertRaises(BufferError):
            # DL Pack does not support float8
            np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            ("INT8", onnx.TensorProto.INT8),
            ("INT16", onnx.TensorProto.INT16),
            ("INT32", onnx.TensorProto.INT32),
            ("INT64", onnx.TensorProto.INT64),
            ("INT4", onnx.TensorProto.INT4),
            *(
                [
                    ("INT2", onnx.TensorProto.INT2),
                ]
                if hasattr(onnx.TensorProto, "INT2")
                else []
            ),
        ]
    )
    def test_tensor_proto_tensor_int(self, _: str, dtype: int):
        tensor_proto = onnx.helper.make_tensor("test_tensor", dtype, [1, 4], [-1, 0, 1, 8])
        tensor = serde.TensorProtoTensor(tensor_proto)
        expected_array = onnx.numpy_helper.to_array(
            tensor_proto
        )  # [-1, 0, 1, 7], 8 is clamped to 7
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        if dtype in (
            onnx.TensorProto.INT4,
            onnx.TensorProto.INT2 if hasattr(onnx.TensorProto, "INT2") else 26,
        ):
            return  # DL Pack does not support int4/int2
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            ("UINT8", onnx.TensorProto.UINT8),
            ("UINT16", onnx.TensorProto.UINT16),
            ("UINT32", onnx.TensorProto.UINT32),
            ("UINT64", onnx.TensorProto.UINT64),
            ("UINT4", onnx.TensorProto.UINT4),
            *(
                [
                    ("INT2", onnx.TensorProto.UINT2),
                ]
                if hasattr(onnx.TensorProto, "UINT2")
                else []
            ),
        ]
    )
    def test_tensor_proto_tensor_uint(self, _: str, dtype: int):
        tensor_proto = onnx.helper.make_tensor("test_tensor", dtype, [1, 3], [0, 1, 8])
        tensor = serde.TensorProtoTensor(tensor_proto)
        expected_array = onnx.numpy_helper.to_array(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        if dtype in (
            onnx.TensorProto.UINT4,
            onnx.TensorProto.UINT2 if hasattr(onnx.TensorProto, "UINT2") else 25,
        ):
            return  # DL Pack does not support uint4/uint2
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            ("COMPLEX64", onnx.TensorProto.COMPLEX64, np.complex64),
            ("COMPLEX128", onnx.TensorProto.COMPLEX128, np.complex128),
        ]
    )
    def test_tensor_proto_tensor_complex(self, _: str, dtype: int, np_dtype: np.dtype):
        expected_array = np.array([[0.0 + 1j, 0.2 - 1j, 0.3]], dtype=np_dtype)
        tensor_proto = onnx.helper.make_tensor(
            "test_tensor", dtype, [1, 3], [0.0 + 1j, 0.2 - 1j, 0.3]
        )
        tensor = serde.TensorProtoTensor(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    def test_tensor_proto_tensor_empty_tensor(self):
        tensor_proto = onnx.helper.make_tensor("test_tensor", onnx.TensorProto.FLOAT, [0], [])
        tensor = serde.TensorProtoTensor(tensor_proto)
        expected_array = onnx.numpy_helper.to_array(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            (name, dtype, array)
            for (name, dtype), array in itertools.product(
                [
                    ("FLOAT", ir.DataType.FLOAT),
                    ("UINT8", ir.DataType.UINT8),
                    ("INT8", ir.DataType.INT8),
                    ("UINT16", ir.DataType.UINT16),
                    ("INT16", ir.DataType.INT16),
                    ("INT32", ir.DataType.INT32),
                    ("INT64", ir.DataType.INT64),
                    ("BOOL", ir.DataType.BOOL),
                    ("FLOAT16", ir.DataType.FLOAT16),
                    ("DOUBLE", ir.DataType.DOUBLE),
                    ("UINT32", ir.DataType.UINT32),
                    ("UINT64", ir.DataType.UINT64),
                    ("COMPLEX64", ir.DataType.COMPLEX64),
                    ("COMPLEX128", ir.DataType.COMPLEX128),
                    ("BFLOAT16", ir.DataType.BFLOAT16),
                    ("FLOAT8E4M3FN", ir.DataType.FLOAT8E4M3FN),
                    ("FLOAT8E4M3FNUZ", ir.DataType.FLOAT8E4M3FNUZ),
                    ("FLOAT8E5M2", ir.DataType.FLOAT8E5M2),
                    ("FLOAT8E5M2FNUZ", ir.DataType.FLOAT8E5M2FNUZ),
                    ("FLOAT8E8M0", ir.DataType.FLOAT8E8M0),
                    ("UINT4", ir.DataType.UINT4),
                    ("INT4", ir.DataType.INT4),
                    ("UINT2", ir.DataType.UINT2),
                    ("INT2", ir.DataType.INT2),
                    ("FLOAT4E2M1", ir.DataType.FLOAT4E2M1),
                ],
                [
                    np.array(
                        [
                            [-1000, -6, -1, -0.0, +0.0],
                            [0.1, 0.25, 1, float("inf"), -float("inf")],
                            [float("NaN"), -float("NaN"), 1000, 6.0, 0.001],
                        ],
                    ),
                    np.array(42),
                    np.array([]),
                    np.array([[[], [], []]]),
                ],
            )
        ]
    )
    def test_round_trip_numpy_conversion_from_raw_data(
        self, _: str, onnx_dtype: ir.DataType, original_array: np.ndarray
    ):
        target_dtype = np.dtype(onnx_dtype.numpy())
        if target_dtype.kind in {"i", "u", "b"}:
            original_array = np.nan_to_num(original_array, nan=0.0, posinf=0.0, neginf=0.0)
        original_array = original_array.astype(target_dtype)
        ir_tensor = ir.Tensor(original_array, name="test_tensor")
        proto = serde.to_proto(ir_tensor)
        if original_array.size > 0:
            self.assertGreater(len(proto.raw_data), 0)
        # tensor_proto_tensor from raw_data
        tensor_proto_tensor = serde.from_proto(proto)
        roundtrip_array = tensor_proto_tensor.numpy()
        if onnx_dtype in {
            ir.DataType.FLOAT8E4M3FNUZ,
            ir.DataType.FLOAT8E5M2FNUZ,
            ir.DataType.FLOAT8E5M2,
            ir.DataType.FLOAT8E4M3FN,
            ir.DataType.FLOAT4E2M1,
            ir.DataType.BFLOAT16,
            ir.DataType.FLOAT8E8M0,
        }:
            # There is a bug in ml_dtypes that causes equality checks to fail for these dtypes
            # See https://github.com/jax-ml/ml_dtypes/issues/301
            self.assertEqual(roundtrip_array.shape, original_array.shape)
            self.assertEqual(roundtrip_array.dtype, original_array.dtype)
            self.assertEqual(roundtrip_array.tobytes(), original_array.tobytes())
        else:
            np.testing.assert_equal(roundtrip_array, original_array, strict=True)


class DeserializeGraphTest(unittest.TestCase):
    def test_deserialize_graph_handles_unsorted_graph(self):
        node_0 = ir.Node(
            "",
            "Op_0",
            inputs=[ir.val("input_0"), ir.val("input_1")],
            num_outputs=2,
            name="node_0",
        )
        node_1 = ir.Node(
            "",
            "Op_1",
            inputs=[node_0.outputs[0]],
            num_outputs=1,
            name="node_1",
        )
        graph = ir.Graph(
            inputs=node_0.inputs,  # type: ignore
            outputs=[node_1.outputs[0]],
            # Unsorted nodes
            nodes=[node_1, node_0],
            name="test_graph",
        )
        graph_proto = serde.serialize_graph(graph)
        deserialized_graph = serde.deserialize_graph(graph_proto)
        self.assertEqual(deserialized_graph[0].op_type, "Op_1")
        self.assertEqual(deserialized_graph[1].op_type, "Op_0")

    def test_deserialize_graph_handles_invalid_output(self):
        # The graph has an output that is not connected to any node, and it does not
        # have shape/type information.
        graph_with_invalid_output = ir.Graph(
            inputs=[],
            outputs=[ir.Value(name="invalid_output")],
            nodes=[],
            name="graph_with_invalid_output",
        )
        graph_proto = serde.serialize_graph(graph_with_invalid_output)
        deserialized_graph = serde.deserialize_graph(graph_proto)
        self.assertEqual(len(deserialized_graph.outputs), 1)
        self.assertEqual(deserialized_graph.outputs[0].name, "invalid_output")
        self.assertEqual(deserialized_graph.outputs[0].type, None)
        self.assertEqual(deserialized_graph.outputs[0].shape, None)
        self.assertEqual(deserialized_graph.outputs[0].dtype, None)

    def test_deserialize_builds_correct_value_connections_for_subgraphs_that_reference_out_of_order_values_in_outer_graph(
        self,
    ):
        model_text = """\
            <
            ir_version: 10,
            opset_import: ["" : 42]
            >
            main_graph (float[2,3] a) => (float[4,5] c)
            <float[3,4] b>
            {
            [node_with_subgraph] c = SubgraphOp () <subgraph: graph = subgraph () => ()
                <float[3,4] b_out>
            {
                [subgraph_node] b_out = SomeOp (b)
            }>
            [b_producer] b = SomeOp (a)
            }
        """
        deserialized_model = serde.from_onnx_text(model_text)
        # Model is unsorted
        self.assertEqual(
            [n.name for n in deserialized_model.graph], ["node_with_subgraph", "b_producer"]
        )
        # Value b in subgraph is the name value defined in the outer graph
        subgraph_node = (
            deserialized_model.graph.node(0).attributes["subgraph"].as_graph().node(0)
        )
        subgraph_value = subgraph_node.inputs[0]
        main_graph_value = deserialized_model.graph.node(1).outputs[0]
        self.assertIs(subgraph_value, main_graph_value)
        self.assertEqual(len(main_graph_value.uses()), 1)
        self.assertEqual(list(main_graph_value.consumers()), [subgraph_node])
        with self.assertRaisesRegex(
            Exception, "Nodes in a graph must be topologically sorted"
        ):
            onnx.checker.check_model(serde.serialize_model(deserialized_model))

        # Graph can be sorted correctly
        deserialized_model.graph.sort()
        self.assertEqual(
            [n.name for n in deserialized_model.graph], ["b_producer", "node_with_subgraph"]
        )

    def test_value_metadata_props_are_preserved(self):
        value = ir.val(
            "test_initializer",
            dtype=ir.DataType.FLOAT,
            shape=(2,),
            const_value=ir.tensor([1.0, 2.0], name="test_initializer"),
            metadata_props={"key": "value"},
        )
        input = ir.val(
            "test_input", dtype=ir.DataType.FLOAT, shape=(2,), metadata_props={"key": "input"}
        )
        node = ir.node("Identity", inputs=[input])
        node.outputs[0].metadata_props["key"] = "intermediate"
        output = ir.val(
            "test_output",
            dtype=ir.DataType.FLOAT,
            shape=(2,),
            metadata_props={"key": "output"},
        )
        node2 = ir.node("Identity", inputs=node.outputs, outputs=[output])
        graph = ir.Graph(
            inputs=[input],
            outputs=[output],
            nodes=[node, node2],
            initializers=[value],
            name="test_graph",
        )
        graph_proto = serde.serialize_graph(graph)
        deserialized_graph = serde.deserialize_graph(graph_proto)

        self.assertEqual(deserialized_graph.inputs[0].metadata_props, {"key": "input"})
        self.assertEqual(deserialized_graph.outputs[0].metadata_props, {"key": "output"})
        intermediate_value = deserialized_graph.node(0).outputs[0]
        self.assertEqual(intermediate_value.metadata_props, {"key": "intermediate"})

        self.assertIn("test_initializer", deserialized_graph.initializers)
        deserialized_value = deserialized_graph.initializers["test_initializer"]
        self.assertEqual(deserialized_value.metadata_props, {"key": "value"})


class SerializationTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("float", ir.AttributeType.FLOAT, 1.5, 1.5),
            ("int_as_float", ir.AttributeType.FLOAT, 1, 1.0),
            ("int", ir.AttributeType.INT, 42, 42),
            ("bool", ir.AttributeType.INT, True, 1),
            ("ints", ir.AttributeType.INTS, [1, 2, 3], (1, 2, 3)),
            ("floats", ir.AttributeType.FLOATS, [1.0, 2.0, 3.0], (1.0, 2.0, 3.0)),
            ("bools", ir.AttributeType.INTS, [True, False], (1, 0)),
            ("string", ir.AttributeType.STRING, "test_string", "test_string"),
        ]
    )
    def test_serialize_attribute(self, _: str, typ: ir.AttributeType, value, expected):
        attr = ir.Attr("test_attr", typ, value)
        with warnings.catch_warnings(record=True) as w:
            # Ensure all warnings are caught, not just the default ones
            warnings.simplefilter("always")
            attr_proto = serde.serialize_attribute(attr)
            self.assertEqual(
                len(w), 0, f"Unexpected warnings: {[str(warn.message) for warn in w]}"
            )
        deserialized_attr = serde.deserialize_attribute(attr_proto)
        self.assertEqual(deserialized_attr.name, attr.name)
        self.assertEqual(deserialized_attr.type, attr.type)
        self.assertEqual(deserialized_attr.value, expected)

    def test_serialize_shape_into_skips_writing_when_value_type_not_known(self):
        shape = ir.Shape((1, 2, 3))
        proto = onnx.TypeProto()
        self.assertIsNone(proto.WhichOneof("value"))
        serde.serialize_shape_into(proto, shape)
        self.assertIsNone(proto.WhichOneof("value"))
        deserialized = serde.deserialize_type_proto_for_shape(proto)
        self.assertIsNone(deserialized, shape)


class QuantizationAnnotationTest(unittest.TestCase):
    """Test that quantization annotations are correctly serialized and deserialized."""

    def setUp(self):
        model_text = """\
ir_version: 8
producer_name: "pytorch"
producer_version: "2.1.1"
graph {
  input {
    name: "input"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  output {
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  node {
    input: "input"
    output: "intermediate_value"
    op_type: "TestOp1"
    domain: "test_domain"
  }
  node {
    input: "intermediate_value"
    output: "output"
    op_type: "TestOp2"
    domain: "test_domain"
  }
  quantization_annotation {
    tensor_name: "input"
    quant_parameter_tensor_names {
      key: "custom_key"
      value: "arbitrary_value_input"
    }
  }
  quantization_annotation {
    tensor_name: "intermediate_value"
    quant_parameter_tensor_names {
      key: "custom_key"
      value: "arbitrary_value_intermediate"
    }
  }
  quantization_annotation {
    tensor_name: "output"
    quant_parameter_tensor_names {
      key: "custom_key"
      value: "arbitrary_value_output"
    }
  }
}"""
        self.model = onnx.ModelProto()
        google.protobuf.text_format.Parse(model_text, self.model)

    def test_deserialize_quantization_annotation(self):
        model = serde.deserialize_model(self.model)
        self.assertEqual(
            model.graph.inputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_input"},
        )
        self.assertEqual(
            model.graph.node(0).outputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_intermediate"},
        )
        self.assertEqual(
            model.graph.outputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_output"},
        )

    def test_serde_roundtrip(self):
        model = serde.deserialize_model(self.model)
        serialized_model = serde.serialize_model(model)
        deserialized_model = serde.deserialize_model(serialized_model)
        self.assertEqual(
            deserialized_model.graph.inputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_input"},
        )
        self.assertEqual(
            deserialized_model.graph.node(0).outputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_intermediate"},
        )
        self.assertEqual(
            deserialized_model.graph.outputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_output"},
        )


class FromProtoDispatchTest(unittest.TestCase):
    """Test from_proto dispatches to the correct deserialize function."""

    def test_from_proto_model(self):
        proto = onnx.parser.parse_model(
            '<ir_version: 10, opset_import: ["": 17]> agraph (float[N] x) => (float[N] z) { z = Relu(x) }'
        )
        result = serde.from_proto(proto)
        self.assertIsInstance(result, ir.Model)

    def test_from_proto_graph(self):
        model_proto = onnx.parser.parse_model(
            '<ir_version: 10, opset_import: ["": 17]> agraph (float[N] x) => (float[N] z) { z = Relu(x) }'
        )
        result = serde.from_proto(model_proto.graph)
        self.assertIsInstance(result, ir.Graph)

    def test_from_proto_node(self):
        node_proto = onnx.helper.make_node("Relu", ["x"], ["y"])
        result = serde.from_proto(node_proto)
        self.assertIsInstance(result, ir.Node)
        self.assertEqual(result.op_type, "Relu")

    def test_from_proto_tensor(self):
        tensor_proto = onnx.numpy_helper.from_array(
            np.array([1.0, 2.0], dtype=np.float32), name="t"
        )
        result = serde.from_proto(tensor_proto)
        self.assertEqual(result.name, "t")

    def test_from_proto_attribute(self):
        attr_proto = onnx.helper.make_attribute("alpha", 2.0)
        result = serde.from_proto(attr_proto)
        self.assertEqual(result.name, "alpha")
        self.assertEqual(result.value, 2.0)

    def test_from_proto_value_info(self):
        vi_proto = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])
        result = serde.from_proto(vi_proto)
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(result.name, "x")

    def test_from_proto_type_proto(self):
        type_proto = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1, 2])
        result = serde.from_proto(type_proto)
        self.assertIsInstance(result, ir.TypeAndShape)

    def test_from_proto_function(self):
        func_proto = onnx.helper.make_function(
            "test_domain",
            "test_func",
            ["x"],
            ["y"],
            [onnx.helper.make_node("Relu", ["x"], ["y"])],
            [onnx.helper.make_opsetid("", 17)],
        )
        result = serde.from_proto(func_proto)
        self.assertIsInstance(result, ir.Function)

    def test_from_proto_tensor_shape(self):
        shape_proto = onnx.TensorShapeProto()
        dim = shape_proto.dim.add()
        dim.dim_value = 3
        dim2 = shape_proto.dim.add()
        dim2.dim_param = "N"
        result = serde.from_proto(shape_proto)
        self.assertIsInstance(result, ir.Shape)

    def test_from_proto_dimension(self):
        dim_proto = onnx.TensorShapeProto.Dimension()
        dim_proto.dim_value = 42
        result = serde.from_proto(dim_proto)
        self.assertEqual(result[0], 42)

    def test_from_proto_opset_imports(self):
        opset_list = [onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("custom", 1)]
        result = serde.from_proto(opset_list)
        self.assertIsInstance(result, dict)
        self.assertEqual(result[""], 17)
        self.assertEqual(result["custom"], 1)

    def test_from_proto_metadata_props(self):
        entries = []
        entry = onnx.StringStringEntryProto()
        entry.key = "key1"
        entry.value = "val1"
        entries.append(entry)
        result = serde.from_proto(entries)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["key1"], "val1")

    def test_from_proto_unsupported_raises(self):
        with self.assertRaises(NotImplementedError):
            serde.from_proto("not a proto")  # type: ignore[arg-type]


class SequenceOptionalTypeDeserializationTest(unittest.TestCase):
    def test_deserialize_sequence_type(self):
        type_proto = onnx.TypeProto()
        type_proto.sequence_type.elem_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        shape_proto = onnx.TensorShapeProto()
        dim = shape_proto.dim.add()
        dim.dim_value = 3
        type_proto.sequence_type.elem_type.tensor_type.shape.CopyFrom(shape_proto)

        ir_type = serde.deserialize_type_proto_for_type(type_proto)
        self.assertIsInstance(ir_type, ir.SequenceType)

        shape = serde.deserialize_type_proto_for_shape(type_proto)
        self.assertIsInstance(shape, ir.Shape)

    def test_deserialize_optional_type(self):
        type_proto = onnx.TypeProto()
        type_proto.optional_type.elem_type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
        shape_proto = onnx.TensorShapeProto()
        dim = shape_proto.dim.add()
        dim.dim_value = 5
        type_proto.optional_type.elem_type.tensor_type.shape.CopyFrom(shape_proto)

        ir_type = serde.deserialize_type_proto_for_type(type_proto)
        self.assertIsInstance(ir_type, ir.OptionalType)

        shape = serde.deserialize_type_proto_for_shape(type_proto)
        self.assertIsInstance(shape, ir.Shape)

    def test_deserialize_sparse_tensor_type(self):
        type_proto = onnx.TypeProto()
        type_proto.sparse_tensor_type.elem_type = onnx.TensorProto.FLOAT
        shape_proto = onnx.TensorShapeProto()
        dim = shape_proto.dim.add()
        dim.dim_value = 10
        type_proto.sparse_tensor_type.shape.CopyFrom(shape_proto)

        ir_type = serde.deserialize_type_proto_for_type(type_proto)
        self.assertIsInstance(ir_type, ir.SparseTensorType)

        shape = serde.deserialize_type_proto_for_shape(type_proto)
        self.assertIsInstance(shape, ir.Shape)


class TypeSerializationTest(unittest.TestCase):
    def test_serialize_sequence_type(self):
        inner = ir.TensorType(ir.DataType.FLOAT)
        seq_type = ir.SequenceType(inner)
        proto = serde.serialize_type(seq_type)
        self.assertTrue(proto.HasField("sequence_type"))

    def test_serialize_optional_type(self):
        inner = ir.TensorType(ir.DataType.DOUBLE)
        opt_type = ir.OptionalType(inner)
        proto = serde.serialize_type(opt_type)
        self.assertTrue(proto.HasField("optional_type"))

    def test_serialize_sparse_tensor_type(self):
        sparse_type = ir.SparseTensorType(ir.DataType.FLOAT)
        proto = serde.serialize_type(sparse_type)
        self.assertTrue(proto.HasField("sparse_tensor_type"))

    def test_sequence_type_roundtrip(self):
        inner = ir.TensorType(ir.DataType.FLOAT)
        seq_type = ir.SequenceType(inner)
        proto = serde.serialize_type(seq_type)
        deserialized = serde.deserialize_type_proto_for_type(proto)
        self.assertIsInstance(deserialized, ir.SequenceType)


class AttributeSerializationCoverageTest(unittest.TestCase):
    """Test serialization of less common attribute types."""

    def test_ints_attribute_roundtrip(self):
        node = ir.Node("", "TestOp", [], attributes=[ir.AttrInt64s("axes", [0, 1, 2])])
        proto = serde.serialize_node(node)
        deserialized = serde.deserialize_node(proto)
        self.assertEqual(list(deserialized.attributes["axes"].as_ints()), [0, 1, 2])

    def test_floats_attribute_roundtrip(self):
        node = ir.Node("", "TestOp", [], attributes=[ir.AttrFloat32s("vals", [1.0, 2.0])])
        proto = serde.serialize_node(node)
        deserialized = serde.deserialize_node(proto)
        self.assertEqual(list(deserialized.attributes["vals"].as_floats()), [1.0, 2.0])

    def test_strings_attribute_roundtrip(self):
        node = ir.Node("", "TestOp", [], attributes=[ir.AttrStrings("names", ["a", "b"])])
        proto = serde.serialize_node(node)
        deserialized = serde.deserialize_node(proto)
        self.assertEqual(list(deserialized.attributes["names"].as_strings()), ["a", "b"])

    def test_tensors_attribute_roundtrip(self):
        t1 = ir.Tensor(np.array([1.0], dtype=np.float32))
        t2 = ir.Tensor(np.array([2.0], dtype=np.float32))
        node = ir.Node("", "TestOp", [], attributes=[ir.AttrTensors("weights", [t1, t2])])
        proto = serde.serialize_node(node)
        deserialized = serde.deserialize_node(proto)
        tensors = deserialized.attributes["weights"].as_tensors()
        self.assertEqual(len(tensors), 2)

    def test_type_proto_attribute_roundtrip(self):
        type_and_shape = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([1, 2]))
        node = ir.Node("", "TestOp", [], attributes=[ir.AttrTypeProto("t", type_and_shape)])
        proto = serde.serialize_node(node)
        deserialized = serde.deserialize_node(proto)
        attr = deserialized.attributes["t"]
        self.assertEqual(attr.type, ir.AttributeType.TYPE_PROTO)

    def test_type_protos_attribute_roundtrip(self):
        ts1 = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([1]))
        ts2 = ir.TypeAndShape(ir.TensorType(ir.DataType.DOUBLE), ir.Shape([2]))
        node = ir.Node("", "TestOp", [], attributes=[ir.AttrTypeProtos("ts", [ts1, ts2])])
        proto = serde.serialize_node(node)
        deserialized = serde.deserialize_node(proto)
        attr = deserialized.attributes["ts"]
        self.assertEqual(attr.type, ir.AttributeType.TYPE_PROTOS)


class FunctionSerializationCoverageTest(unittest.TestCase):
    def test_function_with_value_info_roundtrip(self):
        """Test that functions with typed values serialize and deserialize with value info."""
        model_text = """
            <ir_version: 10, opset_import: ["": 17, "custom": 1]>
            agraph (float[2, 3] x) => (float[2, 3] z) {
                z = custom.my_func(x)
            }
            <domain: "custom", opset_import: ["": 17]>
            my_func (x) => (z) { z = Relu(x) }
        """
        model_proto = onnx.parser.parse_model(model_text)
        model = serde.deserialize_model(model_proto)

        # Add type info to function values
        func = next(iter(model.functions.values()))
        for value in itertools.chain(func.inputs, func.outputs):
            value.type = ir.TensorType(ir.DataType.FLOAT)
            value.shape = ir.Shape([2, 3])

        # Serialize and check
        result_proto = serde.serialize_model(model)
        self.assertGreater(len(result_proto.functions), 0)
        func_proto = result_proto.functions[0]
        value_info_by_name = {
            value_info.name: value_info for value_info in func_proto.value_info
        }
        self.assertEqual(set(value_info_by_name), {"x", "z"})
        for name in ("x", "z"):
            value_info = value_info_by_name[name]
            self.assertEqual(value_info.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
            self.assertEqual(
                [dim.dim_value for dim in value_info.type.tensor_type.shape.dim], [2, 3]
            )

    def test_function_with_ref_attr(self):
        """Test serialization of functions with reference attributes."""
        func_proto = onnx.helper.make_function(
            "test_domain",
            "test_func",
            ["x"],
            ["y"],
            [onnx.helper.make_node("Relu", ["x"], ["y"])],
            [onnx.helper.make_opsetid("", 17)],
            attributes=["my_attr"],
        )
        func = serde.deserialize_function(func_proto)
        self.assertIn("my_attr", func.attributes)

        # Roundtrip
        serialized = serde.serialize_function(func)
        self.assertIn("my_attr", serialized.attribute)


class ExperimentalFunctionValueInfoIR9Test(unittest.TestCase):
    """Test value info handling for functions in IR version <10."""

    def test_model_ir9_with_function_value_info(self):
        """Functions in IR9 have value info stored in the main graph."""
        # Create a model with IR version 9 and functions
        func_proto = onnx.helper.make_function(
            "pkg",
            "my_func",
            ["x"],
            ["y"],
            [onnx.helper.make_node("Relu", ["x"], ["y"])],
            [onnx.helper.make_opsetid("", 17)],
        )
        # Create model proto with ir version 9
        graph_proto = onnx.helper.make_graph(
            [onnx.helper.make_node("pkg.my_func", ["input"], ["output"], domain="pkg")],
            "test_graph",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
        )
        # Add experimental function value info to the graph
        vi = graph_proto.value_info.add()
        vi.name = "pkg::my_func/x"
        vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        vi.type.tensor_type.shape.dim.add().dim_value = 1

        model_proto = onnx.helper.make_model(graph_proto, functions=[func_proto])
        model_proto.ir_version = 9
        model_proto.opset_import.extend([onnx.helper.make_opsetid("pkg", 1)])

        model = serde.deserialize_model(model_proto)
        self.assertEqual(model.ir_version, 9)

        # Serialize back - should use experimental format for IR9
        result_proto = serde.serialize_model(model)
        self.assertEqual(result_proto.ir_version, 9)
        # Verify experimental function value_info entries in main graph
        value_info_by_name = {
            value_info.name: value_info for value_info in result_proto.graph.value_info
        }
        self.assertIn("pkg::my_func/x", value_info_by_name)
        value_info = value_info_by_name["pkg::my_func/x"]
        self.assertEqual(value_info.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in value_info.type.tensor_type.shape.dim], [1])


class ModelWithMetadataPropsTest(unittest.TestCase):
    def test_model_metadata_props_roundtrip(self):
        model = ir.Model(
            graph=ir.Graph([], [], nodes=[], name="g"),
            ir_version=10,
            metadata_props={"key": "value", "author": "test"},
        )
        proto = serde.serialize_model(model)
        deserialized = serde.deserialize_model(proto)
        self.assertEqual(deserialized.metadata_props["key"], "value")
        self.assertEqual(deserialized.metadata_props["author"], "test")

    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration"),
        "ModelProto.configuration is not available",
    )
    def test_model_configuration_roundtrip(self):
        model_proto = onnx.helper.make_model(
            onnx.helper.make_graph([], "test", [], []), ir_version=11
        )
        model_proto.configuration.add(name="conf0", num_devices=2, device=["CPU", "CUDA:0"])
        model = serde.deserialize_model(model_proto)
        self.assertEqual(
            model.device_configurations,
            (
                _multi_device.ModelConfiguration(
                    name="conf0",
                    num_devices=2,
                    device_names=("CPU", "CUDA:0"),
                ),
            ),
        )
        serialized = serde.serialize_model(model)

        self.assertEqual(len(serialized.configuration), 1)
        self.assertEqual(serialized.configuration[0].name, "conf0")
        self.assertEqual(serialized.configuration[0].num_devices, 2)
        self.assertEqual(list(serialized.configuration[0].device), ["CPU", "CUDA:0"])

    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration"),
        "ModelProto.configuration is not available",
    )
    def test_model_configuration_from_dataclass(self):
        model = ir.Model(graph=ir.Graph([], [], nodes=[], name="g"), ir_version=11)
        model.device_configurations = (
            _multi_device.ModelConfiguration(
                name="conf0",
                num_devices=2,
                device_names=("CPU", "CUDA:0"),
            ),
        )
        serialized = serde.serialize_model(model)

        self.assertEqual(len(serialized.configuration), 1)
        self.assertEqual(serialized.configuration[0].name, "conf0")
        self.assertEqual(serialized.configuration[0].num_devices, 2)
        self.assertEqual(list(serialized.configuration[0].device), ["CPU", "CUDA:0"])


class ShapeSerializationEdgeCaseTest(unittest.TestCase):
    def test_serialize_shape_with_denotation(self):
        shape = ir.Shape([1, 2], denotations=["batch", "channel"])
        value = ir.Value(
            name="x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=shape,
        )
        proto = onnx.ValueInfoProto()
        serde.serialize_value_into(proto, value)
        dims = proto.type.tensor_type.shape.dim
        self.assertEqual(dims[0].denotation, "batch")
        self.assertEqual(dims[1].denotation, "channel")

    def test_serialize_shape_without_type_logs_warning(self):
        """When a value has shape but no type, serialization should handle it gracefully."""
        shape = ir.Shape([1, 2])
        value = ir.Value(name="x", shape=shape)
        proto = onnx.ValueInfoProto()
        # Should not raise - just logs a warning
        serde.serialize_value_into(proto, value)

    def test_serialize_sequence_type_with_shape(self):
        """Serialize shape for a sequence type with nested tensor."""
        inner = ir.TensorType(ir.DataType.FLOAT)
        seq_type = ir.SequenceType(inner)
        shape = ir.Shape([3, 4])
        value = ir.Value(name="x", type=seq_type, shape=shape)
        proto = onnx.ValueInfoProto()
        serde.serialize_value_into(proto, value)
        # Shape should be set on the inner tensor type
        self.assertTrue(proto.type.HasField("sequence_type"))


class NodeSerializationTest(unittest.TestCase):
    def test_node_with_domain_and_overload(self):
        node = ir.Node(
            "custom.domain",
            "MyOp",
            [],
            name="my_node",
            overload="v2",
            doc_string="test doc",
            metadata_props={"k": "v"},
        )
        proto = serde.serialize_node(node)
        self.assertEqual(proto.domain, "custom.domain")
        self.assertEqual(proto.name, "my_node")
        self.assertEqual(proto.overload, "v2")
        self.assertEqual(proto.doc_string, "test doc")

    def test_node_with_none_inputs(self):
        """None inputs should serialize as empty strings."""
        x = ir.Value(name="x")
        node = ir.Node("", "Op", [x, None, x])
        proto = serde.serialize_node(node)
        self.assertEqual(list(proto.input), ["x", "", "x"])

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_device_configurations_roundtrip(self):
        node_proto = onnx.helper.make_node("Relu", ["x"], ["y"], name="node")
        node_device_configuration = node_proto.device_configurations.add()
        node_device_configuration.configuration_id = "conf0"
        node_device_configuration.pipeline_stage = 1
        sharding_spec = node_device_configuration.sharding_spec.add()
        sharding_spec.tensor_name = "x"
        sharding_spec.device.extend([0, 1])
        sharded_dim = sharding_spec.sharded_dim.add()
        sharded_dim.axis = 0
        sharded_dim.simple_sharding.add(dim_value=4, num_shards=2)

        node = serde.deserialize_node(node_proto)

        # The sharding is bound to the node's actual input value (object identity).
        self.assertEqual(len(node.device_configurations), 1)
        config = node.device_configurations[0]
        self.assertEqual(config.pipeline_stage, 1)
        # Configuration is a placeholder carrying the id (no model post-pass here).
        self.assertEqual(config.configuration.name, "conf0")
        spec = config.sharding_specs[0]
        self.assertIs(spec.value, node.inputs[0])
        self.assertEqual(spec.value.name, "x")
        self.assertEqual(spec.device, (0, 1))
        self.assertEqual(spec.sharded_dims[0].axis, 0)
        self.assertEqual(spec.sharded_dims[0].simple_shardings[0].dim, 4)
        self.assertEqual(spec.sharded_dims[0].simple_shardings[0].num_shards, 2)

        # sharding_of returns the live spec for the value.
        self.assertEqual(node.sharding_of(node.inputs[0]), (spec,))

        serialized = serde.serialize_node(node)
        self.assertEqual(len(serialized.device_configurations), 1)
        result = serialized.device_configurations[0]
        self.assertEqual(result.configuration_id, "conf0")
        self.assertEqual(result.pipeline_stage, 1)
        self.assertEqual(len(result.sharding_spec), 1)
        self.assertEqual(result.sharding_spec[0].tensor_name, "x")
        self.assertEqual(list(result.sharding_spec[0].device), [0, 1])

    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration")
        and hasattr(onnx.NodeProto(), "device_configurations"),
        "Multi-device protos are not available",
    )
    def test_device_configurations_model_roundtrip_resolves_objects(self):
        # x -> Relu -> y with model-level configuration and node sharding.
        x = ir.Value(name="x", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.FLOAT))
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="node")
        graph = ir.Graph([x], [node.outputs[0]], nodes=[node], opset_imports={"": 18})
        model = ir.Model(graph, ir_version=11)
        conf0 = model.add_device_configuration("conf0", device_names=("CPU", "CUDA:0"))
        node.shard(x, configuration=conf0, axis=0, num_shards=2, device_indices=(0, 1))

        deserialized = serde.deserialize_model(serde.serialize_model(model))
        new_node = deserialized.graph[0]
        new_config = new_node.device_configurations[0]
        # configuration_id resolved to the actual model configuration object.
        self.assertIs(new_config.configuration, deserialized.device_configurations[0])
        # tensor_name resolved to the actual node input value object.
        self.assertIs(new_config.sharding_specs[0].value, new_node.inputs[0])
        self.assertEqual(_multi_device._check_device_configurations(deserialized), [])

    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration")
        and hasattr(onnx.NodeProto(), "device_configurations"),
        "Multi-device protos are not available",
    )
    def test_device_configurations_dangling_id_roundtrips_as_placeholder(self):
        # A node references a configuration that is not declared on the model.
        x = ir.Value(name="x", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.FLOAT))
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="node")
        graph = ir.Graph([x], [node.outputs[0]], nodes=[node], opset_imports={"": 18})
        model = ir.Model(graph, ir_version=11)
        # Register conf0 on the model but point the node at an unrelated id.
        model.add_device_configuration("conf0", device_names=("CPU",))
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=_multi_device.ModelConfiguration("ghost", num_devices=1),
                sharding_specs=(_multi_device.ShardingSpec(value=x),),
            ),
        )

        deserialized = serde.deserialize_model(serde.serialize_model(model))
        new_config = deserialized.graph[0].device_configurations[0]
        # The dangling id is preserved as a placeholder, not silently dropped.
        self.assertEqual(new_config.configuration.name, "ghost")
        self.assertNotIn(new_config.configuration, deserialized.device_configurations)

    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration")
        and hasattr(onnx.NodeProto(), "device_configurations"),
        "Multi-device protos are not available",
    )
    def test_device_configurations_resolution_skips_unconfigured_nodes(self):
        # A model that declares a configuration but also has a node without any
        # device configuration: the resolution post-pass must skip that node.
        x = ir.Value(name="x", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.FLOAT))
        relu = ir.Node("", "Relu", [x], outputs=[ir.Value(name="h")], name="relu")
        ident = ir.Node(
            "", "Identity", [relu.outputs[0]], outputs=[ir.Value(name="y")], name="id"
        )
        graph = ir.Graph([x], [ident.outputs[0]], nodes=[relu, ident], opset_imports={"": 18})
        model = ir.Model(graph, ir_version=11)
        conf = model.add_device_configuration("conf0", device_names=("CPU", "CUDA:0"))
        # Only ``relu`` is sharded; ``ident`` has no device configuration.
        relu.shard(x, configuration=conf, axis=0, num_shards=2)

        deserialized = serde.deserialize_model(serde.serialize_model(model))
        new_relu, new_ident = deserialized.graph[0], deserialized.graph[1]
        self.assertEqual(new_ident.device_configurations, ())
        self.assertIs(
            new_relu.device_configurations[0].configuration,
            deserialized.device_configurations[0],
        )

    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration")
        and hasattr(onnx.NodeProto(), "device_configurations"),
        "Multi-device protos are not available",
    )
    def test_device_configurations_resolved_in_functions(self):
        # A node inside a model-local function references a model configuration;
        # the resolution post-pass must reach function nodes too.
        fx = ir.Value(name="fx", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.FLOAT))
        fnode = ir.Node("", "Relu", [fx], outputs=[ir.Value(name="fy")], name="frelu")
        fgraph = ir.Graph([fx], [fnode.outputs[0]], nodes=[fnode], opset_imports={"": 18})
        func = ir.Function(domain="custom", name="MyFunc", graph=fgraph, attributes=())

        x = ir.Value(name="x", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.FLOAT))
        call = ir.Node("custom", "MyFunc", [x], outputs=[ir.Value(name="y")], name="call")
        graph = ir.Graph(
            [x], [call.outputs[0]], nodes=[call], opset_imports={"": 18, "custom": 1}
        )
        model = ir.Model(graph, ir_version=11, functions=[func])
        conf = model.add_device_configuration("conf0", device_names=("CPU", "CUDA:0"))
        fnode.shard(fx, configuration=conf, axis=0, num_shards=2)

        deserialized = serde.deserialize_model(serde.serialize_model(model))
        new_func = next(iter(deserialized.functions.values()))
        new_fnode = next(iter(new_func))
        # The function node's configuration_id is resolved to the model's object.
        self.assertIs(
            new_fnode.device_configurations[0].configuration,
            deserialized.device_configurations[0],
        )
        self.assertEqual(_multi_device._check_device_configurations(deserialized), [])

    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration")
        and hasattr(onnx.NodeProto(), "device_configurations"),
        "Multi-device protos are not available",
    )
    def test_device_configurations_negative_axis_roundtrip(self):
        # ONNX allows negative axes; a model using one must round-trip and pass
        # the checker (regression for the negative-axis false positive).
        x = ir.Value(name="x", shape=ir.Shape([4, 8]), type=ir.TensorType(ir.DataType.FLOAT))
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="node")
        graph = ir.Graph([x], [node.outputs[0]], nodes=[node], opset_imports={"": 18})
        model = ir.Model(graph, ir_version=11)
        conf = model.add_device_configuration("conf0", num_devices=2)
        node.shard(x, configuration=conf, axis=-1, num_shards=2)

        deserialized = serde.deserialize_model(serde.serialize_model(model))
        spec = deserialized.graph[0].device_configurations[0].sharding_specs[0]
        self.assertEqual(spec.sharded_dims[0].axis, -1)
        self.assertEqual(_multi_device._check_device_configurations(deserialized), [])

    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration")
        and hasattr(onnx.NodeProto(), "device_configurations"),
        "Multi-device protos are not available",
    )
    def test_device_configurations_not_serialized_for_ir10(self):
        # Multi-device metadata requires IR v11 and is intentionally dropped for
        # older model IR versions.
        x = ir.Value(name="x", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.FLOAT))
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="node")
        graph = ir.Graph([x], [node.outputs[0]], nodes=[node], opset_imports={"": 18})

        fx = ir.Value(name="fx", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.FLOAT))
        fnode = ir.Node("", "Relu", [fx], outputs=[ir.Value(name="fy")], name="fnode")
        fgraph = ir.Graph([fx], [fnode.outputs[0]], nodes=[fnode], opset_imports={"": 18})
        func = ir.Function(domain="custom", name="MyFunc", graph=fgraph, attributes=())

        model = ir.Model(graph, ir_version=10, functions=[func])
        conf = model.add_device_configuration("conf0", device_names=("CPU", "CUDA:0"))
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        fnode.shard(fx, configuration=conf, axis=0, num_shards=2)

        proto = serde.serialize_model(model)
        self.assertEqual(len(proto.configuration), 0)
        self.assertEqual(len(proto.graph.node[0].device_configurations), 0)
        self.assertEqual(len(proto.functions[0].node[0].device_configurations), 0)

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_device_configurations_from_dataclass(self):
        x = ir.Value(name="x")
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")])
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=_multi_device.ModelConfiguration(name="conf0", num_devices=2),
                sharding_specs=(
                    _multi_device.ShardingSpec(
                        value=x,
                        device=(0, 1),
                        index_to_device_group_map=(
                            _multi_device.IndexToDeviceGroupMapEntry(
                                key=0,
                                value=(0, 1),
                            ),
                        ),
                        sharded_dims=(
                            _multi_device.ShardedDim(
                                axis=0,
                                simple_shardings=(
                                    _multi_device.SimpleShardedDim(
                                        dim=ir.SymbolicDim("BATCH"),
                                        num_shards=2,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                pipeline_stage=1,
            ),
        )
        serialized = serde.serialize_node(node)

        self.assertEqual(len(serialized.device_configurations), 1)
        result = serialized.device_configurations[0]
        self.assertEqual(result.configuration_id, "conf0")
        self.assertEqual(result.pipeline_stage, 1)
        self.assertEqual(len(result.sharding_spec), 1)
        self.assertEqual(result.sharding_spec[0].tensor_name, "x")
        self.assertEqual(list(result.sharding_spec[0].device), [0, 1])
        self.assertEqual(len(result.sharding_spec[0].index_to_device_group_map), 1)
        self.assertEqual(result.sharding_spec[0].index_to_device_group_map[0].key, 0)
        self.assertEqual(
            list(result.sharding_spec[0].index_to_device_group_map[0].value),
            [0, 1],
        )
        self.assertEqual(result.sharding_spec[0].sharded_dim[0].axis, 0)
        self.assertEqual(
            result.sharding_spec[0].sharded_dim[0].simple_sharding[0].dim_param, "BATCH"
        )

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_serialize_sharding_with_unnamed_value_raises(self):
        x = ir.Value(name="")  # no name
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")])
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=_multi_device.ModelConfiguration("conf0", num_devices=1),
                sharding_specs=(_multi_device.ShardingSpec(value=x),),
            ),
        )
        with self.assertRaises(serde.SerdeError):
            serde.serialize_node(node)


class StringTensorSerializationTest(unittest.TestCase):
    def test_string_tensor_roundtrip(self):
        t = ir.StringTensor(np.array([b"hello", b"world"]), name="str_tensor")
        proto = serde.serialize_tensor(t)
        deserialized = serde.deserialize_tensor(proto)
        self.assertEqual(deserialized.name, "str_tensor")


class GraphSerializationWithSubgraphsTest(unittest.TestCase):
    def test_graphs_attribute_roundtrip(self):
        """Test serialization of GRAPHS attribute (multiple subgraphs)."""
        sub_graph1 = ir.Graph([], [ir.Value(name="o1")], nodes=[], name="sub1")
        sub_graph2 = ir.Graph([], [ir.Value(name="o2")], nodes=[], name="sub2")
        node = ir.Node(
            "",
            "TestOp",
            [],
            attributes=[ir.AttrGraphs("branches", [sub_graph1, sub_graph2])],
        )
        proto = serde.serialize_node(node)
        deserialized = serde.deserialize_node(proto)
        graphs = deserialized.attributes["branches"].as_graphs()
        self.assertEqual(len(graphs), 2)


class ReferenceAttributeSerializationTest(unittest.TestCase):
    def test_reference_attribute_roundtrip(self):
        ref_attr = ir.RefAttr("my_attr", "external_attr", ir.AttributeType.FLOAT)
        node = ir.Node("", "TestOp", [], attributes=[ref_attr])
        proto = serde.serialize_node(node)
        deserialized = serde.deserialize_node(proto)
        attr = deserialized.attributes["my_attr"]
        self.assertTrue(attr.is_ref())
        self.assertEqual(attr.ref_attr_name, "external_attr")


class ValueInfoSerializationTest(unittest.TestCase):
    def test_value_with_doc_string(self):
        value = ir.Value(
            name="x",
            type=ir.TensorType(ir.DataType.FLOAT),
            doc_string="A test value",
        )
        proto = onnx.ValueInfoProto()
        serde.serialize_value_into(proto, value)
        self.assertEqual(proto.doc_string, "A test value")

    def test_value_with_metadata_props(self):
        value = ir.Value(
            name="x",
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        value.metadata_props["key"] = "val"
        proto = onnx.ValueInfoProto()
        serde.serialize_value_into(proto, value)
        self.assertGreater(len(proto.metadata_props), 0)


class SparseTensorTest(unittest.TestCase):
    """Tests for SparseTensor serialization and deserialization."""

    def _make_sparse_tensor_proto(self, name: str = "my_sparse") -> onnx.SparseTensorProto:
        """Create a simple SparseTensorProto for testing."""
        values_proto = onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])
        indices_proto = onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], [0, 2, 5])
        sparse_proto = onnx.helper.make_sparse_tensor(values_proto, indices_proto, [6])
        sparse_proto.values.name = name
        return sparse_proto

    def test_deserialize_sparse_tensor(self):
        proto = self._make_sparse_tensor_proto()
        result = serde.deserialize_sparse_tensor(proto)
        self.assertIsInstance(result, ir.SparseTensor)
        self.assertEqual(result.name, "my_sparse")
        self.assertEqual(result.dims, [6])
        self.assertEqual(result.values.size, 3)
        self.assertEqual(result.indices.size, 3)

    def test_serialize_sparse_tensor(self):
        proto = self._make_sparse_tensor_proto()
        sparse = serde.deserialize_sparse_tensor(proto)
        result_proto = serde.serialize_sparse_tensor(sparse)
        self.assertIsInstance(result_proto, onnx.SparseTensorProto)
        self.assertEqual(list(result_proto.dims), [6])
        self.assertEqual(result_proto.values.name, "my_sparse")

    def test_sparse_tensor_roundtrip(self):
        proto = self._make_sparse_tensor_proto()
        sparse = serde.deserialize_sparse_tensor(proto)
        result_proto = serde.serialize_sparse_tensor(sparse)
        # Check that values are preserved
        np.testing.assert_array_equal(
            onnx.numpy_helper.to_array(result_proto.values), [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            onnx.numpy_helper.to_array(result_proto.indices), [0, 2, 5]
        )
        self.assertEqual(list(result_proto.dims), [6])

    def test_from_proto_sparse_tensor(self):
        proto = self._make_sparse_tensor_proto()
        result = ir.from_proto(proto)
        self.assertIsInstance(result, ir.SparseTensor)
        self.assertEqual(result.name, "my_sparse")

    def test_to_proto_sparse_tensor(self):
        proto = self._make_sparse_tensor_proto()
        sparse = serde.deserialize_sparse_tensor(proto)
        result = ir.to_proto(sparse)
        self.assertIsInstance(result, onnx.SparseTensorProto)
        self.assertEqual(list(result.dims), [6])

    def test_sparse_tensor_attribute_roundtrip(self):
        proto = self._make_sparse_tensor_proto()
        sparse = serde.deserialize_sparse_tensor(proto)
        node = ir.Node("", "TestOp", [], attributes=[ir.AttrSparseTensor("sparse_attr", sparse)])
        node_proto = serde.serialize_node(node)
        restored = serde.deserialize_node(node_proto)
        attr = restored.attributes["sparse_attr"]
        self.assertEqual(attr.type, ir.AttributeType.SPARSE_TENSOR)
        restored_sparse = attr.value
        self.assertIsInstance(restored_sparse, ir.SparseTensor)
        self.assertEqual(restored_sparse.dims, [6])

    def test_sparse_tensors_attribute_roundtrip(self):
        proto1 = self._make_sparse_tensor_proto("sparse1")
        proto2 = self._make_sparse_tensor_proto("sparse2")
        sparse1 = serde.deserialize_sparse_tensor(proto1)
        sparse2 = serde.deserialize_sparse_tensor(proto2)
        node = ir.Node(
            "", "TestOp", [], attributes=[ir.AttrSparseTensors("sparse_attrs", [sparse1, sparse2])]
        )
        node_proto = serde.serialize_node(node)
        restored = serde.deserialize_node(node_proto)
        attr = restored.attributes["sparse_attrs"]
        self.assertEqual(attr.type, ir.AttributeType.SPARSE_TENSORS)
        self.assertEqual(len(attr.value), 2)

    def test_graph_with_sparse_initializer_deserialization(self):
        model_proto = onnx.helper.make_model(
            onnx.helper.make_graph(
                [],
                "test_graph",
                [],
                [onnx.helper.make_tensor_value_info("sparse_val", onnx.TensorProto.FLOAT, None)],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 17)],
        )
        sparse_proto = self._make_sparse_tensor_proto("sparse_val")
        model_proto.graph.sparse_initializer.append(sparse_proto)

        model = serde.deserialize_model(model_proto)
        self.assertIn("sparse_val", model.graph.initializers)
        sparse_init = model.graph.initializers["sparse_val"]
        self.assertIsInstance(sparse_init.const_sparse_value, ir.SparseTensor)
        self.assertEqual(sparse_init.const_sparse_value.dims, [6])

    def test_graph_with_sparse_initializer_serialization(self):
        model_proto = onnx.helper.make_model(
            onnx.helper.make_graph(
                [],
                "test_graph",
                [],
                [onnx.helper.make_tensor_value_info("sparse_val", onnx.TensorProto.FLOAT, None)],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 17)],
        )
        sparse_proto = self._make_sparse_tensor_proto("sparse_val")
        model_proto.graph.sparse_initializer.append(sparse_proto)

        model = serde.deserialize_model(model_proto)
        result_proto = serde.serialize_model(model)
        self.assertEqual(len(result_proto.graph.sparse_initializer), 1)
        sp = result_proto.graph.sparse_initializer[0]
        self.assertEqual(sp.values.name, "sparse_val")
        self.assertEqual(list(sp.dims), [6])

    def test_sparse_tensor_name_property(self):
        sparse_proto = self._make_sparse_tensor_proto("test_name")
        sparse = serde.deserialize_sparse_tensor(sparse_proto)
        self.assertEqual(sparse.name, "test_name")
        # Test name setter
        sparse.name = "new_name"
        self.assertEqual(sparse.name, "new_name")
        self.assertEqual(sparse.values.name, "new_name")

    def test_sparse_tensor_is_sparse_tensor_protocol(self):
        sparse_proto = self._make_sparse_tensor_proto()
        sparse = serde.deserialize_sparse_tensor(sparse_proto)
        self.assertIsInstance(sparse, ir.SparseTensorProtocol)

    def test_value_with_sparse_const_value(self):
        sparse_proto = self._make_sparse_tensor_proto("w")
        sparse = serde.deserialize_sparse_tensor(sparse_proto)
        value = ir.Value(name="w", const_sparse_value=sparse)
        self.assertIsInstance(value.const_sparse_value, ir.SparseTensor)

    def test_serializer_no_warning_for_sparse_initializer_without_const_value(self):
        """Serializer should not warn when a sparse-typed initializer has no const_sparse_value."""
        v = ir.Value(
            None,
            index=None,
            name="sparse_x",
            type=ir.SparseTensorType(ir.DataType.FLOAT),
            shape=ir.Shape([3, 3]),
        )
        graph = ir.Graph(inputs=[], outputs=[], nodes=[], name="test")
        graph.initializers["sparse_x"] = v
        model = ir.Model(graph, ir_version=8)
        with self.assertLogs("onnx_ir.serde", level="WARNING") as log_ctx:
            # Generate an unrelated warning so assertLogs doesn't raise on no warnings
            import logging

            logging.getLogger("onnx_ir.serde").warning("sentinel")
            serde.serialize_model(model)
        # Only the sentinel warning should appear, not the "constant value" warning
        warning_msgs = [r for r in log_ctx.output if "constant value" in r]
        self.assertEqual(warning_msgs, [], "Unexpected warning for sparse-typed initializer")

    def test_serializer_warns_for_dense_initializer_without_const_value(self):
        """Serializer should warn when a dense-typed initializer has no const_value."""
        v = ir.Value(
            None,
            index=None,
            name="dense_x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([3]),
        )
        graph = ir.Graph(inputs=[], outputs=[], nodes=[], name="test")
        graph.initializers["dense_x"] = v
        model = ir.Model(graph, ir_version=8)
        with self.assertLogs("onnx_ir.serde", level="WARNING") as log_ctx:
            serde.serialize_model(model)
        warning_msgs = [r for r in log_ctx.output if "constant value" in r]
        self.assertGreater(len(warning_msgs), 0, "Expected a warning for dense initializer without const_value")


if __name__ == "__main__":
    unittest.main()
