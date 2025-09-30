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
from onnx_ir import _version_utils, serde


class ConvenienceFunctionsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("model", onnx.ModelProto()),
            ("graph", onnx.GraphProto()),
            ("node", onnx.NodeProto(input=["X"], output=["Y"])),
            (
                "tensor",
                onnx.helper.make_tensor("test_tensor", onnx.TensorProto.FLOAT, [1], [1.0]),
            ),
            ("value_info", onnx.ValueInfoProto()),
            ("type", onnx.TypeProto()),
            ("attribute", onnx.AttributeProto()),
        ]
    )
    def test_from_proto(self, _: str, proto):
        serde.from_proto(proto)

    @parameterized.parameterized.expand(
        [
            ("model", ir.Model(ir.Graph([], [], nodes=[]), ir_version=1)),
            ("graph", ir.Graph([], [], nodes=[])),
            (
                "node",
                ir.Node("", "Op", inputs=[], outputs=[ir.Value(name="value")]),
            ),
            (
                "tensor",
                serde.TensorProtoTensor(
                    onnx.helper.make_tensor("test_tensor", onnx.TensorProto.FLOAT, [1], [1.0])
                ),
            ),
            ("value", ir.Value(name="value")),
            ("type", ir.SequenceType(ir.OptionalType(ir.TensorType(ir.DataType.COMPLEX128)))),
            ("attribute", ir.Attr("attribute", ir.AttributeType.FLOAT, 1)),
            ("ref_attribute", ir.RefAttr("ref_attr", "attr", ir.AttributeType.FLOAT)),
            ("graph_view", ir.GraphView([], [], nodes=[])),
        ]
    )
    def test_to_proto(self, _: str, ir_object):
        serde.to_proto(ir_object)

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
        if dtype == onnx.TensorProto.INT4:
            return  # DL Pack does not support int4
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            ("UINT8", onnx.TensorProto.UINT8),
            ("UINT16", onnx.TensorProto.UINT16),
            ("UINT32", onnx.TensorProto.UINT32),
            ("UINT64", onnx.TensorProto.UINT64),
            ("UINT4", onnx.TensorProto.UINT4),
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
        if dtype == onnx.TensorProto.UINT4:
            return  # DL Pack does not support uint4
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
        original_array = original_array.astype(onnx_dtype.numpy())
        ir_tensor = ir.Tensor(original_array, name="test_tensor")
        proto = serde.to_proto(ir_tensor)
        if original_array.size > 0:
            self.assertGreater(len(proto.raw_data), 0)
        # tensor_proto_tensor from raw_data
        tensor_proto_tensor = serde.from_proto(proto)
        roundtrip_array = tensor_proto_tensor.numpy()
        if onnx_dtype in {
            ir.DataType.FLOAT8E5M2FNUZ,
            ir.DataType.FLOAT8E5M2,
            ir.DataType.FLOAT8E4M3FN,
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

    def test_serialize_shape_successful_when_value_type_not_known(self):
        shape = ir.Shape((1, 2, 3))
        proto = onnx.TypeProto()
        self.assertIsNone(proto.WhichOneof("value"))
        serde.serialize_shape_into(proto, shape)
        self.assertEqual(proto.WhichOneof("value"), "tensor_type")
        deserialized = serde.deserialize_type_proto_for_shape(proto)
        self.assertEqual(deserialized, shape)
        self.assertIsNotNone(proto.tensor_type)
        self.assertEqual(proto.tensor_type.elem_type, onnx.TensorProto.UNDEFINED)


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


if __name__ == "__main__":
    unittest.main()
