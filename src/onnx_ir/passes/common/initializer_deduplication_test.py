# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the initializer_deduplication passes."""

import os
import tempfile
import unittest
import weakref
from unittest import mock

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

    def _create_external_model(self, shape=(3,)) -> tuple[ir.Model, str]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => (float[3] sum)
            <float[3] w1 = {1.0, 2.0, 3.0}, float[3] w2 = {1.0, 2.0, 3.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        for initializer in model.graph.initializers.values():
            tensor = ir.ExternalTensor(
                location=f"{initializer.name}.data",
                base_dir=directory.name,
                offset=0,
                length=int(np.prod(shape)) * 4,
                dtype=ir.DataType.FLOAT,
                shape=ir.Shape(shape),
                name=initializer.name,
            )
            initializer.const_value = tensor
            initializer.shape = ir.Shape(shape)
            self.addCleanup(tensor.release)
        model.graph.outputs[0].shape = ir.Shape(shape)
        return model, directory.name

    @parameterized.parameterized.expand([False, True])
    def test_missing_external_data_is_not_deduplicated(self, subgraph):
        model, _ = self._create_external_model()
        graph = model.graph
        initializers = tuple(graph.initializers.values())
        tensors = tuple(value.const_value for value in initializers)
        if subgraph:
            model = ir.Model(
                ir.Graph(
                    [],
                    [],
                    nodes=[ir.node("SubgraphOp", [], attributes={"body": graph})],
                ),
                ir_version=10,
            )

        with self.assertLogs(initializer_deduplication.logger, level="WARNING") as logs:
            result = self.pass_class()(model)

        self.assertFalse(result.modified)
        self.assertEqual(tuple(graph.initializers.values()), initializers)
        self.assertEqual(tuple(value.const_value for value in initializers), tensors)
        self.assertEqual(tuple(graph[0].inputs), initializers)
        self.assertEqual(len(logs.output), 2)
        self.assertIn("w1.data", logs.output[0])
        self.assertIn("base_dir", logs.output[0])

    def test_readable_external_data_is_deduplicated(self):
        model, directory = self._create_external_model()
        for name in ("w1", "w2"):
            np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(
                os.path.join(directory, f"{name}.data")
            )

        result = self.pass_class()(model)

        self.assertTrue(result.modified)
        self.assertEqual(len(model.graph.initializers), 1)
        self.assertIs(model.graph[0].inputs[0], model.graph[0].inputs[1])
        np.testing.assert_array_equal(
            model.graph.initializers["w1"].const_value.numpy(), [1.0, 2.0, 3.0]
        )

    def test_cached_external_data_with_missing_current_path_is_not_deduplicated(self):
        model, directory = self._create_external_model()
        missing_directory = os.path.join(directory, "missing")
        for initializer in model.graph.initializers.values():
            tensor = initializer.const_value
            np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(tensor.path)
            tensor.numpy()
            tensor.base_dir = missing_directory

        with self.assertLogs(initializer_deduplication.logger, level="WARNING") as logs:
            result = self.pass_class()(model)

        self.assertFalse(result.modified)
        self.assertEqual(len(model.graph.initializers), 2)
        self.assertEqual(len(logs.output), 2)
        self.assertTrue(all("was not found" in message for message in logs.output))

    def test_cached_external_data_still_validates_path_security(self):
        model, _ = self._create_external_model()
        tensor = model.graph.initializers["w1"].const_value
        np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(tensor.path)
        tensor.numpy()

        with (
            mock.patch.object(
                ir.ExternalTensor,
                "_check_path_containment",
                side_effect=ValueError("unsafe backing path"),
            ),
            self.assertRaisesRegex(ValueError, "unsafe backing path"),
        ):
            self.pass_class()(model)

    @parameterized.parameterized.expand([((0,),), ((2, 0, 3),)])
    def test_missing_empty_external_data_is_not_deduplicated(self, shape):
        model, _ = self._create_external_model(shape)
        initializers = tuple(model.graph.initializers.values())
        tensors = tuple(value.const_value for value in initializers)

        with self.assertLogs(initializer_deduplication.logger, level="WARNING") as logs:
            result = self.pass_class()(model)

        self.assertFalse(result.modified)
        self.assertEqual(tuple(model.graph.initializers.values()), initializers)
        self.assertEqual(tuple(model.graph[0].inputs), initializers)
        self.assertEqual(tuple(value.const_value for value in initializers), tensors)
        self.assertEqual(len(logs.output), 2)
        for name, message in zip(("w1", "w2"), logs.output):
            self.assertIn(f"{name}.data", message)
            self.assertIn("base_dir", message)

    def test_packed_external_data_validation_error_is_preserved(self):
        model, directory = self._create_external_model((1,))
        with open(os.path.join(directory, "w1.data"), "wb") as file:
            file.write(b"\x40")
        tensor = ir.ExternalTensor(
            location="w1.data",
            base_dir=directory,
            offset=0,
            length=1,
            dtype=ir.DataType.FLOAT6E2M3,
            shape=ir.Shape([1]),
            name="w1",
        )
        model.graph.initializers["w1"].const_value = tensor

        def apply_pass():
            with self.assertRaisesRegex(ValueError, "nonzero padding bits"):
                self.pass_class()(model)

        apply_pass()
        tensor.release()

    def test_release_does_not_raise_with_caller_owned_view(self):
        model, directory = self._create_external_model()
        for name in ("w1", "w2"):
            np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(
                os.path.join(directory, f"{name}.data")
            )
        tensor = model.graph.initializers["w1"].const_value
        caller_view = tensor.numpy()

        result = self.pass_class()(model)

        self.assertTrue(result.modified)
        np.testing.assert_array_equal(caller_view, [1.0, 2.0, 3.0])
        reloaded = tensor.numpy()
        np.testing.assert_array_equal(reloaded, caller_view)
        del reloaded
        del caller_view
        tensor.release()

    def test_release_does_not_mask_active_read_error(self):
        model, _ = self._create_external_model()
        tensor = model.graph.initializers["w1"].const_value
        np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(tensor.path)
        caller_view = tensor.numpy()

        with (
            self.assertRaisesRegex(ValueError, "read failed"),
            initializer_deduplication._release_tensor(tensor),
        ):
            raise ValueError("read failed")

        np.testing.assert_array_equal(caller_view, [1.0, 2.0, 3.0])
        del caller_view
        tensor.release()

    @parameterized.parameterized.expand([ir.DataType.FLOAT, ir.DataType.FLOAT6E2M3])
    def test_external_data_resources_are_released_per_initializer(self, dtype):
        model, directory = self._create_external_model((1,))
        model = ir.Model(ir.Graph([], [], nodes=[]), ir_version=10)
        for index in range(8):
            name = f"w{index}"
            data = ir.tensor(np.array([index % 4], dtype=dtype.numpy()), dtype=dtype)
            with open(os.path.join(directory, f"{name}.data"), "wb") as file:
                file.write(data.tobytes())
            tensor = ir.ExternalTensor(
                location=f"{name}.data",
                base_dir=directory,
                offset=0,
                length=data.nbytes,
                dtype=dtype,
                shape=ir.Shape([1]),
                name=name,
            )
            model.graph.register_initializer(ir.Value(name=name, const_value=tensor))

        mappings = []
        arrays = []
        load = ir.ExternalTensor._load

        def tracked_load(tensor):
            # Only the current initializer may be live when a canonical tensor reloads.
            self.assertLessEqual(sum(not mapping.closed for mapping in mappings), 1)
            self.assertLessEqual(sum(array() is not None for array in arrays), 1)
            load(tensor)
            mappings.append(tensor.raw)
            arrays.append(weakref.ref(tensor.numpy()))

        with mock.patch.object(ir.ExternalTensor, "_load", tracked_load):
            result = self.pass_class()(model)

        self.assertTrue(result.modified)
        self.assertEqual(len(model.graph.initializers), 4)
        self.assertTrue(all(mapping.closed for mapping in mappings))
        self.assertTrue(all(array() is None for array in arrays))

    def test_invalidated_external_data_raises(self):
        model, _ = self._create_external_model()
        model.graph.initializers["w1"].const_value.invalidate()

        with self.assertRaisesRegex(ValueError, "w1.*invalidated"):
            self.pass_class()(model)

    def test_empty_external_data_nonzero_length_raises_without_copying(self):
        model, directory = self._create_external_model((0,))
        tensor = ir.ExternalTensor(
            location="w1.data",
            base_dir=directory,
            offset=0,
            length=1024**3,
            dtype=ir.DataType.FLOAT,
            shape=ir.Shape([0]),
            name="w1",
        )
        model.graph.initializers["w1"].const_value = tensor
        self.addCleanup(tensor.release)

        with (
            mock.patch.object(ir.ExternalTensor, "tofile") as tofile,
            self.assertRaisesRegex(ValueError, "w1.*length"),
        ):
            self.pass_class()(model)
        tofile.assert_not_called()

    @parameterized.parameterized.expand([((0,),), ((2, 0, 3),)])
    def test_readable_empty_external_data_is_deduplicated(self, shape):
        model, directory = self._create_external_model(shape)
        for name in ("w1", "w2"):
            with open(os.path.join(directory, f"{name}.data"), "wb"):
                pass

        result = self.pass_class()(model)

        self.assertTrue(result.modified)
        self.assertEqual(set(model.graph.initializers), {"w1"})
        self.assertIs(model.graph[0].inputs[0], model.graph[0].inputs[1])
        self.assertEqual(model.graph.initializers["w1"].const_value.numpy().shape, shape)

    def test_empty_external_data_path_traversal_raises(self):
        model, _ = self._create_external_model((0,))
        tensor = model.graph.initializers["w1"].const_value
        model.graph.initializers["w1"].const_value = ir.ExternalTensor(
            location=os.path.join("..", "outside.data"),
            base_dir=tensor.base_dir,
            offset=0,
            length=0,
            dtype=ir.DataType.FLOAT,
            shape=ir.Shape([0]),
            name="w1",
        )

        with self.assertRaisesRegex(ValueError, "outside the base directory"):
            self.pass_class()(model)

    def test_empty_external_data_permission_error_raises(self):
        model, directory = self._create_external_model((0,))
        path = os.path.join(directory, "w1.data")

        with (
            mock.patch("builtins.open", side_effect=PermissionError(path)),
            self.assertRaises(PermissionError),
        ):
            self.pass_class()(model)

    def test_missing_external_data_does_not_prevent_other_deduplication(self):
        model, _ = self._create_external_model()
        for name in ("inline1", "inline2"):
            model.graph.register_initializer(
                ir.Value(
                    name=name,
                    const_value=ir.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
                )
            )
        node = ir.node(
            "Add",
            [
                model.graph.initializers["inline1"],
                model.graph.initializers["inline2"],
            ],
        )
        model.graph.append(node)

        with self.assertLogs(initializer_deduplication.logger, level="WARNING"):
            result = self.pass_class()(model)

        self.assertTrue(result.modified)
        self.assertEqual(set(model.graph.initializers), {"w1", "w2", "inline1"})
        self.assertIs(node.inputs[0], node.inputs[1])
        self.assertIsNot(model.graph[0].inputs[0], model.graph[0].inputs[1])

    def test_corrupt_external_data_raises(self):
        model, directory = self._create_external_model()
        np.array([1.0], dtype=np.float32).tofile(os.path.join(directory, "w1.data"))

        with self.assertRaises(ValueError):
            self.pass_class()(model)

    def test_external_data_permission_error_raises(self):
        model, directory = self._create_external_model()
        for name in ("w1", "w2"):
            np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(
                os.path.join(directory, f"{name}.data")
            )

        with (
            mock.patch.object(ir.ExternalTensor, "numpy", side_effect=PermissionError),
            self.assertRaises(PermissionError),
        ):
            self.pass_class()(model)

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

    def test_deduplicates_non_contiguous_initializers(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => (float[3,2] sum)
            <float[3,2] w1 = {0, 1, 2, 3, 4, 5}, float[3,2] w2 = {0, 1, 2, 3, 4, 5}> {
                sum = Add(w1, w2)
            }
            """
        )
        data = np.arange(6, dtype=np.float32).reshape(2, 3).T
        self.assertFalse(data.flags.c_contiguous)
        model.graph.initializers["w1"].const_value = ir.tensor(data)
        model.graph.initializers["w2"].const_value = ir.tensor(data.copy(order="C"))

        result = self.pass_class()(model)

        self.assertTrue(result.modified)
        self.assertEqual(len(model.graph.initializers), 1)
        self.assertIs(model.graph[0].inputs[0], model.graph[0].inputs[1])
        np.testing.assert_array_equal(model.graph.initializers["w1"].const_value.numpy(), data)

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


class DeduplicateHashedInitializersTest(unittest.TestCase):
    def test_matching_digests_are_trusted_without_byte_comparison(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, float[2] w2 = {3.0, 4.0}> {
                sum = Add(w1, w2)
            }
            """
        )

        with mock.patch.object(initializer_deduplication.hashlib, "sha256") as sha256:
            sha256.return_value.hexdigest.return_value = "matching-digest"
            result = initializer_deduplication.DeduplicateHashedInitializersPass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(set(model.graph.initializers), {"w1"})
        self.assertIs(model.graph[0].inputs[0], model.graph[0].inputs[1])


if __name__ == "__main__":
    unittest.main()
