# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the safetensors module."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import ml_dtypes
import numpy as np

import onnx_ir as ir


def _create_initializer(tensor: ir.TensorProtocol) -> ir.Value:
    return ir.Value(
        name=tensor.name,
        shape=tensor.shape,
        type=ir.TensorType(tensor.dtype),
        const_value=tensor,
    )


def _create_simple_model_with_initializers() -> ir.Model:
    """Create a simple model with initializers for testing."""
    tensor_0 = ir.tensor([0.0, 1.0, 2.0], dtype=ir.DataType.FLOAT, name="initializer_0")
    initializer_0 = _create_initializer(tensor_0)
    tensor_1 = ir.tensor([3.0, 4.0, 5.0], dtype=ir.DataType.FLOAT, name="initializer_1")
    initializer_1 = _create_initializer(tensor_1)
    tensor_2 = ir.tensor([6.0, 7.0], dtype=ir.DataType.FLOAT, name="initializer_2")
    initializer_2 = _create_initializer(tensor_2)

    identity_node = ir.Node("", "Identity", inputs=(initializer_0,))
    identity_node.outputs[0].shape = ir.Shape([3])
    identity_node.outputs[0].dtype = ir.DataType.FLOAT
    identity_node.outputs[0].name = "identity_0"

    graph = ir.Graph(
        inputs=[initializer_0, initializer_1, initializer_2],
        outputs=[*identity_node.outputs],
        nodes=[identity_node],
        initializers=[initializer_0, initializer_1, initializer_2],
        name="test_graph",
    )
    return ir.Model(graph, ir_version=10)


def _create_large_model_for_sharding() -> ir.Model:
    """Create a model with large tensors for testing sharding."""
    # Create multiple tensors with varying sizes
    initializers = []
    for i in range(5):
        # Create tensors of increasing size
        size = 1000 * (i + 1)  # 1000, 2000, 3000, 4000, 5000 elements
        tensor = ir.tensor(
            np.random.rand(size).astype(np.float32),
            dtype=ir.DataType.FLOAT,
            name=f"initializer_{i}",
        )
        initializers.append(_create_initializer(tensor))

    identity_node = ir.Node("", "Identity", inputs=(initializers[0],))
    identity_node.outputs[0].shape = ir.Shape([1000])
    identity_node.outputs[0].dtype = ir.DataType.FLOAT
    identity_node.outputs[0].name = "identity_0"

    graph = ir.Graph(
        inputs=initializers,
        outputs=[*identity_node.outputs],
        nodes=[identity_node],
        initializers=initializers,
        name="test_graph",
    )
    return ir.Model(graph, ir_version=10)


class SaveSafetensorsTest(unittest.TestCase):
    def setUp(self):
        if sys.version_info[:2] >= (3, 10):
            self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = self.temp_dir.name

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except (PermissionError, FileNotFoundError) as e:
            print(f"Cleanup error: {e}")

    def test_save_safetensors_creates_safetensors_file(self):
        """Test that save_safetensors creates a .safetensors file."""
        model = _create_simple_model_with_initializers()
        path = os.path.join(self.tmpdir, "model.onnx")

        ir.save_safetensors(model, path, size_threshold_bytes=0)

        # Check that the ONNX file was created
        self.assertTrue(os.path.exists(path))
        # Check that the safetensors file was created
        safetensors_path = os.path.join(self.tmpdir, "model.safetensors")
        self.assertTrue(os.path.exists(safetensors_path))

    def test_save_safetensors_preserves_original_model(self):
        """Test that save_safetensors does not modify the original model."""
        model = _create_simple_model_with_initializers()
        path = os.path.join(self.tmpdir, "model.onnx")

        # Save with external data
        ir.save_safetensors(model, path, size_threshold_bytes=0)

        # Check that original model still has in-memory tensors
        for value in model.graph.initializers.values():
            self.assertIsInstance(value.const_value, ir.Tensor)
            self.assertNotIsInstance(value.const_value, ir.ExternalTensor)

    def test_save_safetensors_loads_correctly(self):
        """Test that a model saved with safetensors can be loaded correctly."""
        model = _create_simple_model_with_initializers()
        path = os.path.join(self.tmpdir, "model.onnx")

        ir.save_safetensors(model, path, size_threshold_bytes=0)

        # Load the model back
        loaded_model = ir.load(path)

        # Check that the loaded model has external tensors
        self.assertEqual(len(loaded_model.graph.initializers), 3)
        for name, value in loaded_model.graph.initializers.items():
            self.assertIsInstance(value.const_value, ir.ExternalTensor)
            # Check that the data is correct
            original_tensor = model.graph.initializers[name].const_value
            np.testing.assert_array_equal(value.const_value.numpy(), original_tensor.numpy())

    def test_save_safetensors_size_threshold(self):
        """Test that size_threshold_bytes correctly filters tensors."""
        model = _create_simple_model_with_initializers()
        path = os.path.join(self.tmpdir, "model.onnx")

        # Save with a high threshold so no tensors are externalized
        # Each tensor is 3 or 2 floats = 12 or 8 bytes
        ir.save_safetensors(model, path, size_threshold_bytes=100)

        loaded_model = ir.load(path)

        # All tensors should still be in memory (not external)
        for value in loaded_model.graph.initializers.values():
            self.assertNotIsInstance(value.const_value, ir.ExternalTensor)

    def test_save_safetensors_with_sharding(self):
        """Test that sharding works correctly."""
        model = _create_large_model_for_sharding()
        path = os.path.join(self.tmpdir, "model.onnx")

        # Set max_shard_size to 10KB to force sharding
        # Total size is approximately 60KB (5 tensors * 1000-5000 floats * 4 bytes)
        max_shard_size = 10000
        ir.save_safetensors(
            model, path, size_threshold_bytes=0, max_shard_size_bytes=max_shard_size
        )

        # Check that the ONNX file was created
        self.assertTrue(os.path.exists(path))

        # Check that multiple shard files were created
        shard_files = [
            f
            for f in os.listdir(self.tmpdir)
            if f.startswith("model-") and f.endswith(".safetensors")
        ]
        self.assertGreater(len(shard_files), 1, "Expected multiple shard files")

        # Check that the index file was created
        index_path = os.path.join(self.tmpdir, "model.safetensors.index.json")
        self.assertTrue(os.path.exists(index_path))

        # Verify the index file structure
        with open(index_path) as f:
            index_data = json.load(f)
        self.assertIn("metadata", index_data)
        self.assertIn("total_size", index_data["metadata"])
        self.assertIn("weight_map", index_data)
        # All initializers should be in the weight map
        self.assertEqual(len(index_data["weight_map"]), 5)

    def test_save_safetensors_no_sharding_when_below_threshold(self):
        """Test that no sharding occurs when total size is below threshold."""
        model = _create_simple_model_with_initializers()
        path = os.path.join(self.tmpdir, "model.onnx")

        # Set a very large max_shard_size
        ir.save_safetensors(model, path, size_threshold_bytes=0, max_shard_size_bytes=1000000)

        # Check that only one safetensors file was created
        safetensors_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".safetensors")]
        self.assertEqual(len(safetensors_files), 1)
        self.assertEqual(safetensors_files[0], "model.safetensors")

        # Check that no index file was created
        index_path = os.path.join(self.tmpdir, "model.safetensors.index.json")
        self.assertFalse(os.path.exists(index_path))

    def test_save_safetensors_callback(self):
        """Test that the callback is called for each tensor."""
        model = _create_simple_model_with_initializers()
        path = os.path.join(self.tmpdir, "model.onnx")

        callback_calls = []

        def callback(tensor: ir.TensorProtocol, info: ir.external_data.CallbackInfo):
            callback_calls.append(
                {
                    "tensor_name": tensor.name,
                    "filename": info.filename,
                    "index": info.index,
                    "total": info.total,
                    "offset": info.offset,
                }
            )

        ir.save_safetensors(model, path, size_threshold_bytes=0, callback=callback)

        # Check that the callback was called for each tensor
        self.assertEqual(len(callback_calls), 3)
        # Check that the filenames are correct
        for call in callback_calls:
            self.assertEqual(call["filename"], "model.safetensors")

    def test_save_safetensors_different_dtypes(self):
        """Test that different data types are saved correctly."""
        tensor_int32 = ir.tensor([1, 2, 3], dtype=ir.DataType.INT32, name="int32_tensor")
        tensor_float16 = ir.tensor(
            [1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT16, name="float16_tensor"
        )
        tensor_bool = ir.tensor(
            [True, False, True], dtype=ir.DataType.BOOL, name="bool_tensor"
        )

        initializers = [
            _create_initializer(tensor_int32),
            _create_initializer(tensor_float16),
            _create_initializer(tensor_bool),
        ]

        identity_node = ir.Node("", "Identity", inputs=(initializers[0],))
        identity_node.outputs[0].shape = ir.Shape([3])
        identity_node.outputs[0].dtype = ir.DataType.INT32
        identity_node.outputs[0].name = "identity_0"

        graph = ir.Graph(
            inputs=initializers,
            outputs=[*identity_node.outputs],
            nodes=[identity_node],
            initializers=initializers,
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=10)

        path = os.path.join(self.tmpdir, "model.onnx")
        ir.save_safetensors(model, path, size_threshold_bytes=0)

        # Load the model back
        loaded_model = ir.load(path)

        # Check that the data is correct for all types
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["int32_tensor"].const_value.numpy(),
            np.array([1, 2, 3], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["float16_tensor"].const_value.numpy(),
            np.array([1.0, 2.0, 3.0], dtype=np.float16),
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["bool_tensor"].const_value.numpy(),
            np.array([True, False, True], dtype=bool),
        )

    def test_save_safetensors_subbyte_types(self):
        """Test that sub-byte data types are saved correctly."""
        # Create tensors with sub-byte types using ml_dtypes
        int4_data = np.array([-8, -1, 0, 1, 2, 7, 1, -5], dtype=ml_dtypes.int4)
        tensor_int4 = ir.tensor(int4_data, dtype=ir.DataType.INT4, name="int4_tensor")

        uint4_data = np.array([0, 1, 2, 7, 15, 8, 4, 12], dtype=ml_dtypes.uint4)
        tensor_uint4 = ir.tensor(uint4_data, dtype=ir.DataType.UINT4, name="uint4_tensor")

        float4_data = np.array([0.0, 1.0, 2.0, -1.0], dtype=ml_dtypes.float4_e2m1fn)
        tensor_float4 = ir.tensor(
            float4_data, dtype=ir.DataType.FLOAT4E2M1, name="float4_tensor"
        )

        # Use multiples of 4 for int2/uint2 since they pack 4 values per byte
        int2_data = np.array([-2, -1, 0, 1, 1, -2, 1, 0], dtype=ml_dtypes.int2)
        tensor_int2 = ir.tensor(int2_data, dtype=ir.DataType.INT2, name="int2_tensor")

        uint2_data = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=ml_dtypes.uint2)
        tensor_uint2 = ir.tensor(uint2_data, dtype=ir.DataType.UINT2, name="uint2_tensor")

        initializers = [
            _create_initializer(tensor_int4),
            _create_initializer(tensor_uint4),
            _create_initializer(tensor_float4),
            _create_initializer(tensor_int2),
            _create_initializer(tensor_uint2),
        ]

        identity_node = ir.Node("", "Identity", inputs=(initializers[0],))
        identity_node.outputs[0].shape = tensor_int4.shape
        identity_node.outputs[0].dtype = ir.DataType.INT4
        identity_node.outputs[0].name = "identity_0"

        graph = ir.Graph(
            inputs=initializers,
            outputs=[*identity_node.outputs],
            nodes=[identity_node],
            initializers=initializers,
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=10)

        path = os.path.join(self.tmpdir, "model.onnx")
        ir.save_safetensors(model, path, size_threshold_bytes=0)

        # Load the model back
        loaded_model = ir.load(path)

        # Verify the tensors are correctly loaded as external tensors
        for tensor_name in [
            "int4_tensor",
            "uint4_tensor",
            "float4_tensor",
            "int2_tensor",
            "uint2_tensor",
        ]:
            self.assertIsInstance(
                loaded_model.graph.initializers[tensor_name].const_value, ir.ExternalTensor
            )

        # Check that the raw data is preserved
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["int4_tensor"].const_value.numpy(),
            int4_data,
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["uint4_tensor"].const_value.numpy(),
            uint4_data,
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["float4_tensor"].const_value.numpy(),
            float4_data,
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["int2_tensor"].const_value.numpy(),
            int2_data,
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["uint2_tensor"].const_value.numpy(),
            uint2_data,
        )

        # Check that the dtype is preserved
        self.assertEqual(
            loaded_model.graph.initializers["int4_tensor"].const_value.dtype,
            ir.DataType.INT4,
        )
        self.assertEqual(
            loaded_model.graph.initializers["uint4_tensor"].const_value.dtype,
            ir.DataType.UINT4,
        )
        self.assertEqual(
            loaded_model.graph.initializers["float4_tensor"].const_value.dtype,
            ir.DataType.FLOAT4E2M1,
        )
        self.assertEqual(
            loaded_model.graph.initializers["int2_tensor"].const_value.dtype,
            ir.DataType.INT2,
        )
        self.assertEqual(
            loaded_model.graph.initializers["uint2_tensor"].const_value.dtype,
            ir.DataType.UINT2,
        )

    def test_save_safetensors_float8_types(self):
        """Test that all float8 data types are saved correctly."""
        # Create tensors with float8 types using ml_dtypes
        f8_e5m2_data = np.array([1.0, 2.0, -1.5], dtype=ml_dtypes.float8_e5m2)
        tensor_f8_e5m2 = ir.tensor(
            f8_e5m2_data, dtype=ir.DataType.FLOAT8E5M2, name="f8_e5m2_tensor"
        )

        f8_e4m3fn_data = np.array([0.5, 1.0, -0.5], dtype=ml_dtypes.float8_e4m3fn)
        tensor_f8_e4m3fn = ir.tensor(
            f8_e4m3fn_data, dtype=ir.DataType.FLOAT8E4M3FN, name="f8_e4m3fn_tensor"
        )

        f8_e4m3fnuz_data = np.array([0.25, 0.75, -0.25], dtype=ml_dtypes.float8_e4m3fnuz)
        tensor_f8_e4m3fnuz = ir.tensor(
            f8_e4m3fnuz_data,
            dtype=ir.DataType.FLOAT8E4M3FNUZ,
            name="f8_e4m3fnuz_tensor",
        )

        f8_e5m2fnuz_data = np.array([1.5, 2.5, -1.5], dtype=ml_dtypes.float8_e5m2fnuz)
        tensor_f8_e5m2fnuz = ir.tensor(
            f8_e5m2fnuz_data,
            dtype=ir.DataType.FLOAT8E5M2FNUZ,
            name="f8_e5m2fnuz_tensor",
        )

        initializers = [
            _create_initializer(tensor_f8_e5m2),
            _create_initializer(tensor_f8_e4m3fn),
            _create_initializer(tensor_f8_e4m3fnuz),
            _create_initializer(tensor_f8_e5m2fnuz),
        ]

        identity_node = ir.Node("", "Identity", inputs=(initializers[0],))
        identity_node.outputs[0].shape = tensor_f8_e5m2.shape
        identity_node.outputs[0].dtype = ir.DataType.FLOAT8E5M2
        identity_node.outputs[0].name = "identity_0"

        graph = ir.Graph(
            inputs=initializers,
            outputs=[*identity_node.outputs],
            nodes=[identity_node],
            initializers=initializers,
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=10)

        path = os.path.join(self.tmpdir, "model.onnx")
        ir.save_safetensors(model, path, size_threshold_bytes=0)

        # Load the model back
        loaded_model = ir.load(path)

        # Verify all float8 types are correctly loaded
        for tensor_name in [
            "f8_e5m2_tensor",
            "f8_e4m3fn_tensor",
            "f8_e4m3fnuz_tensor",
            "f8_e5m2fnuz_tensor",
        ]:
            self.assertIsInstance(
                loaded_model.graph.initializers[tensor_name].const_value,
                ir.ExternalTensor,
            )

        # Check that the raw data is preserved for each type
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["f8_e5m2_tensor"].const_value.numpy(),
            f8_e5m2_data,
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["f8_e4m3fn_tensor"].const_value.numpy(),
            f8_e4m3fn_data,
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["f8_e4m3fnuz_tensor"].const_value.numpy(),
            f8_e4m3fnuz_data,
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["f8_e5m2fnuz_tensor"].const_value.numpy(),
            f8_e5m2fnuz_data,
        )

        # Check that the dtype is preserved for each type
        self.assertEqual(
            loaded_model.graph.initializers["f8_e5m2_tensor"].const_value.dtype,
            ir.DataType.FLOAT8E5M2,
        )
        self.assertEqual(
            loaded_model.graph.initializers["f8_e4m3fn_tensor"].const_value.dtype,
            ir.DataType.FLOAT8E4M3FN,
        )
        self.assertEqual(
            loaded_model.graph.initializers["f8_e4m3fnuz_tensor"].const_value.dtype,
            ir.DataType.FLOAT8E4M3FNUZ,
        )
        self.assertEqual(
            loaded_model.graph.initializers["f8_e5m2fnuz_tensor"].const_value.dtype,
            ir.DataType.FLOAT8E5M2FNUZ,
        )

    def test_save_safetensors_sharding_preserves_data(self):
        """Test that sharded tensors preserve data correctly."""
        model = _create_large_model_for_sharding()
        path = os.path.join(self.tmpdir, "model.onnx")

        # Create a copy of original data for verification
        original_data = {
            name: value.const_value.numpy().copy()
            for name, value in model.graph.initializers.items()
        }

        # Save with sharding
        ir.save_safetensors(model, path, size_threshold_bytes=0, max_shard_size_bytes=10000)

        # Load the model back
        loaded_model = ir.load(path)

        # Verify all data is preserved
        for name, value in loaded_model.graph.initializers.items():
            np.testing.assert_array_equal(value.const_value.numpy(), original_data[name])

    def test_save_safetensors_with_subgraphs(self):
        """Test that initializers in subgraphs are saved correctly."""
        # Create initializers for main graph
        main_tensor = ir.tensor([1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT, name="main_init")
        main_initializer = _create_initializer(main_tensor)

        # Create initializers for then branch
        then_tensor = ir.tensor([4.0, 5.0], dtype=ir.DataType.FLOAT, name="then_init")
        then_initializer = _create_initializer(then_tensor)

        # Create initializers for else branch
        else_tensor = ir.tensor([6.0, 7.0, 8.0], dtype=ir.DataType.FLOAT, name="else_init")
        else_initializer = _create_initializer(else_tensor)

        # Create then branch graph
        then_identity = ir.Node("", "Identity", inputs=(then_initializer,))
        then_identity.outputs[0].shape = ir.Shape([2])
        then_identity.outputs[0].dtype = ir.DataType.FLOAT
        then_identity.outputs[0].name = "then_output"
        then_graph = ir.Graph(
            inputs=[],
            outputs=[*then_identity.outputs],
            nodes=[then_identity],
            initializers=[then_initializer],
            name="then_graph",
        )

        # Create else branch graph
        else_identity = ir.Node("", "Identity", inputs=(else_initializer,))
        else_identity.outputs[0].shape = ir.Shape([3])
        else_identity.outputs[0].dtype = ir.DataType.FLOAT
        else_identity.outputs[0].name = "else_output"
        else_graph = ir.Graph(
            inputs=[],
            outputs=[*else_identity.outputs],
            nodes=[else_identity],
            initializers=[else_initializer],
            name="else_graph",
        )

        # Create condition value for If node
        cond_tensor = ir.tensor([True], dtype=ir.DataType.BOOL, name="condition")
        cond_value = _create_initializer(cond_tensor)

        # Create If node with subgraphs
        if_node = ir.Node(
            "",
            "If",
            inputs=(cond_value,),
            attributes=[
                ir.AttrGraph("then_branch", then_graph),
                ir.AttrGraph("else_branch", else_graph),
            ],
        )
        if_node.outputs[0].shape = ir.Shape([2])
        if_node.outputs[0].dtype = ir.DataType.FLOAT
        if_node.outputs[0].name = "if_output"

        # Create main graph
        graph = ir.Graph(
            inputs=[main_initializer, cond_value],
            outputs=[*if_node.outputs],
            nodes=[if_node],
            initializers=[main_initializer, cond_value],
            name="main_graph",
        )
        model = ir.Model(graph, ir_version=10)

        path = os.path.join(self.tmpdir, "model.onnx")
        ir.save_safetensors(model, path, size_threshold_bytes=0)

        # Load the model back
        loaded_model = ir.load(path)

        # Check main graph initializers
        self.assertIsInstance(
            loaded_model.graph.initializers["main_init"].const_value, ir.ExternalTensor
        )
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["main_init"].const_value.numpy(),
            main_tensor.numpy(),
        )

        # Check then branch initializers
        then_branch = loaded_model.graph.node(0).attributes["then_branch"].as_graph()
        self.assertIsInstance(
            then_branch.initializers["then_init"].const_value, ir.ExternalTensor
        )
        np.testing.assert_array_equal(
            then_branch.initializers["then_init"].const_value.numpy(),
            then_tensor.numpy(),
        )

        # Check else branch initializers
        else_branch = loaded_model.graph.node(0).attributes["else_branch"].as_graph()
        self.assertIsInstance(
            else_branch.initializers["else_init"].const_value, ir.ExternalTensor
        )
        np.testing.assert_array_equal(
            else_branch.initializers["else_init"].const_value.numpy(),
            else_tensor.numpy(),
        )


if __name__ == "__main__":
    unittest.main()
