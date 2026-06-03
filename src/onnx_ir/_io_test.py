# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the _io module."""

import os
import tempfile
import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir import _io


def _create_initializer(tensor: ir.TensorProtocol) -> ir.Value:
    return ir.Value(
        name=tensor.name,
        shape=tensor.shape,
        type=ir.TensorType(tensor.dtype),
        const_value=tensor,
    )


def _create_simple_model_with_initializers() -> ir.Model:
    tensor_0 = ir.tensor([0.0], dtype=ir.DataType.FLOAT, name="initializer_0")
    initializer = _create_initializer(tensor_0)
    tensor_1 = ir.tensor([1.0], dtype=ir.DataType.FLOAT)
    identity_node = ir.Node("", "Identity", inputs=(initializer,))
    identity_node.outputs[0].shape = ir.Shape([1])
    identity_node.outputs[0].dtype = ir.DataType.FLOAT
    identity_node.outputs[0].name = "identity_0"
    const_node = ir.Node(
        "",
        "Constant",
        inputs=(),
        outputs=(
            ir.Value(name="const_0", shape=tensor_1.shape, type=ir.TensorType(tensor_1.dtype)),
        ),
        attributes=ir.convenience.convert_attributes(dict(value=tensor_1)),
    )
    graph = ir.Graph(
        inputs=[initializer],
        outputs=[*identity_node.outputs, *const_node.outputs],
        nodes=[identity_node, const_node],
        initializers=[initializer],
        name="test_graph",
    )
    return ir.Model(graph, ir_version=10)


class IOFunctionsTest(unittest.TestCase):
    def test_load(self):
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(model, path)
            loaded_model = _io.load(path)
        self.assertEqual(loaded_model.ir_version, model.ir_version)
        self.assertEqual(loaded_model.graph.name, model.graph.name)
        self.assertEqual(len(loaded_model.graph.initializers), 1)
        self.assertEqual(len(loaded_model.graph), 2)
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["initializer_0"].const_value.numpy(),
            np.array([0.0]),
        )
        np.testing.assert_array_equal(
            loaded_model.graph.node(1).attributes["value"].as_tensor().numpy(), np.array([1.0])
        )
        self.assertEqual(loaded_model.graph.inputs[0].name, "initializer_0")
        self.assertEqual(loaded_model.graph.outputs[0].name, "identity_0")
        self.assertEqual(loaded_model.graph.outputs[1].name, "const_0")

    def test_save_with_external_data_does_not_modify_model(self):
        model = _create_simple_model_with_initializers()
        self.assertIsInstance(model.graph.initializers["initializer_0"].const_value, ir.Tensor)
        # There may be clean up errors on Windows, so we ignore them
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = "model.data"
            _io.save(model, path, external_data=external_data_file, size_threshold_bytes=0)
            self.assertTrue(os.path.exists(path))
            external_data_path = os.path.join(tmpdir, external_data_file)
            self.assertTrue(os.path.exists(external_data_path))
            loaded_model = _io.load(path)

            # The loaded model contains external data
            initializer_tensor = loaded_model.graph.initializers["initializer_0"].const_value
            self.assertIsInstance(initializer_tensor, ir.ExternalTensor)
            # The attribute is not externalized
            const_attr_tensor = loaded_model.graph.node(1).attributes["value"].as_tensor()
            self.assertIsInstance(const_attr_tensor, ir.TensorProtoTensor)
            np.testing.assert_array_equal(initializer_tensor.numpy(), np.array([0.0]))
            np.testing.assert_array_equal(const_attr_tensor.numpy(), np.array([1.0]))

        # The original model is not changed and can be accessed even if the
        # external data file is deleted
        initializer_tensor = model.graph.initializers["initializer_0"].const_value
        self.assertIsInstance(initializer_tensor, ir.Tensor)
        const_attr_tensor = model.graph.node(1).attributes["value"].as_tensor()
        self.assertIsInstance(const_attr_tensor, ir.Tensor)
        np.testing.assert_array_equal(initializer_tensor.numpy(), np.array([0.0]))
        np.testing.assert_array_equal(const_attr_tensor.numpy(), np.array([1.0]))

    def test_save_with_sharding_creates_multiple_shard_files(self):
        """Test that max_shard_size_bytes creates multiple numbered shard files."""

        # Build a model with 3 tensors, each ~400 bytes (100 float32 elements)
        def make_init(name: str, n: int) -> ir.Value:
            arr = np.zeros(n, dtype=np.float32)
            t = ir.tensor(arr, dtype=ir.DataType.FLOAT, name=name)
            return ir.Value(
                name=name, const_value=t, shape=t.shape, type=ir.TensorType(t.dtype)
            )

        inits = [make_init(f"w{i}", 100) for i in range(3)]
        node = ir.Node("", "Identity", inputs=(inits[0],))
        node.outputs[0].name = "out"
        node.outputs[0].dtype = ir.DataType.FLOAT
        graph = ir.Graph(
            inputs=inits,
            outputs=list(node.outputs),
            nodes=[node],
            initializers=inits,
            name="g",
        )
        model = ir.Model(graph, ir_version=10)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            # 400-byte limit forces each 400-byte tensor into its own shard
            _io.save(
                model,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=400,
            )

            self.assertTrue(os.path.exists(path))
            shard_files = sorted(
                f for f in os.listdir(tmpdir) if f.startswith("model-") and f.endswith(".data")
            )
            self.assertGreater(len(shard_files), 1, "Expected multiple shard files")

            # Load back and verify data integrity
            loaded = _io.load(path)
            for i, init in enumerate(inits):
                loaded_tensor = loaded.graph.initializers[f"w{i}"].const_value
                self.assertIsInstance(loaded_tensor, ir.ExternalTensor)
                np.testing.assert_array_equal(loaded_tensor.numpy(), init.const_value.numpy())

        # Original model must be unchanged (in-memory tensors)
        for init in inits:
            self.assertIsInstance(init.const_value, ir.Tensor)
            self.assertNotIsInstance(init.const_value, ir.ExternalTensor)

    def test_save_with_sharding_single_shard_uses_base_name(self):
        """When tensors fit in one shard the file keeps its original name."""
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(
                model,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=10_000_000,
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "model.data")))
            shard_files = [f for f in os.listdir(tmpdir) if "of-" in f]
            self.assertEqual(shard_files, [], "Should not create numbered shard files")

    def test_save_raise_when_external_data_is_not_relative_path(self):
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = os.path.join(tmpdir, "model.data")
            with self.assertRaises(ValueError):
                _io.save(model, path, external_data=external_data_file)

    def test_save_raise_when_max_shard_size_bytes_is_not_positive(self):
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            with self.assertRaisesRegex(
                ValueError, "max_shard_size_bytes must be greater than 0"
            ):
                _io.save(
                    model,
                    path,
                    external_data="model.data",
                    size_threshold_bytes=0,
                    max_shard_size_bytes=0,
                )

    def test_save_raise_when_max_shard_size_bytes_without_external_data(self):
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            with self.assertRaisesRegex(
                ValueError, "max_shard_size_bytes can only be used together with external_data"
            ):
                _io.save(model, path, max_shard_size_bytes=1024)

    def test_save_sharded_raises_on_foreign_shard_collision(self):
        # The sharded write path refuses to silently clobber existing shard
        # files that aren't already part of the model being saved. This is
        # what enforces "the user knows whose files live in this directory"
        # without bringing back the (very error-prone) stem-based cleanup.
        # 2 initializers + max_shard_size_bytes=200 forces a 2-shard layout
        # whose first shard collides with the foreign file below.
        model, _arrs = self._make_sharded_test_model([100, 100])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            foreign = os.path.join(tmpdir, "model-00001-of-00002.data")
            with open(foreign, "wb") as f:
                f.write(b"foreign")
            path = os.path.join(tmpdir, "model.onnx")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _io.save(
                    model,
                    path,
                    external_data="model.data",
                    size_threshold_bytes=0,
                    max_shard_size_bytes=200,
                )
            # The foreign file must remain untouched after the raise.
            with open(foreign, "rb") as f:
                self.assertEqual(f.read(), b"foreign")

    def test_save_sharded_does_not_raise_on_unrelated_shard_files(self):
        # An unrelated model's shards sitting in the same directory but
        # with *different* filenames must not block our save.
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            unrelated = [
                os.path.join(tmpdir, f"other-{i:05d}-of-00009.data") for i in range(1, 10)
            ]
            for p in unrelated:
                with open(p, "wb") as f:
                    f.write(b"unrelated")
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(
                model,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=10_000_000,
            )
            for p in unrelated:
                self.assertTrue(os.path.exists(p))

    def test_save_loaded_sharded_model_with_different_limit_preserves_data(self):
        def make_init(name: str, n: int) -> tuple[ir.Value, np.ndarray]:
            arr = np.arange(n, dtype=np.float32)
            t = ir.tensor(arr, dtype=ir.DataType.FLOAT, name=name)
            value = ir.Value(
                name=name, const_value=t, shape=t.shape, type=ir.TensorType(t.dtype)
            )
            return value, arr

        init_0, arr_0 = make_init("w0", 150)  # 600 bytes
        init_1, arr_1 = make_init("w1", 100)  # 400 bytes
        init_2, arr_2 = make_init("w2", 100)  # 400 bytes
        inits = [init_0, init_1, init_2]
        node = ir.Node("", "Identity", inputs=(inits[0],))
        node.outputs[0].name = "out"
        node.outputs[0].dtype = ir.DataType.FLOAT
        graph = ir.Graph(
            inputs=inits,
            outputs=list(node.outputs),
            nodes=[node],
            initializers=inits,
            name="g",
        )
        model = ir.Model(graph, ir_version=10)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(
                model,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=1000,
            )
            loaded = _io.load(path)
            _io.save(
                loaded,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=800,
            )
            reloaded = _io.load(path)
            np.testing.assert_array_equal(
                reloaded.graph.initializers["w0"].const_value.numpy(), arr_0
            )
            np.testing.assert_array_equal(
                reloaded.graph.initializers["w1"].const_value.numpy(), arr_1
            )
            np.testing.assert_array_equal(
                reloaded.graph.initializers["w2"].const_value.numpy(), arr_2
            )

    def _make_sharded_test_model(self, sizes):
        """Build a tiny model with one initializer per ``sizes`` element."""

        def make_init(name, n):
            arr = np.arange(n, dtype=np.float32)
            t = ir.tensor(arr, dtype=ir.DataType.FLOAT, name=name)
            return (
                ir.Value(name=name, const_value=t, shape=t.shape, type=ir.TensorType(t.dtype)),
                arr,
            )

        pairs = [make_init(f"w{i}", n) for i, n in enumerate(sizes)]
        inits = [v for v, _ in pairs]
        arrs = [a for _, a in pairs]
        node = ir.Node("", "Identity", inputs=(inits[0],))
        node.outputs[0].name = "out"
        node.outputs[0].dtype = ir.DataType.FLOAT
        graph = ir.Graph(
            inputs=inits,
            outputs=list(node.outputs),
            nodes=[node],
            initializers=inits,
            name="g",
        )
        return ir.Model(graph, ir_version=10), arrs

    def test_resave_loaded_model_with_changed_shard_count_preserves_data(self):
        # Round-2 regression: re-saving a loaded sharded model with a *different*
        # shard count used to crash with FileNotFoundError because the old
        # shards (under different names) got deleted before being read. With
        # the simplified contract there is no automatic cleanup at all, so the
        # old shard files remain on disk as garbage — but the new save still
        # succeeds and the data round-trips cleanly via ir.load.
        model, arrs = self._make_sharded_test_model([125, 125, 125, 125])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(
                model,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=500,
            )
            loaded = _io.load(path)
            _io.save(
                loaded,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=1500,
            )
            reloaded = _io.load(path)
            for i, expected in enumerate(arrs):
                np.testing.assert_array_equal(
                    reloaded.graph.initializers[f"w{i}"].const_value.numpy(), expected
                )

    def test_resave_same_shard_count_with_boundary_shift_preserves_data(self):
        # Round-4 regression (@titaiwangms): when the new shard layout has the
        # *same* shard count but a tensor migrates from shard i to shard j > i,
        # a per-shard materialization pass would still leave that tensor
        # pointing at file i while we are writing shard j. By the time shard i
        # is rewritten and truncated, the migrated tensor's data is gone. The
        # fix is a global pre-materialization pass across all destination
        # paths before any shard is opened for writing.
        model, _arrs = self._make_sharded_test_model([75, 75, 75, 125])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            # First save: layout is [w0,w1,w2][w3] (2 shards under 900 bytes).
            _io.save(
                model,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=900,
            )
            loaded = _io.load(path)
            # Enlarge w1 so the shard boundary shifts: new layout is
            # [w0,w1][w2,w3] — still 2 shards, but w2 migrates from shard 1
            # to shard 2.
            big = np.arange(150, dtype=np.float32) + 10000
            loaded.graph.initializers["w1"].const_value = ir.tensor(big, name="w1")
            _io.save(
                loaded,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=900,
            )
            reloaded = _io.load(path)
            # w2's original data must survive even though its shard changed.
            np.testing.assert_array_equal(
                reloaded.graph.initializers["w2"].const_value.numpy(),
                np.arange(75, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                reloaded.graph.initializers["w0"].const_value.numpy(),
                np.arange(75, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                reloaded.graph.initializers["w1"].const_value.numpy(), big
            )
            np.testing.assert_array_equal(
                reloaded.graph.initializers["w3"].const_value.numpy(),
                np.arange(125, dtype=np.float32),
            )

    def test_resave_loaded_sharded_model_as_single_file_preserves_data(self):
        # sharded -> non-sharded (max_shard_size_bytes=None) transition. The
        # single-file write path is permissive and just overwrites model.data;
        # old shard files remain (cleanup is the caller's responsibility), but
        # the round-tripped data must be correct.
        model, arrs = self._make_sharded_test_model([100, 100, 100])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(
                model,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=500,
            )
            loaded = _io.load(path)
            _io.save(
                loaded,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "model.data")))
            reloaded = _io.load(path)
            for i, expected in enumerate(arrs):
                np.testing.assert_array_equal(
                    reloaded.graph.initializers[f"w{i}"].const_value.numpy(), expected
                )

    def test_save_does_not_delete_unrelated_models_shards(self):
        # Regression test for cross-model deletion: an unrelated model's
        # ``model-NNNNN-of-NNNNN.data`` files sitting in the same directory
        # must survive a save of an unrelated model under the same base name
        # *as long as the filenames don't collide*. (A direct collision is now
        # a FileExistsError, exercised by test_save_sharded_raises_on_foreign_shard_collision.)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            unrelated = [
                os.path.join(tmpdir, f"model-{i:05d}-of-00009.data") for i in range(1, 10)
            ]
            for p in unrelated:
                with open(p, "wb") as f:
                    f.write(b"unrelated")
            model, _ = self._make_sharded_test_model([200, 200])
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(
                model,
                path,
                external_data="model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=500,
            )
            for p in unrelated:
                self.assertTrue(os.path.exists(p), f"unrelated shard {p} was deleted")

    def test_save_with_external_data_invalidates_obsolete_external_tensors(self):
        model = _create_simple_model_with_initializers()
        self.assertIsInstance(model.graph.initializers["initializer_0"].const_value, ir.Tensor)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = "model.data"
            _io.save(model, path, external_data=external_data_file, size_threshold_bytes=0)
            loaded_model = _io.load(path)
            # Now if we load the model back, create a different initializer and save
            # the model to the same external data file, the existing external tensor
            # should be invalidated
            tensor_2 = ir.tensor([2.0], dtype=ir.DataType.FLOAT, name="initializer_2")
            initializer_2 = _create_initializer(tensor_2)
            loaded_model.graph.initializers["initializer_2"] = initializer_2
            _io.save(
                loaded_model, path, external_data=external_data_file, size_threshold_bytes=0
            )
            initializer_0_tensor = loaded_model.graph.initializers["initializer_0"].const_value
            self.assertIsInstance(initializer_0_tensor, ir.ExternalTensor)
            self.assertFalse(initializer_0_tensor.valid())
            with self.assertRaisesRegex(ValueError, "is invalidated"):
                # The existing model has to be modified to use in memory tensors
                # for the values to stay correct. Saving again should raise an error
                _io.save(
                    loaded_model,
                    path,
                    external_data=external_data_file,
                    size_threshold_bytes=0,
                )


if __name__ == "__main__":
    unittest.main()
