# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile
import typing
import unittest

import numpy as np
import onnx
import onnx.external_data_helper

import onnx_ir as ir
from onnx_ir import external_data


class ExternalDataTest(unittest.TestCase):
    def test_set_base_dir_sets_base_dir_for_all_external_tensors(self):
        attr_tensor = onnx.helper.make_tensor(
            name="test_constant",
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=b"\x01\x00\x00\x00",
            raw=True,
        )
        graph = onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node(
                    "Constant",
                    [],
                    ["test"],
                    value=attr_tensor,
                )
            ],
            name="test",
            inputs=[],
            outputs=[],
            initializer=[
                onnx.helper.make_tensor(
                    name="test_tensor",
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[1],
                    vals=b"\x01\x00\x00\x00",
                    raw=True,
                ),
            ],
        )
        model_proto = onnx.helper.make_model(graph)
        onnx.external_data_helper.convert_model_to_external_data(
            model_proto, location="tempdir", size_threshold=0, convert_attribute=True
        )
        model = ir.serde.deserialize_model(model_proto)
        expected_dir = "something_else"
        external_data.set_base_dir(model.graph, expected_dir)

        initializer_tensor = model.graph.initializers["test_tensor"].const_value
        assert isinstance(initializer_tensor, ir.ExternalTensor)
        self.assertEqual(initializer_tensor.base_dir, expected_dir)
        attr_tensor = model.graph.node(0).attributes["value"].value
        self.assertEqual(attr_tensor.base_dir, expected_dir)


class OffsetCalcTest(unittest.TestCase):
    """Test the offset calculation for the external tensor class."""

    def test_align_offset_false(self):
        # Tensor size > Align Threshold
        current_offset = 20000
        tensor_size = 1048
        new_offset = external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset, tensor_size, align_offset=False
        )
        self.assertEqual(current_offset, new_offset)

    def test_align_with_small_align_threshold(self):
        # Tensor size < Align Threshold
        current_offset = 20000
        tensor_size = 1048
        new_offset = external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset,
            tensor_size,
            align_threshold=1000,
        )
        self.assertNotEqual(current_offset, new_offset)

    def test_align_with_large_align_threshold(self):
        # Tensor size > Align Threshold
        current_offset = 20000
        tensor_size = 1048
        new_offset = external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset,
            tensor_size,
        )
        self.assertEqual(current_offset, new_offset)

    def test_allocation_granularity_diff(self):
        # Tensor size > Align Threshold
        current_offset = 20000
        tensor_size = 1048577
        new_offset_1 = external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset,
            tensor_size,
            allocation_granularity=4000,
        )
        new_offset_2 = external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset,
            tensor_size,
        )
        self.assertNotEqual(current_offset, new_offset_1)
        self.assertNotEqual(current_offset, new_offset_2)
        self.assertNotEqual(new_offset_1, new_offset_2)


class OffloadExternalTensorTest(unittest.TestCase):
    """Test the memory mapped external tensor class."""

    def setUp(self):
        # File paths
        if sys.version_info[:2] >= (3, 10):
            self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # pylint: disable=consider-using-with
        else:
            self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.external_data_name = "external_tensors.bin"
        self.base_path = self.temp_dir.name
        self.ext_data_1 = "external_data_1.bin"
        self.ext_data_2 = "external_data_2.bin"
        # Data for the tensors
        self.data = np.random.rand(2, 42).astype(np.float32)
        self.data_other = np.random.rand(2, 42).astype(np.float32)
        self.data_float16 = np.random.rand(2, 42).astype(np.float16)
        self.data_ext1_1 = np.random.rand(1, 42).astype(np.float32)
        self.data_ext1_2 = np.random.rand(4, 42).astype(np.float16)
        self.data_ext2_1 = np.random.rand(5, 42).astype(np.float16)
        self.custom_data = np.random.rand(3, 42).astype(np.float32)
        # Model Assignments
        self.model = self._simple_model()
        self.model_with_external_data_same_path = self._model_with_external_data_same_path()
        self.model_with_external_data_diff_path = self._model_with_external_data_diff_path()
        self.model_with_custom_tensor_class = self._model_with_custom_tensor_class()
        self.model_with_mixed_external_data = self._model_with_mixed_external_data()

    def tearDown(self) -> None:
        # Handle exceptions for windows and python versions < 3.10
        try:
            self.temp_dir.cleanup()
        except PermissionError as e:
            print(f"PermissionError: {e}")
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"An unexpected error occurred: {e}")

    def _simple_model(self) -> ir.Model:
        tensor1 = ir.Tensor(
            self.data,
            dtype=ir.DataType.FLOAT,
            shape=ir.Shape(self.data.shape),
            name="tensor1",
        )
        tensor2 = ir.Tensor(
            self.data_float16,
            dtype=ir.DataType.FLOAT16,
            shape=ir.Shape(self.data_float16.shape),
            name="tensor2",
        )
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
            initializers=[
                ir.Value(name="tensor1", const_value=tensor1),
                ir.Value(name="tensor2", const_value=tensor2),
            ],
            # Unsorted nodes
            nodes=[node_1, node_0],
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=8)
        return model

    def _setup_custom_tensor_class(self, name, value):
        class CustomTensorType(ir.TensorProtocol):
            def __init__(
                self,
                value: np.ndarray,
            ):
                self.name = name
                self._raw = value
                if isinstance(value, np.ndarray):
                    self._dtype = ir._enums.DataType.from_numpy(value.dtype)
                self._shape = ir.Shape(getattr(value, "shape"), frozen=True)  # noqa: B009

            @property
            def dtype(self) -> ir._enums.DataType:
                """The data type of the tensor. Immutable."""
                return self._dtype

            @property
            def shape(self) -> ir.Shape:
                """The shape of the tensor. Immutable."""
                return self._shape

            @property
            def nbytes(self) -> int:
                return len(self.tobytes())

            def __array__(self, dtype: typing.Any = None) -> np.ndarray:
                if isinstance(self._raw, np.ndarray):
                    return self._raw
                else:
                    return TypeError

            def numpy(self) -> np.ndarray:
                return self._raw

            def tobytes(self) -> bytes:
                if isinstance(self._raw, np.ndarray):
                    return self._raw.tobytes()
                else:
                    return TypeError

        return CustomTensorType(value)

    def _model_with_external_data_same_path(self) -> ir.Model:
        model = self._simple_model()
        raw_data = self.data_other.tobytes()
        # Save the data to disk
        file_path = os.path.join(self.base_path, self.external_data_name)
        with open(file_path, "wb") as f:
            f.write(raw_data)
        tensor_same_file = ir.ExternalTensor(
            location=self.external_data_name,
            offset=0,
            length=len(raw_data),
            dtype=ir.DataType.FLOAT,
            name="tensor_same_file",
            shape=ir.Shape(self.data_other.shape),
            base_dir=self.base_path,
        )
        model.graph.initializers["tensor_same_file"] = ir.Value(
            name="tensor_same_file", const_value=tensor_same_file
        )
        return model

    def _model_with_external_data_diff_path(self) -> ir.Model:
        model = self._simple_model()
        # File 1
        file_path_1 = os.path.join(self.base_path, self.ext_data_1)
        with open(file_path_1, "wb") as f:
            f.write(self.data_ext1_1.tobytes())
            f.write(self.data_ext1_2.tobytes())
        tensor_ext1_1 = ir.ExternalTensor(
            location=self.ext_data_1,
            offset=0,
            length=len(self.data_ext1_1.tobytes()),
            dtype=ir.DataType.FLOAT,
            name="tensor_ext1_1",
            shape=ir.Shape(self.data_ext1_1.shape),
            base_dir=self.base_path,
        )
        tensor_ext1_2 = ir.ExternalTensor(
            location=self.ext_data_1,
            offset=len(self.data_ext1_1.tobytes()),
            length=len(self.data_ext1_2.tobytes()),
            dtype=ir.DataType.FLOAT16,
            name="tensor_ext1_2",
            shape=ir.Shape(self.data_ext1_2.shape),
            base_dir=self.base_path,
        )
        # File 2
        file_path_2 = os.path.join(self.base_path, self.ext_data_2)
        with open(file_path_2, "wb") as f:
            f.write(self.data_ext2_1.tobytes())
        tensor_ext2_1 = ir.ExternalTensor(
            location=self.ext_data_2,
            offset=0,
            length=len(self.data_ext2_1.tobytes()),
            dtype=ir.DataType.FLOAT16,
            name="tensor_ext2_1",
            shape=ir.Shape(self.data_ext2_1.shape),
            base_dir=self.base_path,
        )
        model.graph.initializers["tensor_ext1_1"] = ir.Value(
            name="tensor_ext1_1", const_value=tensor_ext1_1
        )
        model.graph.initializers["tensor_ext1_2"] = ir.Value(
            name="tensor_ext1_2", const_value=tensor_ext1_2
        )
        model.graph.initializers["tensor_ext2_1"] = ir.Value(
            name="tensor_ext2_1", const_value=tensor_ext2_1
        )
        return model

    def _model_with_custom_tensor_class(self) -> ir.Model:
        model = self._simple_model()
        custom_tensor = self._setup_custom_tensor_class("custom_tensor", self.custom_data)
        model.graph.initializers["custom_tensor"] = ir.Value(
            name="custom_tensor", const_value=custom_tensor
        )
        return model

    def _model_with_mixed_external_data(self) -> ir.Model:
        model = self._simple_model()
        model_same_path = self.model_with_external_data_same_path
        model_diff_path = self.model_with_external_data_diff_path
        model_custom_tensor = self.model_with_custom_tensor_class
        model.graph.initializers["tensor_same_file"] = ir.Value(
            name="tensor_same_file",
            const_value=model_same_path.graph.initializers["tensor_same_file"].const_value,
        )
        model.graph.initializers["tensor_ext1_1"] = ir.Value(
            name="tensor_ext1_1",
            const_value=model_diff_path.graph.initializers["tensor_ext1_1"].const_value,
        )
        model.graph.initializers["tensor_ext1_2"] = ir.Value(
            name="tensor_ext1_2",
            const_value=model_diff_path.graph.initializers["tensor_ext1_2"].const_value,
        )
        model.graph.initializers["tensor_ext2_1"] = ir.Value(
            name="tensor_ext2_1",
            const_value=model_diff_path.graph.initializers["tensor_ext2_1"].const_value,
        )
        model.graph.initializers["custom_tensor"] = ir.Value(
            name="custom_tensor",
            const_value=model_custom_tensor.graph.initializers["custom_tensor"].const_value,
        )
        return model

    def test_external_data_simple(self):
        model_with_external_data = external_data.unload_from_model(
            self.model, self.base_path, self.external_data_name
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())

    def test_same_path_external_data(self):
        model_with_external_data = external_data.unload_from_model(
            self.model_with_external_data_same_path,
            self.base_path,
            self.external_data_name,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "tensor_same_file"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())

    def test_external_data_diff_paths(self):
        model_with_external_data = external_data.unload_from_model(
            self.model_with_external_data_diff_path,
            self.base_path,
            self.external_data_name,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "tensor_ext1_1"
        ].const_value
        external_tensor4 = model_with_external_data.graph.initializers[
            "tensor_ext1_2"
        ].const_value
        external_tensor5 = model_with_external_data.graph.initializers[
            "tensor_ext2_1"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext2_1.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext2_1.tobytes())

    def test_custom_tensor_in_initializers(self):
        model_with_external_data = external_data.unload_from_model(
            self.model_with_custom_tensor_class,
            self.base_path,
            self.external_data_name,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "custom_tensor"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.custom_data.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.custom_data.tobytes())

    def test_mixed_external_data(self):
        model_with_external_data = external_data.unload_from_model(
            self.model_with_mixed_external_data, self.base_path, self.external_data_name
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "tensor_same_file"
        ].const_value
        external_tensor4 = model_with_external_data.graph.initializers[
            "custom_tensor"
        ].const_value
        external_tensor5 = model_with_external_data.graph.initializers[
            "tensor_ext1_1"
        ].const_value
        external_tensor6 = model_with_external_data.graph.initializers[
            "tensor_ext1_2"
        ].const_value
        external_tensor7 = model_with_external_data.graph.initializers[
            "tensor_ext2_1"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.custom_data.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor6.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor7.numpy().tobytes(), self.data_ext2_1.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.custom_data.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor6.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor7.numpy().tobytes(), self.data_ext2_1.tobytes())

    def test_external_data_sorted(self):
        model_with_external_data = external_data.unload_from_model(
            self.model_with_mixed_external_data,
            self.base_path,
            self.external_data_name,
        )
        file_path = os.path.join(self.base_path, self.external_data_name)
        expected_tensor_order = [
            model_with_external_data.graph.initializers["tensor2"].const_value.tobytes(),
            model_with_external_data.graph.initializers["tensor_ext1_1"].const_value.tobytes(),
            model_with_external_data.graph.initializers["tensor1"].const_value.tobytes(),
            model_with_external_data.graph.initializers[
                "tensor_same_file"
            ].const_value.tobytes(),
            model_with_external_data.graph.initializers["tensor_ext1_2"].const_value.tobytes(),
            model_with_external_data.graph.initializers["tensor_ext2_1"].const_value.tobytes(),
            model_with_external_data.graph.initializers["custom_tensor"].const_value.tobytes(),
        ]
        sorted_tensor_order = [
            self.data_float16.tobytes(),
            self.data_ext1_1.tobytes(),
            self.data.tobytes(),
            self.data_other.tobytes(),
            self.data_ext1_2.tobytes(),
            self.data_ext2_1.tobytes(),
            self.custom_data.tobytes(),
        ]
        with open(file_path, "r+b") as data_file:
            current_offset = 0
            for i, tensor_bytes in enumerate(sorted_tensor_order):
                data_file.seek(current_offset)
                tensor_length = len(tensor_bytes)
                tensor_data = data_file.read(tensor_length)
                current_offset += tensor_length
                self.assertEqual(tensor_data, tensor_bytes)
                self.assertEqual(tensor_data, expected_tensor_order[i])


class ShardFilenameTest(unittest.TestCase):
    """Test the shard filename generation helper."""

    def test_single_shard_returns_original_name(self):
        self.assertEqual(external_data._get_shard_filename("model.data", 1, 1), "model.data")

    def test_multiple_shards_generates_numbered_filename(self):
        self.assertEqual(
            external_data._get_shard_filename("model.data", 1, 3),
            "model-00001-of-00003.data",
        )
        self.assertEqual(
            external_data._get_shard_filename("model.data", 2, 3),
            "model-00002-of-00003.data",
        )
        self.assertEqual(
            external_data._get_shard_filename("model.data", 3, 3),
            "model-00003-of-00003.data",
        )

    def test_filename_without_extension(self):
        self.assertEqual(
            external_data._get_shard_filename("model", 2, 5),
            "model-00002-of-00005",
        )

    def test_filename_with_dotted_directory_and_no_extension(self):
        self.assertEqual(
            external_data._get_shard_filename("my.dir/model", 2, 5),
            os.path.join("my.dir", "model-00002-of-00005"),
        )

    def test_five_digit_padding(self):
        result = external_data._get_shard_filename("weights.bin", 42, 100)
        self.assertEqual(result, "weights-00042-of-00100.bin")


class ShardTensorsTest(unittest.TestCase):
    """Test the tensor sharding helper."""

    def _make_tensor(self, name: str, nbytes: int) -> ir.TensorProtocol:
        """Create a float32 tensor with the requested byte size."""
        n_floats = max(1, nbytes // 4)
        data = np.zeros(n_floats, dtype=np.float32)
        return ir.Tensor(data, dtype=ir.DataType.FLOAT, name=name)

    def test_no_tensors(self):
        shards = external_data._shard_tensors([], 1000)
        self.assertEqual(shards, [[]])

    def test_single_tensor_below_limit(self):
        t = self._make_tensor("t0", 400)
        shards = external_data._shard_tensors([t], 1000)
        self.assertEqual(len(shards), 1)
        self.assertIs(shards[0][0], t)

    def test_tensors_fit_in_one_shard(self):
        tensors = [self._make_tensor(f"t{i}", 200) for i in range(4)]
        shards = external_data._shard_tensors(tensors, 1000)
        self.assertEqual(len(shards), 1)
        self.assertEqual(len(shards[0]), 4)

    def test_tensors_split_into_multiple_shards(self):
        tensors = [self._make_tensor(f"t{i}", 400) for i in range(5)]
        # limit = 800: shards of 2, 2, 1
        shards = external_data._shard_tensors(tensors, 800)
        self.assertEqual(len(shards), 3)
        self.assertEqual([len(s) for s in shards], [2, 2, 1])

    def test_tensor_larger_than_limit_gets_its_own_shard(self):
        t_big = self._make_tensor("big", 2000)
        t_small = self._make_tensor("small", 100)
        with self.assertLogs(external_data.logger, level="WARNING") as logs:
            shards = external_data._shard_tensors([t_big, t_small], 500)
        self.assertEqual(len(shards), 2)
        self.assertIs(shards[0][0], t_big)
        self.assertIs(shards[1][0], t_small)
        self.assertRegex(logs.output[0], r"exceeds max_shard_size_bytes")

    def test_sharding_accounts_for_alignment(self):
        t0 = self._make_tensor("t0", external_data._ALIGN_THRESHOLD + 4)
        t1 = self._make_tensor("t1", external_data._ALIGN_THRESHOLD + 4)
        # Naively this would fit in one shard by nbytes sum, but alignment padding
        # when writing forces a split.
        shards = external_data._shard_tensors([t0, t1], t0.nbytes + t1.nbytes)
        self.assertEqual(len(shards), 2)
        self.assertEqual([len(s) for s in shards], [1, 1])


class ShardedExternalDataTest(unittest.TestCase):
    """Integration tests for sharded ONNX external data via unload_from_model."""

    def setUp(self):
        if sys.version_info[:2] >= (3, 10):
            self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = self.temp_dir.name

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except (PermissionError, FileNotFoundError) as e:
            print(f"Cleanup error: {e}")

    def _make_model(self, sizes: list[int]) -> tuple[ir.Model, list[np.ndarray]]:
        """Build a simple model with float32 initializers of the given byte sizes."""
        arrays = [np.random.rand(max(1, s // 4)).astype(np.float32) for s in sizes]
        initializers = []
        for i, arr in enumerate(arrays):
            t = ir.Tensor(arr, dtype=ir.DataType.FLOAT, name=f"w{i}")
            v = ir.Value(name=f"w{i}", const_value=t)
            initializers.append(v)

        node = ir.Node("", "Identity", inputs=(initializers[0],))
        node.outputs[0].name = "out"
        node.outputs[0].dtype = ir.DataType.FLOAT

        graph = ir.Graph(
            inputs=initializers,
            outputs=list(node.outputs),
            nodes=[node],
            initializers=initializers,
            name="g",
        )
        return ir.Model(graph, ir_version=10), arrays

    def test_sharding_creates_multiple_files(self):
        model, _ = self._make_model([400, 400, 400])
        # max_shard=500 bytes forces a new shard after each ~400-byte tensor
        external_data.unload_from_model(
            model,
            self.base_path,
            "model.data",
            size_threshold_bytes=0,
            max_shard_size_bytes=500,
        )
        shard_files = sorted(f for f in os.listdir(self.base_path) if f.startswith("model-"))
        self.assertGreater(len(shard_files), 1, "Expected multiple shard files")
        # Check that each initializer points to a shard file
        for value in model.graph.initializers.values():
            t = value.const_value
            self.assertIsInstance(t, ir.ExternalTensor)
            self.assertIn("-of-", t.location)

    def test_sharding_data_is_correct(self):
        model, arrays = self._make_model([400, 800, 400, 800])
        external_data.unload_from_model(
            model,
            self.base_path,
            "model.data",
            size_threshold_bytes=0,
            max_shard_size_bytes=1000,
        )
        for i, arr in enumerate(arrays):
            ext = model.graph.initializers[f"w{i}"].const_value
            np.testing.assert_array_equal(ext.numpy(), arr)

    def test_no_sharding_when_limit_not_set(self):
        model, _ = self._make_model([400, 400, 400])
        external_data.unload_from_model(
            model,
            self.base_path,
            "model.data",
            size_threshold_bytes=0,
        )
        for value in model.graph.initializers.values():
            t = value.const_value
            self.assertIsInstance(t, ir.ExternalTensor)
            self.assertEqual(t.location, "model.data")

    def test_single_shard_uses_original_filename(self):
        # When all tensors fit in one shard the file should keep its original name
        model, _ = self._make_model([100, 100])
        external_data.unload_from_model(
            model,
            self.base_path,
            "model.data",
            size_threshold_bytes=0,
            max_shard_size_bytes=10_000,
        )
        for value in model.graph.initializers.values():
            t = value.const_value
            self.assertIsInstance(t, ir.ExternalTensor)
            self.assertEqual(t.location, "model.data")

    def test_sharding_limit_must_be_positive(self):
        model, _ = self._make_model([100, 100])
        with self.assertRaisesRegex(ValueError, "max_shard_size_bytes must be greater than 0"):
            external_data.unload_from_model(
                model,
                self.base_path,
                "model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=0,
            )

    def test_model_unchanged_after_unload_and_load(self):
        model, _ = self._make_model([400, 400, 400])
        # Store originals before mutating model
        originals = {
            name: val.const_value.numpy().copy()
            for name, val in model.graph.initializers.items()
        }
        external_data.unload_from_model(
            model,
            self.base_path,
            "model.data",
            size_threshold_bytes=0,
            max_shard_size_bytes=500,
        )
        for name, orig in originals.items():
            np.testing.assert_array_equal(
                model.graph.initializers[name].const_value.numpy(), orig
            )

    def test_callback_receives_global_indices_and_total(self):
        model, _ = self._make_model([400, 400, 400])
        infos: list[external_data.CallbackInfo] = []

        def cb(tensor: ir.TensorProtocol, info: external_data.CallbackInfo) -> None:
            infos.append(info)

        external_data.unload_from_model(
            model,
            self.base_path,
            "model.data",
            size_threshold_bytes=0,
            max_shard_size_bytes=500,
            callback=cb,
        )
        self.assertEqual(len(infos), 3)
        # total should always equal the total number of tensors (3)
        self.assertTrue(all(i.total == 3 for i in infos))
        # indices should be 0, 1, 2 across all shards
        self.assertEqual(sorted(i.index for i in infos), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
