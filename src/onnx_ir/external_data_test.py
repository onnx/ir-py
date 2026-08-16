# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile
import threading
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


class AlignmentTest(unittest.TestCase):
    """Test the external data offset alignment policy."""

    def test_dense_packing_is_the_default(self):
        # No alignment object means offsets are never advanced.
        large = external_data._DEFAULT_ALIGN_THRESHOLD + 1
        tensors = [
            ir.Tensor(np.zeros(large, dtype=np.uint8), name="a"),
            ir.Tensor(np.zeros(large, dtype=np.uint8), name="b"),
        ]
        self.assertEqual(external_data._estimate_shard_size_bytes(tensors), 2 * large)

    def test_tensor_at_or_below_threshold_is_not_aligned(self):
        self.assertEqual(
            external_data._align_offset(20000, 1000, 65536, align_threshold=1000), 20000
        )

    def test_tensor_above_threshold_is_aligned(self):
        new_offset = external_data._align_offset(20000, 1048, 65536, align_threshold=1000)
        self.assertNotEqual(new_offset, 20000)
        self.assertEqual(new_offset % 65536, 0)

    def test_already_aligned_offset_is_unchanged(self):
        self.assertEqual(
            external_data._align_offset(65536 * 3, 1048, 65536, align_threshold=1000),
            65536 * 3,
        )

    def test_alignment_is_floored_at_one_page(self):
        # A granularity below a 4KB page is raised to 4096.
        offset = external_data._align_offset(20000, 1048, 4000, align_threshold=1000)
        self.assertEqual(offset % 4096, 0)
        self.assertEqual(offset, 20480)


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
                self._shape = ir.Shape(getattr(value, "shape"), frozen=True)  # ruff: ignore[get-attr-with-constant]

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

    def test_external_data_written_in_declaration_order(self):
        model_with_external_data = external_data.unload_from_model(
            self.model_with_mixed_external_data,
            self.base_path,
            self.external_data_name,
        )
        file_path = os.path.join(self.base_path, self.external_data_name)
        # Every tensor here is below the alignment threshold, so they are all in
        # the packed prefix and keep their initializer declaration order.
        initializers = model_with_external_data.graph.initializers
        expected_tensor_order = [
            value.const_value.tobytes() for value in initializers.values()
        ]
        with open(file_path, "r+b") as data_file:
            current_offset = 0
            for i, tensor_bytes in enumerate(expected_tensor_order):
                data_file.seek(current_offset)
                tensor_data = data_file.read(len(tensor_bytes))
                current_offset += len(tensor_bytes)
                self.assertEqual(
                    tensor_data,
                    tensor_bytes,
                    f"Tensor at declaration index {i} is not at the expected offset",
                )

    def test_tensors_are_packed_densely_in_declaration_order(self):
        # Dense packing means offset N+1 == offset N + length N, with no gaps,
        # and the on-disk order is exactly the declaration order.
        threshold = external_data._DEFAULT_ALIGN_THRESHOLD
        tensors = [
            ir.Tensor(np.zeros(16, dtype=np.uint8), name="small_a"),
            ir.Tensor(np.full(threshold + 1, 1, dtype=np.uint8), name="large_a"),
            ir.Tensor(np.zeros(16, dtype=np.uint8), name="small_b"),
            ir.Tensor(np.full(threshold + 1, 2, dtype=np.uint8), name="large_b"),
        ]
        result = external_data.convert_tensors_to_external(
            tensors, self.base_path, "dense.bin"
        )
        self.assertEqual([t.name for t in result], [t.name for t in tensors])
        expected_offset = 0
        for original, external in zip(tensors, result):
            self.assertEqual(external.offset, expected_offset, f"{external.name}")
            expected_offset += original.nbytes
        # The file has no padding at all.
        self.assertEqual(
            os.path.getsize(os.path.join(self.base_path, "dense.bin")),
            sum(t.nbytes for t in tensors),
        )

    def test_alignment_is_opt_in_and_aligns_large_tensors(self):
        alignment = 65536
        threshold = external_data._DEFAULT_ALIGN_THRESHOLD
        tensors = [
            ir.Tensor(np.zeros(16, dtype=np.uint8), name="small_a"),
            ir.Tensor(np.full(threshold + 1, 1, dtype=np.uint8), name="large_a"),
            ir.Tensor(np.zeros(16, dtype=np.uint8), name="small_b"),
            ir.Tensor(np.full(threshold + 1, 2, dtype=np.uint8), name="large_b"),
        ]
        result = external_data.convert_tensors_to_external(
            tensors, self.base_path, "aligned.bin", alignment=alignment
        )
        # Declaration order is preserved regardless of the alignment policy.
        self.assertEqual([t.name for t in result], [t.name for t in tensors])
        by_name = {tensor.name: tensor for tensor in result}
        # Small tensors are never aligned; large ones always are.
        self.assertEqual(by_name["small_a"].offset, 0)
        for name in ("large_a", "large_b"):
            self.assertEqual(by_name[name].offset % alignment, 0)
        # Reading back yields the original bytes despite the padding gaps.
        for original, external in zip(tensors, result):
            np.testing.assert_array_equal(external.numpy(), original.numpy())


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

    def test_shard_count_above_five_digits_uses_natural_width(self):
        # ``:05d`` is a *minimum* width: the shard index keeps 5 zero-padded
        # digits, but a total ≥ 100_000 spills to its natural 6-digit width.
        # The cleanup regex (``\d{5,}``) must still match these.
        result = external_data._get_shard_filename("model.data", 1, 100_000)
        self.assertEqual(result, "model-00001-of-100000.data")
        result_big = external_data._get_shard_filename("model.data", 99_999, 100_000)
        self.assertEqual(result_big, "model-99999-of-100000.data")


class ShardTensorsTest(unittest.TestCase):
    """Test the tensor sharding helper."""

    def _make_tensor(self, name: str, nbytes: int) -> ir.TensorProtocol:
        """Create a float32 tensor with the requested byte size (rounded down to 4)."""
        n_floats = max(1, nbytes // 4)
        data = np.zeros(n_floats, dtype=np.float32)
        return ir.Tensor(data, dtype=ir.DataType.FLOAT, name=name)

    def _make_uint8_tensor(self, name: str, nbytes: int) -> ir.TensorProtocol:
        """Create a uint8 tensor with exactly ``nbytes`` bytes (1 byte per element)."""
        data = np.zeros(max(1, nbytes), dtype=np.uint8)
        return ir.Tensor(data, dtype=ir.DataType.UINT8, name=name)

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

    def test_sharding_is_dense_by_default(self):
        # Without alignment a shard's size is exactly the sum of its tensors,
        # so two tensors summing to the limit fit in a single shard.
        t0 = self._make_uint8_tensor("t0", 1000)
        t1 = self._make_uint8_tensor("t1", 1000)
        shards = external_data._shard_tensors([t0, t1], 2000)
        self.assertEqual(len(shards), 1)

    def test_sharding_accounts_for_alignment_when_enabled(self):
        alignment = 65536
        t0 = self._make_uint8_tensor("t0", external_data._DEFAULT_ALIGN_THRESHOLD + 4)
        t1 = self._make_uint8_tensor("t1", external_data._DEFAULT_ALIGN_THRESHOLD + 4)
        # These fit in one shard by raw byte sum, but the alignment padding
        # inserted before ``t1`` when writing forces a split.
        shards = external_data._shard_tensors([t0, t1], t0.nbytes + t1.nbytes, alignment)
        self.assertEqual([len(shard) for shard in shards], [1, 1])

    def test_shards_never_exceed_the_limit_they_estimate(self):
        # Property test: for arbitrary size mixes, with and without alignment,
        # every shard must fit within the limit once laid out on disk (unless it
        # holds a single oversized tensor, which is allowed by contract).
        alignment = 65536
        threshold = external_data._DEFAULT_ALIGN_THRESHOLD
        size_choices = [1, 100, threshold - 1, threshold, threshold + 1, 3 * threshold]
        rng = np.random.default_rng(0)
        for policy in (None, alignment):
            for _ in range(200):
                count = int(rng.integers(1, 40))
                sizes = [int(rng.choice(size_choices)) for _ in range(count)]
                tensors = [self._make_uint8_tensor(f"t{i}", s) for i, s in enumerate(sizes)]
                limit = int(rng.choice([threshold, 4 * threshold, 16 * threshold]))
                shards = external_data._shard_tensors(tensors, limit, policy)
                # Every tensor is placed exactly once, in declaration order.
                self.assertEqual(
                    [t.name for shard in shards for t in shard],
                    [t.name for t in tensors],
                )
                for shard in shards:
                    size = external_data._estimate_shard_size_bytes(shard, policy)
                    if len(shard) > 1:
                        self.assertLessEqual(size, limit, f"sizes={sizes} limit={limit}")

    def test_estimated_size_matches_written_size(self):
        # The size the sharder predicts must match what actually lands on disk.
        alignment = 65536
        for policy in (None, alignment):
            tensors = [
                self._make_uint8_tensor("small", 128),
                self._make_uint8_tensor("large", external_data._DEFAULT_ALIGN_THRESHOLD + 4),
                self._make_uint8_tensor("small2", 64),
            ]
            with tempfile.TemporaryDirectory() as tmp_dir:
                external_data.convert_tensors_to_external(
                    tensors, tmp_dir, "estimate.bin", alignment=policy
                )
                actual = os.path.getsize(os.path.join(tmp_dir, "estimate.bin"))
            self.assertEqual(external_data._estimate_shard_size_bytes(tensors, policy), actual)


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

    def test_cleanup_leaves_unowned_zero_indexed_shard_files_alone(self):
        # Shard indices are 1-indexed and the 0-indexed file *is* invalid, but
        # we deliberately leave it alone unless the model being saved actually
        # references it. This avoids the cross-model deletion bug where a
        # stem-based glob would wipe out shards belonging to a different model
        # that happens to live in the same directory.
        unrelated = os.path.join(self.base_path, "model-00000-of-00001.data")
        with open(unrelated, "wb") as f:
            f.write(b"stale")
        model, _ = self._make_model([100, 100])
        external_data.unload_from_model(
            model,
            self.base_path,
            "model.data",
            size_threshold_bytes=0,
            max_shard_size_bytes=10_000,
        )
        self.assertTrue(os.path.exists(unrelated))

    def test_cleanup_does_not_delete_unrelated_models_shards_in_same_directory(self):
        # Regression test for the cross-model-deletion bug: an unrelated model
        # in the same directory wrote ``model-00001-of-00009.data`` ...
        # ``model-00009-of-00009.data``. Saving *our* model under the same
        # base name with a smaller shard count must not touch those files.
        other_paths = [
            os.path.join(self.base_path, f"model-{i:05d}-of-00009.data") for i in range(1, 10)
        ]
        for p in other_paths:
            with open(p, "wb") as f:
                f.write(b"other-model-data")
        model, _ = self._make_model([100, 100])
        external_data.unload_from_model(
            model,
            self.base_path,
            "model.data",
            size_threshold_bytes=0,
            max_shard_size_bytes=10_000,
        )
        for p in other_paths:
            self.assertTrue(os.path.exists(p), f"unrelated model's shard {p} was deleted")

    def test_save_sharded_raises_on_foreign_destination_collision(self):
        # Direct filename collision with a file the model doesn't own must
        # raise FileExistsError rather than silently overwriting. 2 tensors
        # of 400 bytes with max_shard_size_bytes=400 yields a 2-shard layout
        # whose first shard filename is ``model-00001-of-00002.data``.
        foreign = os.path.join(self.base_path, "model-00001-of-00002.data")
        with open(foreign, "wb") as f:
            f.write(b"foreign")
        model, _ = self._make_model([400, 400])
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            external_data.unload_from_model(
                model,
                self.base_path,
                "model.data",
                size_threshold_bytes=0,
                max_shard_size_bytes=400,
            )
        with open(foreign, "rb") as f:
            self.assertEqual(f.read(), b"foreign")


class ParallelWriteTest(unittest.TestCase):
    """Tests for the concurrent external data write pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_tensors(self, count=24, size=40_000):
        rng = np.random.default_rng(7)
        return [
            ir.Tensor(rng.integers(0, 255, size=size, dtype=np.uint8), name=f"t{i}")
            for i in range(count)
        ]

    def test_parallel_output_is_byte_identical_to_serial(self):
        tensors = self._make_tensors()
        serial_dir = os.path.join(self.base_path, "serial")
        parallel_dir = os.path.join(self.base_path, "parallel")
        os.makedirs(serial_dir)
        os.makedirs(parallel_dir)

        serial = external_data.convert_tensors_to_external(
            tensors, serial_dir, "model.data", max_workers=None
        )
        parallel = external_data.convert_tensors_to_external(
            tensors, parallel_dir, "model.data", max_workers=8
        )

        with open(os.path.join(serial_dir, "model.data"), "rb") as f:
            serial_bytes = f.read()
        with open(os.path.join(parallel_dir, "model.data"), "rb") as f:
            parallel_bytes = f.read()
        self.assertEqual(serial_bytes, parallel_bytes)

        # Offsets, lengths and order must match too.
        self.assertEqual(
            [(t.name, t.offset, t.length) for t in serial],
            [(t.name, t.offset, t.length) for t in parallel],
        )

    def test_parallel_roundtrip_preserves_values(self):
        tensors = self._make_tensors()
        result = external_data.convert_tensors_to_external(
            tensors, self.base_path, "model.data", max_workers=4
        )
        for original, external in zip(tensors, result):
            np.testing.assert_array_equal(external.numpy(), original.numpy())

    def test_parallel_with_alignment_matches_serial(self):
        alignment = 4096
        tensors = self._make_tensors(count=12)
        serial_dir = os.path.join(self.base_path, "serial")
        parallel_dir = os.path.join(self.base_path, "parallel")
        os.makedirs(serial_dir)
        os.makedirs(parallel_dir)
        external_data.convert_tensors_to_external(
            tensors, serial_dir, "m.data", alignment=alignment, align_threshold=1024
        )
        external_data.convert_tensors_to_external(
            tensors,
            parallel_dir,
            "m.data",
            max_workers=6,
            alignment=alignment,
            align_threshold=1024,
        )
        with open(os.path.join(serial_dir, "m.data"), "rb") as f:
            serial_bytes = f.read()
        with open(os.path.join(parallel_dir, "m.data"), "rb") as f:
            parallel_bytes = f.read()
        # Padding bytes must be zeros in both cases, not stale file content.
        self.assertEqual(serial_bytes, parallel_bytes)

    def test_callback_is_invoked_once_per_tensor(self):
        tensors = self._make_tensors(count=16)
        seen = []
        lock = threading.Lock()

        def callback(tensor, info):
            with lock:
                seen.append((tensor.name, info.index, info.total))

        external_data.convert_tensors_to_external(
            tensors, self.base_path, "model.data", callback=callback, max_workers=8
        )
        self.assertEqual(len(seen), len(tensors))
        self.assertEqual({name for name, _, _ in seen}, {t.name for t in tensors})
        self.assertEqual(sorted(index for _, index, _ in seen), list(range(len(tensors))))
        self.assertTrue(all(total == len(tensors) for _, _, total in seen))

    def test_exception_in_worker_propagates(self):
        class ExplodingTensor(ir.Tensor):
            def tofile(self, file):
                raise RuntimeError("boom")

        tensors = self._make_tensors(count=4)
        tensors[2] = ExplodingTensor(np.zeros(40_000, dtype=np.uint8), name=tensors[2].name)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            external_data.convert_tensors_to_external(
                tensors, self.base_path, "model.data", max_workers=4
            )

    def test_byte_budget_bounds_in_flight_bytes(self):
        budget = external_data._ByteBudget(1000)
        first = budget.acquire(600)
        self.assertEqual(first, 600)
        acquired = threading.Event()

        def worker():
            budget.acquire(600)
            acquired.set()

        thread = threading.Thread(target=worker)
        thread.start()
        # The second acquire must block while the first is outstanding.
        self.assertFalse(acquired.wait(timeout=0.2))
        budget.release(first)
        self.assertTrue(acquired.wait(timeout=5))
        thread.join()

    def test_byte_budget_admits_tensor_larger_than_capacity(self):
        # An oversized tensor must not deadlock; it is admitted on its own.
        budget = external_data._ByteBudget(100)
        amount = budget.acquire(10_000)
        self.assertEqual(amount, 100)
        budget.release(amount)

    def test_parallel_sharded_save_matches_serial(self):
        def build_model():
            rng = np.random.default_rng(11)
            graph = ir.Graph(inputs=[], outputs=[], nodes=[], initializers=[], name="g")
            for i in range(12):
                tensor = ir.Tensor(
                    rng.integers(0, 255, size=30_000, dtype=np.uint8), name=f"w{i}"
                )
                graph.register_initializer(ir.Value(name=f"w{i}", const_value=tensor))
            return ir.Model(graph, ir_version=10)

        serial_dir = os.path.join(self.base_path, "serial")
        parallel_dir = os.path.join(self.base_path, "parallel")
        os.makedirs(serial_dir)
        os.makedirs(parallel_dir)
        external_data.unload_from_model(
            build_model(), serial_dir, "model.data", max_shard_size_bytes=100_000
        )
        external_data.unload_from_model(
            build_model(),
            parallel_dir,
            "model.data",
            max_shard_size_bytes=100_000,
            max_workers=4,
        )
        serial_files = sorted(os.listdir(serial_dir))
        self.assertEqual(serial_files, sorted(os.listdir(parallel_dir)))
        self.assertGreater(len(serial_files), 1)
        for name in serial_files:
            with open(os.path.join(serial_dir, name), "rb") as f:
                expected = f.read()
            with open(os.path.join(parallel_dir, name), "rb") as f:
                self.assertEqual(f.read(), expected, name)


if __name__ == "__main__":
    unittest.main()
