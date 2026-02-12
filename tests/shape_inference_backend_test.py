# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test shape inference against ONNX backend node tests.

For each backend test model, the test:
1. Loads the model and the test input tensors from ``.pb`` files.
2. Injects the input tensors as graph initializers so that constant inputs
   (e.g. ``axes`` for Reduce ops) are available to shape inference.
3. Saves the expected output dtype and shape from the model proto.
4. Clears the output type/shape information.
5. Runs symbolic shape inference.
6. Asserts the inferred dtype and shape match the expected values.

When the inferred shape is ``None`` or symbolic where concrete is expected,
the test fails unless explicitly added to a skip list.
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np
import onnx
import onnx.backend.test
import onnx.numpy_helper
import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import _ops as _ops

_ONNX_BACKEND_NODE_TEST_DIR = pathlib.Path(onnx.backend.test.__file__).parent / "data" / "node"

# Build parametrized test args: (test_name, model_path)
_test_args = [
    (model_dir.name, model_dir / "model.onnx")
    for model_dir in sorted(_ONNX_BACKEND_NODE_TEST_DIR.iterdir())
    if (model_dir / "model.onnx").exists()
]

# Tests where shape inference produces incorrect results due to data-dependent
# shapes (e.g. operator inputs like split sizes or axes are graph inputs, not
# constants). Each entry should be investigated individually.
_SKIP_DATA_DEPENDENT: set[str] = {
    # Compress: output size depends on the boolean condition tensor values.
    "test_compress_0",
    "test_compress_1",
    "test_compress_negative_axis",
    # Unique with axis: output size is data-dependent (number of unique values).
    "test_unique_sorted_with_axis",
    "test_unique_sorted_with_axis_3d",
    "test_unique_sorted_with_negative_axis",
}

# Tests where the inferred shape is symbolic where the expected is concrete.
# These should be investigated to see if concrete inference is possible.
_SKIP_SYMBOLIC_SHAPE: set[str] = set()


def _load_test_inputs(model_dir: pathlib.Path) -> list[onnx.TensorProto]:
    """Load input tensors from the first test_data_set in a backend test dir."""
    data_dir = model_dir / "test_data_set_0"
    if not data_dir.exists():
        return []
    tensors = []
    for pb_file in sorted(data_dir.glob("input_*.pb")):
        tensor = onnx.TensorProto()
        tensor.ParseFromString(pb_file.read_bytes())
        tensors.append(tensor)
    return tensors


def _inject_inputs_as_initializers(
    proto: onnx.ModelProto,
    input_tensors: list[onnx.TensorProto],
) -> None:
    """Add test input tensors as graph initializers.

    This makes constant inputs (like ``axes`` for Reduce ops) visible to
    shape inference via ``get_const_tensor``.  The tensor names in the ``.pb``
    files match the graph input names.
    """
    existing_names = {init.name for init in proto.graph.initializer}
    input_names = [inp.name for inp in proto.graph.input]
    for tensor in input_tensors:
        name = tensor.name
        if not name:
            continue
        if name in existing_names:
            continue
        if name in input_names:
            proto.graph.initializer.append(tensor)


def _shapes_compatible(
    expected: ir.Shape,
    inferred: ir.Shape,
) -> tuple[bool, bool]:
    """Check that the inferred shape is compatible with the expected shape.

    Returns:
        (compatible, has_symbolic): compatible is True if shapes match,
        has_symbolic is True if inferred has symbolic dims where expected
        has concrete ones.
    """
    if len(expected) != len(inferred):
        return False, False
    has_symbolic = False
    for exp_dim, inf_dim in zip(expected, inferred):
        exp_val = exp_dim.value if isinstance(exp_dim, ir.SymbolicDim) else exp_dim
        inf_val = inf_dim.value if isinstance(inf_dim, ir.SymbolicDim) else inf_dim
        if isinstance(exp_val, int) and isinstance(inf_val, int):
            if exp_val != inf_val:
                return False, False
        elif isinstance(exp_val, int) and not isinstance(inf_val, int):
            has_symbolic = True
    return True, has_symbolic


class ShapeInferenceBackendTest(unittest.TestCase):
    @parameterized.parameterized.expand(_test_args)
    def test_shape_inference_matches_expected(self, _: str, model_path: pathlib.Path) -> None:
        test_name = model_path.parent.name

        if test_name in _SKIP_DATA_DEPENDENT:
            self.skipTest("Data-dependent shape (see skip list)")
        if test_name in _SKIP_SYMBOLIC_SHAPE:
            self.skipTest("Symbolic shape where concrete expected (see skip list)")

        proto = onnx.load(model_path)

        # Inject test inputs as initializers so constant inputs are available
        input_tensors = _load_test_inputs(model_path.parent)
        _inject_inputs_as_initializers(proto, input_tensors)

        model = ir.serde.deserialize_model(proto)

        # Save expected output dtype and shape, then clear them
        expected: dict[str, tuple[ir.DataType | None, ir.Shape | None]] = {}
        for out in model.graph.outputs:
            expected[out.name] = (out.dtype, out.shape)
            out.dtype = None
            out.shape = None

        from onnx_ir.shape_inference import infer_symbolic_shapes

        infer_symbolic_shapes(model)

        for out in model.graph.outputs:
            exp_dtype, exp_shape = expected[out.name]

            # Check dtype
            if exp_dtype is not None:
                self.assertIsNotNone(
                    out.dtype,
                    f"Output '{out.name}': dtype is None, expected {exp_dtype}",
                )
                self.assertEqual(
                    out.dtype,
                    exp_dtype,
                    f"Output '{out.name}': dtype mismatch",
                )

            # Check shape
            if exp_shape is not None:
                self.assertIsNotNone(
                    out.shape,
                    f"Output '{out.name}': shape is None, expected {exp_shape}",
                )
                compatible, has_symbolic = _shapes_compatible(exp_shape, out.shape)
                if has_symbolic:
                    self.fail(
                        f"Output '{out.name}': inferred symbolic dims where "
                        f"concrete expected: expected={exp_shape}, got={out.shape}. "
                        f"Add to _SKIP_SYMBOLIC_SHAPE if this is expected."
                    )
                self.assertTrue(
                    compatible,
                    f"Output '{out.name}': shape mismatch: "
                    f"expected={exp_shape}, got={out.shape}",
                )


if __name__ == "__main__":
    unittest.main()
