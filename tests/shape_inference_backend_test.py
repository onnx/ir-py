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

import onnx
import onnx.backend.test
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
_SKIP_DATA_DEPENDENT: set[str] = set()

# Tests using ops from domains we haven't implemented (ai.onnx.ml,
# ai.onnx.preview.training).
_SKIP_UNSUPPORTED_OPS: set[str] = set()

# Expanded multi-op models where shape info is lost through If/Loop/complex
# subgraphs or where intermediate Constant/Shape ops don't preserve shapes.
_SKIP_EXPANDED_MODELS: set[str] = {
    "test_affine_grid_2d_align_corners_expanded",
    "test_affine_grid_2d_expanded",
    "test_affine_grid_3d_align_corners_expanded",
    "test_affine_grid_3d_expanded",
    "test_layer_normalization_2d_axis0_expanded",
    "test_layer_normalization_2d_axis0_expanded_ver18",
    "test_layer_normalization_2d_axis1_expanded",
    "test_layer_normalization_2d_axis1_expanded_ver18",
    "test_layer_normalization_2d_axis_negative_1_expanded",
    "test_layer_normalization_2d_axis_negative_1_expanded_ver18",
    "test_layer_normalization_2d_axis_negative_2_expanded",
    "test_layer_normalization_2d_axis_negative_2_expanded_ver18",
    "test_layer_normalization_3d_axis0_epsilon_expanded",
    "test_layer_normalization_3d_axis0_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis1_epsilon_expanded",
    "test_layer_normalization_3d_axis1_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis2_epsilon_expanded",
    "test_layer_normalization_3d_axis2_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis_negative_1_epsilon_expanded",
    "test_layer_normalization_3d_axis_negative_1_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis_negative_2_epsilon_expanded",
    "test_layer_normalization_3d_axis_negative_2_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis_negative_3_epsilon_expanded",
    "test_layer_normalization_3d_axis_negative_3_epsilon_expanded_ver18",
    "test_layer_normalization_4d_axis0_expanded",
    "test_layer_normalization_4d_axis0_expanded_ver18",
    "test_layer_normalization_4d_axis1_expanded",
    "test_layer_normalization_4d_axis1_expanded_ver18",
    "test_layer_normalization_4d_axis2_expanded",
    "test_layer_normalization_4d_axis2_expanded_ver18",
    "test_layer_normalization_4d_axis3_expanded",
    "test_layer_normalization_4d_axis3_expanded_ver18",
    "test_layer_normalization_4d_axis_negative_1_expanded",
    "test_layer_normalization_4d_axis_negative_1_expanded_ver18",
    "test_layer_normalization_4d_axis_negative_2_expanded",
    "test_layer_normalization_4d_axis_negative_2_expanded_ver18",
    "test_layer_normalization_4d_axis_negative_3_expanded",
    "test_layer_normalization_4d_axis_negative_3_expanded_ver18",
    "test_layer_normalization_4d_axis_negative_4_expanded",
    "test_layer_normalization_4d_axis_negative_4_expanded_ver18",
    "test_layer_normalization_default_axis_expanded",
    "test_layer_normalization_default_axis_expanded_ver18",
    "test_range_float_type_positive_delta_expanded",
    "test_range_int32_type_negative_delta_expanded",
    "test_rms_normalization_2d_axis0_expanded",
    "test_rms_normalization_2d_axis1_expanded",
    "test_rms_normalization_2d_axis_negative_1_expanded",
    "test_rms_normalization_2d_axis_negative_2_expanded",
    "test_rms_normalization_3d_axis0_epsilon_expanded",
    "test_rms_normalization_3d_axis1_epsilon_expanded",
    "test_rms_normalization_3d_axis2_epsilon_expanded",
    "test_rms_normalization_3d_axis_negative_1_epsilon_expanded",
    "test_rms_normalization_3d_axis_negative_2_epsilon_expanded",
    "test_rms_normalization_3d_axis_negative_3_epsilon_expanded",
    "test_rms_normalization_4d_axis0_expanded",
    "test_rms_normalization_4d_axis1_expanded",
    "test_rms_normalization_4d_axis2_expanded",
    "test_rms_normalization_4d_axis3_expanded",
    "test_rms_normalization_4d_axis_negative_1_expanded",
    "test_rms_normalization_4d_axis_negative_2_expanded",
    "test_rms_normalization_4d_axis_negative_3_expanded",
    "test_rms_normalization_4d_axis_negative_4_expanded",
    "test_rms_normalization_default_axis_expanded",
}

# Tests where the inferred shape is symbolic where the expected is concrete.
# These are ops with inherently data-dependent output shapes, or ops where
# constant folding would be needed to resolve the concrete shape.
_SKIP_SYMBOLIC_SHAPE: set[str] = set()

# Tests where inference fails due to missing support for sequence types,
# Scan subgraphs, or specific op features (CenterCropPad axes, Resize
# not_smaller policy).
_SKIP_INCOMPLETE_SUPPORT: set[str] = set()

_ALL_SKIPS = (
    _SKIP_DATA_DEPENDENT
    | _SKIP_UNSUPPORTED_OPS
    | _SKIP_EXPANDED_MODELS
    | _SKIP_SYMBOLIC_SHAPE
    | _SKIP_INCOMPLETE_SUPPORT
)


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


def _run_inference_and_compare(
    test_case: unittest.TestCase,
    model_path: pathlib.Path,
) -> tuple[bool, bool, bool]:
    """Run shape inference on a model and compare with expected outputs.

    Returns:
        (all_ok, has_symbolic, has_other_error): all_ok is True when every
        output matches.  has_symbolic is True when at least one output has a
        symbolic dim where a concrete one was expected.  has_other_error is
        True when a non-symbolic mismatch occurred (dtype None, shape None,
        rank mismatch, concrete value mismatch).
    """
    proto = onnx.load(model_path)
    input_tensors = _load_test_inputs(model_path.parent)
    _inject_inputs_as_initializers(proto, input_tensors)

    model = ir.serde.deserialize_model(proto)

    expected: dict[
        str, tuple[ir.DataType | None, ir.Shape | None, ir.TypeProtocol | None]
    ] = {}
    for out in model.graph.outputs:
        expected[out.name] = (out.dtype, out.shape, out.type)
        out.type = None

    from onnx_ir.shape_inference import infer_symbolic_shapes

    infer_symbolic_shapes(model)

    all_ok = True
    has_symbolic = False
    has_other_error = False

    for out in model.graph.outputs:
        exp_dtype, exp_shape, exp_type = expected[out.name]

        # For sequence/optional types, just check the type class matches
        if exp_type is not None and isinstance(exp_type, ir.SequenceType):
            if out.type is None or not isinstance(out.type, ir.SequenceType):
                all_ok = False
                has_other_error = True
            continue

        if exp_dtype is not None:
            if out.dtype is None or out.dtype != exp_dtype:
                all_ok = False
                has_other_error = True

        if exp_shape is not None:
            if out.shape is None:
                all_ok = False
                has_other_error = True
            else:
                compatible, symbolic = _shapes_compatible(exp_shape, out.shape)
                if not compatible:
                    all_ok = False
                    has_other_error = True
                elif symbolic:
                    all_ok = False
                    has_symbolic = True

    return all_ok, has_symbolic, has_other_error


class ShapeInferenceBackendTest(unittest.TestCase):
    @parameterized.parameterized.expand(_test_args)
    def test_shape_inference_matches_expected(self, _: str, model_path: pathlib.Path) -> None:
        test_name = model_path.parent.name

        if test_name in _ALL_SKIPS:
            self.skipTest("See skip list for reason")

        proto = onnx.load(model_path)

        # Inject test inputs as initializers so constant inputs are available
        input_tensors = _load_test_inputs(model_path.parent)
        _inject_inputs_as_initializers(proto, input_tensors)

        model = ir.serde.deserialize_model(proto)

        # Save expected output dtype and shape, then clear them
        expected: dict[
            str, tuple[ir.DataType | None, ir.Shape | None, ir.TypeProtocol | None]
        ] = {}
        for out in model.graph.outputs:
            expected[out.name] = (out.dtype, out.shape, out.type)
            out.type = None

        from onnx_ir.shape_inference import infer_symbolic_shapes

        infer_symbolic_shapes(model)

        for out in model.graph.outputs:
            exp_dtype, exp_shape, exp_type = expected[out.name]

            # For sequence/optional types, just check type class matches
            if exp_type is not None and isinstance(exp_type, ir.SequenceType):
                self.assertIsNotNone(
                    out.type,
                    f"Output '{out.name}': type is None, expected SequenceType",
                )
                self.assertIsInstance(
                    out.type,
                    ir.SequenceType,
                    f"Output '{out.name}': expected SequenceType, got {type(out.type).__name__}",
                )
                continue

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


# Build test args for _SKIP_SYMBOLIC_SHAPE validation
_symbolic_skip_test_args = [
    (name, _ONNX_BACKEND_NODE_TEST_DIR / name / "model.onnx")
    for name in sorted(_SKIP_SYMBOLIC_SHAPE)
    if (_ONNX_BACKEND_NODE_TEST_DIR / name / "model.onnx").exists()
]


class SymbolicShapeSkipValidationTest(unittest.TestCase):
    """Verify _SKIP_SYMBOLIC_SHAPE entries fail due to symbolic dims only.

    If an entry starts passing, it should be removed from the skip list.
    If an entry fails for a non-symbolic reason, the skip list category is wrong.
    """

    @parameterized.parameterized.expand(_symbolic_skip_test_args, skip_on_empty=True)
    def test_skip_is_symbolic_not_other_error(self, _: str, model_path: pathlib.Path) -> None:
        all_ok, has_symbolic, has_other_error = _run_inference_and_compare(self, model_path)
        if all_ok:
            self.fail(
                f"Test passes now — remove '{model_path.parent.name}' "
                f"from _SKIP_SYMBOLIC_SHAPE"
            )
        self.assertTrue(
            has_symbolic,
            f"'{model_path.parent.name}' does not have symbolic dim issues; "
            f"it has other errors (has_other_error={has_other_error}). "
            f"Move it to the correct skip list.",
        )


if __name__ == "__main__":
    unittest.main()
