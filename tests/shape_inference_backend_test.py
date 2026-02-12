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

# Tests using ops from domains we haven't implemented (ai.onnx.ml,
# ai.onnx.preview.training).
_SKIP_UNSUPPORTED_OPS: set[str] = {
    "test_adagrad",
    "test_adagrad_multiple",
    "test_adam",
    "test_adam_multiple",
    "test_ai_onnx_ml_array_feature_extractor",
    "test_ai_onnx_ml_binarizer",
    "test_ai_onnx_ml_label_encoder_string_int",
    "test_ai_onnx_ml_label_encoder_string_int_no_default",
    "test_ai_onnx_ml_label_encoder_tensor_mapping",
    "test_ai_onnx_ml_label_encoder_tensor_value_only_mapping",
    "test_ai_onnx_ml_tree_ensemble_set_membership",
    "test_ai_onnx_ml_tree_ensemble_single_tree",
    "test_momentum",
    "test_momentum_multiple",
    "test_nesterov_momentum",
}

# Expanded multi-op models where shape info is lost through If/Loop/complex
# subgraphs or where intermediate Constant/Shape ops don't preserve shapes.
_SKIP_EXPANDED_MODELS: set[str] = {
    "test_affine_grid_2d_align_corners_expanded",
    "test_affine_grid_2d_expanded",
    "test_affine_grid_3d_align_corners_expanded",
    "test_affine_grid_3d_expanded",
    "test_attention_3d_causal_expanded",
    "test_attention_3d_diff_heads_sizes_causal_expanded",
    "test_attention_3d_diff_heads_sizes_expanded",
    "test_attention_3d_diff_heads_sizes_scaled_expanded",
    "test_attention_3d_diff_heads_sizes_softcap_expanded",
    "test_attention_3d_expanded",
    "test_attention_3d_gqa_causal_expanded",
    "test_attention_3d_gqa_expanded",
    "test_attention_3d_gqa_scaled_expanded",
    "test_attention_3d_gqa_softcap_expanded",
    "test_attention_3d_scaled_expanded",
    "test_attention_3d_softcap_expanded",
    "test_attention_3d_transpose_verification_expanded",
    "test_attention_4d_causal_expanded",
    "test_attention_4d_diff_heads_sizes_causal_expanded",
    "test_attention_4d_diff_heads_sizes_expanded",
    "test_attention_4d_diff_heads_sizes_scaled_expanded",
    "test_attention_4d_diff_heads_sizes_softcap_expanded",
    "test_attention_4d_expanded",
    "test_attention_4d_fp16_expanded",
    "test_attention_4d_gqa_causal_expanded",
    "test_attention_4d_gqa_expanded",
    "test_attention_4d_gqa_scaled_expanded",
    "test_attention_4d_gqa_softcap_expanded",
    "test_attention_4d_scaled_expanded",
    "test_attention_4d_softcap_expanded",
    "test_attention_4d_with_qk_matmul_expanded",
    "test_blackmanwindow_expanded",
    "test_blackmanwindow_symmetric_expanded",
    "test_center_crop_pad_crop_and_pad_expanded",
    "test_center_crop_pad_crop_axes_chw_expanded",
    "test_center_crop_pad_crop_axes_hwc_expanded",
    "test_center_crop_pad_crop_expanded",
    "test_center_crop_pad_crop_negative_axes_hwc_expanded",
    "test_center_crop_pad_pad_expanded",
    "test_hammingwindow_expanded",
    "test_hammingwindow_symmetric_expanded",
    "test_hannwindow_expanded",
    "test_hannwindow_symmetric_expanded",
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
_SKIP_SYMBOLIC_SHAPE: set[str] = {
    # Col2Im: output spatial dims depend on const inputs
    "test_col2im",
    "test_col2im_5d",
    "test_col2im_dilations",
    "test_col2im_pads",
    "test_col2im_strides",
    # Compress: output size is data-dependent
    "test_compress_default_axis",
    # ImageDecoder: image dimensions are data-dependent
    "test_image_decoder_decode_bmp_rgb",
    "test_image_decoder_decode_jpeg2k_rgb",
    "test_image_decoder_decode_jpeg_bgr",
    "test_image_decoder_decode_jpeg_grayscale",
    "test_image_decoder_decode_jpeg_rgb",
    "test_image_decoder_decode_png_rgb",
    "test_image_decoder_decode_pnm_rgb",
    "test_image_decoder_decode_tiff_rgb",
    "test_image_decoder_decode_webp_rgb",
    # Loop: output shape depends on loop iterations
    "test_loop11",
    # MaxUnpool: output shape depends on optional output_shape input
    "test_maxunpool_export_without_output_shape",
    # MelWeightMatrix: output dims depend on const inputs
    "test_melweightmatrix",
    # NonMaxSuppression: output size is data-dependent
    "test_nonmaxsuppression_center_point_box_format",
    "test_nonmaxsuppression_flipped_coordinates",
    "test_nonmaxsuppression_identical_boxes",
    "test_nonmaxsuppression_iou_threshold_boundary",
    "test_nonmaxsuppression_limit_output_size",
    "test_nonmaxsuppression_single_box",
    "test_nonmaxsuppression_suppress_by_IOU",
    "test_nonmaxsuppression_suppress_by_IOU_and_scores",
    "test_nonmaxsuppression_two_batches",
    "test_nonmaxsuppression_two_classes",
    # NonZero: output size is data-dependent
    "test_nonzero_example",
    # OneHot: depth dimension comes from const input
    "test_onehot_negative_indices",
    "test_onehot_with_axis",
    "test_onehot_with_negative_axis",
    "test_onehot_without_axis",
    # Range: output size depends on start/limit/delta values
    "test_range_float_type_positive_delta",
    "test_range_int32_type_negative_delta",
    # STFT: frame count depends on signal length and hop/window sizes
    "test_stft",
    "test_stft_with_window",
    # StringSplit: output dim depends on string content
    "test_string_split_basic",
    "test_string_split_consecutive_delimiters",
    "test_string_split_empty_string_delimiter",
    "test_string_split_empty_tensor",
    "test_string_split_maxsplit",
    "test_string_split_no_delimiter",
    # StringNormalizer: output dim may change due to stopwords
    "test_strnormalizer_export_monday_casesensintive_lower",
    "test_strnormalizer_export_monday_casesensintive_nochangecase",
    "test_strnormalizer_export_monday_casesensintive_upper",
    "test_strnormalizer_export_monday_empty_output",
    "test_strnormalizer_export_monday_insensintive_upper_twodim",
    # TfIdfVectorizer: output dim depends on vocabulary
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip0",
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip5",
    "test_tfidfvectorizer_tf_batch_uniandbigrams_skip5",
    "test_tfidfvectorizer_tf_only_bigrams_skip0",
    "test_tfidfvectorizer_tf_onlybigrams_levelempty",
    "test_tfidfvectorizer_tf_onlybigrams_skip5",
    "test_tfidfvectorizer_tf_uniandbigrams_skip5",
    # Unique: output size is data-dependent
    "test_unique_length_1",
    "test_unique_not_sorted_without_axis",
    "test_unique_sorted_without_axis",
}

# Tests where inference fails due to missing support for sequence types,
# Scan subgraphs, or specific op features (CenterCropPad axes, Resize
# not_smaller policy).
_SKIP_INCOMPLETE_SUPPORT: set[str] = {
    # CenterCropPad: axes attribute not fully handled
    "test_center_crop_pad_crop_axes_chw",
    # Resize: keep_aspect_ratio_policy edge cases
    "test_resize_downsample_sizes_nearest_not_smaller",
    # Scan: subgraph type propagation not implemented
    "test_scan9_sum",
    "test_scan_sum",
    # Sequence ops: sequence type inference not implemented
    "test_sequence_insert_at_back",
    "test_sequence_insert_at_front",
    "test_sequence_map_add_1_sequence_1_tensor",
    "test_sequence_map_add_2_sequences",
    "test_sequence_map_extract_shapes",
    "test_sequence_map_identity_1_sequence",
    "test_sequence_map_identity_1_sequence_1_tensor",
    "test_sequence_map_identity_2_sequences",
    "test_split_to_sequence_1",
    "test_split_to_sequence_2",
    "test_split_to_sequence_nokeepdims",
}

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
