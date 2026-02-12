# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test shape inference against ONNX backend node tests.

For each backend test model, the test:
1. Saves the expected output dtype and shape from the model proto.
2. Clears the output type/shape information.
3. Runs symbolic shape inference.
4. Asserts the inferred dtype and shape match the expected values.

Outputs where inference produces ``None`` (incomplete inference due to
data-dependent shapes or missing constant inputs) are not treated as
failures — only *incorrect* inferred values are failures.
"""

from __future__ import annotations

import logging
import pathlib
import unittest

import onnx
import onnx.backend.test
import parameterized

import onnx_ir as ir
import onnx_ir.shape_inference
from onnx_ir.shape_inference import _ops as _ops

logger = logging.getLogger(__name__)

_ONNX_BACKEND_NODE_TEST_DIR = pathlib.Path(onnx.backend.test.__file__).parent / "data" / "node"

# Build parametrized test args: (test_name, model_path)
_test_args = [
    (model_dir.name, model_dir / "model.onnx")
    for model_dir in sorted(_ONNX_BACKEND_NODE_TEST_DIR.iterdir())
    if (model_dir / "model.onnx").exists()
]


def _is_supported(model: ir.Model) -> bool:
    """Check if all ops in the model have registered shape inference."""
    return all(
        onnx_ir.shape_inference.registry.has(node.domain or "", node.op_type)
        for node in model.graph
    )


def _shapes_compatible(
    expected: ir.Shape | None,
    inferred: ir.Shape | None,
) -> bool:
    """Check that the inferred shape is compatible with the expected shape.

    The inferred shape may contain generated symbolic dims (e.g. ``_d0``)
    where the expected shape has concrete values.  As long as ranks match and
    every concrete expected dim matches the inferred dim, we consider them
    compatible.  A ``None`` expected shape matches any inferred shape.
    """
    if expected is None:
        return True
    if inferred is None:
        # Incomplete inference, not wrong — caller decides how to handle
        return True
    if len(expected) != len(inferred):
        return False
    for exp_dim, inf_dim in zip(expected, inferred):
        exp_val = exp_dim.value if isinstance(exp_dim, ir.SymbolicDim) else exp_dim
        inf_val = inf_dim.value if isinstance(inf_dim, ir.SymbolicDim) else inf_dim
        if isinstance(exp_val, int) and isinstance(inf_val, int):
            if exp_val != inf_val:
                return False
        # If expected is concrete but inferred is symbolic, that's acceptable
        # (we may not have enough info to resolve the dim).
    return True


class ShapeInferenceBackendTest(unittest.TestCase):
    @parameterized.parameterized.expand(_test_args)
    def test_shape_inference_matches_expected(self, _: str, model_path: pathlib.Path) -> None:
        proto = onnx.load(model_path)
        model = ir.serde.deserialize_model(proto)

        if not _is_supported(model):
            self.skipTest("Contains ops without registered shape inference")

        # Save expected output dtype and shape, then clear them
        expected: dict[str, tuple[ir.DataType | None, ir.Shape | None]] = {}
        for out in model.graph.outputs:
            expected[out.name] = (out.dtype, out.shape)
            out.dtype = None
            out.shape = None

        onnx_ir.shape_inference.infer_symbolic_shapes(model)

        for out in model.graph.outputs:
            exp_dtype, exp_shape = expected[out.name]

            # Check dtype: None inferred dtype is acceptable (incomplete)
            if exp_dtype is not None and out.dtype is not None:
                self.assertEqual(
                    out.dtype,
                    exp_dtype,
                    f"Output '{out.name}': dtype mismatch",
                )

            # Check shape compatibility
            self.assertTrue(
                _shapes_compatible(exp_shape, out.shape),
                f"Output '{out.name}': shape mismatch: expected={exp_shape}, got={out.shape}",
            )


if __name__ == "__main__":
    unittest.main()
