# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import pathlib
import unittest

import onnx
import onnx.backend.test
import parameterized

import onnx_ir as ir
import onnx_ir.testing

onnx_backend_test_path = pathlib.Path(onnx.backend.test.__file__).parent / "data"

model_paths = list(onnx_backend_test_path.rglob("*.onnx"))
test_args = [
    (f"{model_path.parent.name}_{model_path.name}", model_path) for model_path in model_paths
]


class SerdeTest(unittest.TestCase):
    @parameterized.parameterized.expand(test_args)
    def test_serialization_deserialization_produces_same_model(
        self, _: str, model_path: pathlib.Path
    ) -> None:
        model = onnx.load(model_path)
        # Fix the missing graph name of some test models
        model.graph.name = "main_graph"
        onnx.checker.check_model(model)

        # Profile the serialization and deserialization process
        ir_model = ir.serde.deserialize_model(model)
        serialized = ir.serde.serialize_model(ir_model)

        onnx_ir.testing.assert_onnx_proto_equal(serialized, model)
        onnx.checker.check_model(serialized)


if __name__ == "__main__":
    unittest.main()
