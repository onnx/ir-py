# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import pathlib
import unittest

from onnx_ir._onnx_compat import onnx  # noqa: TID251
import parameterized

import onnx_ir as ir
import onnx_ir.testing
from onnx_ir._onnx_compat import use_onnx_light

model_folder_path = pathlib.Path(__file__).resolve().parent.parent / "testdata"

# The backend test data is only available with the standard onnx package
from onnx_ir._onnx_compat import _USE_ONNX_LIGHT  # noqa: TID251

if _USE_ONNX_LIGHT:
    onnx_backend_test_path = None
else:
    import importlib as _importlib

    _backend_test = _importlib.import_module("onnx.backend.test")
    onnx_backend_test_path = pathlib.Path(_backend_test.__file__).parent / "data"

assert model_folder_path.exists()
if onnx_backend_test_path is not None:
    assert onnx_backend_test_path.exists()

model_paths = [
    *model_folder_path.rglob("*.textproto"),
    *(onnx_backend_test_path.rglob("*.onnx") if onnx_backend_test_path else []),
]
test_args = [
    (f"{model_path.parent.name}_{model_path.name}", model_path) for model_path in model_paths
]


def initialize_with_data(model: onnx.ModelProto) -> None:
    for tensor_proto in model.graph.initializer:
        if (
            tensor_proto.raw_data != b""
            or len(tensor_proto.float_data) != 0
            or len(tensor_proto.int32_data) != 0
            or len(tensor_proto.int64_data) != 0
            or len(tensor_proto.string_data) != 0
            or len(tensor_proto.uint64_data) != 0
        ):
            continue
        # This does not handle string tensors, but it's ok for our purposes
        tensor = ir.from_proto(tensor_proto)
        data = b"\0" * tensor.nbytes
        tensor_proto.raw_data = data


@unittest.skipIf(use_onnx_light, "textproto not supported yet")
class SerdeTest(unittest.TestCase):
    @parameterized.parameterized.expand(test_args)
    def test_serialization_deserialization_produces_same_model(
        self, _: str, model_path: pathlib.Path
    ) -> None:
        model = onnx.load(model_path)
        initialize_with_data(model)
        # Fix the missing graph name of some test models
        model.graph.name = "main_graph"
        onnx.checker.check_model(model)

        # Profile the serialization and deserialization process
        ir_model = ir.serde.deserialize_model(model)
        serialized = ir.serde.serialize_model(ir_model)

        onnx_ir.testing.assert_onnx_proto_equal(
            serialized, model, ignore_initializer_value_proto=True
        )
        onnx.checker.check_model(serialized)


if __name__ == "__main__":
    unittest.main()
