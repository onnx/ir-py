# Portions of this file are derived from work by Microsoft Corporation under the MIT License. This was modified by bjeffrey92 for use in ir-py.
# See below for original license and copyright.

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSES/MIT.txt in the project root for
# license information.
# Original source: https://github.com/microsoft/onnxconverter-common/blob/209a47e18e6a4c3474273a0b2a5e8f1fda481643/tests/test_float16.py

import os
import typing

import numpy as np
import onnx
import onnxruntime as _ort
from onnx import onnx_pb as onnx_proto

from onnx_ir.passes.common import onnx_float_16


def _ort_inference(
    mdl: onnx_proto.ModelProto, inputs: dict[str, typing.Any]
) -> typing.Sequence[typing.Any]:
    sess = _ort.InferenceSession(mdl.SerializeToString())
    return sess.run(None, inputs)


def test_convert_to_float16() -> None:
    model32_name = "image_classifier32.onnx"
    working_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(working_path, "data")
    model_path = os.path.join(data_path, model32_name)
    onnx_model32 = onnx.load(model_path)
    input_x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    output_32 = _ort_inference(onnx_model32, {"modelInput": input_x})

    onnx_model16 = onnx_float_16.convert_float_to_float16(onnx_model32, keep_io_types=False)
    output_16 = _ort_inference(onnx_model16, {"modelInput": input_x.astype(np.float16)})
    assert np.allclose(output_16, output_32, atol=1e-2)

    onnx_model16 = onnx_float_16.convert_float_to_float16(onnx_model32, keep_io_types=True)
    output_16 = _ort_inference(onnx_model16, {"modelInput": input_x})
    assert np.allclose(output_16, output_32, atol=1e-2)


def test_convert_to_float16_with_truncated() -> None:
    np_array = np.array([1e-10, -2.0, 15, -1e-9, 65536.1, -100000])
    onnx_float_16.convert_np_to_float16(np_array)


def test_convert_to_float16_with_subgraph() -> None:
    model32_name = "test_subgraph.onnx"
    working_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(working_path, "data")
    model_path = os.path.join(data_path, model32_name)
    onnx_model32 = onnx.load(model_path)
    x = np.array([1.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    output_32 = _ort_inference(onnx_model32, {"x": x, "y": y})

    onnx_model16 = onnx_float_16.convert_float_to_float16(onnx_model32, keep_io_types=True)
    actual = _ort_inference(onnx_model16, {"x": x, "y": y})
    assert np.allclose(actual, output_32, atol=1e-2)
    assert actual[0].dtype == np.float32

    onnx_model16 = onnx_float_16.convert_float_to_float16(onnx_model32, keep_io_types=False)
    actual = _ort_inference(
        onnx_model16, {"x": x.astype(np.float16), "y": y.astype(np.float16)}
    )
    assert np.allclose(actual, output_32, atol=1e-2)
    assert actual[0].dtype == np.float16
