# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Turn an ONNX model into a textproto and strip all tensors.

Usage:
    python create_test_model.py <onnx_model_path>
"""

import argparse

from onnx_ir._onnx_compat import onnx


def strip_tensor_data(tensor: onnx.TensorProto) -> None:
    """Strip data from the tensor proto."""
    tensor.raw_data = b""
    del tensor.float_data[:]
    del tensor.int32_data[:]
    del tensor.int64_data[:]
    del tensor.string_data[:]
    del tensor.uint64_data[:]


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to textproto and strip tensor data."
    )
    parser.add_argument("onnx_model", type=str, help="Path to the ONNX model file.")
    args = parser.parse_args()

    output_path = args.onnx_model.replace(".onnx", ".textproto")

    # Load the ONNX model
    model = onnx.load(args.onnx_model, load_external_data=False)

    for tensor in model.graph.initializer:
        strip_tensor_data(tensor)

    # Save the model as a textproto
    onnx.save(model, output_path)
    print(f"Model saved as {output_path}")


if __name__ == "__main__":
    main()
