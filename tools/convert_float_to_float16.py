#!/usr/bin/env python3
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Example script demonstrating float16 conversion.

This script shows how to use the ConvertFloatToFloat16Pass to convert
an ONNX model from float32 to float16.
"""

import argparse
import sys

import onnx_ir as ir
from onnx_ir.passes.common import ConvertFloatToFloat16Pass


def main():
    """Convert an ONNX model from float32 to float16."""
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model from float32 to float16"
    )
    parser.add_argument("input", help="Input ONNX model path")
    parser.add_argument("output", help="Output ONNX model path")
    parser.add_argument(
        "--keep-io-types",
        action="store_true",
        help="Keep input/output types as float32",
    )
    parser.add_argument(
        "--min-positive-val",
        type=float,
        default=1e-7,
        help="Minimum positive value for clamping (default: 1e-7)",
    )
    parser.add_argument(
        "--max-finite-val",
        type=float,
        default=1e4,
        help="Maximum finite value for clamping (default: 1e4)",
    )
    parser.add_argument(
        "--op-block-list",
        nargs="*",
        help="List of operator types to not convert (default: uses built-in list)",
    )
    parser.add_argument(
        "--node-block-list",
        nargs="*",
        help="List of node names to not convert",
    )

    args = parser.parse_args()

    # Load the model
    print(f"Loading model from {args.input}...")
    try:
        model = ir.load(args.input)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    # Create the conversion pass
    pass_kwargs = {
        "min_positive_val": args.min_positive_val,
        "max_finite_val": args.max_finite_val,
        "keep_io_types": args.keep_io_types,
    }

    if args.op_block_list is not None:
        pass_kwargs["op_block_list"] = args.op_block_list

    if args.node_block_list:
        pass_kwargs["node_block_list"] = args.node_block_list

    conversion_pass = ConvertFloatToFloat16Pass(**pass_kwargs)

    # Apply the conversion
    print("Converting model to float16...")
    result = conversion_pass(model)

    if not result.modified:
        print(
            "Warning: Model was not modified. It may already be in float16 or have no float32 tensors."
        )

    # Save the converted model
    print(f"Saving converted model to {args.output}...")
    try:
        ir.save(result.model, args.output)
    except Exception as e:
        print(f"Error saving model: {e}", file=sys.stderr)
        return 1

    print("Conversion complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
