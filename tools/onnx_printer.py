#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""This script prints information about an ONNX IR model in a tabular format.

Usage:
    python onnx_printer.py <path_to_model> [--inline] [--no_wrap]

Example:
    python onnx_printer.py model.onnx --no_wrap > model_info.txt
"""

from __future__ import annotations

import argparse

import tabulate

import onnx_ir as ir
import onnx_ir.passes.common


def _create_io_row(value: ir.Value) -> list[str]:
    if value.is_graph_input():
        type = "Input"
    elif value.is_graph_output():
        type = "Output"
    elif value.is_initializer():
        type = "Initializer"
    else:
        raise ValueError(f"{value} is not a graph input, output, or initializer")

    return [
        type,
        value.name,
        str(value.const_value) if value.const_value is not None else str(value),
        "",
    ]


def _format_attributes_text(node: ir.Node) -> str:
    attr = [
        f"{k}={v.value!r}"
        if v.type != ir.AttributeType.GRAPH
        else f"{k}=GRAPH('{v.value.name}')"
        for k, v in node.attributes.items()
    ]
    return "{" + ", ".join(attr) + "}"


def _create_node_row(node: ir.Node) -> list[str]:
    return [
        node.op_type if not node.domain else f"{node.domain}::{node.op_type}",
        "[" + ", ".join([v.name if v is not None else '""' for v in node.outputs]) + "]",
        "[" + ", ".join([v.name if v is not None else '""' for v in node.inputs]) + "] ",
        _format_attributes_text(node),
    ]


def _create_header_row() -> list[str]:
    return [
        "Op",
        "Outputs",
        "Inputs",
        "Attrs",
    ]


def main(path: str, inline: bool, wrap: bool) -> None:
    model = ir.load(path)
    if inline:
        onnx_ir.passes.common.InlinePass()(model)
    print(f"IR Version: {model.ir_version}")
    print(f"Model Producer: {model.producer_name} {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Opsets: {model.opset_imports}")
    print(f"Inputs: {len(model.graph.inputs)}")
    print(f"Outputs: {len(model.graph.outputs)}")
    print(f"Initializers: {len(model.graph.initializers)}")
    print(f"Nodes (including subgraphs): {len(tuple(model.graph.all_nodes()))}")

    for graph in model.graphs():
        print()
        if graph is model.graph:
            print(f"Graph: {graph.name}")
        else:
            print(f"Subgraph: {graph.name}")
        rows = []
        for input in graph.inputs:
            rows.append(_create_io_row(input))
        for initializer in graph.initializers.values():
            rows.append(_create_io_row(initializer))
        for node in graph:
            rows.append(_create_node_row(node))
        for output in graph.outputs:
            rows.append(_create_io_row(output))

        print(
            tabulate.tabulate(
                rows,
                headers=_create_header_row(),
                maxcolwidths=[20, 20, 30, 25] if wrap else None,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print ONNX IR model information.")
    parser.add_argument("path", type=str, help="Path to the ONNX IR model file.")
    parser.add_argument(
        "--inline", action="store_true", help="Inline all functions before printing."
    )
    parser.add_argument(
        "--no_wrap", action="store_true", help="Wrap long lines in the output."
    )
    args = parser.parse_args()
    main(args.path, inline=args.inline, wrap=not args.no_wrap)
