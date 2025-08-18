#!/usr/bin/env python
from __future__ import annotations

import argparse
import tabulate
import onnx_ir as ir


def _create_io_row(value: ir.Value) -> list[str]:
    shape_text = str(value.shape) if value.shape is not None else "?"
    type_text = str(value.type) if value.type is not None else "?"
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
        "",
        f"<{type_text},{shape_text}>",
    ]


def _format_attributes_text(node: ir.Node) -> str:
    attr = [
        f"{k}={v.value!r}" if v.type != ir.AttributeType.GRAPH else f"{k}=GRAPH"
        for k, v in node.attributes.items()
    ]
    return "{" + ", ".join(attr) + "}"


def _create_node_row(node: ir.Node) -> list[str]:
    return [
        node.op_type if not node.domain else f"{node.domain}::{node.op_type}",
        "[" + ", ".join([v.name if v is not None else '""' for v in node.outputs]) + "]",
        "[" + ", ".join([v.name if v is not None else '""' for v in node.inputs]) + "] ",
        _format_attributes_text(node),
        str(node.name) if node.name else "",
    ]


def _create_header_row() -> list[str]:
    return [
        "Op",
        "Outputs",
        "InputsAttrs",
        "Name",
    ]


def main(path):
    model = ir.load(path)
    print(f"IR Version: {model.ir_version}")
    print(f"Model Producer: {model.producer_name} {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Opsets: {model.opset_imports}")
    print(f"Inputs: {len(model.graph.inputs)}")
    print(f"Outputs: {len(model.graph.outputs)}")
    print(f"Initializers: {len(model.graph.initializers)}")
    print(f"Nodes: {len(model.graph)}")
    print()
    rows = []
    for input in model.graph.inputs:
        rows.append(_create_io_row(input))
    for initializer in model.graph.initializers.values():
        rows.append(_create_io_row(initializer))
    for node in model.graph:
        rows.append(_create_node_row(node))
    for output in model.graph.outputs:
        rows.append(_create_io_row(output))
    print(
        tabulate.tabulate(
            rows,
            headers=_create_header_row(),
            maxcolwidths=[20, 40, 30, 30, 20],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print ONNX IR model information.")
    parser.add_argument("path", type=str, help="Path to the ONNX IR model file.")
    args = parser.parse_args()
    main(args.path)
