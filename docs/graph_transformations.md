# Graph transformation patterns

This page documents practical graph-editing patterns exposed by `onnx_ir` source
APIs, especially those in `onnx_ir.convenience`, `onnx_ir.traversal`, and
`onnx_ir.analysis`.

## Replace all downstream uses of a value

Use `ir.convenience.replace_all_uses_with` when replacing one value-producing
node with another.

```python
import onnx_ir as ir

model = ir.load("model.onnx")
graph = model.graph

target = next(node for node in graph if node.op_type == "Relu")
new_node = ir.node("Identity", inputs=target.inputs, name="relu_replacement")
graph.insert_after(target, [new_node])

ir.convenience.replace_all_uses_with(
    target.outputs,
    new_node.outputs,
    replace_graph_outputs=True,
)
graph.remove([target], safe=True)
```

## Replace a node block with a new block

Use `replace_nodes_and_values` when multiple old/new nodes and outputs need to be
rewired as one operation.

```python
ir.convenience.replace_nodes_and_values(
    graph_or_function=graph,
    insertion_point=anchor_node,
    old_nodes=[old_a, old_b],
    new_nodes=[new_x, new_y],
    old_values=[old_b.outputs[0]],
    new_values=[new_y.outputs[0]],
)
```

## Rename values safely (including initializers)

`rename_values` handles initializer rename corner cases and collisions.

```python
ir.convenience.rename_values(
    values=[graph.initializers["w0"], graph.initializers["w1"]],
    names=["encoder.w0", "encoder.w1"],
)
```

## Iterate recursively over nested subgraphs

Use `traversal.RecursiveGraphIterator` to process control-flow bodies (`If`,
`Loop`, etc.) in one pass.

```python
import onnx_ir as ir

for node in ir.traversal.RecursiveGraphIterator(model.graph):
    if node.op_type == "Dropout":
        node.attributes["ratio"] = ir.AttrFloat32("ratio", 0.0)
```

## Analyze implicit captures in subgraphs

`analysis.analyze_implicit_usage` helps detect outer-scope values captured by
nested subgraphs.

```python
import onnx_ir as ir

implicit = ir.analysis.analyze_implicit_usage(model.graph)
for subgraph, captured in implicit.items():
    print(subgraph.name, [v.name for v in captured])
```

## Extract a bounded subgraph

Use `convenience.extract` to carve out a model region with explicit frontier
inputs/outputs.

```python
subgraph = ir.convenience.extract(
    model.graph,
    inputs=["x", "w"],
    outputs=["y"],
)
```
