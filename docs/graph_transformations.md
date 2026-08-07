# Graph transformation patterns

This page documents practical graph-editing patterns exposed by `onnx_ir` source
APIs, especially those in `onnx_ir.convenience`, `onnx_ir.traversal`, and
`onnx_ir.analysis`.

## Choose the scopes to transform

Direct iteration processes only one graph. Use recursive traversal for nested
graph attributes, and process model-local functions separately:

```python
import onnx_ir as ir


def iter_model_nodes(model: ir.Model):
    yield from ir.traversal.RecursiveGraphIterator(model.graph)
    for function in model.functions.values():
        yield from ir.traversal.RecursiveGraphIterator(function)
```

Not every rewrite should apply to every scope. Decide explicitly whether the
transformation supports control-flow subgraphs, implicit captures, and functions.
When replacing values, use `replace_graph_outputs=True` if graph outputs should
also be redirected.

## Mutate a graph during iteration

It is safe to insert, remove, or move nodes while iterating over a graph. The
iterator preserves its position using the node's original location:

- Nodes inserted after the current node are visited during the same iteration.
- Nodes inserted before the current node are not visited during that iteration.
- If the current node is removed or moved, iteration continues from the node that
  followed it at its original location.

For example, this loop can replace nodes without first copying the graph's node
list:

```python
import onnx_ir as ir

for node in graph:
    if node.op_type != "Dropout":
        continue

    replacement = ir.node("Identity", inputs=node.inputs)
    graph.insert_after(node, [replacement])
    node.outputs[0].replace_all_uses_with(
        replacement.outputs[0],
        replace_graph_outputs=True,
    )
    graph.remove([node], safe=True)
```

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
    graph,
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
inputs and outputs. Inputs and outputs may be specified as `Value` objects or by
name.

```python
subgraph = ir.convenience.extract(
    model.graph,
    inputs=["x", "w"],
    outputs=["y"],
)
```

Extraction walks backward from the requested outputs until it reaches the
requested inputs. It:

- includes the nodes required to compute the outputs, preserving their original
  order;
- includes required initializers automatically;
- follows outer-scope values captured by nested graph attributes;
- preserves the source name, documentation, opset imports, and serialized
  metadata;
- returns an independent cloned {py:class}`onnx_ir.Graph`.

The requested inputs must fully bound the extracted region. If a required
non-initializer value enters the region but is not listed in `inputs`, extraction
raises `ValueError` rather than creating a graph with an undeclared dependency.
At least one output is required, and supplied `Value` objects must belong to the
source graph unless the source is a {py:class}`onnx_ir.GraphView`.

## Preserve invariants and use targeted repair

A well-formed transformation should maintain the invariants it does not intend to
change, including use-def links, unique required names, and topological order.
Avoid routinely running cleanup passes after every rewrite: each pass adds another
model traversal and may be expensive for large models.

Use a normalization or analysis pass only when the transformation's contract
requires it:

- Run `NameFixPass` if the rewrite can introduce missing or duplicate names.
- Run `TopologicalSortPass` if nodes may no longer be in topological order.
- Run shape inference only when the rewrite invalidates shape/type information and
  a later stage requires it.
- Run `CheckerPass` at explicit validation boundaries or while diagnosing a
  transformation, rather than after every pass.

Document these effects in reusable passes so callers know which invariants remain
valid and which follow-up work, if any, is necessary.
