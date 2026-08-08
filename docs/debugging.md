# Debugging transformations

ONNX IR provides readable object displays, ONNX text conversion, mutation
journaling, and construction tapes for inspecting model transformations.

## Print or display IR objects

`Model`, `Graph`, `Node`, `Value`, and tensor objects have compact string
representations:

```python
print(model.graph)
print(node)
```

Call `display()` for syntax-highlighted terminal output when `rich` is installed:

```console
pip install rich
```

```python
model.display()
model.graph.display(page=True)
```

`page=True` opens a terminal pager, which is useful for large graphs.

## Use ONNX text format

Convert a model to the standard ONNX textual syntax when comparing structure with
other ONNX tools:

```python
text = ir.to_onnx_text(model)
print(text)

round_tripped = ir.from_onnx_text(text)
```

Pass `exclude_initializers=True` to `to_onnx_text` when large constant values
would obscure the graph structure. Text conversion serializes through ONNX and is
best used for diagnostics or interchange rather than inside performance-sensitive
transformation loops.

## Record mutations with a journal

The alpha {py:mod}`onnx_ir.journaling` API records supported IR operations,
including stack traces and object references:

```python
from onnx_ir.journaling import Journal

with Journal() as journal:
    transform(model)

journal.display()
```

Inspect `journal.entries` to filter by operation or class, call
`entry.display()` for full details, or attach a hook for real-time logging:

```python
journal = Journal()
journal.add_hook(lambda entry: print(entry.operation, entry.class_name))
```

Keep the journal scope focused around the suspicious transformation. Recording a
large pipeline creates substantial diagnostic output and stack-trace data.

## Inspect graph relationships

Use object relationships rather than names when diagnosing connectivity:

```python
print(value.producer())
print(list(value.uses()))
print(list(value.consumers()))
print(node.predecessors())
print(node.successors())
print(node.graph)
```

For nested graphs, use `RecursiveGraphIterator` and
`analysis.analyze_implicit_usage` to reveal control-flow bodies and outer-scope
captures.

## Isolate a failing region

{py:func}`onnx_ir.convenience.extract` can clone a bounded region into a smaller
graph for inspection or reproduction:

```python
region = ir.convenience.extract(
    model.graph,
    inputs=["x", "weight"],
    outputs=["y"],
)
region.display()
```

Extraction reports undeclared frontier dependencies instead of silently creating
an incomplete graph.

## Validate between pipeline stages

When a long pass pipeline fails, insert focused checks to locate the first invalid
stage:

```python
import onnx_ir.passes.common as common_passes

checker = common_passes.CheckerPass()

for pass_ in passes:
    result = pass_(model)
    model = result.model
    checker(model)
```

This per-stage checking is a diagnostic technique, not a recommended production
pipeline: it serializes and checks the whole model after each pass.

Topological sorting, name fixing, shape inference, and ONNX checking test different
properties. Apply the checks relevant to the invariant each stage promises.

## Use Tape to reproduce construction bugs

{py:class}`onnx_ir.tape.Tape` records nodes and initializers while constructing a
graph. It is useful for reducing a failing model to a short sequence of operations
that can be copied into a regression test.

See [Constructing models](model_construction.md) for a complete example.
