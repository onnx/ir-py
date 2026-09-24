# Subgraphs and functions

ONNX represents control flow with graph-valued attributes and reusable
model-local operations with functions. Both contain nodes, but they have different
ownership, capture, and invocation semantics.

## Nested subgraphs

Operators such as `If`, `Loop`, and `Scan` store graphs in their attributes.
Subgraphs declare their own inputs, outputs, and initializers, and may also
reference values from an enclosing graph. These references are implicit captures
rather than declared subgraph inputs. Subgraph `opset_imports` are ignored;
control-flow subgraphs use the main graph's opset imports.

Use {py:class}`onnx_ir.traversal.RecursiveGraphIterator` to visit a graph and all
graphs nested in node attributes:

```python
import onnx_ir as ir

for node in ir.traversal.RecursiveGraphIterator(model.graph):
    inspect(node)
```

`model.graph.all_nodes()` is a convenient alias for the same basic traversal.
Use iterator callbacks when an analysis needs to know when traversal enters or
exits a graph scope.

## Analyze implicit captures

{py:func}`onnx_ir.analysis.analyze_implicit_usage` reports the outer-scope values
used by each nested graph:

```python
implicit_usage = ir.analysis.analyze_implicit_usage(model.graph)

for subgraph, captured_values in implicit_usage.items():
    print(subgraph.name, [value.name for value in captured_values])
```

Capture analysis is useful before extraction, cloning, moving a subgraph, or
rewriting the values visible to a control-flow body.

When cloning a subgraph independently, `graph.clone()` rejects outer-scope values
by default. Pass `allow_outer_scope_values=True` to preserve references to the
original enclosing values.

## Model-local functions

A {py:class}`onnx_ir.Function` defines a reusable operator identified by
`(domain, name, overload)`. A call node invokes it by using the same operator
identifier.

Functions own a graph internally but should normally be accessed through the
function object:

```python
for function in model.functions.values():
    print(function.identifier())
    for node in function:
        inspect(node)
```

Unlike graph-valued attributes, functions are stored separately on
`model.functions`; recursive traversal of the main graph does not visit them.
Whole-model transformations should process functions explicitly.

Functions can declare reference attributes whose values are supplied by call
nodes. They also carry their own opset imports, which must remain compatible when
their bodies are copied into another scope.

## Inline function calls

Use {py:class}`onnx_ir.passes.common.InlinePass` to replace calls with cloned
function bodies:

```python
import onnx_ir.passes.common as common_passes

result = common_passes.InlinePass()(model)
model = result.model
print(result.id_count)
```

By default, all eligible calls are inlined. Pass a `criteria` callback to select
which functions to inline. The pass rejects cyclic function dependencies and
opset conflicts rather than producing an ambiguous model.

`InlinePass` maintains graph connectivity, names, and ordering for the inlined
body. Do not add normalization passes solely because inlining occurred. Use a
targeted repair or validation pass only if another transformation in the pipeline
invalidated the corresponding invariant.

## Scope-aware transformation checklist

Before implementing a transformation, decide whether it applies to:

- only direct nodes in the main graph;
- the main graph and nested subgraphs;
- model-local functions and their subgraphs;
- every scope in the model.

Also account for implicit captures, graph outputs, and names inherited from
enclosing scopes. Opset imports are per main graph/function scope; subgraph
`opset_imports` are ignored.
