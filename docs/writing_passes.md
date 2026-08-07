# Writing transformation passes

Passes package graph transformations as composable operations over an
{py:class}`onnx_ir.Model`. A pass returns a
{py:class}`onnx_ir.passes.PassResult` containing the resulting model and a
`modified` flag.

Use a pass when a transformation should be reusable, testable, composed with
other transformations, or guarded by explicit preconditions and postconditions.
For individual graph-editing operations, see
[Graph transformation patterns](graph_transformations.md).

## Choose a pass type

Most transformations should inherit from
{py:class}`onnx_ir.passes.InPlacePass`. An in-place pass mutates and returns the
same model object.

Use {py:class}`onnx_ir.passes.FunctionalPass` when the input model must remain
unchanged. A functional pass must return a different model object, typically by
cloning before transformation:

```python
import onnx_ir as ir


class FunctionalRewrite(ir.passes.FunctionalPass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        new_model = model.clone()
        # Transform new_model.
        return ir.passes.PassResult(new_model, modified=True)
```

An existing in-place pass can also be converted into a functional pass with
{py:func}`onnx_ir.passes.functionalize`.

## Pass lifecycle

The pass infrastructure calls three methods in order:

1. `requires(model)` checks preconditions.
2. `call(model)` performs the transformation and returns `PassResult`.
3. `ensures(model)` checks postconditions on the result.

`requires` and `ensures` are optional. Raise
{py:class}`onnx_ir.passes.PreconditionError` or
{py:class}`onnx_ir.passes.PostconditionError` when an invariant is not satisfied.
Other exceptions from these methods are wrapped in the corresponding error type.

```python
class RequiresOpset13(ir.passes.InPlacePass):
    def requires(self, model: ir.Model) -> None:
        if model.graph.opset_imports.get("", 0) < 13:
            raise ir.passes.PreconditionError("Requires the default opset to be >= 13")

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        # Transform model and update modified.
        return ir.passes.PassResult(model, modified=modified)
```

The framework verifies that the returned model identity matches the declared pass
type. It raises {py:class}`onnx_ir.passes.PassError` if an in-place pass returns a
new model or a functional pass returns its input model.

## Transform every applicable scope

{py:class}`onnx_ir.traversal.RecursiveGraphIterator` visits a graph and the
subgraphs stored in node attributes. It does not implicitly process model-local
functions, which are stored separately.

```python
def iter_model_nodes(model: ir.Model):
    yield from ir.traversal.RecursiveGraphIterator(model.graph)
    for function in model.functions.values():
        yield from ir.traversal.RecursiveGraphIterator(function)
```

Not every pass should recurse. Decide explicitly whether the transformation
applies only to the main graph, to nested subgraphs, to functions, or to all of
them. Also account for optional node inputs, which are represented by `None`.

## Example: eliminate Identity nodes

This in-place pass processes the main graph, nested subgraphs, and model-local
functions. Graph iteration remains safe while matching nodes are removed.

```python
class IdentityEliminationPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for node in iter_model_nodes(model):
            if node.domain != "" or node.op_type != "Identity":
                continue
            if len(node.inputs) != 1 or node.inputs[0] is None:
                continue
            if len(node.outputs) != 1:
                continue

            node.outputs[0].replace_all_uses_with(
                node.inputs[0],
                replace_graph_outputs=True,
            )
            assert node.graph is not None
            node.graph.remove(node, safe=True)
            modified = True

        return ir.passes.PassResult(model, modified=modified)
```

When replacing or removing values, preserve required type, shape, name, constant,
and metadata information. Use `safe=True` for node removal so dangling consumers
or graph outputs are reported rather than silently producing an invalid graph.

## Analysis metadata and invalidation

IR objects expose `meta` for arbitrary analysis state that is not serialized.
When a transformation changes information an analysis depends on, update the
cached value or mark it invalid:

```python
value.meta.invalidate("shape_analysis")
```

The validity of a metadata entry is defined by the analysis that owns it. A value
of `None` may still be a valid result, so use `meta.is_valid(key)` rather than
interpreting the stored value itself as a validity flag.

Use `metadata_props` only for string key-value metadata that should be serialized
into the ONNX model.

## Compose passes

Use {py:class}`onnx_ir.passes.Sequential` to run passes once in order:

```python
import onnx_ir.passes.common as common_passes

pipeline = ir.passes.Sequential(
    IdentityEliminationPass(),
    common_passes.TopologicalSortPass(),
    common_passes.CheckerPass(),
)
result = pipeline(model)
```

Use {py:class}`onnx_ir.passes.PassManager` to repeat a sequence for a bounded
number of steps and optionally stop when no pass reports a modification:

```python
pipeline = ir.passes.PassManager(
    [
        IdentityEliminationPass(),
        common_passes.CommonSubexpressionEliminationPass(),
    ],
    steps=5,
    early_stop=True,
)
result = pipeline(model)
```

The `modified` flag is part of the pass contract. Report it accurately so pass
managers can detect convergence and callers can avoid unnecessary work.

## Testing a pass

Tests should cover both the transformation and its contract:

- matching and non-matching graphs;
- graph inputs, outputs, initializers, and optional inputs;
- nested subgraphs and functions when supported;
- accurate `modified` values;
- required precondition and postcondition failures;
- preservation of names, types, shapes, and metadata;
- topological order and successful ONNX checking when the output is expected to
  be a valid model.

Prefer small, directly constructed graphs that make ownership and use-def
relationships explicit. Compare object identity when checking whether a value was
rewired; names alone are not sufficient.
