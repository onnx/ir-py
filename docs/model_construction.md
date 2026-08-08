# Constructing models

ONNX IR provides Python-native constructors for values, tensors, nodes, graphs,
and models. Construct objects directly when the graph structure is explicit, or
use {py:class}`onnx_ir.tape.Tape` when building a longer sequence of operations.

## Build a model directly

Create graph inputs and outputs with {py:func}`onnx_ir.val`. Static dimensions
are integers; strings create symbolic dimensions.

```python
import onnx_ir as ir

x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 4])
y = ir.val("y", dtype=ir.DataType.FLOAT, shape=["batch", 4])
```

Initializers are values with constant tensor data:

```python
bias = ir.val(
    "bias",
    const_value=ir.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=ir.DataType.FLOAT,
    ),
)
```

Create nodes with {py:func}`onnx_ir.node`. Plain Python attribute values are
converted to ONNX attributes automatically.

```python
add = ir.node("Add", inputs=[x, bias], name="add_bias")
relu = ir.node("Relu", inputs=add.outputs, outputs=[y], name="relu")
```

Assemble the graph and model:

```python
graph = ir.Graph(
    inputs=[x],
    outputs=[y],
    nodes=[add, relu],
    initializers=[bias],
    opset_imports={"": 21},
    name="main_graph",
)
model = ir.Model(graph, ir_version=10)
```

Values and nodes are connected by object identity. The strings used as names are
for serialization, diagnostics, and lookup; they do not create graph edges.

## Add nodes incrementally

Graph mutation methods establish ownership, assign missing names, and maintain
use-def relationships:

```python
old_output = graph.outputs[0]
old_output.name = "relu_output"
new_output = ir.val("y", type=old_output.type, shape=old_output.shape)

sigmoid = ir.node("Sigmoid", inputs=[old_output], outputs=[new_output])
graph.append(sigmoid)
graph.outputs[0] = new_output
```

A node can belong to only one graph. Remove it from its current graph before
moving it to another graph.

Use `graph.register_initializer(value)` to add an initializer after graph
construction. The value must be named, have `const_value`, and have no producer.

## Use Tape for sequential construction

{py:class}`onnx_ir.tape.Tape` records nodes and initializers while returning
values that can be passed directly to later operations:

```python
tape = ir.tape.Tape()

weight = tape.initializer(
    ir.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=ir.DataType.FLOAT,
        name="weight",
    )
)
x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 2])

matmul = tape.op("MatMul", [x, weight])
output = tape.op("Relu", [matmul])

graph = ir.Graph(
    inputs=[x],
    outputs=[output],
    nodes=tape.nodes,
    initializers=tape.initializers,
    opset_imports={"": 21},
    name="main_graph",
)
model = ir.Model(graph, ir_version=10)
```

Use `op_multi_out` for operators with multiple outputs. `tape.used_opsets`
records the domain and explicit version requested by each operation, but callers
still choose the graph's final opset imports.

## Preserve public interfaces

Graph input and output names are part of a model's external interface. Keep them
explicit when constructing or rewriting models. Intermediate names may be omitted
and assigned automatically when nodes enter a graph.

Run {py:class}`onnx_ir.passes.common.NameFixPass` only when construction may have
introduced missing or duplicate names:

```python
import onnx_ir.passes.common as common_passes

model = common_passes.NameFixPass()(model).model
```

## Preserve construction invariants

Construction APIs intentionally permit intermediate states that may not yet form
a valid ONNX model. Build nodes in topological order, keep public names unique,
and provide required type information directly when possible.

Use targeted passes only for invariants the construction process did not preserve.
See [Preserve invariants and use targeted repair](invariant-preservation)
for the authoritative checklist.

At an explicit validation boundary, the checker can confirm the completed model:

```python
common_passes.CheckerPass(full_check=True)(model)
```

See [Model I/O and external data workflows](model_io.md) for shape inference and
serialization options.
