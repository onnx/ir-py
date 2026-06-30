# Migrating from `onnx.helper` to `onnx_ir` APIs

This page shows practical migrations from common `onnx.helper` model-building
patterns to `onnx_ir` APIs.

## Why migrate

`onnx_ir` keeps ONNX concepts (Model/Graph/Node/Value), but gives you:

- More convenient constructors (`ir.val`, `ir.node`, `ir.tensor`)
- Better graph mutation ergonomics
- Utilities for value replacement, extraction, and transformation workflows
- Direct serialization back to ONNX

## Mapping cheatsheet

| `onnx.helper` pattern | `onnx_ir` pattern |
|---|---|
| `make_tensor_value_info` | `ir.val(name, dtype=..., shape=...)` |
| `make_tensor` | `ir.tensor(...)` |
| `make_node` | `ir.node(op_type, inputs, attributes=...)` |
| `make_graph` | `ir.Graph(inputs=..., outputs=..., nodes=..., initializers=...)` |
| `make_model` | `ir.Model(graph, ir_version=...)` |
| `onnx.save(model_proto, path)` | `ir.save(model, path)` |

## Example 1: Build a small model from scratch

### With `onnx.helper`

```python
import onnx
from onnx import TensorProto

x = onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
y = onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
out = onnx.helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])

bias = onnx.helper.make_tensor(
    "bias",
    TensorProto.FLOAT,
    dims=[2, 3],
    vals=[1.0] * 6,
)

add = onnx.helper.make_node("Add", inputs=["x", "bias"], outputs=["tmp"], name="add_bias")
relu = onnx.helper.make_node("Relu", inputs=["tmp"], outputs=["out"], name="relu")

graph = onnx.helper.make_graph(
    [add, relu],
    "g",
    inputs=[x, y],
    outputs=[out],
    initializer=[bias],
)
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])
```

### With `onnx_ir`

```python
import onnx_ir as ir

x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])
out = ir.val("out", dtype=ir.DataType.FLOAT, shape=[2, 3])

bias_tensor = ir.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], name="bias")
bias = ir.val("bias", const_value=bias_tensor)

add = ir.node("Add", inputs=[x, bias], name="add_bias")
relu = ir.node("Relu", inputs=add.outputs, outputs=[out], name="relu")

graph = ir.Graph(
    inputs=[x],
    outputs=[out],
    nodes=[add, relu],
    initializers=[bias],
    opset_imports={"": 20},
    name="g",
)
model = ir.Model(graph, ir_version=10)

ir.save(model, "model.onnx")
```

## Example 2: Create nodes with Python attributes directly

With `onnx.helper`, attributes often require explicit helper calls.
With `ir.node`, plain Python values are converted automatically.

```python
import onnx_ir as ir

x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1, 3, 8, 8])

conv = ir.node(
    "Conv",
    inputs=[x, ir.val("w"), ir.val("b")],
    attributes={
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "strides": [1, 1],
        "group": 1,
    },
    name="conv0",
)
```

## Example 3: Graph rewrite (replace a node output)

This is a common migration pain-point when using protobuf-level APIs directly.

```python
import onnx_ir as ir

model = ir.load("model.onnx")
graph = model.graph

# Suppose we replace a node producing old_out with a new node producing new_out.
old_node = next(node for node in graph if node.name == "old_node")
inp = old_node.inputs[0]
new_node = ir.node("Identity", [inp], name="new_node")
graph.insert_after(old_node, [new_node])

# Redirect all downstream users and graph outputs.
ir.convenience.replace_all_uses_with(
    old_node.outputs,
    new_node.outputs,
    replace_graph_outputs=True,
)

graph.remove([old_node], safe=True)
ir.save(model, "rewritten.onnx")
```

## Example 4: Extract a bounded subgraph

```python
import onnx_ir as ir

model = ir.load("model.onnx")

subgraph = ir.convenience.extract(
    model.graph,
    inputs=["input_0", "weight_0"],
    outputs=["layer3_out"],
)

submodel = ir.Model(subgraph, ir_version=model.ir_version)
ir.save(submodel, "subgraph.onnx")
```

## Migration tips

1. Start by replacing `make_tensor_value_info`/`make_node` with `ir.val`/`ir.node`.
2. Keep names explicit while migrating to preserve external interfaces.
3. Prefer value-based rewrites (`replace_all_uses_with`) over positional list surgery.
4. Use `ir.save`/`ir.load` at the boundaries and keep transformation logic in IR.
