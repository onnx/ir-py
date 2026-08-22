# Performance and large-model workflows

ONNX IR is designed to manipulate large models without eagerly materializing all
tensor data or repeatedly converting between Python objects and protobuf.

## Keep models in IR form

Load once, perform a sequence of analyses and transformations in memory, and
serialize at the boundary:

```python
import onnx_ir as ir

model = ir.load("model.onnx")

for pass_ in passes:
    model = pass_(model).model

ir.save(model, "optimized.onnx")
```

Avoid converting to `ModelProto` between passes. Serialization walks the model
and may materialize or copy data depending on the tensor backend.

## Avoid materializing large tensors

{py:class}`onnx_ir.ExternalTensor` uses memory mapping for ONNX external data.
Inspect `shape`, `dtype`, `size`, and `nbytes` before calling `numpy()` or
`tobytes()`, because those operations access the tensor data.

Prefer {py:class}`onnx_ir.TensorProtocol` in APIs so callers can provide in-memory,
external, lazy, packed, or framework-backed tensors without an eager conversion.

See [Tensor Representation in the IR](tensors.md) and
[Model I/O and external data workflows](model_io.md).

## Choose traversal deliberately

- `len(graph)`, appending, inserting, and removing nodes are efficient operations.
- Accessing the first or last graph node is constant time; arbitrary indexing is
  linear time.
- Use direct graph iteration when only one scope matters.
- Use `RecursiveGraphIterator` once when nested subgraphs matter instead of
  repeatedly rediscovering them.
- Convert nodes to a list or name mapping only when repeated random access
  justifies the additional memory.

Combine related analyses in a single traversal when practical, but keep
transformations understandable and preserve accurate invalidation of cached
metadata.

## Prefer in-place passes

{py:class}`onnx_ir.passes.InPlacePass` avoids cloning the model and is the usual
choice for transformations. Use functional passes only when preserving the input
model is part of the API contract.

`model.clone()` creates new graphs, nodes, and values but shares initializer and
constant tensors. `deep_copy=True` additionally copies analysis metadata; it does
not imply copying tensor storage.

{py:class}`onnx_ir.GraphView` is cheaper than cloning when an analysis only needs
a fixed view over existing nodes.

## Batch graph edits

Use value-based rewrites and batch helpers rather than repeated global searches:

- inspect `Value.producer()`, `Value.uses()`, and `Value.consumers()`;
- use `replace_all_uses_with` for rewiring;
- remove multiple known nodes in one `graph.remove` call;
- use `create_value_mapping` once when many name lookups are required.

Graph iterators support mutation, so transformations do not need to copy
`list(graph)` merely to remove or insert nodes safely.

## Save large weights efficiently

Use `ir.save(..., external_data=...)` to keep large tensors outside the protobuf
file. Set `max_shard_size_bytes` when storage systems impose file-size limits.

Use `ir.save_safetensors` when safetensors is the desired external-data format.
This requires `safetensors>=0.7.0` and unique initializer names across graphs and
subgraphs.

## Measure representative models

Performance depends on graph size, tensor storage, filesystem behavior, and the
number of full-model traversals. Benchmark the complete pipeline on representative
models, including load and save when they are part of the production path.
