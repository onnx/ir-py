# Model I/O and external data workflows

This page covers user-facing I/O features implemented in `onnx_ir._io`,
`onnx_ir.external_data`, and `onnx_ir._safetensors`.

## Load and save a model

```python
import onnx_ir as ir

model = ir.load("model.onnx")
# ... mutate model ...
ir.save(model, "updated.onnx")
```

## Save large initializers as ONNX external data

```python
ir.save(
    model,
    "model.onnx",
    external_data="model.data",       # relative to model path
    size_threshold_bytes=1024,        # externalize tensors >= 1KB
    max_shard_size_bytes=256 * 1024,  # optional sharding
)
```

Notes:

- `external_data` must be a **relative** path.
- `max_shard_size_bytes` requires `external_data`.
- Single-file mode overwrites destination data file.
- Sharded mode is stricter and can raise `FileExistsError` for collisions.

## Save external data with progress callback

```python
def callback(tensor, info):
    print(f"[{info.index + 1}/{info.total}] {info.filename} :: {tensor.name}")

ir.save(
    model,
    "model.onnx",
    external_data="model.data",
    size_threshold_bytes=0,
    callback=callback,
)
```

## Save with safetensors backend

```python
ir.save_safetensors(
    model,
    "model.onnx",
    size_threshold_bytes=0,
    max_shard_size_bytes=5 * 1000**3,
)
```

`save_safetensors` writes model weights to `.safetensors` side files while
keeping the ONNX graph in `model.onnx`.

## Important safetensors constraints

- All initializer names across graphs/subgraphs must be unique.
- Tensor attributes in nodes are not externalized to safetensors.
- For large constant nodes you want externalized, consider lifting constants to
  initializers before saving.
