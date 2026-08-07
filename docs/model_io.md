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

## Use normalization and validation when needed

The IR permits temporarily incomplete or invalid states so multi-step
transformations can be expressed naturally. Prefer transformations that preserve
names, topological order, and type/shape information when possible. Normalization
and inference passes traverse the model, so do not add them to every save path by
default.

Apply only the operations required by the preceding transformations:

```python
import onnx_ir.passes.common as common_passes

# Only if a rewrite may have disturbed node order.
model = common_passes.TopologicalSortPass()(model).model

# Only if stale shapes/types are needed by a later stage.
model = common_passes.ShapeInferencePass()(model).model

# At an explicit validation boundary, if desired.
common_passes.CheckerPass(full_check=True)(model)

ir.save(model, "updated.onnx")
```

These stages serve different purposes:

- `TopologicalSortPass` stably sorts the main graph, nested subgraphs, and model
  functions. A cycle raises `ValueError`.
- `ShapeInferencePass` asks ONNX shape inference to update value types and shapes.
  If inference fails, it logs a warning and leaves the model unchanged.
- `CheckerPass` serializes through the ONNX boundary and runs
  `onnx.checker.check_model`; it does not modify the model.
- `ir.save` performs final serialization and external-data handling.

### Use symbolic shape inference

For richer symbolic inference, use the optional
[`onnx-shape-inference`](https://pypi.org/project/onnx-shape-inference/)
package. It operates directly on ONNX IR, represents dimension arithmetic with
SymPy expressions, propagates shape-tensor data through patterns such as
`Shape -> Slice -> Concat -> Reshape`, and supports custom operator inference
functions.

```console
pip install onnx-shape-inference
```

Use it instead of the built-in `ShapeInferencePass` when a later stage needs
richer symbolic relationships:

```python
from onnx_shape_inference import infer_symbolic_shapes

model = infer_symbolic_shapes(model)
```

The default `refine` merge policy preserves compatible existing information while
adding inferred details. Use `policy="strict"` to report conflicts between
declared and inferred shapes. Other policies and extension APIs are documented in
the [onnx-shape-inference project](https://github.com/justinchuby/onnx-shape-inference).

Shape and type information is not automatically recomputed after every graph edit.
Run inference when downstream transformations or consumers depend on updated
information, and use pass preconditions or postconditions for requirements that
are specific to your pipeline.

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
