# Multi-Device Configurations

ONNX IR version 11 introduced metadata for describing how a model is partitioned
across multiple devices: model-level **device configurations** and per-node
**sharding specifications** and **pipeline stages**. The IR exposes this metadata
in an *object-bound* form, so it integrates naturally with the rest of the graph.

## Object-bound by design

In the ONNX protobuf, sharding metadata refers to tensors and configurations by
*name* (`tensor_name`, `configuration_id`). The IR instead binds directly to the
{py:class}`onnx_ir.Value <onnx_ir.Value>` and
{py:class}`onnx_ir.ModelConfiguration <onnx_ir.ModelConfiguration>` **objects**.
The proto name strings are *derived* from `value.name` and `configuration.name`
when you serialize.

This has two practical consequences:

- **Single source of truth.** There is no second copy of the name to keep in
  sync, so references cannot silently drift out of date.
- **References follow renames.** Renaming a value or reassigning it updates every
  sharding spec that points at it, automatically.

## A quick example

The convenience API is the recommended way to attach multi-device metadata.
{py:meth}`Model.add_device_configuration <onnx_ir.Model.add_device_configuration>`
declares a configuration, and
{py:meth}`Node.shard <onnx_ir.Node.shard>` records a sharding of one of a node's
inputs or outputs.

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    # Build a tiny model: x -> Relu -> y
    x = ir.Value(name="x", shape=ir.Shape([8, 16]), type=ir.TensorType(ir.DataType.FLOAT))
    relu = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="relu0")
    graph = ir.Graph([x], [relu.outputs[0]], nodes=[relu], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)

    # Declare a configuration with two devices and shard ``x`` along axis 0.
    conf = model.add_device_configuration("conf0", device_names=("CPU", "CUDA:0"))
    relu.shard(x, configuration=conf, axis=0, num_shards=2, device_indices=(0, 1))

    # The sharding is bound to the value object, so it follows renames.
    x.name = "input"
    spec = relu.sharding_of(x)[0]
    print(spec.value.name)         # input
    print(spec.value is x)         # True
```

{py:meth}`Node.sharding_of <onnx_ir.Node.sharding_of>` returns the live sharding
specs that target a particular value (matched by object identity). Calling
`shard` again with the same `configuration` reuses its
{py:class}`~onnx_ir.NodeDeviceConfiguration` rather than creating a new one.

## Patterns at a glance

| Goal | How | See section |
| --- | --- | --- |
| Split a tensor along one axis | `node.shard(value, configuration=…, axis=…, num_shards=…)` | *Common sharding patterns* |
| Split across a 2D device mesh | call `shard` once per axis for the same value | *2D device mesh* |
| Replicate a tensor across devices | `ShardingSpec` with a device-group key and empty `sharded_dims` | *Replication across device groups* |
| Mix split + replication | `ShardingSpec` with device-group keys and `sharded_dims` | *Replication across device groups* |
| Place whole subgraphs on a device (pipeline) | `node.set_pipeline_stage(configuration, stage)` | *Pipeline parallelism* |
| Both shard and place a node | `shard` + `set_pipeline_stage` (same configuration) | *End-to-end* |
| Read a node's placement / sharding back | iterate `node.device_configurations`, `node.sharding_of(value)` | *Querying shardings* |

## Common sharding patterns

The examples below build a small `W @ x` matmul and show how typical shardings
map onto the IR. They follow the same vocabulary as systems like
[Shardy](https://openxla.org/shardy/sharding_representation): a tensor dimension
is *sharded* across some devices and *replicated* along the rest.

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    # W: [1024, 4096], x: [4096, 8]  ->  y: [1024, 8]
    w = ir.Value(name="W", shape=ir.Shape([1024, 4096]), type=ir.TensorType(ir.DataType.FLOAT))
    x = ir.Value(name="x", shape=ir.Shape([4096, 8]), type=ir.TensorType(ir.DataType.FLOAT))
    matmul = ir.Node("", "MatMul", [w, x], outputs=[ir.Value(name="y")], name="mm")
    graph = ir.Graph([w, x], [matmul.outputs[0]], nodes=[matmul], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)

    # --- 1D mesh: 4 devices ---
    mesh1d = model.add_device_configuration("mesh1d", num_devices=4)

    # Row-parallel: shard W along axis 0 (its rows) across all 4 devices.
    # Each device holds a [256, 4096] shard; x is replicated.
    matmul.shard(w, configuration=mesh1d, axis=0, num_shards=4, device_indices=(0, 1, 2, 3))

    # Column-parallel would instead shard W along axis 1:
    #   matmul.shard(w, configuration=mesh1d, axis=1, num_shards=4, ...)

    spec = matmul.sharding_of(w)[0]
    print(spec.sharded_dims[0].axis)                       # 0
    print(spec.sharded_dims[0].simple_shardings[0].num_shards)  # 4
```

### 2D device mesh

To shard a tensor across a 2-axis mesh (for example a ``2 x 2`` grid of 4
devices), shard the same value along each axis. The calls merge into a single
{py:class}`~onnx_ir.ShardingSpec` with one {py:class}`~onnx_ir.ShardedDim` per
axis — the canonical representation for a multi-axis mesh.

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    w = ir.Value(name="W", shape=ir.Shape([1024, 4096]), type=ir.TensorType(ir.DataType.FLOAT))
    x = ir.Value(name="x", shape=ir.Shape([4096, 8]), type=ir.TensorType(ir.DataType.FLOAT))
    matmul = ir.Node("", "MatMul", [w, x], outputs=[ir.Value(name="y")], name="mm")
    graph = ir.Graph([w, x], [matmul.outputs[0]], nodes=[matmul], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)

    mesh2x2 = model.add_device_configuration("mesh2x2", num_devices=4)
    # Shard rows along the first mesh axis and columns along the second.
    matmul.shard(w, configuration=mesh2x2, axis=0, num_shards=2, device_indices=(0, 1, 2, 3))
    matmul.shard(w, configuration=mesh2x2, axis=1, num_shards=2, device_indices=(0, 1, 2, 3))

    spec = matmul.sharding_of(w)[0]
    print(len(matmul.sharding_of(w)))                # 1 (single spec)
    print([d.axis for d in spec.sharded_dims])        # [0, 1]
    print(spec.device)                               # (0, 1, 2, 3)
```

### Replication across device groups

A single shard can also be *replicated* across a group of devices. Following the
ONNX multi-device proposal, a ``ShardingSpec.device`` entry is either a direct
device id, or a (typically negative) key into ``index_to_device_group_map`` that
names a group of real device ids the shard is replicated across. This is
expressed by constructing the {py:class}`~onnx_ir.ShardingSpec` directly.

A pure replication (the same tensor on every device, no splitting) uses a single
group key and an empty ``sharded_dim``:

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    w = ir.Value(name="W", shape=ir.Shape([1024, 4096]), type=ir.TensorType(ir.DataType.FLOAT))
    x = ir.Value(name="x", shape=ir.Shape([4096, 8]), type=ir.TensorType(ir.DataType.FLOAT))
    matmul = ir.Node("", "MatMul", [w, x], outputs=[ir.Value(name="y")], name="mm")
    graph = ir.Graph([w, x], [matmul.outputs[0]], nodes=[matmul], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)
    conf = model.add_device_configuration("conf", num_devices=2)

    # Replicate W across devices {0, 1}: device key -1 maps to that group,
    # and there is no sharded_dims (nothing is split).
    replicated = ir.ShardingSpec(
        value=w,
        device=(-1,),
        index_to_device_group_map=(
            ir.IndexToDeviceGroupMapEntry(key=-1, value=(0, 1)),
        ),
    )
    matmul.device_configurations = (
        ir.NodeDeviceConfiguration(configuration=conf, sharding_specs=(replicated,)),
    )
    print(replicated.index_to_device_group_map[0].value)   # (0, 1)
```

Splitting and replication can be mixed: shard ``W`` into 2 row-shards, each
replicated across a 2-device group (4 devices total).

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    w = ir.Value(name="W", shape=ir.Shape([1024, 4096]), type=ir.TensorType(ir.DataType.FLOAT))
    x = ir.Value(name="x", shape=ir.Shape([4096, 8]), type=ir.TensorType(ir.DataType.FLOAT))
    matmul = ir.Node("", "MatMul", [w, x], outputs=[ir.Value(name="y")], name="mm")
    graph = ir.Graph([w, x], [matmul.outputs[0]], nodes=[matmul], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)
    conf = model.add_device_configuration("conf", num_devices=4)

    # Row-shard 0 -> group -1 = devices {0, 1}; row-shard 1 -> group -2 = {2, 3}.
    spec = ir.ShardingSpec(
        value=w,
        device=(-1, -2),
        index_to_device_group_map=(
            ir.IndexToDeviceGroupMapEntry(key=-1, value=(0, 1)),
            ir.IndexToDeviceGroupMapEntry(key=-2, value=(2, 3)),
        ),
        sharded_dims=(
            ir.ShardedDim(
                axis=0,
                simple_shardings=(ir.SimpleShardedDim(dim=1024, num_shards=2),),
            ),
        ),
    )
    matmul.device_configurations = (
        ir.NodeDeviceConfiguration(configuration=conf, sharding_specs=(spec,)),
    )
    print(spec.index_to_device_group_map[1].value)   # (2, 3)
```

### Querying shardings

Walk a node's configurations to read back how each tensor is sharded:

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    w = ir.Value(name="W", shape=ir.Shape([1024, 4096]), type=ir.TensorType(ir.DataType.FLOAT))
    x = ir.Value(name="x", shape=ir.Shape([4096, 8]), type=ir.TensorType(ir.DataType.FLOAT))
    matmul = ir.Node("", "MatMul", [w, x], outputs=[ir.Value(name="y")], name="mm")
    graph = ir.Graph([w, x], [matmul.outputs[0]], nodes=[matmul], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)
    mesh = model.add_device_configuration("mesh2x2", num_devices=4)
    matmul.shard(w, configuration=mesh, axis=0, num_shards=2, device_indices=(0, 1, 2, 3))
    matmul.shard(w, configuration=mesh, axis=1, num_shards=2, device_indices=(0, 1, 2, 3))

    for config in matmul.device_configurations:
        print("configuration:", config.configuration.name)
        for spec in config.sharding_specs:
            sharded = {
                sd.axis: sd.simple_shardings[0].num_shards for sd in spec.sharded_dims
            }
            print(f"  {spec.value.name}: axis->num_shards = {sharded}, devices={spec.device}")

    # Direct lookup for one value:
    for spec in matmul.sharding_of(w):
        print("axes sharded:", [sd.axis for sd in spec.sharded_dims])
```

## Pipeline parallelism (device placement)

Sharding splits a single tensor across devices. *Pipeline parallelism* is the
complementary case: whole blocks of the graph are placed on different devices and
activations are handed off from one stage to the next. It is expressed with the
``pipeline_stage`` of a {py:class}`~onnx_ir.NodeDeviceConfiguration` rather than a
sharding spec. {py:meth}`Node.set_pipeline_stage
<onnx_ir.Node.set_pipeline_stage>` attaches a pure placement (no sharding) to a
node. How a stage maps to a physical device is by convention; a common choice is
``stage == device index`` into ``configuration.device_names``.

Two situations call for placing parts of a model on different devices, with
different split strategies:

- **Capacity (identical devices).** The model is too big for one accelerator, so
  a run of layers is split by position across identical devices (e.g. two GPUs)
  to fit memory. The example below shows this.
- **Affinity (heterogeneous devices).** Different ops run best on different
  hardware (CPU, NPU, GPU), so the split follows what each op is good for — see
  the end-to-end example below.

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    # A 10-"layer" decoder (one node per layer for illustration), too big for a
    # single GPU, split by position across two identical GPUs to fit memory.
    h = [
        ir.Value(name=f"h{i}", shape=ir.Shape(["B", "T", 4096]), type=ir.TensorType(ir.DataType.FLOAT))
        for i in range(11)
    ]
    layers = [
        ir.Node("custom", "DecoderLayer", [h[i]], outputs=[h[i + 1]], name=f"layer{i}")
        for i in range(10)
    ]
    graph = ir.Graph([h[0]], [h[10]], nodes=layers, opset_imports={"": 18, "custom": 1})
    model = ir.Model(graph, ir_version=11)

    # Two identical GPUs: device index 0 = GPU:0, index 1 = GPU:1.
    pipeline = model.add_device_configuration("pipeline", num_devices=2, device_names=("GPU:0", "GPU:1"))

    # First half of the layers -> stage 0 (GPU:0); second half -> stage 1 (GPU:1).
    for i, layer in enumerate(layers):
        layer.set_pipeline_stage(pipeline, 0 if i < 5 else 1)

    # Query the placement back, resolving the stage to a device name.
    device_names = model.device_configurations[0].device_names
    for layer in layers:
        stage = layer.device_configurations[0].pipeline_stage
        print(f"{layer.name}: stage {stage} -> {device_names[stage]}")
```

A node can be both sharded and staged: {py:meth}`Node.shard <onnx_ir.Node.shard>`
and {py:meth}`Node.set_pipeline_stage <onnx_ir.Node.set_pipeline_stage>` share one
{py:class}`~onnx_ir.NodeDeviceConfiguration` per configuration, so a layer can be
tensor-sharded *and* assigned to a stage at once.

## End-to-end: splitting a model across devices

This worked example places a small decoder — an embedding, ten decoder layers,
and an LM head — across a **CPU, NPU, and GPU** by *operator affinity*, each part
going to the device that runs it best:

- **CPU** — the **embedding** (`Gather`): a memory-bound lookup over a large table
  that accelerators handle poorly.
- **NPU** — the **decoder layers**: dense, quantization-friendly matmuls at high
  performance-per-watt; the bulk of the compute.
- **GPU** — the **LM head**: a large hidden→vocab projection that wants the GPU's
  throughput and precision.

The annotations are hints; the runtime inserts the cross-device transfers
implicitly, and the plan round-trips through ``to_proto`` / ``from_proto``.

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    B, T, D, V = "B", "T", 4096, 32000  # batch, seq, hidden, vocab

    # --- Build the graph: Gather embedding -> 10 DecoderLayers -> MatMul head ---
    tokens = ir.Value(name="tokens", shape=ir.Shape([B, T]), type=ir.TensorType(ir.DataType.INT64))
    embed_w = ir.Value(name="embed.weight", shape=ir.Shape([V, D]), type=ir.TensorType(ir.DataType.FLOAT))
    lm_w = ir.Value(name="lm_head.weight", shape=ir.Shape([V, D]), type=ir.TensorType(ir.DataType.FLOAT))

    h = ir.Value(name="h0", shape=ir.Shape([B, T, D]), type=ir.TensorType(ir.DataType.FLOAT))
    embed = ir.Node("", "Gather", [embed_w, tokens], outputs=[h], name="embed")

    nodes = [embed]
    layers = []
    cur = h
    for i in range(10):
        out = ir.Value(name=f"h{i + 1}", shape=ir.Shape([B, T, D]), type=ir.TensorType(ir.DataType.FLOAT))
        layer = ir.Node("custom", "DecoderLayer", [cur], outputs=[out], name=f"layer{i}")
        nodes.append(layer)
        layers.append(layer)
        cur = out

    logits = ir.Value(name="logits", shape=ir.Shape([B, T, V]), type=ir.TensorType(ir.DataType.FLOAT))
    head = ir.Node("", "MatMul", [cur, lm_w], outputs=[logits], name="lm_head")
    nodes.append(head)

    graph = ir.Graph(
        [tokens, embed_w, lm_w], [logits], nodes=nodes, opset_imports={"": 18, "custom": 1}
    )
    model = ir.Model(graph, ir_version=11)

    # --- Heterogeneous plan: device 0 = CPU, 1 = NPU, 2 = GPU ---
    plan = model.add_device_configuration("plan", num_devices=3, device_names=("CPU", "NPU", "GPU"))

    # Place each op on the device that suits it (stage == device index here).
    embed.set_pipeline_stage(plan, 0)              # embedding lookup -> CPU
    for layer in layers:
        layer.set_pipeline_stage(plan, 1)          # decoder layers   -> NPU
    head.set_pipeline_stage(plan, 2)               # LM head          -> GPU

    # --- Inspect the plan ---
    device_names = model.device_configurations[0].device_names
    for node in model.graph:
        if not node.device_configurations:
            continue
        stage = node.device_configurations[0].pipeline_stage
        print(f"{node.name:<8} -> {device_names[stage]}")

    # --- Round-trip: the plan survives serialization ---
    restored = ir.from_proto(ir.to_proto(model))
    placed = [n.device_configurations[0].pipeline_stage for n in restored.graph if n.device_configurations]
    print("nodes placed:", len(placed))
```

If a stage is served by several identical devices (say two NPUs), tensor-shard
the heavy ops across them with {py:meth}`Node.shard <onnx_ir.Node.shard>`: tensor
parallelism stays within a homogeneous group, while pipeline placement spans the
device *types*. To bind a tensor to a device explicitly instead of relying on the
``stage == device index`` convention, attach a placement-only
{py:class}`~onnx_ir.ShardingSpec` — one with a ``device`` but no ``sharded_dims``.

## Removing a configuration

{py:meth}`Model.remove_device_configuration <onnx_ir.Model.remove_device_configuration>`
is the counterpart of `add_device_configuration`. It accepts either the
{py:class}`~onnx_ir.ModelConfiguration` object or its name. By default it removes
only the model-level configuration and leaves node references intact, so any
dangling references remain detectable. Pass ``cascade=True`` to also strip every
node sharding that referenced it, leaving no dangling references behind.

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    x = ir.Value(name="x", shape=ir.Shape([8, 16]), type=ir.TensorType(ir.DataType.FLOAT))
    relu = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="relu0")
    graph = ir.Graph([x], [relu.outputs[0]], nodes=[relu], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)
    conf = model.add_device_configuration("conf0", device_names=("CPU", "CUDA:0"))
    relu.shard(x, configuration=conf, axis=0, num_shards=2)

    # Cascade removal also clears the node's sharding that used ``conf``.
    model.remove_device_configuration(conf, cascade=True)
    print(model.device_configurations)   # ()
    print(relu.device_configurations)    # ()
```

## Validating the configuration

Because metadata can be edited freely, the IR follows the common compiler-IR
convention: intermediate states may be temporarily invalid, and validity is
checked at well-defined points rather than on every edit.

The convenience methods validate eagerly: {py:meth}`Node.shard
<onnx_ir.Node.shard>` raises immediately if you pass a value that is not one of
the node's own inputs or outputs, an out-of-range axis, ``num_shards < 1``, an
axis that is already sharded, or a conflicting ``pipeline_stage``. Serialization
is the hard boundary — it raises rather than emitting a malformed proto (for
example when a sharded value has no name).

An internal checker (``onnx_ir._multi_device._check_device_configurations``) is
also used to surface dangling references and structural problems such as a
configuration that is not declared on the model, a sharded value that is not an
input or output of its node, empty names, out-of-range or duplicated axes, and
device indices outside the configuration's ``num_devices``. It is not part of the
public API yet.

## Serialization

Serialization derives the proto ``tensor_name`` and ``configuration_id`` from the
bound objects' names, and deserialization resolves them back to the corresponding
{py:class}`~onnx_ir.Value` and {py:class}`~onnx_ir.ModelConfiguration` objects.

```python
import onnx_ir as ir

# ... build ``model`` as above ...
proto = ir.to_proto(model)
restored = ir.from_proto(proto)

node = restored.graph[0]
spec = node.device_configurations[0].sharding_specs[0]
assert spec.value is node.inputs[0]                       # value resolved
assert node.device_configurations[0].configuration is restored.device_configurations[0]
```

References that cannot be resolved on load (for example a ``tensor_name`` that is
not present in the graph, or a ``configuration_id`` not declared on the model) are
preserved as lightweight placeholders so the round-trip stays lossless. Serializing
a sharding whose value has no name raises, rather than emitting a malformed proto.

## Working with the dataclasses directly

The convenience API is built on a small set of frozen dataclasses, which you can
also construct directly for full control:

- {py:class}`onnx_ir.ModelConfiguration <onnx_ir.ModelConfiguration>` — a named
  device configuration on the model.
- {py:class}`onnx_ir.NodeDeviceConfiguration <onnx_ir.NodeDeviceConfiguration>` —
  per-node sharding specs and an optional ``pipeline_stage``.
- {py:class}`onnx_ir.ShardingSpec <onnx_ir.ShardingSpec>` — the sharding of a
  single {py:class}`~onnx_ir.Value`.
- {py:class}`onnx_ir.ShardedDim <onnx_ir.ShardedDim>`,
  {py:class}`onnx_ir.SimpleShardedDim <onnx_ir.SimpleShardedDim>`, and
  {py:class}`onnx_ir.IndexToDeviceGroupMapEntry <onnx_ir.IndexToDeviceGroupMapEntry>`
  — the per-axis sharding details.

These mirror the corresponding ONNX protos field-for-field, except that
``ShardingSpec`` holds a `value` object instead of a `tensor_name` string and
``NodeDeviceConfiguration`` holds a `configuration` object instead of a
`configuration_id` string.

`Model.device_configurations` only accepts `ModelConfiguration` objects, and
`Node.device_configurations` only accepts `NodeDeviceConfiguration` objects.
Assigning any other type (for example raw ``bytes`` or a protobuf message) is
rejected at the serialization boundary (surface error:
`onnx_ir.serde.SerdeError`, with the original `TypeError` as `__cause__`).
