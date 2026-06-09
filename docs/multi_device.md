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
    conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
    relu.shard(x, configuration=conf, axis=0, num_shards=2, devices=(0, 1))

    # The sharding is bound to the value object, so it follows renames.
    x.name = "input"
    spec = relu.sharding_of(x)[0]
    print(spec.value.name)         # input
    print(spec.value is x)         # True
```

{py:meth}`Node.sharding_of <onnx_ir.Node.sharding_of>` returns the live sharding
specs that target a particular value (matched by object identity).

Calling `shard` again with the same `configuration` appends to the existing
{py:class}`onnx_ir.NodeDeviceConfiguration <onnx_ir.NodeDeviceConfiguration>`
rather than creating a new one:

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    x = ir.Value(name="x", shape=ir.Shape([8, 16]), type=ir.TensorType(ir.DataType.FLOAT))
    relu = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="relu0")
    graph = ir.Graph([x], [relu.outputs[0]], nodes=[relu], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)
    conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))

    relu.shard(x, configuration=conf, axis=0, num_shards=2)
    relu.shard(relu.outputs[0], configuration=conf, axis=0, num_shards=2)

    # Both specs live under a single configuration.
    print(len(relu.device_configurations))               # 1
    print(len(relu.device_configurations[0].sharding_spec))  # 2
```

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
    matmul.shard(w, configuration=mesh1d, axis=0, num_shards=4, devices=(0, 1, 2, 3))

    # Column-parallel would instead shard W along axis 1:
    #   matmul.shard(w, configuration=mesh1d, axis=1, num_shards=4, ...)

    spec = matmul.sharding_of(w)[0]
    print(spec.sharded_dim[0].axis)                       # 0
    print(spec.sharded_dim[0].simple_sharding[0].num_shards)  # 4
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
    matmul.shard(w, configuration=mesh2x2, axis=0, num_shards=2, devices=(0, 1, 2, 3))
    matmul.shard(w, configuration=mesh2x2, axis=1, num_shards=2, devices=(0, 1, 2, 3))

    spec = matmul.sharding_of(w)[0]
    print(len(matmul.sharding_of(w)))                # 1 (single spec)
    print([d.axis for d in spec.sharded_dim])        # [0, 1]
    print(spec.device)                               # (0, 1, 2, 3)
```

### Replication across device groups

A single shard can be *replicated* across a group of devices. The proto models
this with ``ShardingSpec.device`` entries that are group indices, plus
``index_to_device_group_map`` mapping each index to the devices in the group.
This is expressed by constructing the {py:class}`~onnx_ir.ShardingSpec` directly:

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    w = ir.Value(name="W", shape=ir.Shape([1024, 4096]), type=ir.TensorType(ir.DataType.FLOAT))
    x = ir.Value(name="x", shape=ir.Shape([4096, 8]), type=ir.TensorType(ir.DataType.FLOAT))
    matmul = ir.Node("", "MatMul", [w, x], outputs=[ir.Value(name="y")], name="mm")
    graph = ir.Graph([w, x], [matmul.outputs[0]], nodes=[matmul], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)
    conf = model.add_device_configuration("conf", num_devices=4)

    # Shard W into 2 row-shards across 2 device *groups* (indices 0 and 1).
    # Group 0 = devices {0, 1}, group 1 = devices {2, 3}: each shard is
    # replicated across the two devices in its group.
    spec = ir.ShardingSpec(
        value=w,
        device=(0, 1),
        index_to_device_group_map=(
            ir.IndexToDeviceGroupMapEntry(key=0, value=(0, 1)),
            ir.IndexToDeviceGroupMapEntry(key=1, value=(2, 3)),
        ),
        sharded_dim=(
            ir.ShardedDim(
                axis=0,
                simple_sharding=(ir.SimpleShardedDim(dim=1024, num_shards=2),),
            ),
        ),
    )
    matmul.device_configurations = (
        ir.NodeDeviceConfiguration(configuration=conf, sharding_spec=(spec,)),
    )
    print(spec.index_to_device_group_map[0].value)   # (0, 1)
```

### Querying shardings

Walk a node's configurations to read back how each tensor is sharded. Values are
bound objects, so you get the live {py:class}`~onnx_ir.Value` back, not a name.

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    w = ir.Value(name="W", shape=ir.Shape([1024, 4096]), type=ir.TensorType(ir.DataType.FLOAT))
    x = ir.Value(name="x", shape=ir.Shape([4096, 8]), type=ir.TensorType(ir.DataType.FLOAT))
    matmul = ir.Node("", "MatMul", [w, x], outputs=[ir.Value(name="y")], name="mm")
    graph = ir.Graph([w, x], [matmul.outputs[0]], nodes=[matmul], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)
    mesh = model.add_device_configuration("mesh2x2", num_devices=4)
    matmul.shard(w, configuration=mesh, axis=0, num_shards=2, devices=(0, 1, 2, 3))
    matmul.shard(w, configuration=mesh, axis=1, num_shards=2, devices=(0, 1, 2, 3))

    for config in matmul.device_configurations:
        print("configuration:", config.configuration.name)
        for spec in config.sharding_spec:
            sharded = {
                sd.axis: sd.simple_sharding[0].num_shards for sd in spec.sharded_dim
            }
            print(f"  {spec.value.name}: axis->num_shards = {sharded}, devices={spec.device}")

    # Direct lookup for one value:
    for spec in matmul.sharding_of(w):
        print("axes sharded:", [sd.axis for sd in spec.sharded_dim])
```

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
    conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
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
spec = node.device_configurations[0].sharding_spec[0]
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
rejected at the serialization boundary with a `TypeError`.
