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

## Removing a configuration

{py:meth}`Model.remove_device_configuration <onnx_ir.Model.remove_device_configuration>`
is the counterpart of `add_device_configuration`. It accepts either the
{py:class}`~onnx_ir.ModelConfiguration` object or its name. By default it removes
only the model-level configuration and leaves node references intact, so any
dangling references are surfaced by `check_device_configurations`. Pass
``cascade=True`` to also strip every node sharding that referenced it, leaving no
dangling references behind.

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
    print(ir.check_device_configurations(model))  # []
```

## Validating the configuration

Because metadata can be edited freely, the IR follows the common compiler-IR
convention: intermediate states may be temporarily invalid, and validity is
checked at well-defined points rather than on every edit.
{py:func}`onnx_ir.check_device_configurations <onnx_ir.check_device_configurations>`
returns a list of human-readable problems (an empty list means the model is
valid). It never raises for invariant violations, so you decide whether to warn
or fail.

```{eval-rst}
.. exec_code::

    import onnx_ir as ir

    x = ir.Value(name="x", shape=ir.Shape([8, 16]), type=ir.TensorType(ir.DataType.FLOAT))
    relu = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="relu0")
    graph = ir.Graph([x], [relu.outputs[0]], nodes=[relu], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=11)
    conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
    relu.shard(x, configuration=conf, axis=0, num_shards=2, devices=(0, 1))

    print(ir.check_device_configurations(model))   # []
```

The checker reports:

- references to a {py:class}`~onnx_ir.ModelConfiguration` that is not declared on
  the model,
- shardings whose value is not an input or output of the node,
- values or configurations with empty names (which cannot be serialized), and
- out-of-range axes, ``num_shards`` below one, and device indices outside the
  configuration's ``num_devices``.

The convenience methods validate eagerly as well: {py:meth}`Node.shard
<onnx_ir.Node.shard>` raises immediately if you pass a value that is not one of
the node's own inputs or outputs, an out-of-range axis, or ``num_shards < 1``.

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
