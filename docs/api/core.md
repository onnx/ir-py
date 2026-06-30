# onnx_ir

```{eval-rst}
.. automodule::onnx_ir
```

## Functions and constructors

```{eval-rst}
.. autosummary::
    :toctree: generated/functions
    :template: functiontemplate.rst
    :nosignatures:

    onnx_ir.load
    onnx_ir.save
    onnx_ir.save_safetensors
    onnx_ir.from_proto
    onnx_ir.from_onnx_text
    onnx_ir.to_proto
    onnx_ir.to_onnx_text
    onnx_ir.tensor
    onnx_ir.val
    onnx_ir.node
    onnx_ir.set_value_magic_handler
```

## Classes

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: classtemplate_inherited.rst
    :nosignatures:

    onnx_ir.TensorProtocol
    onnx_ir.Value
    onnx_ir.Node
    onnx_ir.Graph
    onnx_ir.Model
    onnx_ir.GraphView
    onnx_ir.Function
    onnx_ir.Attr
    onnx_ir.Shape
    onnx_ir.SymbolicDim
    onnx_ir.TypeAndShape
    onnx_ir.TensorType
    onnx_ir.SparseTensorType
    onnx_ir.SequenceType
    onnx_ir.OptionalType
    onnx_ir.Tensor
    onnx_ir.ExternalTensor
    onnx_ir.StringTensor
    onnx_ir.LazyTensor
    onnx_ir.PackedTensor
```

## Multi-device configurations

See [Multi-Device Configurations](../multi_device.md) for a guide.

```{eval-rst}
.. currentmodule:: onnx_ir

.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    ModelConfiguration
    NodeDeviceConfiguration
    ShardingSpec
    ShardedDim
    SimpleShardedDim
    IndexToDeviceGroupMapEntry
```

## Enums

```{eval-rst}
.. currentmodule:: onnx_ir

.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    DataType
    AttributeType
```

### Internal Containers

```{eval-rst}
.. currentmodule:: onnx_ir

.. autosummary::
    :toctree: generated
    :template: classtemplate_inherited.rst
    :nosignatures:

    _graph_containers.GraphInitializers
    _graph_containers.Attributes
    _metadata.MetadataStore
```
