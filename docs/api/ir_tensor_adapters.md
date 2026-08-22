# onnx_ir.tensor_adapters

See [Tensor Representation in the IR](../tensors.md) for tensor backends and
[Performance and large-model workflows](../performance.md) for avoiding
unnecessary materialization.

```{eval-rst}
.. automodule:: onnx_ir.tensor_adapters
```

## Adapters for PyTorch

```{eval-rst}
.. currentmodule:: onnx_ir.tensor_adapters

.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    TorchTensor
```

```{eval-rst}
.. autofunction:: onnx_ir.tensor_adapters.from_torch_dtype
.. autofunction:: onnx_ir.tensor_adapters.to_torch_dtype
```
