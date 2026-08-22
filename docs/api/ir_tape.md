# onnx_ir.tape

```{eval-rst}
.. automodule:: onnx_ir.tape
.. currentmodule:: onnx_ir.tape
```

The `onnx_ir.tape` module provides utilities for recording nodes and initializers to construct computational graphs or functions.

See [Constructing models](../model_construction.md) for direct and tape-based
construction examples, and [Debugging transformations](../debugging.md) for using
a tape to reduce construction failures.

## The `Tape` class

The `Tape` class is a recorder that collects nodes and initializers created during the construction of a graph or function. It supports creating nodes with single or multiple outputs and registering initializers.

```{eval-rst}
.. autoclass:: Tape
   :members:
   :undoc-members:
```
