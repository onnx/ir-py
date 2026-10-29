# onnx_ir.passes

The pass API defines transformation contracts, results, composition, lifecycle
errors, and functionalization utilities. Reusable implementations are documented
in {py:mod}`onnx_ir.passes.common`.

See [Writing transformation passes](../writing_passes.md) for implementation,
composition, metadata invalidation, and testing guidance. See
[Model I/O](../model_io.md) for targeted normalization and validation tools.

```{eval-rst}
.. automodule:: onnx_ir.passes
.. currentmodule:: onnx_ir.passes
```

## Pass infrastructure

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    PassBase
    InPlacePass
    FunctionalPass
    Sequential
    PassResult
    PassManager
```

## Errors

```{eval-rst}
.. autoexception:: onnx_ir.passes.InvariantError
.. autoexception:: onnx_ir.passes.PreconditionError
.. autoexception:: onnx_ir.passes.PostconditionError
.. autoexception:: onnx_ir.passes.PassError
```

## Utilities

```{eval-rst}
.. autofunction:: onnx_ir.passes.functionalize
```
