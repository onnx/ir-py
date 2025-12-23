# onnx_ir.passes

```{eval-rst}
.. automodule::onnx_ir.passes
```

## Use built-in passes

Common, reusable passes are implemented in `onnx_ir.passes.common`. You can use {py:class}`onnx_ir.passes.Sequential <onnx_ir.passes.Sequential>` to chain passes or use {py:class}`onnx_ir.passes.PassManager <onnx_ir.passes.PassManager>` which supports early stopping if no changes are made.

## Pass infrastructure

Inherent {py:class}`onnx_ir.passes.InPlacePass <onnx_ir.passes.InPlacePass>` or {py:class}`onnx_ir.passes.FunctionalPass <onnx_ir.passes.FunctionalPass>` to define a pass. You will need to implement the `call` method which returns a {py:class}`onnx_ir.passes.PassResult <onnx_ir.passes.PassResult>`.

Alternatively, inherent the base class `onnx_ir.passes.PassBase <onnx_ir.passes.PassBase>` and override the two properties `changes_input` and `in_place` to set properties of the pass.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    onnx_ir.passes.PassBase
    onnx_ir.passes.InPlacePass
    onnx_ir.passes.FunctionalPass
    onnx_ir.passes.Sequential
    onnx_ir.passes.PassResult
    onnx_ir.passes.PassManager
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
