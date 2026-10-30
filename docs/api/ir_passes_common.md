# onnx_ir.passes.common

Built-in passes provided by the ONNX IR

See [Writing transformation passes](../writing_passes.md) for composition and
custom pass authoring, [Subgraphs and functions](../subgraphs_and_functions.md)
for inlining, and [Model I/O](../model_io.md) for validation workflows.

```{eval-rst}
.. automodule:: onnx_ir.passes.common
.. currentmodule:: onnx_ir.passes.common
```

## Built-in passes

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    AddDefaultAttributesPass
    AddInitializersToInputsPass
    CheckerPass
    ClearMetadataAndDocStringPass
    CommonSubexpressionEliminationPass
    DeduplicateHashedInitializersPass
    DeduplicateInitializersPass
    IdentityEliminationPass
    InlinePass
    LiftConstantsToInitializersPass
    LiftSubgraphInitializersToMainGraphPass
    NameFixPass
    OutputFixPass
    RemoveInitializersFromInputsPass
    RemoveUnusedFunctionsPass
    RemoveUnusedNodesPass
    RemoveUnusedOpsetsPass
    ShapeInferencePass
    TopologicalSortPass
```
