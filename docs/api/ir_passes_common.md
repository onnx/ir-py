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
    VmapPass
```

## Independent batching with VmapPass

{py:class}`VmapPass` transforms a **single-example** model into an ordinary
batched ONNX model. It is a functional pass: the source model is unchanged,
including when an unsupported operation raises
{py:class}`onnx_ir.passes.PreconditionError`.

```python
import onnx_ir as ir
from onnx_ir.passes.common import VmapPass

# The source graph has inputs x: [D], weight: [D, H], and output y: [H].
model = ir.load("single_example.onnx")
batched = VmapPass(
    in_axes=(0, None),  # x is mapped; weight is shared.
    out_axes=-1,
)(model).model
# The new graph takes x: [B, D], weight: [D, H], and returns y: [H, B].
# B is determined at runtime; pass axis_size=32 to specialize it to 32.
ir.save(batched, "batched.onnx")
```

`in_axes` has one entry per graph input, in graph-input order. An integer
**inserts** a batch axis into that input's single-example shape; `None` leaves
the input shared. At least one input must be mapped. By default, the batch length
is inferred at runtime from the first mapped input. All mapped inputs must have
the **same** runtime batch length; this is part of the model's input contract,
not an inserted runtime assertion. Supply `axis_size` to specialize the graph
to a fixed positive number of examples. `out_axes` is either one integer for every
output or a sequence of integers in graph-output order. Negative axes index the
resulting batched tensor, not the original tensor. Every output is mapped:
outputs independent of mapped inputs are repeated along the requested axis.

The pass canonicalizes mapped values to a leading batch dimension internally.
It explicitly pads example ranks for broadcasting, excludes the batch dimension
from reductions and transpose permutations, and promotes/squeezes vector
operands around MatMul. These are tensor-level rewrites, not Loop/Scan lowering;
no new ONNX operator or runtime extension is required.

### Supported operators

Only standard-domain operators in this whitelist are supported, including on
branches that use only shared inputs:

| Category | Operators |
| --- | --- |
| Constants | `Constant` with a tensor `value` attribute; graph initializers are shared |
| Unary elementwise | `Abs`, `Cast`, `Ceil`, `Cos`, `Erf`, `Exp`, `Floor`, `Identity`, `IsInf`, `IsNaN`, `Log`, `Neg`, `Not`, `Reciprocal`, `Relu`, `Round`, `Sigmoid`, `Sign`, `Sin`, `Sqrt`, `Tan`, `Tanh` |
| Broadcasting elementwise | `Add`, `And`, `Div`, `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `Max`, `Mean`, `Min`, `Mul`, `Or`, `Pow`, `Sub`, `Sum`, `Where`, `Xor` |
| Linear algebra | `MatMul`, including vector operands and per-example batch dimensions |
| Axis operations | `Transpose`, `Concat`, `Softmax`, `LogSoftmax` |
| Reductions | `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceMax`, `ReduceMean`, `ReduceMin`, `ReduceProd`, `ReduceSum`, `ReduceSumSquare` |

Reduction axes must be omitted or supplied by a shared 1-D `INT64` initializer
or tensor-valued Constant. Negative axes, `keepdims`, and
`noop_with_empty_axes` retain their per-example meanings. Axes supplied as graph
inputs are rejected, even when they have initializer defaults, since callers
can override them.

### Restrictions and invariants

- The source must be a valid, topologically sorted ONNX graph with
  standard-domain opset **18 or newer**. The pass does not upgrade the opset.
- `axis_size` may be omitted for dynamic batch sizes or set to a fixed positive
  integer. Empty batches are outside the supported input contract. Example
  dimensions may be zero, symbolic, or
  unknown, but graph inputs must be tensors with known ranks.
- Examples must have uniform shapes. Ragged results, sequence/optional values,
  control flow, custom operators, local function calls, randomness, and all
  other operators without a batching rule are rejected. Inline supported local
  functions before mapping them. Unused function definitions are left unchanged.
- Initializers are shared. Mapping an input with an initializer default is
  rejected; a shared input may retain its default.
- Input names, graph/model metadata, and node-output names are preserved.
  Directly returned inputs or initializers receive new output names when an
  adapter is necessary. Internal mapped shape annotations are updated using
  known ranks; the pass does not perform general shape inference.

Nested mapping is not a first-version feature: generated adapter operations
such as Shape, Expand, Unsqueeze, and Squeeze do not themselves have batching
rules. Unsupported graphs fail explicitly rather than falling back to a loop
or silently changing per-example behavior.
