---
authors:
  - '@justinchuby'
  - '@copilot'
created: 2026-01-27
updated: 2026-02-11
---

# Symbolic Shape Inference for ONNX IR

> [!NOTE]
> This design doc mainly created by Copilot and reviewed by @justinchuby

## Overview

This document describes the design of symbolic shape inference capability for the ONNX IR Python library. The feature enables shape propagation through computational graphs while preserving symbolic dimensions and supporting arithmetic expressions.

## Goals

1. **Symbolic Preservation**: Maintain symbolic dimension names (e.g., "batch", "seq_len") through operations
2. **Arithmetic Support**: Enable symbolic arithmetic (e.g., `batch * 2`, `seq_len - 1`)
3. **Backward Compatibility**: Extend existing classes without breaking current usage
4. **Modularity**: Easy to add shape inference for new operators
5. **Opset Awareness**: Support different inference logic for different ONNX opset versions
6. **Unique Dimension Tracking**: Assign unique names to unknown dimensions so relationships can be established across operations

## Non-Goals

- Full constraint solving/unification (future work)
- Bidirectional inference (future work)

## Design

### 1. Enhanced SymbolicDim

The existing `SymbolicDim` class is enhanced to support SymPy expressions:

```python
class SymbolicDim:
    def __init__(self, value: str | None | sympy.Expr) -> None:
        ...
```

**Accepted inputs:**

- `str`: Named symbolic dimension (e.g., `"batch"`)
- `None`: Unknown dimension
- `sympy.Expr`: Symbolic expression (e.g., `sympy.Symbol("batch") + 1`)

**Key properties:**

- `value: str | None` - String representation (pre-computed at init)
- `expr: sympy.Expr | None` - SymPy expression (lazy, created on first access)

**Arithmetic operations:**

```python
dim = ir.SymbolicDim("batch")
dim + 1      # SymbolicDim with expr: batch + 1
dim * 2      # SymbolicDim with expr: batch * 2
dim // 4     # SymbolicDim with expr: floor(batch / 4)
dim % 8      # SymbolicDim with expr: batch mod 8
```

**Design decisions:**

1. **Lazy SymPy creation**: SymPy `Symbol` objects are only created when `.expr` is accessed. This avoids overhead for simple cases where symbolic math isn't needed.

2. **Pre-computed value**: Since `SymbolicDim` is immutable, `value` is computed once at initialization.

3. **Value-based equality**: `__eq__` compares string values, not SymPy expressions. Users should call `simplify()` before comparing complex expressions.

4. **None propagation**: Arithmetic with `None` (unknown) produces `None`:

   ```python
   SymbolicDim(None) + 1  # SymbolicDim(None)
   ```

5. **NotImplemented for unsupported types**: Magic methods return `NotImplemented` for unsupported operand types, following Python conventions.

### 2. Shape Enhancements

The `Shape` class gains methods for working with symbolic dimensions:

```python
class Shape:
    def evaluate(self, bindings: Mapping[str, int]) -> Shape:
        """Substitute symbolic dims with concrete values."""

    def simplify(self) -> Shape:
        """Simplify all symbolic expressions in the shape."""

    def free_symbols(self) -> set[str]:
        """Get all symbolic variable names."""
```

**Example:**

```python
shape = ir.Shape(["batch", 128, "seq"])
shape.evaluate({"batch": 4, "seq": 512})  # Shape([4, 128, 512])
shape.free_symbols()  # {"batch", "seq"}
```

### 3. Shape Inference Registry

An opset-aware registry maps `(domain, op_type, version)` to inference functions with O(1) lookup:

```python
registry = OpShapeInferenceRegistry()

@registry.register("", "Add", since_version=7)  # Version 7 and above
def infer_add_v7(ctx, node):
    ...

@registry.register("", "Reshape", since_version=14)  # Version 14 and above
def infer_reshape_v14(ctx, node):
    ...
```

**Version specification:**

- `since_version=int`: This version and all above until the next registration

**Lookup behavior:**

1. Dispatch to the registered function where `version >= since_version`
   and `version < next_since_version`
2. Uses a cached dict for O(1) lookup after first access
3. Tracks the largest `since_version` for efficient lookup of versions beyond cache

### 4. Shape Inference Context

The context tracks state during inference and applies merge policies:

```python
class ShapeInferenceContext:
    def __init__(
        self,
        opset_imports: Mapping[str, int] | None = None,
        policy: ShapeMergePolicy = "refine",
    ) -> None:
        ...

    def set_shape(self, value: ir.Value, shape: ir.Shape) -> bool: ...
    def set_dtype(self, value: ir.Value, dtype: ir.DataType) -> bool: ...
    def set_shape_and_dtype(self, value, shape, dtype) -> bool: ...
    def set_type(self, value: ir.Value, type_: ir.TypeProtocol) -> bool: ...
    def new_symbolic_dim(self, prefix: str = "_d") -> ir.SymbolicDim: ...
    def bind(self, symbol: str, value: int | sympy.Expr) -> None: ...
    def record_error(self, node: ir.Node, message: str) -> None: ...
```

#### Type Setting Methods

- **`set_shape_and_dtype`**: Convenience for tensor-typed values.
- **`set_type`**: For non-tensor types like `SequenceType` and `OptionalType`.
  Follows the same merge policy as `set_shape` and `set_dtype`.

#### Unique Dimension Naming

When the exact size of a dimension is unknown, operators must **not** use
`ir.SymbolicDim(None)`.  Instead they call `ctx.new_symbolic_dim()` which
returns a `SymbolicDim` with a unique auto-generated name (e.g. `_d0`,
`_d1`, …).  This ensures every unknown dimension has a distinct identity, so
downstream inference can establish relationships between inputs and outputs.

```python
# BAD  – all unknowns are anonymous and indistinguishable
ir.Shape([ir.SymbolicDim(None), ir.SymbolicDim(None)])

# GOOD – each unknown gets a unique name
ir.Shape([ctx.new_symbolic_dim(), ctx.new_symbolic_dim()])
```

#### Error Handling

Two error types distinguish different failure modes:

- **`OpUsageError`** (always raised): Indicates the model is malformed — wrong
  number of inputs, a required input is ``None``, or a required attribute is
  missing.  Helper functions `check_inputs()` and `require_attr()` raise this
  directly, eliminating boilerplate in every op:

  ```python
  (data, shape) = _context.check_inputs(node, "data", "shape")
  axis_attr = _context.require_attr(node, "axis")
  ```

- **`ShapeInferenceError`** (via `ctx.record_error()`): Indicates a semantic
  problem detected during inference — axis out of range, rank mismatch, etc.
  Raised in all policies except ``"skip"``, where it is logged as a warning
  instead.

Both errors are `ValueError` subclasses.

**Merge policies** (`ShapeMergePolicy = Literal["skip", "override", "refine", "strict"]`):

| Policy | Behavior |
|--------|----------|
| `"skip"` | Keep existing shape/dtype if present; semantic errors logged only |
| `"override"` | Always replace with inferred value; semantic errors raised |
| `"refine"` | Update only if inferred is more specific (int > named > None); semantic errors raised |
| `"strict"` | Raise `ValueError` on conflicts; semantic errors raised |

### 5. Broadcasting Utilities

NumPy-style broadcasting for element-wise operations:

```python
def broadcast_shapes(shape1: ir.Shape | None, shape2: ir.Shape | None) -> ir.Shape | None:
    """Compute broadcast result of two shapes."""
```

**Rules:**

1. Prepend 1s to shorter shape to match ranks
2. For each dimension pair:
   - If equal, keep the value
   - If one is 1, use the other
   - If one is concrete and other is symbolic, prefer concrete
   - If incompatible (different concrete values), return `None`

### 6. Operator Implementations

Each operator's inference logic lives in its own file under `onnx_ir/shape_inference/_ops/`:

```
shape_inference/
├── __init__.py
├── _registry.py
├── _context.py
├── _broadcast.py
└── _ops/
    ├── __init__.py         # Imports all modules to trigger registration
    ├── _testing.py         # Test infrastructure (ts, const_value, run_shape_inference, …)
    ├── _elementwise.py     # Generic binary elementwise (Add, Sub, Mul, Div, …)
    ├── _unary.py           # Generic unary passthrough (~40 ops)
    ├── _reduce.py          # Generic Reduce* (10 ops)
    ├── _cast.py            # Cast, CastLike
    ├── _concat.py          # Concat
    ├── _constant.py        # Constant
    ├── _constant_of_shape.py
    ├── _conv.py            # Conv (with auto_pad support)
    ├── _dropout.py         # Dropout (output + mask)
    ├── _expand.py          # Expand
    ├── _gather.py          # Gather
    ├── _gemm.py            # Gemm (transA/transB)
    ├── _matmul.py          # MatMul (batch, 1-D handling)
    ├── _reshape.py         # Reshape (0, -1, allowzero)
    ├── _sequence.py        # Sequence ops (8 ops)
    ├── _shape_ops.py       # Shape, Size, Flatten
    ├── _slice.py           # Slice (v10+, input-based)
    ├── _softmax.py         # Softmax, LogSoftmax, Hardmax
    ├── _split.py           # Split (attr and input-based)
    ├── _squeeze.py         # Squeeze, Unsqueeze (attr and input-based)
    ├── _transpose.py       # Transpose
    └── _where.py           # Where (3-input broadcast)
```

#### Generic Helpers

Shared patterns are implemented as generic functions with stacked decorators
for multiple-op registration, avoiding code duplication:

```python
_reg = _registry.registry.register

@_reg("", "Add", since_version=7)
@_reg("", "Sub", since_version=7)
@_reg("", "Mul", since_version=7)
# ... more ops
def infer_binary_elementwise(ctx, node):
    ...
```

Three generic helpers cover a large number of ops:

- **`_elementwise.py`**: Binary broadcast (arithmetic, comparison, logical)
- **`_unary.py`**: Shape+dtype passthrough; logical unary (output BOOL)
- **`_reduce.py`**: Axes handling (attribute or input), keepdims, noop_with_empty_axes

#### Sequence Operators

Sequence operators produce `SequenceType` outputs rather than `TensorType`.
They use `ctx.set_type()` instead of `ctx.set_shape_and_dtype()`:

```python
@_reg("", "SequenceConstruct", since_version=11)
def infer_sequence_construct(ctx, node):
    elem_type = node.inputs[0].type
    ctx.set_type(node.outputs[0], ir.SequenceType(elem_type))
```

Implemented sequence ops: `SequenceConstruct`, `SequenceEmpty`, `SequenceAt`,
`SequenceLength`, `SequenceInsert`, `SequenceErase`, `SplitToSequence`,
`ConcatFromSequence`.

### 7. Inference Engine and Pass

The core inference logic lives in `_engine.py`, decoupled from the pass
framework.  `SymbolicShapeInferencePass` delegates to `_engine` so that
calling `infer_symbolic_shapes` directly does not wrap errors in `PassError`.

```
infer_symbolic_shapes(model)          ◄── public API in onnx_ir.shape_inference
│
▼
_infer_symbolic_shapes(model)         ◄── _engine.py, returns modified: bool
│
├─ Create ShapeInferenceContext(opset_imports, policy)
│
▼
_process_graph(ctx, model.graph)      ◄── starts from the main graph
│
├─ Name anonymous dims on graph inputs     ◄── SymbolicDim(None) → SymbolicDim("_d0")
│
├─ For each node in topological order:
│   │
│   ├─ Recurse into subgraph attrs         ◄── If/Loop bodies processed in-order
│   │   └─ _process_graph(ctx, subgraph)
│   │
│   ├─ Lookup infer_func from registry     ◄── registry.get(domain, op_type, version)
│   │
│   ├─ [not found] ──► log warning, skip
│   │
│   ├─ [found] ──►
│   │   │
│   │   ├─ Name anonymous dims on inputs   ◄── ensures no None dims enter the op
│   │   │
│   │   ├─ Save old output states
│   │   │
│   │   ├─ Call infer_func(ctx, node)       ◄── see "Op-Level Flow" below
│   │   │
│   │   ├─ Compare output states → set modified=True if changed
│   │   │
│   │   └─ [exception] ──► log warning, continue
│   │
│   └─ next node
│
└─ Return modified: bool
```

`SymbolicShapeInferencePass` is a thin wrapper that calls `_infer_symbolic_shapes`
and returns a `PassResult`:

```python
class SymbolicShapeInferencePass(ir.passes.InPlacePass):
    def call(self, model):
        modified = _infer_symbolic_shapes(model, policy=self.policy, ...)
        return ir.passes.PassResult(model, modified)
```

**Op-Level Flow** (inside each `infer_func`):

```
infer_func(ctx, node)
│
├─ check_inputs(node, "A", "B", ...)       ◄── validates count & non-None
│   └─ [fail] ──► raise OpUsageError       ◄── always raised, caught by engine
│
├─ require_attr(node, "axis")              ◄── optional, for ops needing attrs
│   └─ [fail] ──► raise OpUsageError
│
├─ Graceful degradation                    ◄── e.g. shape is None → set dtype only
│   └─ ctx.set_shape_and_dtype(out, None, dtype)
│
├─ Semantic validation                     ◄── e.g. rank check, axis range
│   └─ [fail] ──► ctx.record_error(node, msg)
│               └─ policy="skip" → log warning
│               └─ otherwise     → raise ShapeInferenceError
│
├─ Compute output shape                   ◄── op-specific logic, broadcast, etc.
│
└─ ctx.set_shape_and_dtype(out, shape, dtype)
    │
    └─ _check_no_anonymous_dims(shape)     ◄── rejects SymbolicDim(None) in outputs
```

**Convenience function:**

```python
from onnx_ir.shape_inference import infer_symbolic_shapes

model = infer_symbolic_shapes(model, policy="refine")
```

## API Summary

### Public API (`onnx_ir.shape_inference`)

```python
# Types
ShapeMergePolicy = Literal["skip", "override", "refine", "strict"]

# Classes
ShapeInferenceContext
ShapeInferenceError
OpUsageError
OpShapeInferenceRegistry

# Functions
broadcast_shapes(shape1, shape2) -> Shape | None
check_inputs(node, *names) -> tuple[Value, ...]
require_attr(node, name) -> Attr
infer_symbolic_shapes(model, *, policy, warn_on_missing) -> Model

# Registry instance
registry: OpShapeInferenceRegistry
```

### Enhanced Core Classes (`onnx_ir`)

```python
# SymbolicDim - accepts str | None | sympy.Expr
# Supports: +, -, *, //, % with int or SymbolicDim

# Shape
Shape.evaluate(bindings) -> tuple[int, ...] | Shape
Shape.simplify() -> Shape
Shape.free_symbols() -> set[str]
```

## Supported Operators

### Tier 1: Generic Helpers (~70 ops)

| Category | Ops | Module |
|----------|-----|--------|
| Arithmetic | Add, Sub, Mul, Div, Mod, Pow, BitShift | `_elementwise.py` |
| Comparison | Equal, Less, Greater, LessOrEqual, GreaterOrEqual | `_elementwise.py` |
| Logical | And, Or, Xor | `_elementwise.py` |
| Unary | Neg, Abs, Ceil, Floor, Relu, Sigmoid, Tanh, Exp, Log, Sin, Cos, … | `_unary.py` |
| Logical Unary | Not, IsNaN, IsInf | `_unary.py` |
| Reduce | ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1/L2, … | `_reduce.py` |

### Tier 2: Individual Ops

| Op | Module | Notes |
|----|--------|-------|
| Reshape | `_reshape.py` | Handles 0, -1, allowzero |
| Concat | `_concat.py` | Sum along axis |
| Squeeze / Unsqueeze | `_squeeze.py` | Attribute and input-based axes |
| Gather | `_gather.py` | Any axis |
| Slice | `_slice.py` | v10+ input-based only |
| Split | `_split.py` | Attribute and input-based split |
| Expand | `_expand.py` | Broadcast to target shape |
| MatMul | `_matmul.py` | Batch, 1-D handling |
| Gemm | `_gemm.py` | transA/transB |
| Conv | `_conv.py` | auto_pad (VALID, SAME_UPPER, SAME_LOWER) |
| Cast / CastLike | `_cast.py` | dtype from attribute or input |
| Constant | `_constant.py` | Shape/dtype from attribute |
| ConstantOfShape | `_constant_of_shape.py` | Shape from const input |
| Shape / Size / Flatten | `_shape_ops.py` | start/end for Shape v15+ |
| Softmax / LogSoftmax / Hardmax | `_softmax.py` | Passthrough |
| Where | `_where.py` | 3-input broadcast |
| Dropout | `_dropout.py` | Output + optional mask (BOOL) |
| Transpose | `_transpose.py` | Perm attribute |

### Tier 3: Sequence Ops

| Op | Module | Notes |
|----|--------|-------|
| SequenceConstruct | `_sequence.py` | Tensor inputs → SequenceType |
| SequenceEmpty | `_sequence.py` | dtype attribute |
| SequenceAt | `_sequence.py` | SequenceType → element TensorType |
| SequenceLength | `_sequence.py` | Scalar INT64 |
| SequenceInsert / Erase | `_sequence.py` | Preserves sequence type |
| SplitToSequence | `_sequence.py` | Tensor → SequenceType |
| ConcatFromSequence | `_sequence.py` | SequenceType → Tensor |

## Usage Examples

### Basic Shape Inference

```python
import onnx_ir as ir
from onnx_ir.shape_inference import infer_symbolic_shapes

# Load model
model = ir.from_onnx(onnx_model)

# Run inference
infer_symbolic_shapes(model)

# Check results
for node in model.graph:
    for output in node.outputs:
        print(f"{output.name}: shape={output.shape}, dtype={output.dtype}")
```

### Custom Operator Registration

```python
from onnx_ir.shape_inference import registry, ShapeInferenceContext, check_inputs

@registry.register("com.custom", "MyOp", since_version=1)
def infer_my_op(ctx: ShapeInferenceContext, node: ir.Node) -> None:
    (data,) = check_inputs(node, "data")
    if data.shape is not None:
        ctx.set_shape(node.outputs[0], data.shape)
```

### Symbolic Arithmetic

```python
batch = ir.SymbolicDim("batch")
seq = ir.SymbolicDim("seq")

# Create shape with arithmetic
shape = ir.Shape([batch, seq // 2, 256])

# Evaluate with concrete values
concrete = shape.evaluate({"batch": 4, "seq": 128})  # Shape([4, 64, 256])

# Get free symbols
symbols = shape.free_symbols()  # {"batch", "seq"}
```

## Future Work

1. **Constraint System**: Track dimension equality constraints and propagate bindings
2. **Bidirectional Inference**: Infer input shapes from output constraints
3. **Model Local Functions**: Inline shape inference for functions defined in `model.functions`, propagating shapes through function call boundaries
4. **Subgraph Support**: Infer shapes inside subgraphs used by control-flow operators (If, Loop, Scan), propagating outer-scope bindings into the subgraph and back
5. **Newer Operators**: Add inference for operators introduced in recent opsets, such as `Attention`, `GroupNormalization`, `RotaryEmbedding`, and other transformer-related ops
6. **ConvTranspose**: Implement shape inference for ConvTranspose operator
7. **SequenceMap**: Implement body-graph-based inference for SequenceMap
8. **QLinearConv / ConvInteger**: Quantized convolution operators

## Dependencies

- `sympy>=1.13`: Symbolic mathematics library for expression handling
