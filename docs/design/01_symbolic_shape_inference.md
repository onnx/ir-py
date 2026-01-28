# Symbolic Shape Inference for ONNX IR

Authors: @justinchuby, @copilot

Created: 1/2026

> [!NOTE]
> This design doc was iterated with Copilot

## Overview

This document describes the design of symbolic shape inference capability for the ONNX IR Python library. The feature enables shape propagation through computational graphs while preserving symbolic dimensions and supporting arithmetic expressions.

## Goals

1. **Symbolic Preservation**: Maintain symbolic dimension names (e.g., "batch", "seq_len") through operations
2. **Arithmetic Support**: Enable symbolic arithmetic (e.g., `batch * 2`, `seq_len - 1`)
3. **Backward Compatibility**: Extend existing classes without breaking current usage
4. **Modularity**: Easy to add shape inference for new operators
5. **Opset Awareness**: Support different inference logic for different ONNX opset versions

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
    def evaluate(self, bindings: Mapping[str, int]) -> list[int] | None:
        """Substitute symbolic dims with concrete values."""

    def simplify(self) -> Shape:
        """Simplify all symbolic expressions in the shape."""

    def free_symbols(self) -> set[str]:
        """Get all symbolic variable names."""
```

**Example:**

```python
shape = ir.Shape(["batch", 128, "seq"])
shape.evaluate({"batch": 4, "seq": 512})  # [4, 128, 512]
shape.free_symbols()  # {"batch", "seq"}
```

### 3. Shape Inference Registry

An opset-aware registry maps `(domain, op_type, version)` to inference functions:

```python
registry = OpShapeInferenceRegistry()

@registry.register("", "Add", versions=1)  # Version 1 and above
def infer_add(node, ctx):
    ...

@registry.register("", "Reshape", versions=range(5, 14))  # Versions 5-13 only
def infer_reshape_v5(node, ctx):
    ...
```

**Version specification:**

- `versions=None`: All versions (wildcard)
- `versions=int`: This version and all above (minimum version)
- `versions=range(a, b)`: Exact versions a to b-1 only

**Lookup behavior:**

1. Check exact version match (for range registrations)
2. Fall back to highest minimum-version registration that applies
3. Fall back to wildcard registration

### 4. Shape Inference Context

The context tracks state during inference and applies merge policies:

```python
class ShapeInferenceContext:
    def __init__(
        self,
        model: ir.Model,
        opset: int | None = None,
        policy: ShapeMergePolicy = "refine",
    ) -> None:
        ...

    def set_shape(self, value: ir.Value, shape: ir.Shape) -> bool: ...
    def set_dtype(self, value: ir.Value, dtype: ir.DataType) -> bool: ...
    def bind(self, symbol: str, value: int | sympy.Expr) -> None: ...
```

**Merge policies** (`ShapeMergePolicy = Literal["skip", "override", "refine", "strict"]`):

| Policy | Behavior |
|--------|----------|
| `"skip"` | Keep existing shape/dtype if present |
| `"override"` | Always replace with inferred value |
| `"refine"` | Update only if inferred is more specific (int > named > None) |
| `"strict"` | Raise `ValueError` on conflicts |

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
    ├── __init__.py
    ├── _add.py
    └── _transpose.py
```

**Example: Add operator**

```python
@registry.register("", "Add", versions=1)
def infer_add(node: ir.Node, ctx: ShapeInferenceContext) -> None:
    a, b = node.inputs[0], node.inputs[1]
    output = node.outputs[0]

    # Infer shape via broadcasting
    result_shape = broadcast_shapes(a.shape, b.shape)

    # Infer dtype (same as inputs for Add)
    result_dtype = a.dtype or b.dtype

    ctx.set_shape_and_dtype(output, result_shape, result_dtype)
```

### 7. Inference Pass

The main pass traverses the graph in topological order:

```python
class SymbolicShapeInferencePass(ir.passes.InPlacePass):
    def __init__(
        self,
        policy: ShapeMergePolicy = "refine",
        warn_on_missing: bool = True,
    ) -> None:
        ...

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        ...
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
OpShapeInferenceRegistry

# Functions
broadcast_shapes(shape1, shape2) -> Shape | None
infer_symbolic_shapes(model, *, policy, warn_on_missing) -> Model

# Registry instance
registry: OpShapeInferenceRegistry
```

### Enhanced Core Classes (`onnx_ir`)

```python
# SymbolicDim - accepts str | None | sympy.Expr
# Supports: +, -, *, //, % with int or SymbolicDim

# Shape
Shape.evaluate(bindings) -> list[int] | None
Shape.simplify() -> Shape
Shape.free_symbols() -> set[str]
```

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
from onnx_ir.shape_inference import registry, ShapeInferenceContext

@registry.register("com.custom", "MyOp", versions=1)
def infer_my_op(node: ir.Node, ctx: ShapeInferenceContext) -> None:
    # Custom inference logic
    input_shape = node.inputs[0].shape
    if input_shape is not None:
        # Output has same shape as input
        ctx.set_shape(node.outputs[0], input_shape)
```

### Symbolic Arithmetic

```python
batch = ir.SymbolicDim("batch")
seq = ir.SymbolicDim("seq")

# Create shape with arithmetic
shape = ir.Shape([batch, seq // 2, 256])

# Evaluate with concrete values
concrete = shape.evaluate({"batch": 4, "seq": 128})  # [4, 64, 256]

# Get free symbols
symbols = shape.free_symbols()  # {"batch", "seq"}
```

## Future Work

1. **Constraint System**: Track dimension equality constraints and propagate bindings
2. **Bidirectional Inference**: Infer input shapes from output constraints
3. **ONNX Fallback**: Use ONNX's native shape inference for unregistered ops
4. **More Operators**: Expand coverage beyond Add and Transpose
5. **Serialization**: Serialize complex SymPy expressions to ONNX `dim_param` strings

## Dependencies

- `sympy>=1.13`: Symbolic mathematics library for expression handling
