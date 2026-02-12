# Shape Inference Op Development Skill

This skill provides guidance for implementing and testing shape inference operators in `src/onnx_ir/shape_inference/_ops/`.

## Overview

Shape inference operators live under `src/onnx_ir/shape_inference/_ops/`. Each file registers one or more ops with the global registry. Tests live alongside implementation files with a `_test.py` suffix.

## Implementing an Op

### File Structure

```python
# src/onnx_ir/shape_inference/_ops/_my_op.py
from __future__ import annotations

__all__ = ["infer_my_op"]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "MyOp", since_version=1)
def infer_my_op(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    (data,) = _context.check_inputs(node, "data")
    # ... inference logic ...
    ctx.set_shape_and_dtype(node.outputs[0], output_shape, data.dtype)
```

### Import Conventions

In **source files** (`_ops/*.py`), import modules only — not names:

```python
# GOOD
from onnx_ir.shape_inference import _context, _registry

# BAD
from onnx_ir.shape_inference._context import check_inputs
```

In **test files** (`_ops/*_test.py`), importing the `ts` helper directly is an approved exception:

```python
from onnx_ir.shape_inference._ops._testing import ts
```

### Registration Pattern

For ops that share the same logic, stack `@register` decorators on a single function:

```python
_reg = _registry.registry.register

@_reg("", "Add", since_version=7)
@_reg("", "Sub", since_version=7)
@_reg("", "Mul", since_version=7)
def infer_binary_elementwise(ctx, node):
    ...
```

### Precondition Checks

Use `check_inputs()` and `require_attr()` for model validation. These raise `OpUsageError` for malformed models:

```python
(data, shape) = _context.check_inputs(node, "data", "shape")
axis_attr = _context.require_attr(node, "axis")
```

### Unknown Dimensions

When the exact size of a dimension is unknown, **always** use `ctx.new_symbolic_dim()` — never `ir.SymbolicDim(None)`:

```python
# GOOD — each unknown gets a unique name (_d0, _d1, ...)
output_dims = [ctx.new_symbolic_dim() for _ in range(rank)]

# BAD — anonymous dims are indistinguishable
output_dims = [ir.SymbolicDim(None) for _ in range(rank)]
```

### Graceful Degradation

When input shape is unknown, set dtype only and return early:

```python
if data.shape is None:
    ctx.set_shape_and_dtype(node.outputs[0], None, data.dtype)
    return
```

### Semantic Errors

Use `ctx.record_error()` for runtime semantic issues (axis out of range, rank mismatch):

```python
if axis >= rank:
    ctx.record_error(node, f"Axis {axis} out of range for rank {rank}")
    return
```

## Testing an Op

### The `ts()` Helper

All tests use `ts()` (TypeAndShape) for concise assertions. It creates an `ir.TypeAndShape` object for direct comparison with `assertEqual`:

```python
from onnx_ir.shape_inference._ops._testing import ts

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64

# Concrete shape
ts(FLOAT, [3, 4])              # Tensor(FLOAT), Shape([3, 4])

# Named symbolic dims (propagated from inputs)
ts(FLOAT, ["batch", 128])      # Tensor(FLOAT), Shape([SymbolicDim("batch"), 128])

# Auto-generated symbolic dims (from ctx.new_symbolic_dim())
ts(FLOAT, [3, "_d0"])          # Tensor(FLOAT), Shape([3, SymbolicDim("_d0")])

# Unknown rank (shape is None)
ts(FLOAT)                      # Tensor(FLOAT), shape=None
```

### Symbolic Arithmetic in Dim Expressions

When an op's output dimension can be computed from input symbolic dims and known
integer parameters (kernel size, stride, padding, blocksize, etc.), use
SymbolicDim arithmetic instead of `ctx.new_symbolic_dim()`:

```python
# GOOD — preserves relationship between input and output dims
out_dim = (in_dim + pad_begin + pad_end - effective_kernel) // stride + 1

# BAD — loses the relationship
out_dim = ctx.new_symbolic_dim()
```

SymbolicDim supports `+`, `-`, `*`, `//`, `%` with integers and other
SymbolicDims. Results are new SymbolicDim objects with SymPy expressions:

```python
H = ir.SymbolicDim("H")
H + 2       # SymbolicDim("H + 2")
H * 3       # SymbolicDim("3*H")
H // 4      # SymbolicDim("floor(H/4)")
(H - 3) // 1 + 1  # SymbolicDim("H - 2")
```

**Use `ctx.new_symbolic_dim()` only when** the output size is truly data-dependent
or unknowable at graph construction time:

- Data-dependent ops: NonZero, Compress, Unique, StringSplit
- Runtime-only values: TopK k, OneHot depth, Range length
- Control flow: If/Loop trip counts
- Kernel size unknown (weight shape not available)

### Auto-Generated Dim Counter

The `_d` counter resets at the start of each `run_shape_inference()` call. The
test helpers also name anonymous `SymbolicDim(None)` dims on inputs before
calling the op (matching engine behavior), so `_d0` may appear from that naming:

```python
# Conv uses arithmetic expressions for computed spatial dims
actual = run_shape_inference("", "Conv", [ts(FLOAT, ["N", 3, "H", "W"]), ...], ...)
self.assertEqual(actual, [ts(FLOAT, ["N", 16, "H - 2", "W - 2"])])

# Named dims propagate through passthrough ops (unary, cast, softmax, etc.)
actual = run_shape_inference("", "Relu", [ts(FLOAT, ["N", "C"])], ...)
self.assertEqual(actual, [ts(FLOAT, ["N", "C"])])

# _d dims still appear for truly data-dependent ops
actual = run_shape_inference("", "NonZero", [ts(INT64, [3, 4])], ...)
self.assertEqual(actual, [ts(INT64, [2, "_d0"])])
```

### When Dims Are Named vs Auto-Generated

**Named dims propagate through**: unary ops, Cast, Softmax, Dropout, elementwise broadcast, Where, MatMul, Gemm, Concat (non-concat axis), Squeeze, Unsqueeze, Transpose, Slice (non-sliced axis), Gather (data shape passthrough), Conv (batch dim), LSTM/GRU (seq/batch dims), Attention.

**Expression-based dims (arithmetic on input symbolic dims)**: Conv/Pooling spatial (`H - 2`), ConvTranspose spatial (`H + 2`), Pad (`N + 2`), Tile (`2*N`), DepthToSpace/SpaceToDepth (`floor(C/4)`, `2*H`), Concat axis (`a + b`), Flatten (`2*N`), DFT (preserves input dims).

**Auto-generated `_d` dims appear for**: data-dependent outputs (NonZero, Compress, Unique), runtime-only values (TopK k, OneHot depth, Range length), control flow (If/Loop), unknown kernel sizes, Resize fallback, STFT frame/freq bins.

### Test Structure

Use parameterized tests to cover multiple cases concisely:

```python
class MyOpTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        ("concrete", [3, 4], [3, 4]),
        ("symbolic", ["N", 4], ["N", 4]),
    ])
    def test_basic(self, _name, input_shape, expected_shape):
        actual = run_shape_inference("", "MyOp", [ts(FLOAT, input_shape)], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_missing_shape(self):
        actual = run_shape_inference("", "MyOp", [ts(FLOAT)], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT)])

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "MyOp", [None], opset_version=21)
```

### Testing with Constant Inputs

Use `const_value()` for ops that read constant inputs:

```python
from onnx_ir.shape_inference._ops._testing import const_value

x = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4, 5]))
k = const_value([3])
actual = run_shape_inference_with_values("", "TopK", [x, k], opset_version=21, num_outputs=2)
self.assertEqual(actual[0], ts(FLOAT, [3, 4, 3]))
```

### Assertion Style

Always prefer `ts()` over verbose multi-line assertions:

```python
# GOOD — single line covers type + shape
self.assertEqual(actual, [ts(FLOAT, [3, "_d0"])])

# BAD — verbose and fragile
self.assertIsNotNone(actual[0].shape)
self.assertEqual(actual[0].shape.rank(), 2)
self.assertEqual(actual[0].shape[0], 3)
self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
self.assertEqual(actual[0].type.dtype, FLOAT)
```

## Running Tests and Lint

```bash
# Run all shape inference tests
python -m pytest src/onnx_ir/shape_inference/_ops/ -x -q

# Run a single test file
python -m pytest src/onnx_ir/shape_inference/_ops/_slice_test.py -x -q

# Lint and auto-fix
lintrunner -a --output oneline
```

## Registering New Op Files

After creating a new op file, import it in `src/onnx_ir/shape_inference/_ops/__init__.py` to trigger registration:

```python
from onnx_ir.shape_inference._ops import _my_op  # noqa: F401
```

## References

- Design doc: `docs/design/01_symbolic_shape_inference.md`
- Test infrastructure: `src/onnx_ir/shape_inference/_ops/_testing.py`
- Registry: `src/onnx_ir/shape_inference/_registry.py`
- Context: `src/onnx_ir/shape_inference/_context.py`
- Broadcasting: `src/onnx_ir/shape_inference/_broadcast.py`
