# Migration Guide

This guide helps teams move safely to ONNX IR and maintain reliable upgrades
afterward.

## Upgrade checklist

1. **Pin dependencies**
   - Pin `onnx-ir` and `onnx` versions in your environment.
2. **Run compatibility tests**
   - Load, transform, and serialize representative production models.
3. **Validate external data handling**
   - Ensure `base_dir` is set when loading untrusted model artifacts.
4. **Audit internal imports**
   - Replace imports from private modules (for example `onnx_ir._*`) with public
     APIs whenever possible.
5. **Check dtype behavior**
   - Re-test non-NumPy-native dtype paths (`bfloat16`, `float8`, `int4`, `int2`).
6. **Baseline performance**
   - Compare latency and memory for key graph transformations.

## Common migration risks

### Reliance on private internals

If your code imports private symbols, it may break across versions. Prefer the
documented API modules and top-level exports.

### External tensor loading assumptions

Security checks for external tensors are intentionally strict when `base_dir` is
configured. If your models use symlinks/hardlinks, validate your artifact pipeline.

### Adapter-specific tensor assumptions

When using framework-backed tensors, ensure conversion and serialization paths
preserve dtype and layout expectations.

## Suggested CI gates for migration

- Golden model round-trip tests (`load -> transform -> serialize -> reload`)
- Compatibility suite across your supported Python/ONNX matrix
- Security regression tests for external data path handling
- Memory and throughput smoke benchmarks
