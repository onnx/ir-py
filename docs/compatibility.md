# Compatibility

This page documents compatibility guarantees and tested integration boundaries.

## Runtime compatibility

- **Python**: `>=3.9`
- **ONNX**: `>=1.16`
- **Core dependencies**: `numpy`, `typing_extensions>=4.10`, `ml_dtypes>=0.5.0`,
  `sympy>=1.13`

These constraints are defined in `pyproject.toml` and represent the minimum runtime
requirements.

## Platform support

ONNX IR is expected to run on Linux, macOS, and Windows in standard CPython
environments where supported dependency versions are available.

For production usage, maintain and validate your own platform matrix in CI based on:

- Python versions you support internally
- OS/architecture targets you deploy to
- ONNX and framework versions in your stack

## Tensor adapter ecosystem

ONNX IR can interoperate with multiple tensor backends through protocol-based
adapters (for example NumPy and PyTorch-backed arrays).

For non-NumPy-native dtypes (for example bfloat16/float8/int4/int2), behavior
depends on `ml_dtypes` availability and the tensor adapter path. Validate
serialization and round-tripping in your CI for all critical model families.

## Guidance for production consumers

Before promoting to production:

1. Freeze a tested dependency matrix.
2. Run representative model load/transform/serialize tests against that matrix.
3. Add canary tests for external data models and large tensors.
4. Re-validate before any dependency upgrade.
