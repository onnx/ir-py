# Troubleshooting

## Installation and import issues

### `ModuleNotFoundError: onnx_ir`

- Confirm installation happened in the active environment.
- Run `python -m pip show onnx-ir`.
- Reinstall with `python -m pip install --upgrade onnx-ir`.

### Dependency version conflicts

- Check `onnx` and `numpy` versions in your environment.
- Recreate a clean virtualenv and install from a pinned requirements file.

## External tensor loading errors

### Path containment or security check failures

If loading external data fails due to path/symlink/hardlink checks:

1. Ensure `base_dir` points to the model artifact directory.
2. Remove unsafe path traversal and absolute paths from external tensor metadata.
3. Verify symlink targets stay within `base_dir`.
4. Check hard link counts in your artifact directory.

See [Security](security.md) for the full threat model and policy.

## Serialization/deserialization issues

### Model fails after transformation

- Validate graph invariants after each pass.
- Confirm value names and producer/consumer relationships are still consistent.
- Test round-trip serialization in CI.

### Unexpected dtype behavior

- Recheck dtype mappings for non-NumPy-native types.
- Ensure `ml_dtypes` is installed and version-compatible.
- Validate that adapter-backed tensors return expected byte representation.

## Performance regressions

- Compare against a stable baseline using the same dependency versions.
- Profile model sizes with and without external tensor paths.
- Verify transformations are not introducing repeated tensor materialization.

## Need help

- Open an issue: <https://github.com/onnx/ir-py/issues>
- Ask questions: <https://github.com/onnx/ir-py/discussions>
