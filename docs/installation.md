# Installation

## Requirements

- Python 3.9+
- `pip` 23+

## Install from PyPI

```bash
pip install onnx-ir
```

## Install from source

```bash
git clone https://github.com/onnx/ir-py.git
cd ir-py
pip install -e .
```

## Verify installation

```bash
python -c "import onnx_ir as ir; print(ir.__version__)"
```

## Optional integrations

Install optional packages only for the workflows that need them:

| Package | Purpose |
|---|---|
| `onnx-shape-inference` | IR-native symbolic shape inference with SymPy expressions and shape-data propagation |
| `safetensors>=0.7.0` | Save model weights with `ir.save_safetensors` |
| `rich` | Syntax-highlighted `display()` output and paging |
| `torch` | PyTorch tensor adapter and dtype conversion utilities |

```bash
pip install onnx-shape-inference "safetensors>=0.7.0" rich
```

ONNX IR does not require these packages for its core model, graph, and tensor
APIs. See [Model I/O](model_io.md), [Tensor Representation](tensors.md), and
[Debugging transformations](debugging.md) for integration-specific usage.

## Recommended setup for production workflows

1. Use a dedicated virtual environment per project.
2. Pin exact versions in your lockfile (`requirements.txt`, `uv.lock`, or similar).
3. Run with a tested ONNX version from your environment matrix.
4. For external tensor loading from untrusted artifacts, always set `base_dir`.

See [Security](security.md) for details.
