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

## Recommended setup for production workflows

1. Use a dedicated virtual environment per project.
2. Pin exact versions in your lockfile (`requirements.txt`, `uv.lock`, or similar).
3. Run with a tested ONNX version from your environment matrix.
4. For external tensor loading from untrusted artifacts, always set `base_dir`.

See [Compatibility](compatibility.md) and [Security](security.md) for details.
