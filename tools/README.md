# ONNX IR Tools

This directory contains command-line tools for working with ONNX models using the ONNX IR library.

## Available Tools

### convert_float_to_float16.py

Convert ONNX models from float32 to float16 precision.

**Usage:**

```bash
python tools/convert_float_to_float16.py input.onnx output.onnx [options]
```

**Options:**

- `--keep-io-types`: Keep input/output types as float32 (recommended for deployment)
- `--min-positive-val FLOAT`: Minimum positive value for clamping (default: 1e-7)
- `--max-finite-val FLOAT`: Maximum finite value for clamping (default: 1e4)
- `--op-block-list OP1 OP2 ...`: List of operator types to not convert
- `--node-block-list NODE1 NODE2 ...`: List of node names to not convert

**Examples:**

```bash
# Basic conversion
python tools/convert_float_to_float16.py model.onnx model_fp16.onnx

# Keep input/output as float32
python tools/convert_float_to_float16.py model.onnx model_fp16.onnx --keep-io-types

# Block specific operators
python tools/convert_float_to_float16.py model.onnx model_fp16.onnx --op-block-list Resize TopK
```

**Programmatic Usage:**

```python
import onnx_ir as ir
from onnx_ir.passes.common import ConvertFloatToFloat16Pass

# Load model
model = ir.load("model.onnx")

# Create and apply conversion pass
pass_ = ConvertFloatToFloat16Pass(
    min_positive_val=1e-7,
    max_finite_val=1e4,
    keep_io_types=False,
)
result = pass_(model)

# Save converted model
if result.modified:
    ir.save(result.model, "model_fp16.onnx")
```

### onnx_printer.py

Pretty-print ONNX models in a human-readable format.

### create_test_model.py

Utility for creating test ONNX models for development and testing.
