# ONNX IR

ONNX IR is an in-memory IR that supports the full ONNX spec for graph construction,
analysis, and transformation.

This docsite is organized for fast onboarding, production usage, and complete API
references.

## Why ONNX IR

- Full ONNX model coverage, including a subset of structurally invalid models that
  still need repair workflows.
- Efficient tensor handling, including external data and zero-copy friendly pathways.
- Robust graph mutation semantics suitable for compiler and transformation passes.
- Pythonic APIs that preserve intuitive mapping to ONNX concepts.

## Start here

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started
installation
compatibility
```

```{toctree}
:maxdepth: 2
:caption: Reliability and Operations

stability
migration
troubleshooting
faq
security
release_checklist
```

```{toctree}
:maxdepth: 2
:caption: User Guides

tensors
multi_device
model_io
graph_transformations
onnx_helper_migration
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
