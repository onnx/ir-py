# <img src="docs/_static/logo-light.png" alt="ONNX IR" width="250"/>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-ir.svg)](https://pypi.org/project/onnx-ir)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-ir.svg)](https://pypi.org/project/onnx-ir)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/onnx/ir-py/graph/badge.svg?token=SPQ3G9T78Z)](https://codecov.io/gh/onnx/ir-py)
[![PyPI Downloads](https://static.pepy.tech/badge/onnx-ir/month)](https://pepy.tech/projects/onnx-ir)

An in-memory IR that supports the full ONNX spec, designed for graph construction, analysis and transformation.

## Getting Started

[onnx-ir documentation](https://onnx.ai/ir-py/)

### Installation

Via pip:

```
pip install onnx-ir
```

Or from source:

```
pip install git+https://github.com/onnx/ir-py.git
```

## Features ✨

- Full ONNX spec support: all valid models representable by ONNX protobuf, and a subset of invalid models (so you can load and fix them).
- Low memory footprint: mmap'ed external tensors; unified interface for ONNX TensorProto, Numpy arrays and PyTorch Tensors etc. No tensor size limitation. Zero copies.
- Straightforward access patterns: Access value information and traverse the graph topology at ease.
- Robust mutation: Create as many iterators as you like on the graph while mutating it.
- Speed: Performant graph manipulation, serialization/deserialization to Protobuf.
- Pythonic and familiar APIs: Classes define Pythonic apis and still map to ONNX protobuf concepts in an intuitive way.
- No protobuf dependency: The IR does not require protobuf once the model is converted to the IR representation, decoupling from the serialization format.

## Concept Diagram

![Concept Diagram](docs/resource/onnx-ir-entities.svg)

## Code Organization 🗺️

- [`_protocols.py`](src/onnx_ir/_protocols.py): Interfaces defined for all entities in the IR.
- [`_core.py`](src/onnx_ir/_core.py): Implementation of the core entities in the IR, including `Model`, `Graph`, `Node`, `Value`, and others.
- [`_enums.py`](src/onnx_ir/_enums.py): Definition of the type enums that correspond to the `DataType` and `AttributeType` in `onnx.proto`.
- [`_name_authority.py`](src/onnx_ir/_name_authority.py): The authority for giving names to entities in the graph, used internally.
- [`_linked_list.py`](src/onnx_ir/_linked_list.py): The data structure as the node container in the graph that supports robust iteration and mutation. Internal.
- [`_metadata.py`](src/onnx_ir/_metadata.py): Metadata store for all entities in the IR.
