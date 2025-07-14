# ONNX IR

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-ir.svg)](https://pypi.org/project/onnx-ir)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-ir.svg)](https://pypi.org/project/onnx-ir)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/onnx/ir-py/graph/badge.svg?token=SPQ3G9T78Z)](https://codecov.io/gh/onnx/ir-py)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-onnx%2Fir--py-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/onnx/ir-py)
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

## Code Organization 🗺️

- [`_protocols.py`](src/onnx_ir/_protocols.py): Interfaces defined for all entities in the IR.
- [`_core.py`](src/onnx_ir/_core.py): Implementation of the core entities in the IR, including `Model`, `Graph`, `Node`, `Value`, and others.
- [`_enums.py`](src/onnx_ir/_enums.py): Definition of the type enums that correspond to the `DataType` and `AttributeType` in `onnx.proto`.
- [`_name_authority.py`](src/onnx_ir/_name_authority.py): The authority for giving names to entities in the graph, used internally.
- [`_linked_list.py`](src/onnx_ir/_linked_list.py): The data structure as the node container in the graph that supports robust iteration and mutation. Internal.
- [`_metadata.py`](src/onnx_ir/_metadata.py): Metadata store for all entities in the IR.
