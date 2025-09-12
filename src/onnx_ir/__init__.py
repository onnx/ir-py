# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""In-memory intermediate representation for ONNX graphs."""

__all__ = [
    # Modules
    "serde",
    "traversal",
    "convenience",
    "external_data",
    "tape",
    # IR classes
    "Tensor",
    "ExternalTensor",
    "StringTensor",
    "LazyTensor",
    "PackedTensor",
    "SymbolicDim",
    "Shape",
    "TensorType",
    "OptionalType",
    "SequenceType",
    "SparseTensorType",
    "TypeAndShape",
    "Value",
    "Attr",
    "RefAttr",
    "Node",
    "Function",
    "Graph",
    "GraphView",
    "Model",
    # Constructors
    "AttrFloat32",
    "AttrFloat32s",
    "AttrGraph",
    "AttrGraphs",
    "AttrInt64",
    "AttrInt64s",
    "AttrSparseTensor",
    "AttrSparseTensors",
    "AttrString",
    "AttrStrings",
    "AttrTensor",
    "AttrTensors",
    "AttrTypeProto",
    "AttrTypeProtos",
    "Input",
    # Protocols
    "ArrayCompatible",
    "DLPackCompatible",
    "TensorProtocol",
    "ValueProtocol",
    "ModelProtocol",
    "NodeProtocol",
    "GraphProtocol",
    "GraphViewProtocol",
    "AttributeProtocol",
    "ReferenceAttributeProtocol",
    "SparseTensorProtocol",
    "SymbolicDimProtocol",
    "ShapeProtocol",
    "TypeProtocol",
    "MapTypeProtocol",
    "FunctionProtocol",
    # Enums
    "AttributeType",
    "DataType",
    # Types
    "OperatorIdentifier",
    # Protobuf compatible types
    "TensorProtoTensor",
    # Conversion functions
    "from_proto",
    "from_onnx_text",
    "to_proto",
    "to_onnx_text",
    # Convenience constructors
    "tensor",
    "node",
    "val",
    # Pass infrastructure
    "passes",
    # IO
    "load",
    "save",
    # Flags
    "DEBUG",
]

import types

from onnx_ir import convenience, external_data, passes, serde, tape, traversal
from onnx_ir._convenience._constructors import node, tensor, val
from onnx_ir._core import (
    Attr,
    AttrFloat32,
    AttrFloat32s,
    AttrGraph,
    AttrGraphs,
    AttrInt64,
    AttrInt64s,
    AttrSparseTensor,
    AttrSparseTensors,
    AttrString,
    AttrStrings,
    AttrTensor,
    AttrTensors,
    AttrTypeProto,
    AttrTypeProtos,
    ExternalTensor,
    Function,
    Graph,
    GraphView,
    Input,
    LazyTensor,
    Model,
    Node,
    OptionalType,
    PackedTensor,
    RefAttr,
    SequenceType,
    Shape,
    SparseTensorType,
    StringTensor,
    SymbolicDim,
    Tensor,
    TensorType,
    TypeAndShape,
    Value,
)
from onnx_ir._enums import (
    AttributeType,
    DataType,
)
from onnx_ir._io import load, save
from onnx_ir._protocols import (
    ArrayCompatible,
    AttributeProtocol,
    DLPackCompatible,
    FunctionProtocol,
    GraphProtocol,
    GraphViewProtocol,
    MapTypeProtocol,
    ModelProtocol,
    NodeProtocol,
    OperatorIdentifier,
    ReferenceAttributeProtocol,
    ShapeProtocol,
    SparseTensorProtocol,
    SymbolicDimProtocol,
    TensorProtocol,
    TypeProtocol,
    ValueProtocol,
)
from onnx_ir.serde import TensorProtoTensor, from_onnx_text, from_proto, to_onnx_text, to_proto

DEBUG = False


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        obj = global_dict[name]
        if hasattr(obj, "__module__") and not isinstance(obj, types.GenericAlias):
            obj.__module__ = __name__


__set_module()
__version__ = "0.1.10"
