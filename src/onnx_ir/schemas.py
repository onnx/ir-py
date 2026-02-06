# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

__all__ = [
    "OpSignature",
    "Parameter",
    "AttributeParameter",
    "TypeConstraintParam",
]

import dataclasses
import functools
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import onnx  # noqa: TID251

from onnx_ir import _core, _enums, _protocols, serde


# A special value to indicate that the default value is not specified
class _Empty:
    def __repr__(self) -> str:
        return "_EMPTY_DEFAULT"


_EMPTY_DEFAULT = _Empty()


@functools.cache
def _all_value_types():
    return frozenset(
        {_core.TensorType(dtype) for dtype in _enums.DataType}
        | {_core.SequenceType(_core.TensorType(dtype)) for dtype in _enums.DataType}
        | {_core.OptionalType(_core.TensorType(dtype)) for dtype in _enums.DataType}
    )


@dataclasses.dataclass(frozen=True)
class TypeConstraintParam:
    """Type constraint for a parameter.

    Attributes:
        name: Name of the parameter. E.g. "TFloat"
        allowed_types: Allowed types for the parameter.
        description: Human-readable description of the type constraint.
    """

    name: str
    allowed_types: frozenset[_protocols.TypeProtocol]
    description: str = ""

    def __post_init__(self):
        if not self.allowed_types:
            raise ValueError(
                f"Type constraint '{self.name}' must have at least one allowed type."
            )

        if not isinstance(self.allowed_types, frozenset):
            object.__setattr__(self, "allowed_types", frozenset(self.allowed_types))

    def __str__(self) -> str:
        allowed_types_str = " | ".join(str(t) for t in self.allowed_types)
        return f"{self.name}={allowed_types_str}"

    @classmethod
    def any_tensor(cls, name: str, description: str = "") -> TypeConstraintParam:
        return cls(
            name, frozenset(_core.TensorType(dtype) for dtype in _enums.DataType), description
        )

    @classmethod
    def any_value(cls, name: str, description: str = "") -> TypeConstraintParam:
        return cls(name, _all_value_types(), description)  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True)
class Parameter:
    """A formal parameter of an operator."""

    name: str
    type_constraint: TypeConstraintParam
    required: bool
    variadic: bool
    homogeneous: bool = True
    min_arity: int = 1
    # TODO: Add differentiation_category
    default: Any = _EMPTY_DEFAULT

    def __str__(self) -> str:
        type_str = self.type_constraint.name
        if self.has_default():
            return f"{self.name}: {type_str} = {self.default}"
        return f"{self.name}: {type_str}"

    def has_default(self) -> bool:
        return self.default is not _EMPTY_DEFAULT

    def is_param(self) -> bool:
        """This parameter is an ONNX input or output parameter, as opposed to an ONNX attribute parameter."""
        return True

    def is_attribute(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class AttributeParameter:
    """A parameter in the function signature that represents an ONNX attribute."""

    name: str
    type: _enums.AttributeType
    required: bool
    default: _core.Attr | None = None

    def __str__(self) -> str:
        type_str = self.type.name
        if self.has_default():
            return f"{self.name}: {type_str} = {self.default}"
        return f"{self.name}: {type_str}"

    def has_default(self) -> bool:
        return self.default is not None

    def is_param(self) -> bool:
        return False

    def is_attribute(self) -> bool:
        """This parameter is an ONNX attribute parameter, as opposed to an ONNX input or output parameter."""
        return True


def _get_type_from_str(
    type_str: str,
) -> _core.TensorType | _core.SequenceType | _core.OptionalType:
    """Converter a type_str from ONNX OpSchema to _protocols.TypeProtocol.

    A type str has the form of "tensor(float)" or composite type like "seq(tensor(float))".
    """
    # Split the type_str a sequence types and dtypes
    # 1. Remove the ending ")"
    striped = type_str.rstrip(")")
    # 2. Split the type_str by "("
    type_parts = striped.split("(")

    # Convert the dtype to _enums.DataType
    dtype = _enums.DataType[type_parts[-1].upper()]

    # Create a place holder type first
    type_: _protocols.TypeProtocol = _core.TensorType(_enums.DataType.UNDEFINED)

    # Construct the type
    for type_part in reversed(type_parts[:-1]):
        if type_part == "tensor":
            type_ = _core.TensorType(dtype)
        elif type_part == "seq":
            type_ = _core.SequenceType(type_)
        elif type_part == "optional":
            type_ = _core.OptionalType(type_)
        else:
            raise ValueError(f"Unknown type part: '{type_part}' in type '{type_str}'")
    return type_  # type: ignore[return-value]


def _convert_formal_parameter(
    param: onnx.defs.OpSchema.FormalParameter,
    type_constraints: Mapping[str, TypeConstraintParam],
) -> Parameter:
    """Convert a formal parameter from ONNX OpSchema to Parameter."""
    if param.type_str in type_constraints:
        type_constraint = type_constraints[param.type_str]
    else:
        # param.type_str can be a plain type like 'int64'.
        type_constraint = TypeConstraintParam(
            name=param.name,
            allowed_types=frozenset((_get_type_from_str(param.type_str),)),
        )
    return Parameter(
        name=param.name,
        type_constraint=type_constraint,
        required=param.option != onnx.defs.OpSchema.FormalParameterOption.Optional,
        variadic=param.option == onnx.defs.OpSchema.FormalParameterOption.Variadic,
        homogeneous=param.is_homogeneous,
        min_arity=param.min_arity,
    )


@dataclasses.dataclass
class OpSignature:
    """Schema for an operator.

    Attributes:
        domain: Domain of the operator. E.g. "".
        name: Name of the operator. E.g. "Add".
        overload: Overload name of the operator.
        params: Input parameters. When the op is an ONNX function definition,
          the order is according to the function signature. This mean we can
          interleave ONNX inputs and ONNX attributes in the list.
        outputs: Output parameters.
        since_version: The version of the operator set. E.g. 1.
    """

    domain: str
    name: str
    overload: str
    params: Sequence[Parameter | AttributeParameter]
    outputs: Sequence[Parameter]
    params_map: Mapping[str, Parameter | AttributeParameter] = dataclasses.field(
        init=False, repr=False
    )
    since_version: int = 1

    def __post_init__(self):
        params_map: dict[str, Parameter | AttributeParameter] = {}
        for param in self.params:
            if param.name in params_map:
                raise ValueError(
                    f"Duplicate parameter name {param.name!r} in OpSignature "
                    f"{self.domain!r}::{self.name!r}"
                )
            params_map[param.name] = param
        self.params_map = params_map

    def get(
        self,
        name: str,
        default: Parameter | AttributeParameter | None = None,
    ) -> Parameter | AttributeParameter | None:
        return self.params_map.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self.params_map

    def __iter__(self) -> Iterator[Parameter | AttributeParameter]:
        return iter(self.params)

    def __str__(self) -> str:
        domain = self.domain or "''"
        overload = f"::{self.overload}" if self.overload else ""
        params = ", ".join(str(param) for param in self.params)
        outputs = ", ".join(str(param.type_constraint.name) for param in self.outputs)
        type_constraints = {}
        for param in self.params:
            if isinstance(param, Parameter):
                type_constraints[param.type_constraint.name] = param.type_constraint
        for param in self.outputs:
            type_constraints[param.type_constraint.name] = param.type_constraint
        type_constraints_str = ", ".join(
            str(type_constraint) for type_constraint in type_constraints.values()
        )
        return f"{domain}::{self.name}{overload}({params}) -> ({outputs}) where {type_constraints_str}"

    @property
    def inputs(self) -> Sequence[Parameter]:
        """Returns the input parameters."""
        return [param for param in self.params if isinstance(param, Parameter)]

    @property
    def attributes(self) -> Sequence[AttributeParameter]:
        """Returns the attribute parameters."""
        return [param for param in self.params if isinstance(param, AttributeParameter)]

    @classmethod
    def from_op_schema(cls, op_schema: onnx.defs.OpSchema) -> OpSignature:
        """Produce an OpSignature from an ONNX OpSchema."""
        type_constraints = {
            constraint.type_param_str: TypeConstraintParam(
                name=constraint.type_param_str,
                allowed_types=frozenset(
                    _get_type_from_str(type_str) for type_str in constraint.allowed_type_strs
                ),
                description=constraint.description,
            )
            for constraint in op_schema.type_constraints
        }

        params = [
            _convert_formal_parameter(param, type_constraints) for param in op_schema.inputs
        ]

        for param in op_schema.attributes.values():
            default_attr = (
                serde.deserialize_attribute(param.default_value)
                if param.default_value is not None
                else None
            )
            if default_attr is not None:
                # Set the name of the default attribute because it may have a different name from the parameter
                default_attr.name = param.name
            params.append(
                AttributeParameter(
                    name=param.name,
                    type=_enums.AttributeType(param.type),  # type: ignore[arg-type]
                    required=param.required,
                    default=default_attr,  # type: ignore[arg-type]
                )
            )

        outputs = [
            _convert_formal_parameter(param, type_constraints) for param in op_schema.outputs
        ]

        return cls(
            domain=op_schema.domain,
            name=op_schema.name,
            overload="",
            params=params,
            outputs=outputs,
            since_version=op_schema.since_version,
        )
