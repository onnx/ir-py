# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for testing."""

from __future__ import annotations

__all__ = [
    "assert_onnx_proto_equal",
    "topologically_equal",
    "assert_topologically_equal",
]

import difflib
import math
from collections.abc import Collection, Sequence
from typing import Any

import google.protobuf.message
import onnx


def _opset_import_key(opset_import: onnx.OperatorSetIdProto) -> tuple[str, int]:
    return (opset_import.domain, opset_import.version)


def _value_info_key(value_info: onnx.ValueInfoProto) -> str:
    return value_info.name


def _function_key(function: onnx.FunctionProto) -> tuple[str, str, str]:
    return (function.domain, function.name, getattr(function, "overload", ""))


def _find_duplicates(with_duplicates: Collection[Any]) -> list[Any]:
    """Return a list of duplicated elements in a collection."""
    seen = set()
    duplicates = []
    for x in with_duplicates:
        if x in seen:
            duplicates.append(x)
        seen.add(x)
    return duplicates


def assert_onnx_proto_equal(
    actual: google.protobuf.message.Message | Any,
    expected: google.protobuf.message.Message | Any,
    ignore_initializer_value_proto: bool = False,
) -> None:
    """Assert that two ONNX protos are equal.

    Equality is defined as having the same fields with the same values. When
    a field takes the default value, it is considered equal to the field
    not being set.

    Sequential fields with name `opset_import`, `value_info`, and `functions` are
    compared disregarding the order of their elements.

    Args:
        actual: The first ONNX proto.
        expected: The second ONNX proto.
        ignore_initializer_value_proto: Ignore value protos for initializers if there
            are extra ones in the actual proto.
    """
    assert type(actual) is type(expected), (
        f"Type not equal: {type(actual)} != {type(expected)}"
    )

    a_fields = {field.name: value for field, value in actual.ListFields()}
    b_fields = {field.name: value for field, value in expected.ListFields()}
    all_fields = sorted(set(a_fields.keys()) | set(b_fields.keys()))
    if isinstance(actual, onnx.GraphProto) and isinstance(expected, onnx.GraphProto):
        actual_initializer_names = {i.name for i in actual.initializer}
        expected_initializer_names = {i.name for i in expected.initializer}
    else:
        actual_initializer_names = set()
        expected_initializer_names = set()

    # Record and report all errors
    errors = []
    for field in all_fields:  # pylint: disable=too-many-nested-blocks
        # Obtain the default value if the field is not set. This way we can compare the two fields.
        a_value = getattr(actual, field)
        b_value = getattr(expected, field)
        if (
            isinstance(a_value, Sequence)
            and isinstance(b_value, Sequence)
            and not isinstance(a_value, (str, bytes))
            and not isinstance(b_value, (str, bytes))
        ):
            # Check length first
            a_keys: list[Any] = []
            b_keys: list[Any] = []
            if field == "opset_import":
                a_value = sorted(a_value, key=_opset_import_key)
                b_value = sorted(b_value, key=_opset_import_key)
                a_keys = [_opset_import_key(opset_import) for opset_import in a_value]
                b_keys = [_opset_import_key(opset_import) for opset_import in b_value]
            elif field == "value_info":
                if (
                    ignore_initializer_value_proto
                    and isinstance(actual, onnx.GraphProto)
                    and isinstance(expected, onnx.GraphProto)
                ):
                    # Filter out initializers from the value_info list
                    a_value = [
                        value_info
                        for value_info in a_value
                        if value_info.name not in actual_initializer_names
                    ]
                    b_value = [
                        value_info
                        for value_info in b_value
                        if value_info.name not in expected_initializer_names
                    ]
                a_value = sorted(a_value, key=_value_info_key)
                b_value = sorted(b_value, key=_value_info_key)
                a_keys = [_value_info_key(value_info) for value_info in a_value]
                b_keys = [_value_info_key(value_info) for value_info in b_value]
            elif field == "functions":
                a_value = sorted(a_value, key=_function_key)
                b_value = sorted(b_value, key=_function_key)
                a_keys = [_function_key(functions) for functions in a_value]
                b_keys = [_function_key(functions) for functions in b_value]

            if a_keys != b_keys:
                keys_only_in_actual = set(a_keys) - set(b_keys)
                keys_only_in_expected = set(b_keys) - set(a_keys)
                error_message = (
                    f"Field {field} not equal: keys_only_in_actual={keys_only_in_actual}, keys_only_in_expected={keys_only_in_expected}. "
                    f"Field type: {type(a_value)}. "
                    f"Duplicated a_keys: {_find_duplicates(a_keys)}, duplicated b_keys: {_find_duplicates(b_keys)}"
                )
                errors.append(error_message)
            elif len(a_value) != len(b_value):
                error_message = (
                    f"Field {field} not equal: len(a)={len(a_value)}, len(b)={len(b_value)} "
                    f"Field type: {type(a_value)}"
                )
                errors.append(error_message)
            else:
                # Check every element
                for i in range(len(a_value)):  # pylint: disable=consider-using-enumerate
                    actual_value_i = a_value[i]
                    expected_value_i = b_value[i]
                    if isinstance(
                        actual_value_i, google.protobuf.message.Message
                    ) and isinstance(expected_value_i, google.protobuf.message.Message):
                        try:
                            assert_onnx_proto_equal(
                                actual_value_i,
                                expected_value_i,
                                ignore_initializer_value_proto=ignore_initializer_value_proto,
                            )
                        except AssertionError as e:
                            error_message = f"Field {field} index {i} in sequence not equal. type(actual_value_i): {type(actual_value_i)}, type(expected_value_i): {type(expected_value_i)}, actual_value_i: {actual_value_i}, expected_value_i: {expected_value_i}"
                            error_message = (
                                str(e) + "\n\nCaused by the above error\n\n" + error_message
                            )
                            errors.append(error_message)
                    elif actual_value_i != expected_value_i:
                        if (
                            isinstance(actual_value_i, float)
                            and isinstance(expected_value_i, float)
                            and math.isnan(actual_value_i)
                            and math.isnan(expected_value_i)
                        ):
                            # Consider NaNs equal
                            continue
                        error_message = f"Field {field} index {i} in sequence not equal. type(actual_value_i): {type(actual_value_i)}, type(expected_value_i): {type(expected_value_i)}"
                        for line in difflib.ndiff(
                            str(actual_value_i).splitlines(),
                            str(expected_value_i).splitlines(),
                        ):
                            error_message += "\n" + line
                        errors.append(error_message)
        elif isinstance(a_value, google.protobuf.message.Message) and isinstance(
            b_value, google.protobuf.message.Message
        ):
            assert_onnx_proto_equal(
                a_value, b_value, ignore_initializer_value_proto=ignore_initializer_value_proto
            )
        elif a_value != b_value:
            if (
                isinstance(a_value, float)
                and isinstance(b_value, float)
                and math.isnan(a_value)
                and math.isnan(b_value)
            ):
                # Consider NaNs equal
                continue
            error_message = (
                f"Field {field} not equal. field_actual: {a_value}, field_expected: {b_value}"
            )
            errors.append(error_message)
    if errors:
        raise AssertionError(
            f"Protos not equal: {type(actual)} != {type(expected)}\n" + "\n".join(errors)
        )


# Import graph comparison functions
from onnx_ir._graph_comparison import assert_topologically_equal, topologically_equal
