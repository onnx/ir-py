# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "PassBase",
    "PassResult",
    "PassManager",
    "Sequential",
    "InPlacePass",
    "FunctionalPass",
    "functionalize",
    "pass_fn",
    # Errors
    "InvariantError",
    "PreconditionError",
    "PostconditionError",
    "PassError",
]

from onnx_ir.passes._pass_infra import (
    FunctionalPass,
    InPlacePass,
    InvariantError,
    PassBase,
    PassError,
    PassManager,
    PassResult,
    PostconditionError,
    PreconditionError,
    Sequential,
    functionalize,
    pass_fn,
)


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        global_dict[name].__module__ = __name__


__set_module()
