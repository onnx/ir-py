# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Transformation passes and their composition infrastructure.

Batching and automatic differentiation implementations are developed separately
in the private ``_batching`` and ``_autodiff`` modules. They share private
domain/operator/opset dispatch, but have independent rule contracts and coverage.
These internals are experimental, not a supported plugin interface. Public pass
entry points belong in :mod:`onnx_ir.passes.common`.
"""

__all__ = [
    "PassBase",
    "PassResult",
    "PassManager",
    "Sequential",
    "InPlacePass",
    "FunctionalPass",
    "functionalize",
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
)


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        global_dict[name].__module__ = __name__


__set_module()
