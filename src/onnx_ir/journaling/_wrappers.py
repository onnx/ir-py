# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Wrappers for IR classes to enable journaling."""

from __future__ import annotations
from typing import Any

from onnx_ir import _core
from onnx_ir.journaling import _journaling


def _wrap_init(journal: _journaling.Journal, func):
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        func(self, *args, **kwargs)
        journal.record(
            operation="init",
            obj=self,
            details=repr(self),
        )


def store_original_functions():
    return {
        "TensorBase.__init__": _core.TensorBase.__init__,
    }


def wrap_ir_classes(journal: _journaling.Journal):
    _core.TensorBase.__init__ = _wrap_init(journal, _core.TensorBase.__init__)
