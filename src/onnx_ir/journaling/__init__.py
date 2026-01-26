# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Journaling system for ONNX IR operations."""

from __future__ import annotations

__all__ = ["Journal", "JournalEntry", "get_current_journal"]

from onnx_ir.journaling._journaling import Journal, JournalEntry, get_current_journal


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        global_dict[name].__module__ = __name__


__set_module()
