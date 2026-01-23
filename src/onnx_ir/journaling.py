"""Journaling system for ONNX IR operations."""

from __future__ import annotations
from typing import Any
import weakref


__all__ = ["Journal", "JournalEntry", "get_journal"]

import dataclasses
import traceback
from collections.abc import Sequence

from typing_extensions import Self

_current_journal: Journal | None = None


@dataclasses.dataclass(frozen=True)
class JournalEntry:
    """A single journal entry recording an operation on the IR."""

    operation: str
    class_: type
    class_name: str
    object: weakref.ref | None
    stack_trace: str
    details: str | None


def get_journal() -> Journal | None:
    """Get the current journal, if any."""
    return _current_journal


def _get_stack_trace() -> str:
    """Get a string representation of the current stack trace."""
    stack = traceback.extract_stack()[:-3]
    formatted_stack = traceback.format_list(stack)
    return "".join(formatted_stack)


class Journal:
    """A journaling system to record operations on the ONNX IR.

    It can be used as a context manager to automatically record operations within a block.

    Example::
        journal = Journal()

        with onnx_ir.Journal() as journal:
            # Perform operations on the ONNX IR
            pass

        entries = journal.get_entries()
    """

    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []
        self._previous_journal: Journal | None = None

    def record(self, obj: Any, operation: str, *, details: str | None = None) -> None:
        """Record a new journal entry."""
        entry = JournalEntry(
            operation,
            class_=obj.__class__,
            class_name=obj.__class__.__name__,
            object=weakref.ref(obj),
            stack_trace=_get_stack_trace(),
            details=details,
        )
        self._entries.append(entry)

    def get_entries(self) -> Sequence[JournalEntry]:
        """Get all recorded journal entries."""
        return self._entries

    def __enter__(self) -> Self:
        global _current_journal
        self._previous_journal = _current_journal
        _current_journal = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _current_journal
        _current_journal = self._previous_journal
