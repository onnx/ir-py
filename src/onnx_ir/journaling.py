"""Journaling system for ONNX IR operations."""

from __future__ import annotations

import weakref
from typing import Any

__all__ = ["Journal"]

import dataclasses
import traceback
from collections.abc import Sequence
import time

from typing_extensions import Self

_current_journal: Journal | None = None


@dataclasses.dataclass(frozen=True)
class _JournalEntry:
    """A single journal entry recording an operation on the IR.

    Attributes:
        timestamp: The time at which the operation was performed.
        operation: The name of the operation performed.
        class_: The class of the object on which the operation was performed.
        class_name: The name of the class of the object.
        ref: A weak reference to the object on which the operation was performed.
            To access the object, call object().
        object_id: The unique identifier of the object (id()).
        stack_trace: The stack trace at the time of the operation.
        details: Additional details about the operation.
    """

    timestamp: float
    operation: str
    class_: type
    class_name: str
    ref: weakref.ref | None
    object_id: int
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
        for entry in entries:
            print(f"Operation: {entry.operation} on {entry.class_name}")

    You can also filter the entries by operation or class name using the `filter` method::
        filtered_entries = journal.filter(operation="set_name", class_name="Node")
    """

    def __init__(self) -> None:
        self._entries: list[_JournalEntry] = []
        self._previous_journal: Journal | None = None

    def __enter__(self) -> Self:
        global _current_journal
        self._previous_journal = _current_journal
        _current_journal = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _current_journal
        _current_journal = self._previous_journal

    def record(self, obj: Any, operation: str, details: str | None = None) -> None:
        """Record a new journal entry."""
        entry = _JournalEntry(
            timestamp=time.time(),
            operation=operation,
            class_=obj.__class__,
            class_name=obj.__class__.__name__,
            ref=weakref.ref(obj),
            object_id=id(obj),
            stack_trace=_get_stack_trace(),
            details=details,
        )
        self._entries.append(entry)

    def get_entries(self) -> Sequence[_JournalEntry]:
        """Get all recorded journal entries."""
        return self._entries

    def filter(
        self, *, operation: str | None = None, class_name: str | None = None
    ) -> Sequence[_JournalEntry]:
        """Filter journal entries by operation and/or class name."""
        result = self._entries
        if operation is not None:
            result = [entry for entry in result if entry.operation == operation]
        if class_name is not None:
            result = [entry for entry in result if entry.class_name == class_name]
        return result

    def display(self) -> None:
        """Display all journal entries."""
        for entry in self._entries:
            obj = entry.ref() if entry.ref is not None else None
            details = f" [{entry.details}]" if entry.details else ""
            print(f"{entry.operation} | {entry.class_name}(id={entry.object_id}) | {obj} | {details}")
