"""Journaling system for ONNX IR operations."""

from __future__ import annotations

import weakref
from typing import Any

__all__ = ["Journal", "JournalEntry", "get_journal"]

import dataclasses
import datetime
import time
import traceback
from collections.abc import Callable, Sequence

from typing_extensions import Self

_current_journal: Journal | None = None


@dataclasses.dataclass(frozen=True)
class JournalEntry:
    """A single journal entry recording an operation on the IR.

    Attributes:
        timestamp: The time at which the operation was performed.
        operation: The name of the operation performed.
        class_: The class of the object on which the operation was performed.
        class_name: The name of the class of the object.
        ref: A weak reference to the object on which the operation was performed.
            To access the object, call ``ref()``. Note that ``ref`` may be ``None``,
            and ``ref()`` may return ``None`` if the object has been garbage-collected.
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
    stack_trace: list[traceback.FrameSummary]
    details: str | None

    def display(self) -> None:
        """Display the journal entry in a detailed multi-line format."""
        # Header with timestamp
        timestamp = datetime.datetime.fromtimestamp(self.timestamp).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        print(f"\033[1m{'=' * 80}\033[0m")
        print(f"\033[1mTimestamp:\033[0m {timestamp}")
        print(f"\033[1mOperation:\033[0m {self.operation}")
        print(f"\033[1mClass:\033[0m {self.class_name} (id={self.object_id})")

        # Object representation
        if self.ref is None:
            object_repr = "<no ref>"
        elif (obj := self.ref()) is not None:
            object_repr = repr(obj)
        else:
            object_repr = "<deleted>"
        print("\033[1mObject:\033[0m")
        for line in object_repr.split("\n"):
            print(f"  {line}")

        # Details
        if self.details:
            print("\033[1mDetails:\033[0m")
            for line in self.details.split("\n"):
                print(f"  {line}")

        # Stack trace - find user code frame
        if self.stack_trace:
            user_frame = None
            for f in reversed(self.stack_trace):
                filename = f.filename.replace("\\", "/")
                if "onnx_ir" not in filename or "onnx_ir/passes" in filename:
                    user_frame = f
                    break

            print("\033[1mUser Code Location:\033[0m")
            if user_frame is not None:
                print(
                    f"  \033[90m{user_frame.filename}:{user_frame.lineno} in {user_frame.name}\033[0m"
                )
                if user_frame.line:
                    print(f"  \033[90m>>> {user_frame.line}\033[0m")
            else:
                print("  \033[90m<unknown>\033[0m")

            print("\033[1mFull Stack Trace:\033[0m")
            for f in self.stack_trace:
                print(f"  \033[90m{f.filename}:{f.lineno} in {f.name}\033[0m")
                if f.line:
                    print(f"    \033[90m{f.line}\033[0m")
        print(f"\033[1m{'=' * 80}\033[0m")


def get_journal() -> Journal | None:
    """Get the current journal, if any."""
    return _current_journal


def _get_stack_trace() -> list[traceback.FrameSummary]:
    return traceback.extract_stack()[:-3]


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
        self._entries: list[JournalEntry] = []
        self._previous_journal: Journal | None = None
        self._hooks: list[Callable[[JournalEntry], None]] = []

    def __enter__(self) -> Self:
        global _current_journal
        self._previous_journal = _current_journal
        _current_journal = self
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        global _current_journal
        _current_journal = self._previous_journal

    def record(self, obj: Any, operation: str, details: str | None = None) -> None:
        """Record a new journal entry."""
        entry = JournalEntry(
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
        for hook in self._hooks:
            hook(entry)

    def add_hook(self, hook: Callable[[JournalEntry], None]) -> None:
        """Add a hook that will be called whenever a new journal entry is recorded."""
        self._hooks.append(hook)

    def clear_hooks(self) -> None:
        """Clear all hooks."""
        self._hooks.clear()

    def get_entries(self) -> Sequence[JournalEntry]:
        """Get all recorded journal entries."""
        return self._entries

    def filter(
        self, *, operation: str | None = None, class_name: str | None = None
    ) -> Sequence[JournalEntry]:
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
            details = f" [{entry.details}]" if entry.details else ""
            timestamp = datetime.datetime.fromtimestamp(entry.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
            if entry.stack_trace:
                # Find the first frame that is not from internal onnx_ir modules
                frame = None
                for f in reversed(entry.stack_trace):
                    # Normalize path separators for cross-platform compatibility
                    filename = f.filename.replace("\\", "/")
                    if "onnx_ir" not in filename or "onnx_ir/passes" in filename:
                        frame = f
                        break
                if frame is not None:
                    location = f"{frame.filename}:{frame.lineno} in {frame.name}"
                else:
                    location = "<unknown>"
            else:
                location = "<unknown>"
            print()
            print(f"[{timestamp}] \033[90m{location}\033[0m")
            if entry.ref is None:
                object_repr = "<no ref>"
            elif (obj := entry.ref()) is not None:
                object_repr = repr(obj).replace("\n", "\\n")
            else:
                object_repr = "<deleted>"
            details_text = details.replace("\n", "\\n")
            if len(details_text) > 100:
                details_text = details_text[:95] + "[...]"
            print(
                f"Class: {entry.class_name}(id={entry.object_id}). Operation: {entry.operation}. Object: {object_repr}."
            )
            if details:
                print(f"\033[90mDetails: {details_text}\033[0m")
