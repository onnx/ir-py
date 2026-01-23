# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the journaling module."""

from __future__ import annotations

import unittest

from onnx_ir import journaling


class _DummyObject:
    """A simple object that can be weakly referenced."""

    def __init__(self, name: str = "dummy"):
        self.name = name

    def __repr__(self) -> str:
        return f"DummyObject({self.name!r})"


class JournalTest(unittest.TestCase):
    def test_journal_records_entries(self):
        journal = journaling.Journal()
        obj = _DummyObject("test")

        with journal:
            journal.record(obj, "test_operation", details="test details")

        entries = journal.get_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.operation, "test_operation")
        self.assertEqual(entry.class_name, "_DummyObject")
        self.assertEqual(entry.details, "test details")
        self.assertEqual(entry.object_id, id(obj))
        self.assertIsNotNone(entry.ref)
        self.assertIs(entry.ref(), obj)

    def test_journal_context_manager_sets_current_journal(self):
        self.assertIsNone(journaling.get_journal())

        journal = journaling.Journal()
        with journal:
            self.assertIs(journaling.get_journal(), journal)

        self.assertIsNone(journaling.get_journal())

    def test_journal_context_manager_restores_previous_journal(self):
        outer_journal = journaling.Journal()
        inner_journal = journaling.Journal()

        with outer_journal:
            self.assertIs(journaling.get_journal(), outer_journal)
            with inner_journal:
                self.assertIs(journaling.get_journal(), inner_journal)
            self.assertIs(journaling.get_journal(), outer_journal)

        self.assertIsNone(journaling.get_journal())

    def test_journal_filter_by_operation(self):
        journal = journaling.Journal()
        obj = _DummyObject()

        with journal:
            journal.record(obj, "op1")
            journal.record(obj, "op2")
            journal.record(obj, "op1")

        filtered = journal.filter(operation="op1")
        self.assertEqual(len(filtered), 2)
        for entry in filtered:
            self.assertEqual(entry.operation, "op1")

    def test_journal_filter_by_class_name(self):
        journal = journaling.Journal()

        with journal:
            journal.record(_DummyObject("str1"), "op1")
            journal.record(_DummyObject("str2"), "op3")

        filtered = journal.filter(class_name="_DummyObject")
        self.assertEqual(len(filtered), 2)
        for entry in filtered:
            self.assertEqual(entry.class_name, "_DummyObject")

    def test_journal_filter_by_operation_and_class_name(self):
        journal = journaling.Journal()

        with journal:
            journal.record(_DummyObject("obj1"), "op1")
            journal.record(_DummyObject("obj2"), "op2")

        filtered = journal.filter(operation="op1", class_name="_DummyObject")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].operation, "op1")
        self.assertEqual(filtered[0].class_name, "_DummyObject")

    def test_journal_hook_is_called_on_record(self):
        journal = journaling.Journal()
        obj = _DummyObject()
        recorded_entries: list[journaling.JournalEntry] = []

        def hook(entry: journaling.JournalEntry) -> None:
            recorded_entries.append(entry)

        journal.add_hook(hook)

        with journal:
            journal.record(obj, "op1")
            journal.record(obj, "op2")

        self.assertEqual(len(recorded_entries), 2)
        self.assertEqual(recorded_entries[0].operation, "op1")
        self.assertEqual(recorded_entries[1].operation, "op2")

    def test_journal_clear_hooks(self):
        journal = journaling.Journal()
        obj = _DummyObject()
        recorded_entries: list[journaling.JournalEntry] = []

        def hook(entry: journaling.JournalEntry) -> None:
            recorded_entries.append(entry)

        journal.add_hook(hook)

        with journal:
            journal.record(obj, "op1")
            journal.clear_hooks()
            journal.record(obj, "op2")

        self.assertEqual(len(recorded_entries), 1)
        self.assertEqual(recorded_entries[0].operation, "op1")

    def test_journal_entry_has_timestamp(self):
        journal = journaling.Journal()
        obj = _DummyObject()

        with journal:
            journal.record(obj, "test_operation")

        entry = journal.get_entries()[0]
        self.assertIsInstance(entry.timestamp, float)
        self.assertGreater(entry.timestamp, 0)

    def test_journal_entry_has_stack_trace(self):
        journal = journaling.Journal()
        obj = _DummyObject()

        with journal:
            journal.record(obj, "test_operation")

        entry = journal.get_entries()[0]
        self.assertIsInstance(entry.stack_trace, list)
        self.assertGreater(len(entry.stack_trace), 0)

    def test_journal_entry_ref_returns_none_after_object_deleted(self):
        journal = journaling.Journal()

        with journal:
            obj = _DummyObject()
            journal.record(obj, "test_operation")
            entry = journal.get_entries()[0]
            self.assertIs(entry.ref(), obj)

        # Delete the object
        del obj

        # The weak reference should now return None
        self.assertIsNone(entry.ref())


class JournalEntryTest(unittest.TestCase):
    def test_journal_entry_display_does_not_raise(self):
        """Test that display() doesn't raise an exception."""
        journal = journaling.Journal()
        obj = _DummyObject("test")

        with journal:
            journal.record(obj, "test_operation", details="some details")

        entry = journal.get_entries()[0]
        # Should not raise
        entry.display()

    def test_journal_entry_display_handles_deleted_object(self):
        """Test that display() handles deleted objects gracefully."""
        journal = journaling.Journal()

        with journal:
            obj = _DummyObject()
            journal.record(obj, "test_operation")

        entry = journal.get_entries()[0]
        del obj

        # Should not raise even though object is deleted
        entry.display()


class JournalDisplayTest(unittest.TestCase):
    def test_journal_display_does_not_raise(self):
        """Test that Journal.display() doesn't raise an exception."""
        journal = journaling.Journal()

        with journal:
            journal.record(_DummyObject("obj1"), "op1", details="details1")
            journal.record(_DummyObject("obj3"), "op3", details="details3")

        # Should not raise
        journal.display()

    def test_journal_display_empty_journal(self):
        """Test that display() works on an empty journal."""
        journal = journaling.Journal()
        # Should not raise
        journal.display()


if __name__ == "__main__":
    unittest.main()
