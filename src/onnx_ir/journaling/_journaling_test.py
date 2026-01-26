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

        entries = journal.entries
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.operation, "test_operation")
        self.assertEqual(entry.class_name, "_DummyObject")
        self.assertEqual(entry.details, "test details")
        self.assertEqual(entry.object_id, id(obj))
        self.assertIsNotNone(entry.ref)
        self.assertIs(entry.ref(), obj)

    def test_journal_context_manager_sets_current_journal(self):
        self.assertIsNone(journaling.get_current_journal())

        journal = journaling.Journal()
        with journal:
            self.assertIs(journaling.get_current_journal(), journal)

        self.assertIsNone(journaling.get_current_journal())

    def test_journal_context_manager_restores_previous_journal(self):
        outer_journal = journaling.Journal()
        inner_journal = journaling.Journal()

        with outer_journal:
            self.assertIs(journaling.get_current_journal(), outer_journal)
            with inner_journal:
                self.assertIs(journaling.get_current_journal(), inner_journal)
            self.assertIs(journaling.get_current_journal(), outer_journal)

        self.assertIsNone(journaling.get_current_journal())

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

        entry = journal.entries[0]
        self.assertIsInstance(entry.timestamp, float)
        self.assertGreater(entry.timestamp, 0)

    def test_journal_entry_has_stack_trace(self):
        journal = journaling.Journal()
        obj = _DummyObject()

        with journal:
            journal.record(obj, "test_operation")

        entry = journal.entries[0]
        self.assertIsInstance(entry.stack_trace, list)
        self.assertGreater(len(entry.stack_trace), 0)

    def test_journal_entry_ref_returns_none_after_object_deleted(self):
        journal = journaling.Journal()

        with journal:
            obj = _DummyObject()
            journal.record(obj, "test_operation")
            entry = journal.entries[0]
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

        entry = journal.entries[0]
        # Should not raise
        entry.display()

    def test_journal_entry_display_handles_deleted_object(self):
        """Test that display() handles deleted objects gracefully."""
        journal = journaling.Journal()

        with journal:
            obj = _DummyObject()
            journal.record(obj, "test_operation")

        entry = journal.entries[0]
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


class JournalIntegrationTest(unittest.TestCase):
    """Integration tests for journaling with real IR objects.

    These tests verify that wrap_ir_classes() and restore_ir_classes() work
    correctly with Node, Value, and Graph objects.
    """

    def test_journal_records_node_creation(self):
        """Test that creating a Node inside a Journal context records entries."""
        import onnx_ir as ir

        journal = journaling.Journal()

        with journal:
            node = ir.node("Add", inputs=[ir.val("x"), ir.val("y")], outputs=[ir.val("sum")])

        # Should have recorded entries for Value inits and Node init
        entries = journal.entries
        self.assertGreater(len(entries), 0, "Journal should have recorded entries")

        # Check that we recorded Value and Node init operations
        class_names = {entry.class_name for entry in entries}
        self.assertIn("Value", class_names, "Should record Value creation")
        self.assertIn("Node", class_names, "Should record Node creation")

        # Check that operations are "init"
        operations = {entry.operation for entry in entries}
        self.assertIn("init", operations, "Should record init operations")

        # Verify no exceptions were raised and node was created successfully
        self.assertIsNotNone(node)
        self.assertEqual(node.op_type, "Add")

    def test_journal_records_node_property_modifications(self):
        """Test that modifying Node properties records entries."""
        import onnx_ir as ir

        journal = journaling.Journal()

        # Create node outside journal first
        node = ir.node("Add", inputs=[ir.val("x"), ir.val("y")], outputs=[ir.val("sum")])

        initial_entry_count = len(journal.entries)

        with journal:
            # Modify node properties
            node.name = "my_add_node"
            node.domain = "custom.domain"

        # Should have recorded the property changes
        new_entries = journal.entries[initial_entry_count:]
        self.assertGreater(len(new_entries), 0, "Should record property modifications")

        # Check operations include set_name and set_domain
        operations = [entry.operation for entry in new_entries]
        self.assertIn("set_name", operations, "Should record name change")
        self.assertIn("set_domain", operations, "Should record domain change")

        # Verify the properties were actually set
        self.assertEqual(node.name, "my_add_node")
        self.assertEqual(node.domain, "custom.domain")

    def test_journal_records_value_creation(self):
        """Test that creating Value objects records entries."""
        import onnx_ir as ir

        journal = journaling.Journal()

        with journal:
            value1 = ir.val("input1")
            value2 = ir.val("input2")

        # Should have recorded Value creation
        entries = journal.entries
        self.assertGreaterEqual(len(entries), 2, "Should record at least 2 Value creations")

        # All entries should be for Value init
        for entry in entries:
            self.assertEqual(entry.class_name, "Value")
            self.assertEqual(entry.operation, "init")

        # Verify values were created successfully
        self.assertIsNotNone(value1)
        self.assertIsNotNone(value2)
        self.assertEqual(value1.name, "input1")
        self.assertEqual(value2.name, "input2")

    def test_journal_records_graph_creation(self):
        """Test that creating a Graph records entries."""
        import onnx_ir as ir

        journal = journaling.Journal()

        with journal:
            # Create inputs and outputs
            x = ir.val("x")
            y = ir.val("y")
            z = ir.val("z")

            # Create a simple node
            add_node = ir.node("Add", inputs=[x, y], outputs=[z])

            # Create a graph
            graph = ir.Graph(
                inputs=[x, y],
                outputs=[z],
                nodes=[add_node],
            )

        # Should have recorded multiple entries including Graph init
        entries = journal.entries
        self.assertGreater(len(entries), 0, "Should record entries")

        # Check that Graph was recorded
        class_names = [entry.class_name for entry in entries]
        self.assertIn("Graph", class_names, "Should record Graph creation")

        # Verify graph was created successfully
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.inputs), 2)
        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(len(list(graph)), 1)

    def test_journal_with_multiple_operations(self):
        """Test journal with multiple operations on different IR objects."""
        import onnx_ir as ir

        journal = journaling.Journal()

        with journal:
            # Create values
            a = ir.val("a")
            b = ir.val("b")
            c = ir.val("c")

            # Create node
            mul_node = ir.node("Mul", inputs=[a, b], outputs=[c])

            # Modify node
            mul_node.name = "multiply"

        # Should have recorded all operations
        entries = journal.entries
        self.assertGreater(len(entries), 3, "Should record multiple operations")

        # Verify operations are tracked correctly
        operations = [entry.operation for entry in entries]
        self.assertIn("init", operations)
        self.assertIn("set_name", operations)

        # Verify no exceptions and all objects exist
        self.assertEqual(a.name, "a")
        self.assertEqual(mul_node.op_type, "Mul")
        self.assertEqual(mul_node.name, "multiply")

    def test_journal_context_manager_wraps_and_restores_classes(self):
        """Test that Journal context manager properly wraps and restores IR classes."""
        import onnx_ir as ir

        journal = journaling.Journal()

        # Create a node outside the journal - should not be recorded
        node_before = ir.node("Add", inputs=[ir.val("x")], outputs=[ir.val("y")])
        entries_before = len(journal.entries)

        with journal:
            # Create a node inside journal - should be recorded
            node_inside = ir.node("Sub", inputs=[ir.val("a")], outputs=[ir.val("b")])

        entries_during = len(journal.entries) - entries_before
        self.assertGreater(entries_during, 0, "Operations inside journal should be recorded")

        # Create a node after journal - should not be recorded to this journal
        node_after = ir.node("Mul", inputs=[ir.val("p")], outputs=[ir.val("q")])
        entries_after = len(journal.entries) - entries_before - entries_during

        # No new entries should be added after exiting journal context
        self.assertEqual(entries_after, 0, "Operations after journal should not be recorded")

        # Verify all nodes were created successfully
        self.assertIsNotNone(node_before)
        self.assertIsNotNone(node_inside)
        self.assertIsNotNone(node_after)


if __name__ == "__main__":
    unittest.main()
