# onnx_ir.journaling

```{eval-rst}
.. automodule:: onnx_ir.journaling
.. currentmodule:: onnx_ir.journaling
```

The `onnx_ir.journaling` module provides a journaling system for tracking and debugging operations performed on ONNX IR objects. This is useful for understanding how a model is transformed, debugging unexpected behavior, or auditing changes made to a model.

## Quick Start

Use the `Journal` class as a context manager to record operations:

```python
import onnx_ir as ir
from onnx_ir.journaling import Journal

model = ir.load("model.onnx")

with Journal() as journal:
    # Perform operations on the model
    for node in model.graph:
        node.name = f"renamed_{node.name}"

# View all recorded operations
journal.display()

# Or filter by specific criteria
filtered_entries = [
    entry for entry in journal.entries
    if entry.operation == "set_name" and entry.class_name == "Node"
]
for entry in filtered_entries:
    print(f"{entry.operation} on {entry.class_name}")
```

## The `Journal` class

The `Journal` class is the main interface for recording operations on the ONNX IR. It captures details about each operation including the object involved, a timestamp, and a stack trace for debugging.

```{eval-rst}
.. autoclass:: Journal
   :members:
   :undoc-members:
```

## The `JournalEntry` class

Each recorded operation is stored as a `JournalEntry`. This dataclass contains all the information about a single operation.

```{eval-rst}
.. autoclass:: JournalEntry
   :members:
   :undoc-members:
```

## Using Hooks

You can add hooks to be notified whenever a new entry is recorded. This is useful for real-time monitoring or custom logging:

```python
from onnx_ir.journaling import Journal, JournalEntry

def my_hook(entry: JournalEntry) -> None:
    print(f"Operation recorded: {entry.operation} on {entry.class_name}")

journal = Journal()
journal.add_hook(my_hook)

with journal:
    # Operations will trigger the hook as they are recorded
    ...
```

## Inspecting Entries

The `JournalEntry.display()` method provides a detailed, multi-line view of an entry including:

- Timestamp
- Operation name
- Class and object ID
- Object representation
- Details (if provided)
- User code location (the first stack frame outside of onnx_ir internals)
- Full stack trace

```python
for entry in journal.entries:
    entry.display()  # Detailed multi-line output
```

The `Journal.display()` method provides a more compact summary of all entries, suitable for quick overview.
