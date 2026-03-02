---
theme: seriph
title: "ONNX IR: Efficient Representation of ONNX In-Memory"
info: |
  ## ONNX IR: Efficient Representation of ONNX In-Memory
  Data Structure and API Design
layout: cover
class: text-center

transition: slide-left
mdc: true
overviewSnapshots: true
---

# ONNX IR

## Efficient In-Memory Representation of ONNX

<div class="text-xl mt-2 op-80">
Data Structure and API Design
</div>

<div class="abs-bl mx-14 my-12 flex flex-col text-left">
  <div class="text-sm op-60">
    Speaker Name · ONNX Project
  </div>
</div>

<style>
h1 {
  font-size: 3.5em !important;
  font-weight: 800 !important;
  line-height: 1.1 !important;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
h2 {
  font-size: 1.5em !important;
  font-weight: 400 !important;
  opacity: 0.9;
}
</style>

<!--
Welcome everyone. Today I'll talk about the ONNX IR — a Python in-memory representation for ONNX models that's designed from the ground up for mutation, lazy loading, and graph transformation passes. If you've ever tried to modify an ONNX model using protobuf objects directly, you know the pain we're solving.
-->

---
transition: fade-out
---

# Agenda

<v-clicks>

- 🔍 **The Problem** — Why not just use protobuf?
- 🏗️ **Design Principles** — Object identity, back-references, mutation-friendly
- 📦 **Initializers** — From string-matched lists to unified Values
- 🧩 **Tensor Protocol** — One interface for NumPy, PyTorch, external data
- 🔗 **Node Storage** — Linked lists for safe graph mutation
- 🔄 **Safe Iteration** — Mutate the graph while you traverse it
- 🛠️ **Convenience APIs** — Replace uses, create nodes, manipulate graphs
- ⚙️ **Passes & PassManager** — Composable graph transformations
- 📊 **Summary** — Protobuf vs IR at a glance

</v-clicks>

<!--
Here's what we'll cover in the next 20 minutes. We'll start with the motivation, walk through the key data structures and design decisions, and finish with how you actually use the IR to write graph passes.
-->

---
transition: slide-up
---

# The Problem: Why Not Just Use Protobuf?

ONNX protobuf was designed for **serialization**, not **manipulation**.

<v-clicks>

- **No object identity** — nodes and values are just repeated message fields
- **String-based references** — initializers matched to inputs by name (O(n) lookup)
- **Unsafe mutation** — removing a node from a `repeated` field shifts indices
- **No back-references** — "who uses this value?" requires a full graph scan
- **Eager loading** — every tensor must be fully deserialized into memory
- **No mixed opsets** — version is per-graph, not per-node

</v-clicks>

<div v-click class="mt-8 p-4 bg-red-500/10 rounded-lg">
  <strong>Bottom line:</strong> Protobuf is great for interchange, terrible for transformation.
</div>

<!--
If you've written an ONNX optimizer, you've hit these problems. Removing a node can invalidate your iteration. Finding all uses of a value means scanning every node. Loading a 2GB model means 2GB in memory even if you only need the graph structure. The IR solves all of these.
-->

---

# Key Design Principles

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

<v-click>

### Object Identity
Every `Node`, `Value`, and `Graph` is a **unique Python object** with stable identity.

```python
value = node.outputs[0]
assert value is other_ref  # Same object
```

</v-click>

<v-click>

### Back-References
Every `Value` knows its **producer** and all its **consumers**.

```python
value.producer()   # → Node that created it
value.uses()       # → All Usage(node, idx) pairs
```

</v-click>

</div>
<div>

<v-click>

### Mutation-Friendly
Data structures are designed for **safe in-place modification**.

```python
for node in graph:
    if should_remove(node):
        graph.remove(node)  # ✅ Safe! O(1)
```

</v-click>

<v-click>

### Lazy Loading
Tensors load on **first access**, not on model load.

```python
model = ir.load("large_model.onnx")
# 2GB stays on disk until accessed
data = value.const_value.numpy()
```

</v-click>

</div>
</div>

<!--
These four principles guide every design decision. Object identity means Python `is` comparisons work. Back-references make "find all uses" O(1). Mutation-friendly data structures make iteration safe. Lazy loading lets you work with models much larger than memory.
-->

---
layout: two-cols
layoutClass: gap-8
---

# Initializers: The Protobuf Way

In ONNX protobuf, initializers are a **separate list** matched by name:

```protobuf
message GraphProto {
  repeated NodeProto node = 1;
  repeated TensorProto initializer = 5;
  repeated ValueInfoProto input = 11;
}
```

<div class="mt-4">

**Problems:**
- String matching: O(n) to find an initializer
- Initializers duplicated in `input` list
- No connection between Value and its data
- Must deserialize entire TensorProto

</div>

```python
# Protobuf: find initializer for an input
for init in graph.initializer:
    if init.name == input_name:  # O(n) scan
        return init
```

::right::

# Initializers: The IR Way

In the IR, initializers are a **dict** on `Value` objects:

```python
class Graph:
    initializers: dict[str, Value]
    # O(1) lookup by name
```

<div class="mt-4">

**Benefits:**
- O(1) lookup: `graph.initializers["weight"]`
- Value **owns** its tensor via `const_value`
- No duplication — one object, one source of truth
- Lazy: tensor loads on access

</div>

```python
# IR: direct access
value = graph.initializers["weight"]
tensor = value.const_value  # TensorProtocol
data = tensor.numpy()       # Load on demand
```

<!--
This is one of the most visible differences. In protobuf, you have this awkward dance of matching initializer names to input names. In the IR, a Value object directly holds its constant tensor. The graph maintains a dict for O(1) lookup. Much cleaner.
-->

---

# The Value Object: Unifying Constants and Computations

A `Value` is the central abstraction — it represents any data flowing through the graph.

```python
class Value:
    name: str | None              # Optional name
    shape: Shape | None           # Tensor shape (symbolic dims supported)
    type: TypeProtocol | None     # Full type info
    const_value: TensorProtocol | None  # Constant data (if any)

    def producer(self) -> Node | None:      # What node created this?
    def uses(self) -> Collection[Usage]:    # All Usage(node, idx) tuples
    def index(self) -> int | None:          # Which output of the producer?
```

<div class="grid grid-cols-3 gap-4 mt-6">
<div class="p-4 bg-blue-500/10 rounded-lg text-center">
  <strong>Graph Input</strong><br>
  <code>producer() → None</code><br>
  <code>const_value → None</code>
</div>
<div class="p-4 bg-green-500/10 rounded-lg text-center">
  <strong>Initializer</strong><br>
  <code>producer() → None</code><br>
  <code>const_value → Tensor</code>
</div>
<div class="p-4 bg-purple-500/10 rounded-lg text-center">
  <strong>Computed Value</strong><br>
  <code>producer() → Node</code><br>
  <code>const_value → None</code>
</div>
</div>

<!--
The Value object is the unifying concept. Whether it's a graph input, an initializer weight, or the output of a computation — it's all the same Value class. The difference is just which fields are populated. This means every API that works with Values works uniformly across all three cases. And the back-references — producer and uses — give you O(1) navigation in both directions.
-->

---

# The Tensor Protocol

`TensorProtocol` is a **structural typing interface** — any object with the right methods works.

```python
@runtime_checkable
class TensorProtocol(ArrayCompatible, DLPackCompatible, Protocol):
    name: str | None
    shape: ShapeProtocol
    dtype: DataType
    def numpy(self) -> np.ndarray: ...     # NumPy interop
    def tobytes(self) -> bytes: ...        # Serialization
    def __dlpack__(self, *, stream=...) -> Any: ...  # Zero-copy interop
```

<div class="grid grid-cols-3 gap-4 mt-6">
<div class="p-4 bg-blue-500/10 rounded-lg">
  <strong>Tensor</strong><br>
  In-memory NumPy array<br>
  <code class="text-sm">Tensor(np.array([1,2,3]))</code>
</div>
<div class="p-4 bg-green-500/10 rounded-lg">
  <strong>ExternalTensor</strong><br>
  Memory-mapped file (lazy)<br>
  <code class="text-sm">ExternalTensor(path, offset, length, ...)</code>
</div>
<div class="p-4 bg-purple-500/10 rounded-lg">
  <strong>TorchTensor</strong><br>
  PyTorch tensor (zero-copy)<br>
  <code class="text-sm">TorchTensor(torch.randn(3,4))</code>
</div>
</div>

<!--
This is the adapter pattern in action. The TensorProtocol defines what a tensor must look like — shape, dtype, numpy access, dlpack for zero-copy. Then we have multiple implementations. Tensor wraps a NumPy array. ExternalTensor memory-maps a file. TorchTensor wraps a PyTorch tensor. Your pass code doesn't care which one it gets — it just calls .numpy() or .dtype and it works.
-->

---

# ExternalTensor: Lazy Loading in Practice

Load a 2GB model without using 2GB of memory.

```python
# Simplified — actual class inherits TensorBase + TensorProtocol
class ExternalTensor(TensorBase):
    def __init__(self, location: str, offset: int | None, length: int | None,
                 dtype: DataType, shape: Shape, name: str, base_dir: str = ""):
        self._location = location    # Relative path to data file
        self._offset = offset        # Byte offset into file

    def numpy(self) -> np.ndarray:
        # Memory-maps the file on access — data stays on disk until needed
        path = os.path.join(self._base_dir, self._location)
        return np.memmap(path, dtype=..., offset=self._offset, ...)
```

<div class="mt-4 p-4 bg-yellow-500/10 rounded-lg">

**Why this matters:** Graph transformation passes often only touch the graph structure, not tensor data. With `ExternalTensor`, you can load and optimize a model's graph without ever reading the weights into memory.

</div>

<!--
ExternalTensor is where lazy loading becomes real. When you load an ONNX model with external data, each tensor is represented as a path, offset, and length. The actual data stays on disk until someone calls .numpy(). This means a pass that only manipulates graph structure — like removing identity nodes — never touches the tensor data at all. Huge memory savings.
-->

---

# Node Storage: The Ski Lift 🚡

Think of graph nodes like a **ski lift** — and you'll understand the `DoublyLinkedSet`.

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

<v-clicks>

- 🪢 **The cable** = the linked list chain
- 🪑 **The chairs** = `_LinkBox` containers
- ⛷️ **The skiers** = `Node` objects sitting in chairs
- 🎫 **The ticket booth** = `dict[id, _LinkBox]` for O(1) lookup

</v-clicks>

<div v-click class="mt-4 p-4 bg-green-500/10 rounded-lg">

You can **detach a chair** (remove a node) or **attach a new one** (insert) without stopping the lift. Riders ahead keep going — <strong>safe iteration</strong>.

</div>

</div>
<div v-click>

<div class="mt-2 p-4 bg-blue-500/10 rounded-lg">

**It's also a set!** Each skier gets exactly one chair — adding a duplicate **moves** it to the new position instead of creating a copy.

```python
# O(1) uniqueness — duplicates are moved, not copied
if id(node) in self._value_ids_to_boxes:
    self.remove(node)  # Detach from old position
# Then insert at new position
```

</div>

</div>
</div>

<!--
The ski lift metaphor makes this intuitive. The cable is always running — that's your iterator. Chairs are the _LinkBox wrappers. Skiers are your Node objects. You can detach a chair mid-ride and the cable keeps moving. New chairs clip on without stopping anything. And like a lift pass system, the dict tracks everyone by identity — no duplicates.
-->

---

# DoublyLinkedSet: Architecture

<div class="text-center mt-2 mb-4">

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                 │
   ▼              prev ◄──►  next                                    │
┌────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ _root  │◄──►│  _LinkBox    │◄──►│  _LinkBox    │◄──►│  _LinkBox    │
│sentinel│    │  value=NodeA │    │  value=NodeB │    │  value=NodeC │
└────────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                     │                   │                   │
                     ▼                   ▼                   ▼
                  Node A              Node B              Node C

   _value_ids_to_boxes = { id(A): Box₁, id(B): Box₂, id(C): Box₃ }
```

</div>

<v-click>

| Operation | Complexity | How |
|-----------|-----------|-----|
| **Append / Insert** | O(1) | Rewire prev/next pointers |
| **Remove** | O(1) | Dict lookup → erase box → rewire |
| **Contains** | O(1) | `id(node) in _value_ids_to_boxes` |
| **Length** | O(1) | Maintained counter `_length` |
| **Iterate** | O(n) | Walk the chain, skip erased |
| **Index [0] / [-1]** | O(1) | `_root.next` / `_root.prev` |

</v-click>

<!--
Here's the actual architecture. It's a circular doubly-linked list with a sentinel root node. Each element is wrapped in a _LinkBox. The dict maps object identity to boxes for O(1) lookup. The circular design means append is just "insert before root" — always O(1). The sentinel simplifies edge cases: there's always a valid prev and next.
-->

---

# Safe Iteration: The Erased-Flag Pattern

The key insight: **erased nodes' forward pointers remain valid**, so iterators follow the chain.

```
Before removal:              After removing NodeB:
  ┌───┐   ┌───┐   ┌───┐      ┌───┐           ┌───┐
  │ A │◄─►│ B │◄─►│ C │      │ A │◄─────────►│ C │
  └───┘   └───┘   └───┘      └───┘   ┌───┐   └───┘
                                      │ B │ ← erased (value=None)
                     Iterator ──►     └─┬─┘   but next still → C ✓
                                        └──────────►
```

```python
def __iter__(self) -> Iterator[T]:
    box = self._root.next
    while box is not self._root:
        if box.owning_list is not self:  # Detect moved nodes
            raise RuntimeError("Element not in list")
        if not box.erased:        # Skip removed nodes
            yield box.value
        box = box.next            # Forward pointer still valid after erasure
```

<div class="mt-2 p-4 bg-green-500/10 rounded-lg text-sm">

**The chair detaches from the cable, but its hook still points to the next chair.** Active iterators follow the chain safely — no index bookkeeping, no deferred deletion.

</div>

<!--
Here's the erased-flag pattern in detail. When you remove NodeB, its box gets unlinked from the chain — A now points to C. But B's own next pointer still points to C. So if an iterator was sitting on B, it follows next to C without any issue. The owning_list check also detects if a node was moved to a different graph during iteration.
-->

---

# Safe Iteration in Passes: A Real Example

Identity elimination — remove redundant `Identity` nodes while iterating the graph.

```python
class IdentityEliminationPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if node.op_type == "Identity":
                input_value = node.inputs[0]
                output_value = node.outputs[0]

                # Redirect all consumers of the output to the input
                ir.convenience.replace_all_uses_with(output_value, input_value)

                # Safe to remove during iteration!
                model.graph.remove(node, safe=True)
                modified = True

        return ir.passes.PassResult(model, modified)
```

<div class="mt-2 p-3 bg-blue-500/10 rounded-lg text-sm">
<strong>No deferred deletion.</strong> No index tracking. No second pass. Just iterate, check, remove.
</div>

<!--
This is what it looks like in practice. The identity elimination pass iterates over all nodes, finds Identity ops, redirects their consumers to use the input directly, and removes the node. Notice there's no bookkeeping — no "collect nodes to delete" list, no index adjustment. The linked list handles it all. This is the kind of clean pass code the IR enables.
-->

---

# Convenience APIs

The `ir.convenience` module and constructor helpers for common operations.

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

### Smart Constructors

```python
# Create tensors from anything
ir.tensor([1, 2, 3])        # → int64
ir.tensor(np.zeros((2,3)))  # → float64
ir.tensor(torch_tensor)     # → TorchTensor

# Create nodes with Pythonic attrs
ir.node("Conv", inputs=[x, w, b],
    attributes={"kernel_shape": [3, 3]},
    num_outputs=1)

# Create typed values
ir.val("x", ir.DataType.FLOAT, ["N", 42])
```

</div>
<div>

### Graph Manipulation

```python
# Replace all consumers of old → new
ir.convenience.replace_all_uses_with(
    old_value, new_value
)

# Atomic node+value replacement
ir.convenience.replace_nodes_and_values(
    graph, insertion_point=node,
    old_nodes=[a, b], new_nodes=[fused],
    old_values=[a.outputs[0]],
    new_values=[fused.outputs[0]],
)

# Resolve constant data
tensor = ir.convenience.get_const_tensor(val)
```

</div>
</div>

<div class="mt-4 p-3 bg-yellow-500/10 rounded-lg text-sm">
<strong>Why these matter:</strong> <code>ir.tensor()</code> auto-detects dtype from any input. <code>replace_all_uses_with</code> saves you from manually iterating <code>value.uses()</code>. These encode common patterns so passes stay concise.
</div>

<!--
These convenience APIs are the bread and butter of pass writing. replace_all_uses_with is probably the most used — it handles the tedious work of finding every consumer of a value and switching them to a new one. extract lets you carve out a subgraph, which is great for testing. These encode patterns you'd otherwise repeat in every pass.
-->

---

# Graph Passes & PassManager

Passes are the primary way to transform ONNX models in the IR.

```python
class PassBase(abc.ABC):
    @abc.abstractmethod
    def call(self, model: Model) -> PassResult: ...

class InPlacePass(PassBase):     # Modifies model in-place
    in_place = True

class FunctionalPass(PassBase):  # Returns a new model
    in_place = False
```

### PassManager: Composable & Iterative

```python
passes = ir.passes.PassManager(
    [
        IdentityEliminationPass(),
        CommonSubexpressionEliminationPass(),
        RemoveUnusedNodesPass(),
    ],
    steps=3,            # Run the sequence up to 3 times
    early_stop=True,    # Stop if no pass reports changes
)
result = passes(model)  # Returns PassResult(model, modified=True/False)
```

<!--
The pass infrastructure is simple but powerful. You subclass InPlacePass or FunctionalPass, implement call(), and return a PassResult. PassManager runs a sequence of passes, optionally repeating until a fixed point — meaning it keeps running until no pass reports changes. This is how you compose optimizations that enable each other.
-->

---

# Built-in Passes

The IR ships with production-ready optimization passes.

| Pass | What It Does |
|------|-------------|
| `IdentityEliminationPass` | Removes redundant `Identity` nodes |
| `CommonSubexpressionEliminationPass` | Deduplicates identical subexpressions |
| `RemoveUnusedNodesPass` | Dead code elimination — removes nodes with no consumers |
| `DeduplicateInitializersPass` | Merges duplicate weight tensors |
| `LiftConstantsToInitializersPass` | Converts `Constant` nodes to graph initializers |
| `TopologicalSortPass` | Orders nodes in dependency order |

<v-click>

<div class="mt-2 p-3 bg-gray-500/10 rounded-lg text-sm">
<strong>Plus:</strong> <code>ShapeInferencePass</code>, <code>CheckerPass</code>, <code>InlinePass</code>, <code>AddDefaultAttributesPass</code>, <code>ClearMetadataAndDocStringPass</code>, and more.
</div>

</v-click>

<div class="mt-2 p-3 bg-green-500/10 rounded-lg text-sm">
All passes follow the same <code>PassBase</code> interface — compose freely with <code>PassManager</code>.
</div>

<!--
These are the passes that ship with the IR today. You can compose any subset of them. The PassManager's fixed-point iteration means, for example, that after CSE creates new dead nodes, the unused removal pass will clean them up on the next iteration. Each pass is focused and single-purpose — the composition is where the power comes from.
-->

---

# Writing Your Own Pass

Three steps: subclass, implement `call`, return `PassResult`.

```python {all|3-7|8-13|all}
class FuseMatMulAddPass(ir.passes.InPlacePass):
    """Fuse MatMul + Add into Gemm."""
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if node.op_type != "Add":
                continue
            producer = node.inputs[0].producer()
            if producer is None or producer.op_type != "MatMul":
                continue
            gemm = ir.Node("", "Gemm",
                inputs=[producer.inputs[0], producer.inputs[1], node.inputs[1]],
                num_outputs=1)
            ir.convenience.replace_nodes_and_values(
                model.graph, insertion_point=node,
                old_nodes=[producer, node], new_nodes=[gemm],
                old_values=[node.outputs[0]], new_values=[gemm.outputs[0]])
            modified = True
        return ir.passes.PassResult(model, modified)
```

<!--
Writing your own pass is straightforward. Subclass InPlacePass, implement call, iterate nodes, and use replace_nodes_and_values for safe atomic replacement. The RecursiveGraphIterator handles subgraphs automatically. This MatMul+Add→Gemm fusion is a common real-world optimization pattern.
-->

---
layout: center
class: text-center
---

# 🖥️ Live Demo

<div class="text-xl text-gray-400 mt-4">

`ir.load()` → `PassManager()` → `ir.save()`

</div>

<div class="mt-8 text-gray-500">

Loading a model, running optimization passes, inspecting the result

</div>

<!--
[Optional demo slide — skip if short on time] Show a quick terminal demo: load an ONNX model with ir.load(), run a PassManager with identity elimination and unused removal, then save the optimized model. Show the node count before and after.
-->

---
layout: two-cols
layoutClass: gap-4
---

# Protobuf vs IR: Summary

| Aspect | Protobuf | IR |
|--------|----------|------|
| **Purpose** | Serialization | Manipulation |
| **Initializers** | String-matched list | `dict[str, Value]` |
| **Node storage** | Repeated field | DoublyLinkedSet |
| **Tensor data** | Eager `TensorProto` | Lazy `TensorProtocol` |
| **Back-refs** | None | `producer()`, `uses()` |
| **Mutation** | Unsafe | Safe during iteration |

::right::

<div class="mt-12">

### When to Use What

<div class="p-4 bg-blue-500/10 rounded-lg mt-4">
  <strong>Use Protobuf for:</strong><br>
  Saving/loading, interchange, validation
</div>

<div class="p-4 bg-green-500/10 rounded-lg mt-4">
  <strong>Use IR for:</strong><br>
  Optimization, analysis, passes, tooling
</div>

<div class="p-4 bg-gray-500/10 rounded-lg mt-4">

```python
import onnx_ir as ir

model = ir.load("model.onnx")  # → IR
# ... transform ...
ir.save(model, "optimized.onnx")  # → Proto
```

</div>
</div>

<!--
Here's the full comparison at a glance. Protobuf is the right choice for serialization — it's the interchange format. The IR is the right choice for everything else — analysis, optimization, tooling. And converting between them is a one-liner in each direction.
-->

---

# What You Can Do With ONNX IR

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

### Graph Optimization
- Remove identity nodes, dead code
- Common subexpression elimination
- Constant folding
- Custom fusion patterns

### Model Analysis
- Traverse nodes with `RecursiveGraphIterator`
- Inspect shapes, types, tensor data
- Validate with `CheckerPass`

</div>
<div>

### Memory-Efficient Processing
- Lazy-load multi-GB models
- Work with graph structure without loading weights
- Memory-map external tensor data

### Framework Integration
- Zero-copy with PyTorch via DLPack
- NumPy interop on all tensors
- Custom tensor adapters for any backend

</div>
</div>

<div class="mt-6 p-4 bg-blue-500/10 rounded-lg text-center">

```
pip install onnx-ir
```

</div>

<!--
To sum up — the IR gives you a mutation-friendly, memory-efficient, framework-agnostic way to work with ONNX models. Whether you're writing an optimizer, building a tool, or integrating with a framework, the IR's design makes your code cleaner and your tools faster. And it's just a pip install away.
-->

---
layout: center
class: text-center
---

# Thank You

<div class="text-2xl mt-4 mb-8 text-gray-400">
Questions?
</div>

<div class="grid grid-cols-3 gap-8 mt-8 text-sm text-gray-500">
<div>

📦 `pip install onnx-ir`

</div>
<div>

🔗 github.com/onnx/ir-py

</div>
<div>

📖 onnx.ai/ir-py

</div>
</div>

<!--
Thank you! I'm happy to take questions about the IR design, specific passes, or how to integrate it into your workflow.
-->
