---
theme: default
title: "ONNX IR: Efficient In-Memory Representation"
info: |
  ## ONNX IR
  Data structure and API design for representing ONNX in memory.
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
---

# ONNX IR

## Efficient In-Memory Representation of ONNX

Data structure and API design

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/onnx/ir-py" target="_blank" class="text-xl slidev-icon-btn !border-none">
    <carbon-logo-github />
  </a>
</div>

<!--
Welcome. Today we'll look at onnx-ir — an in-memory IR for ONNX models designed for graph construction, analysis, and transformation. We'll explore the data structures, why they differ from protobuf, and useful APIs.
-->

---

# Why an In-Memory IR?

ONNX protobuf is a **serialization format**, not a manipulation format.

<v-clicks>

**Protobuf limitations:**
- 🔗 **String-based references** — nodes reference values by name strings
- 🔍 **No usage tracking** — finding consumers requires scanning all nodes
- ⚠️ **Unsafe mutation** — no support for modifying during iteration
- 📦 **Protobuf dependency** — need protobuf library for everything
- 📏 **2GB size limit** — protobuf message size cap

</v-clicks>

<v-click>

**What we want:**
- Direct object references (SSA-like)
- O(1) usage tracking
- Safe graph mutation during iteration
- Zero-copy tensor access, no size limits
- Protobuf-free after loading

</v-click>

<!--
The ONNX protobuf format is great for serialization and interchange, but it has fundamental limitations for in-memory manipulation. String-based lookups are slow, there's no way to find consumers of a value without scanning, and you can't safely modify a graph while iterating over it.
-->

---

# The IR Hierarchy

<div class="grid grid-cols-2 gap-8">
<div>

```
Model
├── graph: Graph
│   ├── inputs: list[Value]
│   ├── outputs: list[Value]
│   ├── initializers: dict[str, Value]
│   └── nodes: DoublyLinkedSet[Node]
│       └── Node
│           ├── inputs: list[Value | None]
│           ├── outputs: list[Value]
│           ├── attributes: dict[str, Attr]
│           ├── domain: str
│           ├── op_type: str
│           └── version: int | None
└── opset_imports: dict[str, int]
```

</div>
<div>

**Key classes:**
- **`Model`** — Top-level container
- **`Graph`** — Mutable graph (linked list)
- **`GraphView`** — Immutable snapshot (tuple)
- **`Node`** — Operation with typed attributes
- **`Value`** — Data flow entity (SSA-like)
- **`Shape`** — With symbolic dimensions
- **`Attr`** — Strongly typed attributes

<br/>

> 💡 None of these classes have `to_onnx` or `from_protobuf` methods — the IR is **protobuf-free** after loading.

</div>
</div>

<!--
Here's the entity hierarchy. At the top is Model, which contains a Graph. The Graph has inputs, outputs, initializers, and nodes stored in a doubly linked set. Each Node has inputs and outputs that are Value objects — direct references, not strings. Note that each node carries its own version, supporting mixed opset models.
-->

---

# Values: SSA-Like References

Values are the **edges** of the graph — they connect nodes with direct object references.

<div class="grid grid-cols-2 gap-4">
<div>

**Protobuf (string-based):**
```protobuf
graph {
  node {
    input: ["x"]
    output: ["y"]
    op_type: "Relu"
  }
  node {
    input: ["y"]
    output: ["z"]
    op_type: "Sigmoid"
  }
}
```

Finding consumers of `"y"` → scan all nodes 😩

</div>
<div>

**IR (direct references):**
```python
# Direct object access
y = relu_node.outputs[0]

# Who produces this value?
y.producer()  # → relu_node

# Who consumes it?
y.uses()  # → {Usage(sigmoid_node, idx=0)}

# Graph membership
y.is_graph_output()  # → False
y.is_initializer()   # → False
```

Finding consumers of `y` → O(1) 🚀

</div>
</div>

<!--
This is the biggest difference. In protobuf, everything is connected by string names — to find who uses a value, you'd have to scan every node. In the IR, each Value object knows its producer node and all its consumers. This makes graph traversal and transformation dramatically faster and more intuitive.
-->

---

# Initializers: How They're Stored

Initializers are **Values with `const_value` set**, stored in a dict on the Graph.

```python
# Graph.initializers → GraphInitializers(UserDict[str, Value])
graph.initializers                     # dict-like: {name: Value}
graph.initializers["weight"]           # Access by name → Value
graph.initializers["weight"].const_value  # → TensorProtocol (the actual data)
```

<v-click>

**Key design decisions:**
- Initializers are `Value` objects — same type as graph inputs/outputs
- The **tensor data** lives on `value.const_value`, not the Value itself
- `GraphInitializers` enforces invariants:
  - Value must have a name
  - Value cannot be produced by a node (must be a "free" value)
  - Tracks graph ownership — a Value belongs to exactly one graph

</v-click>

<v-click>

```python
# Adding an initializer
value = ir.Value(name="bias", const_value=ir.tensor(np.zeros(10)))
graph.initializers.add(value)        # or: graph.initializers["bias"] = value

# Checking membership
value.is_initializer()  # → True
value.graph             # → the owning graph
```

</v-click>

<!--
Initializers are values with constant data attached. They're stored in a dict for O(1) name lookup. The GraphInitializers class extends UserDict and enforces invariants — values must be named, can't be produced by a node, and track which graph they belong to. This is different from protobuf where initializers are a separate repeated field of TensorProto.
-->

---

# TensorProtocol: Unified Interface

A single protocol for **all** tensor types — numpy, PyTorch, disk-backed, lazy, packed.

```python
@runtime_checkable
class TensorProtocol(ArrayCompatible, DLPackCompatible, Protocol):
    name: str | None
    shape: ShapeProtocol
    dtype: DataType
    raw: Any                              # The backing data (any type)

    def numpy(self) -> np.ndarray: ...    # Universal access
    def tobytes(self) -> bytes: ...       # For serialization
    def __array__(self, dtype=None): ...  # np.array(tensor) works
    def __dlpack__(self, *, stream=...): ...  # Zero-copy framework interop
```

<v-click>

**Two interop protocols baked in:**
- **`ArrayCompatible`** (`__array__`) — works with `np.array(tensor)`
- **`DLPackCompatible`** (`__dlpack__`) — zero-copy exchange with PyTorch, JAX, TensorFlow

> 💡 You never need to check what kind of tensor you have — just call `.numpy()` or pass it to your framework.

</v-click>

<!--
The TensorProtocol is the unified interface for all tensor types. It inherits from both ArrayCompatible and DLPackCompatible, which means any IR tensor works with numpy's np.array() and supports zero-copy exchange with other frameworks through DLPack. The raw attribute holds the actual backing data, which could be anything — a numpy array, an mmap object, a PyTorch tensor.
-->

---

# Tensor Implementations

<div class="grid grid-cols-2 gap-6">
<div>

| Type | Backing | Use Case |
|------|---------|----------|
| **`Tensor`** | numpy / DLPack | General purpose |
| **`ExternalTensor`** | mmap'd file | Large models |
| **`StringTensor`** | `bytes` seq | String data |
| **`LazyTensor`** | callable | Deferred eval |
| **`PackedTensor`** | packed bytes | 2/4-bit quant |
| **`TorchTensor`** | torch.Tensor | PyTorch interop |

</div>
<div>

**ExternalTensor — mmap for large models:**
```python
class ExternalTensor:
    """Disk-backed tensor via memory mapping."""

    def _load(self):
        with open(self.path, "rb") as f:
            self.raw = mmap.mmap(
                f.fileno(), 0,
                access=mmap.ACCESS_READ
            )

    def numpy(self):
        return np.frombuffer(
            self.raw, dtype=...
        )  # No copy!
```

No 2GB limit. Data stays on disk until accessed.

</div>
</div>

<v-click>

**Zero-copy philosophy:** Data flows from disk → mmap → numpy view → DLPack → framework tensor. No copies at any step.

</v-click>

<!--
We have multiple tensor implementations, all satisfying TensorProtocol. The most interesting is ExternalTensor which uses mmap to memory-map files on disk, avoiding the 2GB protobuf limit entirely. When you call numpy(), it returns a view via np.frombuffer — no copy. The zero-copy chain goes all the way from disk to your framework.
-->

---

# Graph: DoublyLinkedSet

Nodes are stored in a **doubly-linked ordered set** — not a Python list.

<div class="grid grid-cols-2 gap-4">
<div>

**Why not a `list`?**

| Operation | `list` | `DoublyLinkedSet` |
|-----------|--------|-------------------|
| Append | O(1)* | **O(1)** |
| Remove by value | **O(n)** | **O(1)** |
| Insert before/after | **O(n)** | **O(1)** |
| `__contains__` | O(n) | **O(1)** |
| Random access `[i]` | O(1) | O(n)† |
| `len()` | O(1) | O(1) |
| Iterate + mutate | ❌ **Broken** | ✅ **Safe** |

<div class="text-xs mt-1">

*amortized &nbsp; †O(1) for `[0]` and `[-1]`

</div>

</div>
<div>

**It's also a set — no duplicates:**

```python
graph.append(node_a)
graph.append(node_a)  # Silently moves,
                      # no duplicate!
```

Backed by `dict[id(value), _LinkBox]` — identity-based membership in O(1).

If a node is already in the set, inserting it again **moves** it to the new position instead of creating a duplicate.

> 💡 You never need to worry about accidentally adding a node twice.

</div>
</div>

<!--
The heart of the graph is the DoublyLinkedSet — a doubly-linked list that also enforces set semantics. Why not a regular Python list? Because remove and insert are O(n), contains is O(n), and you can't safely modify a list while iterating. With our linked set, all those operations are O(1), and it guarantees no duplicate nodes. If you accidentally append a node that's already in the graph, it just moves it — no duplicates, no crash.
-->

---

# The Ski Lift: LinkBox Architecture

Think of it like a **ski lift** 🎿 — chairs on a cable, skiers in chairs.

<div class="text-xs">

```
  The circular cable (prev ⇄ next pointers)
  ╭──────────────────────────────────────────────────────────────────────────────────╮
  │                                                                                  │
  │    ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐            │
  ╰──▶ │ Root 🏔️ │ ⇄───▶ │ LinkBox │ ⇄───▶ │ LinkBox │ ⇄───▶ │ LinkBox │ ⇄─────────╯
       │(sentinel)│       │  chair  │       │  chair  │       │  chair  │
       │ val=None │       │         │       │         │       │         │
       └─────────┘       └────┬────┘       └────┬────┘       └────┬────┘
                              │ .value          │ .value          │ .value
                              ▼                 ▼                 ▼
                          Node A 🎿         Node B 🎿         Node C 🎿
                          (skier)           (skier)           (skier)

       ╔══════════════════════════════════════════════════════════════════╗
       ║  Lookup board:  { id(A): Box₁,  id(B): Box₂,  id(C): Box₃ }  ║
       ╚══════════════════════════════════════════════════════════════════╝
```

</div>

<div class="grid grid-cols-4 gap-3 text-sm mt-2">
<div>🔗 <b>Cable</b> = prev/next chain</div>
<div>💺 <b>Chairs</b> = _LinkBox containers</div>
<div>🎿 <b>Skiers</b> = Node objects</div>
<div>🏔️ <b>Station</b> = _root sentinel</div>
</div>

<v-click>

**Why separate the chair from the skier?** When a skier gets off (node removed), the chair's cable connections still work. The iterator follows the cable, skipping empty chairs. The cable is never broken.

</v-click>

<!--
Think of the DoublyLinkedSet as a circular ski lift. The LinkBox objects are chairs on the cable, connected by prev and next pointers. The actual Node objects are skiers sitting in those chairs. The root sentinel is the loading station — an empty chair connecting the loop. The lookup board is a dict mapping node identity to its LinkBox for O(1) access.

The key insight is separating the chair from the skier. When a node is removed, we take the skier off the chair, but the cable connections — prev and next — still work. The iterator follows the cable from chair to chair, skipping empty ones. This is why you can safely remove nodes during iteration — the cable is never broken.
-->

---

# Removal, Insertion, and the Iterator

<div class="grid grid-cols-2 gap-4">
<div>

**Removing a node (skier gets off):**
```
Before:  [A] ⇄ [B] ⇄ [C]
Remove B:
  1. A.next → C, C.prev → A  (cable)
  2. B.value = None           (empty chair)

After:   [A] ⇄ [C]
              ↗
         [B̸]  ← ghost box, next still → C
```
Iterator at B? Follows `box.next` → arrives at C ✅

</div>
<div>

**Inserting a node (new skier boards):**
```
Before:  [A] ⇄ [B] ⇄ [C]
Insert D after B:
  1. Create new LinkBox for D
  2. B.next → D, D.prev → B
  3. D.next → C, C.prev → D

After:   [A] ⇄ [B] ⇄ [D] ⇄ [C]
```
Iterator at B? Will visit D next ✅
Iterator at C? D is behind, won't revisit ✅

</div>
</div>

<v-click>

**The set guarantee during insertion:**
```python
def _insert_one_after(self, box, new_value):
    if id(new_value) in self._value_ids_to_boxes:
        self.remove(new_value)   # ← Already in set? Move it, don't duplicate!
    new_box = _LinkBox(self, new_value)
    # ... wire up prev/next ...
```

> 🎿 A skier can only sit in **one chair at a time**. Boarding a new chair automatically vacates the old one.

</v-click>

<!--
Let's see how removal and insertion work. When you remove node B, the cable reconnects A to C directly, and B's chair is marked empty. But critically, the ghost box's next pointer still points to C — so if the iterator was looking at B, it follows next and arrives at C safely.

For insertion, we create a new LinkBox chair and splice it into the cable. If the iterator is at B when we insert D after it, D is ahead on the cable and will be visited. If the iterator is already past that point at C, D is behind and won't cause a revisit.

And the set guarantee: if you try to insert a node that's already somewhere in the set, it's automatically removed from its old position first. One skier, one chair.
-->

---

# Safe Iteration During Mutation

The key property that enables robust graph passes.

```python
def __iter__(self):
    box = self._root.next
    while box is not self._root:
        if not box.erased:         # Skip empty chairs
            yield box.value        # Yield the skier
        box = box.next             # Follow the cable →
```

<v-clicks>

**Guarantees:**
- ✅ **Insert after current** → new nodes **will be** visited
- ✅ **Insert before current** → new nodes **will not be** visited (no infinite loops)
- ✅ **Remove current node** → iterator continues from original next
- ✅ **Move current node** → iterator resumes from the original location
- ✅ **Multiple concurrent iterators** → all work independently
- ✅ **No duplicates** → set semantics prevent double-processing

</v-clicks>

<v-click>

```python
# This is completely safe:
for node in graph:
    if should_replace(node):
        graph.insert_before(node, [new_node])  # new_node won't be visited
        graph.remove([node], safe=True)         # iteration continues from next
```

</v-click>

<!--
Here's the actual iterator. It walks the cable from box to box. If a box is erased — an empty chair — it skips it. The iterator always follows the next pointer, which is why all the mutation guarantees hold. Insert after? The new box is ahead, you'll see it. Insert before? It's behind, you won't loop. Remove? The ghost box's next still points forward. Multiple iterators? Each follows its own position on the cable independently. And since it's a set, you'll never accidentally process a node twice.
-->

---

# Graph Passes in Practice

<div class="grid grid-cols-2 gap-4">
<div>

**Identity Elimination:**
```python
# Forward iteration — replace and remove
for node in ir.traversal
        .RecursiveGraphIterator(graph):
    if node.op_type == "Identity":
        # Rewire: users of output
        #   now use input directly
        ir.convenience.replace_all_uses_with(
            node.outputs[0],
            node.inputs[0]
        )
        graph.remove([node], safe=True)
```

</div>
<div>

**Unused Node Removal:**
```python
# Reverse iteration — bottom-up cleanup
graph_outputs = set(graph.outputs)

for node in reversed(graph):
    if all(
        len(o.uses()) == 0
        and o not in graph_outputs
        for o in node.outputs
    ):
        graph.remove([node], safe=True)
```

</div>
</div>

<v-click>

> 💡 **Forward** iteration for rewriting patterns. **Reverse** iteration for dead code elimination (ensures consumers are checked before producers).

</v-click>

<!--
Here are two common pass patterns. Identity elimination walks forward, rewiring each identity's consumers to use the input directly, then removes the node. Unused removal walks backward — if no output of a node is used and none are graph outputs, the node is dead and can be removed. Reverse order ensures that when you remove a node, its would-be consumers have already been checked. Both patterns are safe because of the linked list.
-->

---

# Convenience APIs

Helpers that make common operations easy and safe.

```python
from onnx_ir import convenience
```

<div class="grid grid-cols-2 gap-4">
<div>

**`get_const_tensor(value)`**
```python
# Gets constant data from a Value — handles
# both initializers and Constant nodes
tensor = convenience.get_const_tensor(value)
if tensor is not None:
    data = tensor.numpy()
```

**`replace_all_uses_with(old, new)`**
```python
# Rewire all consumers of old_val
# to use new_val instead
convenience.replace_all_uses_with(
    old_val, new_val
)
```

</div>
<div>

**`replace_nodes_and_values(...)`**
```python
# Atomic replacement: swap old nodes/values
# for new ones at an insertion point
convenience.replace_nodes_and_values(
    graph,
    insertion_point=node,
    old_nodes=[relu, bias_add],
    new_nodes=[fused_op],
    old_values=[relu.outputs[0]],
    new_values=[fused_op.outputs[0]],
)
# Propagates metadata, rewires consumers,
# inserts new nodes, removes old ones
```

**`create_value_mapping(graph)`**
```python
# Build name → Value dict for the whole graph
mapping = convenience.create_value_mapping(graph)
```

</div>
</div>

<!--
These convenience APIs handle the most common graph manipulation patterns. get_const_tensor abstracts over both initializers and Constant nodes. replace_all_uses_with rewires all consumers. replace_nodes_and_values does an atomic swap — it propagates metadata like shape and type from old values to new ones, rewires consumers, inserts the new nodes, and removes the old ones in one operation.
-->

---

# Key Takeaways

<v-clicks>

1. 🧠 **SSA-like Values** — Direct object references replace string lookups. O(1) producer/consumer access.

2. 📦 **Initializers as Values** — Unified treatment. Tensor data on `const_value`. Dict-based for O(1) name access.

3. 🔌 **TensorProtocol** — One interface for numpy, PyTorch, mmap, lazy, packed tensors. Zero-copy all the way.

4. 🔗 **DoublyLinkedSet** — O(1) insert/remove. Safe iteration during mutation. Enables robust graph passes.

5. 🛠️ **Convenience APIs** — `get_const_tensor`, `replace_all_uses_with`, `replace_nodes_and_values`.

6. 🚫 **No protobuf dependency** — The IR is completely protobuf-free after loading.

</v-clicks>

---
layout: center
---

# Thank You

<div class="text-center">

**onnx-ir** — An in-memory IR for ONNX

`pip install onnx-ir`

<br/>

<div class="flex gap-4 justify-center">
  <a href="https://github.com/onnx/ir-py" target="_blank">GitHub: onnx/ir-py</a>
  <span>•</span>
  <a href="https://onnx.ai/ir-py/" target="_blank">Docs: onnx.ai/ir-py</a>
  <span>•</span>
  <a href="https://pypi.org/project/onnx-ir" target="_blank">PyPI: onnx-ir</a>
</div>

</div>

<!--
Thank you! Check out the project on GitHub, the docs at onnx.ai/ir-py, and install it with pip install onnx-ir. Questions?
-->
