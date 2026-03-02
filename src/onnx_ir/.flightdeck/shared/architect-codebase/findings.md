# ONNX IR Codebase: Technical Deep-Dive

> Detailed architecture analysis for presentation slides.
> Covers storage, protocols, data structures, APIs, design differences, and usage patterns.

---

## 1. How Initializers Are Stored and Represented

### The Key Insight: Initializers Are Values, Not Tensors

In ONNX protobuf, initializers are stored as a flat list of `TensorProto` objects at the graph level, completely separate from the graph's inputs and node outputs. In `onnx_ir`, **initializers are `Value` objects with a `const_value` property set to a tensor**. This unifies the concept: a Value can be a graph input, a node output, an initializer, or even all three simultaneously.

### Storage Architecture

```
Graph
├── _inputs: GraphInputs (UserList[Value])
├── _outputs: GraphOutputs (UserList[Value])
├── _initializers: GraphInitializers (UserDict[str, Value])  ← dict keyed by name
└── _nodes: DoublyLinkedSet[Node]
```

**Graph.__init__** converts the initializer sequence into a name→Value dict:

```python
# _core.py, Graph.__init__
self._initializers = _graph_containers.GraphInitializers(
    self, {initializer.name: initializer for initializer in initializers}
)
```

### GraphInitializers Container

`GraphInitializers` extends `collections.UserDict[str, Value]` with graph-aware invariants:

```python
class GraphInitializers(collections.UserDict[str, "_core.Value"]):
    def __setitem__(self, key: str, value: _core.Value) -> None:
        # Validates: value must have no producer node
        # Validates: key must match value.name
        # Sets: value._is_initializer = True
        # Sets: value._graph = self._graph

    def __delitem__(self, key: str) -> None:
        # Unsets: value._is_initializer = False
        # Conditionally clears value._graph (only if not still owned as input/output)

    def add(self, value: _core.Value) -> None:
        self[value.name] = value
```

### Value Ownership Tracking

Each `Value` tracks its role via private flags:

```python
class Value:
    _is_graph_input: bool
    _is_graph_output: bool
    _is_initializer: bool
    _graph: Graph | None

    def _owned_by_graph(self) -> bool:
        return self._is_graph_input or self._is_graph_output or self._is_initializer
```

A single Value can be **both a graph input AND an initializer** — this mirrors ONNX's design where initializers can optionally appear in inputs for runtime overriding.

### The `const_value` Property

```python
@property
def const_value(self) -> TensorProtocol | None:
    """The backing constant tensor for the value.

    If the Value has a const_value and is part of a graph initializers
    dictionary, the value is an initialized value. Its const_value
    will appear as an initializer in the GraphProto when serialized.

    If the Value is not part of a graph initializers dictionary,
    the const_value field will be ignored during serialization.
    """
    return self._const_value
```

### `register_initializer` Convenience Method

```python
def register_initializer(self, value: Value) -> None:
    # Checks: value must have a name
    # Checks: no duplicate name (or must be same object)
    # Checks: value.const_value must be set
    self._initializers.add(value)
```

### Why This Design?

| ONNX Protobuf | onnx_ir |
|---|---|
| `graph.initializer` is `repeated TensorProto` | `graph.initializers` is `dict[str, Value]` |
| Tensors live separate from values | Tensor data lives on the Value via `const_value` |
| Name-based matching to connect initializer→input | Same Value object, no name matching needed |
| No O(1) lookup by name | O(1) dict lookup by name |
| Hard to tell if a value is an initializer | `value.is_initializer()` property |

**Benefits:**
- **Unified value model** — one type for inputs, outputs, and initializers
- **O(1) initializer lookup** by name (dict vs. linear scan)
- **No name-matching ambiguity** — object identity, not string matching
- **Easy to convert** Constant nodes ↔ initializers (just set/unset `const_value` and register)

---

## 2. The Tensor Protocol

### Protocol Hierarchy

```
ArrayCompatible (Protocol)     DLPackCompatible (Protocol)
    ↓                              ↓
    └──────── TensorProtocol ──────┘   (runtime_checkable)
                    ↑
            TensorBase (ABC)
            ↗        ↘
    Tensor[T]    ExternalTensor    StringTensor
```

### TensorProtocol Definition

```python
@typing.runtime_checkable
class TensorProtocol(ArrayCompatible, DLPackCompatible, Protocol):
    name: str | None
    shape: ShapeProtocol
    dtype: DataType
    doc_string: str | None
    raw: Any                              # Raw underlying data (backend-specific)
    metadata_props: MutableMapping[str, str]  # Serializable metadata
    meta: MutableMapping[str, Any]            # Non-serializable pass metadata

    @property
    def size(self) -> int: ...      # Number of elements
    @property
    def nbytes(self) -> int: ...    # Number of bytes

    def numpy(self) -> np.ndarray: ...
    def tobytes(self) -> bytes: ...          # ONNX-spec little-endian
    def __array__(self, dtype=None) -> np.ndarray: ...
    def __dlpack__(self, *, stream=None) -> Any: ...
    def __dlpack_device__(self) -> Any: ...
```

### Why a Protocol (not an ABC)?

Using `typing.Protocol` allows **structural subtyping** — any object with the right methods/properties satisfies `TensorProtocol` without explicitly inheriting from it. This means:
- PyTorch tensors can be wrapped without modifying PyTorch
- Third-party frameworks can participate without importing onnx_ir
- Tools can depend on the protocol without depending on implementations

### Concrete Tensor Implementations

#### `Tensor[TArrayCompatible]` — In-Memory Tensor

```python
class Tensor(TensorBase, Generic[TArrayCompatible]):
    def __init__(
        self,
        value: TArrayCompatible,  # numpy array, DLPack-compatible, etc.
        dtype: DataType | None = None,
        shape: Shape | None = None,
        name: str | None = None,
    ) -> None:
```

**Key design decisions:**
- **Immutable** — once created, data cannot be changed
- **Zero-copy** — stores the original `value` in `self._raw` without copying
- **Lazy conversion** — `.numpy()` called only when needed
- Supports numpy protocol (`__array__`) and DLPack protocol (`__dlpack__`)
- Special handling for `ml_dtypes` (bfloat16, float8, int4)

#### `ExternalTensor` — Disk-Backed Tensor

```python
class ExternalTensor(TensorBase):
    def __init__(
        self,
        location: os.PathLike | str,  # Relative path to data file
        offset: int | None,           # Byte offset in file
        length: int | None,           # Byte length
        dtype: DataType,
        shape: Shape,
        name: str,
        base_dir: os.PathLike | str = "",
    ) -> None:
```

**Key design decisions:**
- **Memory-mapped** — data loaded on demand via `mmap`
- `path` computed as `base_dir / location`
- Enables working with models too large for RAM
- Same `TensorProtocol` interface — callers don't need to know data is on disk

### Tensor Adapters

#### `TorchTensor` — PyTorch Adapter

```python
class TorchTensor(_core.Tensor):
    def __init__(self, tensor: torch.Tensor, name: str | None = None):
        super().__init__(tensor, dtype=from_torch_dtype(tensor.dtype), name=name)
```

**Features:**
- Wraps `torch.Tensor` to satisfy `TensorProtocol`
- Lazy dtype mapping (initialized on first use)
- Special `numpy()` handling for bfloat16, float8 (view as uint16/uint8 first)
- Direct `tobytes()` / `tofile()` via ctypes for zero-copy serialization
- Handles `FakeTensor` gracefully via `unset_fake_temporarily()`

**Dtype conversion functions:**

```python
from_torch_dtype(dtype: torch.dtype) -> ir.DataType
to_torch_dtype(dtype: ir.DataType) -> torch.dtype
```

Supports 18+ dtypes including float8 variants (E4M3FN, E5M2, E8M0) and sub-byte types (INT2, UINT2).

### How This Enables Lazy Loading

```
Model loaded from disk
  └─ ExternalTensor(location="weights.bin", offset=0, length=4096)
       └─ .numpy() → memory-maps file, returns np.ndarray  (only when accessed)
       └─ .raw → the raw mmap bytes

Model exported from PyTorch
  └─ TorchTensor(torch.randn(3,4))
       └─ .numpy() → tensor.numpy(force=True)  (stays on GPU until needed)
       └─ .raw → the original torch.Tensor

Model constructed in code
  └─ Tensor(np.array([1,2,3]))
       └─ .numpy() → returns the array directly (zero-copy)
       └─ .raw → the np.ndarray
```

The uniform `TensorProtocol` means passes and tools never need to know the backing storage.

---

## 3. How Nodes Are Stored in a Graph

### The DoublyLinkedSet

Nodes are stored in a `DoublyLinkedSet[Node]` — a doubly-linked list with set semantics (no duplicates, O(1) membership test).

```python
# _core.py, Graph.__init__
self._nodes: _linked_list.DoublyLinkedSet[Node] = _linked_list.DoublyLinkedSet()
```

### Internal Architecture

```
          ┌──────────────────────────────────────────────────┐
          │                                                  │
          ▼                                                  │
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  _root   │◄──►│  Box(A)  │◄──►│  Box(B)  │◄──►│  Box(C)  │
    │ sentinel │    │ value=A  │    │ value=B  │    │ value=C  │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘

    _value_ids_to_boxes = { id(A): Box(A), id(B): Box(B), id(C): Box(C) }
```

**`_LinkBox`** — the internal node wrapper:

```python
class _LinkBox(Generic[T]):
    __slots__ = ("next", "owning_list", "prev", "value")

    def __init__(self, owner: DoublyLinkedSet[T], value: T | None) -> None:
        self.prev: _LinkBox[T] = self  # Circular initially
        self.next: _LinkBox[T] = self
        self.value: T | None = value   # None = erased
        self.owning_list: DoublyLinkedSet[T] = owner

    @property
    def erased(self) -> bool:
        return self.value is None
```

### O(1) Operations

| Operation | Complexity | How |
|---|---|---|
| `append(node)` | O(1) | Insert before `_root` + dict insert |
| `remove(node)` | O(1) | `id(node)` dict lookup → erase box → rewire pointers |
| `insert_before(ref, nodes)` | O(k) | Dict lookup for ref → insert k nodes |
| `insert_after(ref, nodes)` | O(k) | Dict lookup for ref → insert k nodes |
| `__contains__(node)` | O(1) | `id(node) in _value_ids_to_boxes` |
| `__getitem__(0)` or `[-1]` | O(1) | `_root.next` / `_root.prev` |
| `__getitem__(i)` | O(n) | Walk from nearest end |

### Safe Mutation During Iteration

This is the **killer feature** for graph passes. The iterator:

```python
def __iter__(self) -> Iterator[T]:
    box = self._root.next
    while box is not self._root:
        if box.owning_list is not self:
            raise RuntimeError(f"Element {box!r} is not in the list")
        if not box.erased:       # Skip nodes that were removed
            assert box.value is not None
            yield box.value
        box = box.next           # Follow the chain (even if current was erased)
```

**What's safe during iteration:**

| Mutation | Safe? | Why |
|---|---|---|
| Remove current node | ✅ | Box is marked `erased`, `box.next` still valid |
| Remove future node | ✅ | Iterator will skip it (`erased` check) |
| Remove past node | ✅ | Iterator already passed it |
| Insert after current | ✅ | `box.next` now points to new box |
| Insert before current | ✅ | Iterator already has reference to current box |
| Move node to different list | ✅ | `owning_list` check detects it |

**Example — a pass that removes Identity nodes while iterating:**

```python
for node in graph:
    if node.op_type == "Identity":
        # Replace uses of this node's output with its input
        node.outputs[0].replace_all_uses_with(node.inputs[0])
        graph.remove(node)  # Safe! Iterator continues to next node
```

### Why Not a Plain List?

| Plain List | DoublyLinkedSet |
|---|---|
| O(n) remove (shift elements) | O(1) remove |
| Insertion invalidates iteration | Safe mutation during iteration |
| O(n) insert-before | O(1) insert-before |
| O(1) random access | O(n) random access (rarely needed) |
| No duplicate detection | Set semantics (no duplicates) |

Graph passes constantly insert and remove nodes while iterating. A plain list would require collecting changes and applying them afterward, making pass code more complex and error-prone.

---

## 4. Convenience APIs

### Module Structure

```
convenience.py              → Public re-exports
_convenience/
    __init__.py             → Core convenience functions
    _constructors.py        → tensor(), node(), val() factories
    _extractor.py           → extract() for subgraph extraction
```

### Constructor Helpers

#### `ir.tensor(value, dtype=None, name=None)` → TensorProtocol

Creates a tensor from nearly any input format:

```python
ir.tensor([1, 2, 3])                    # → Tensor(int64, shape=[3])
ir.tensor(3.14)                          # → Tensor(float32, shape=[])
ir.tensor(np.zeros((2,3)))              # → Tensor(float64, shape=[2,3])
ir.tensor(torch_tensor)                  # → TorchTensor
ir.tensor(onnx_tensor_proto)            # → deserialized Tensor
ir.tensor(["hello", "world"])           # → StringTensor
```

#### `ir.node(op_type, inputs, attributes=None, ...)` → Node

Creates nodes with Python-native attribute specification:

```python
add_node = ir.node(
    "Add",
    inputs=[x, y],
    name="my_add",
)

conv_node = ir.node(
    "Conv",
    inputs=[input_val, weight, bias],
    attributes={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]},
    num_outputs=1,
)
```

Attributes are auto-converted: `int` → `AttrInt64`, `float` → `AttrFloat32`, `[int]` → `AttrInt64s`, etc.

#### `ir.val(name, dtype=None, shape=None)` → Value

Creates values with simplified type specification:

```python
x = ir.val("x", ir.DataType.FLOAT, ["N", 42, 3])
# x.name  → 'x'
# x.type  → Tensor(FLOAT)
# x.shape → Shape([SymbolicDim('N'), 42, 3])
```

### Graph Manipulation

#### `replace_all_uses_with(values, replacements)`

```python
# Before: A's output feeds B and C
# After:  D's output feeds B and C
ir.convenience.replace_all_uses_with(node_a.outputs, node_d.outputs)
```

#### `replace_nodes_and_values(graph, insertion_point, old_nodes, new_nodes, old_values, new_values)`

Atomic node+value replacement with metadata propagation:

```python
ir.convenience.replace_nodes_and_values(
    graph,
    insertion_point=matmul_node,
    old_nodes=[matmul_node, add_node],
    new_nodes=[fused_gemm_node],
    old_values=[matmul_node.outputs[0], add_node.outputs[0]],
    new_values=[fused_gemm_node.outputs[0], fused_gemm_node.outputs[0]],
)
```

#### `get_const_tensor(value)` → TensorProtocol | None

Resolves a value to its constant tensor (if any):

```python
tensor = ir.convenience.get_const_tensor(value)
if tensor is not None:
    data = tensor.numpy()  # Now we have the actual data
```

Handles:
- Values with `const_value` set (initializers)
- Values produced by `Constant` nodes (all attribute variants)

#### `create_value_mapping(graph)` → dict[str, Value]

```python
mapping = ir.convenience.create_value_mapping(graph)
value = mapping["input_0"]  # O(1) lookup by name
```

#### `extract(graph, inputs, outputs)` → Graph

Extracts a subgraph between specified inputs and outputs:

```python
subgraph = ir.convenience.extract(
    graph,
    inputs=[graph.inputs[0], graph.inputs[1]],
    outputs=[some_intermediate_value],
)
```

Validates that the subgraph is properly bounded (no dangling references).

### Attribute Conversion

#### `convert_attribute(name, attr, attr_type=None)` → Attr

```python
ir.convenience.convert_attribute("alpha", 1.0)          # → AttrFloat32
ir.convenience.convert_attribute("axes", [0, 1])         # → AttrInt64s
ir.convenience.convert_attribute("body", some_graph)     # → AttrGraph
```

Supports 20+ Python types including ONNX protobuf objects (auto-deserialized).

---

## 5. Key Differences from Serialization Format (ONNX Protobuf)

### 1. Initializers Are Values, Not Separate Tensors

| Protobuf | onnx_ir |
|---|---|
| `graph.initializer: repeated TensorProto` | `graph.initializers: dict[str, Value]` |
| Tensor data detached from value identity | Tensor stored as `value.const_value` |
| Match by name string | Match by object identity |

### 2. Nodes Use a Linked List, Not a Repeated Field

| Protobuf | onnx_ir |
|---|---|
| `graph.node: repeated NodeProto` (flat list) | `graph._nodes: DoublyLinkedSet[Node]` |
| O(n) insertion/removal | O(1) insertion/removal |
| Iteration invalidated by mutation | Safe mutation during iteration |

### 3. Values Are First-Class Objects with Back-References

| Protobuf | onnx_ir |
|---|---|
| Values are just strings (names) | `Value` objects with identity |
| No way to find consumers | `value.uses()` → all consuming nodes |
| No way to find producer | `value.producer()` → producing node |
| Name-based wiring | Object reference wiring |

This is perhaps the biggest architectural shift. In protobuf, connecting a node's output to another node's input requires name-string matching across the entire graph. In `onnx_ir`, it's a direct object reference.

### 4. Per-Node Opset Versioning

| Protobuf | onnx_ir |
|---|---|
| `model.opset_import` (global) | `node.version` (per-node, optional) |
| All nodes share opset version | Each node can have its own version |

This enables **mixed-version graphs** where some nodes use newer opset features while others use older semantics.

### 5. Protocol-Based Type System

| Protobuf | onnx_ir |
|---|---|
| Concrete `TensorProto` class | `TensorProtocol` (structural typing) |
| One tensor representation | Multiple: `Tensor`, `ExternalTensor`, `TorchTensor`, ... |
| Must load all data into memory | Lazy loading via `ExternalTensor` |

### 6. Graphs vs GraphViews

| Protobuf | onnx_ir |
|---|---|
| One representation | `Graph` (mutable) vs `GraphView` (read-only) |
| No way to express read-only access | Type system enforces it |

### 7. Rich Metadata System

| Protobuf | onnx_ir |
|---|---|
| `metadata_props` only (serializable) | `metadata_props` + `meta` (non-serializable) |
| No way to attach pass state | `value.meta["my_pass_data"] = ...` |

The `meta` dict allows passes to attach intermediate analysis results (shapes, types, custom annotations) without polluting the serialized format.

### 8. Functions as First-Class Citizens

| Protobuf | onnx_ir |
|---|---|
| `model.functions: repeated FunctionProto` | `model.functions: dict[OperatorIdentifier, Function]` |
| Flat list, lookup by name | Dict keyed by `(domain, name, overload)` |
| No overload support | Full overload support |

### Summary: Why These Differences?

The in-memory IR is optimized for **analysis and transformation**, while protobuf is optimized for **serialization and interchange**. Key principles:

1. **Object identity over string matching** — eliminates an entire class of name-collision bugs
2. **O(1) graph mutations** — passes run fast on large models
3. **Safe iteration** — pass authors don't need to reason about iterator invalidation
4. **Lazy loading** — work with 100GB+ models without loading everything into RAM
5. **Backend flexibility** — tensors can stay on GPU, on disk, or in any framework
6. **Metadata for passes** — intermediate analysis without serialization overhead

---

## 6. What You Can Do With It: Graph Transformation Passes

### Pass Infrastructure

```python
class PassBase(abc.ABC):
    in_place: bool          # Modifies model in place?
    changes_input: bool     # Changes the input model?

    def call(self, model: ir.Model) -> PassResult: ...
    def requires(self, model: ir.Model) -> None: ...   # Preconditions
    def ensures(self, model: ir.Model) -> None: ...    # Postconditions

# Convenience base classes
class InPlacePass(PassBase):   # in_place=True, changes_input=True
class FunctionalPass(PassBase): # in_place=False, changes_input=False
```

### Composing Passes

```python
# Sequential execution
pipeline = ir.passes.Sequential(
    TopologicalSortPass(),
    CommonSubexpressionEliminationPass(),
    RemoveUnusedNodesPass(),
)

# With repetition and early stopping
manager = ir.passes.PassManager(
    passes=[
        IdentityEliminationPass(),
        RemoveUnusedNodesPass(),
    ],
    steps=10,          # Max iterations
    early_stop=True,   # Stop when no changes
)

result = manager(model)  # Returns PassResult(model, modified)
```

### Available Passes (in `passes/common/`)

#### Optimization Passes

| Pass | What it does |
|---|---|
| `CommonSubexpressionEliminationPass` | Finds duplicate operations (same op, same inputs) and replaces them with a single copy |
| `IdentityEliminationPass` | Removes redundant `Identity` nodes by rewiring value references |
| `DeduplicateInitializersPass` | Finds initializers with identical data and deduplicates them |
| `DeduplicateHashedInitializersPass` | Memory-efficient variant using SHA-512 for large models |

#### Constant Manipulation

| Pass | What it does |
|---|---|
| `LiftConstantsToInitializersPass` | Converts `Constant` nodes to graph initializers (configurable size limit) |
| `LiftSubgraphInitializersToMainGraphPass` | Promotes subgraph initializers to main graph level |
| `RemoveInitializersFromInputsPass` | Removes initializers from `graph.inputs` list |
| `AddInitializersToInputsPass` | Adds initializers to `graph.inputs` (for runtime override) |

#### Cleanup Passes

| Pass | What it does |
|---|---|
| `RemoveUnusedNodesPass` | Dead code elimination — removes nodes with no consumers |
| `RemoveUnusedFunctionsPass` | Removes functions not called from main graph |
| `RemoveUnusedOpsetsPass` | Removes opset imports not used by any node |
| `ClearMetadataAndDocStringPass` | Strips metadata for smaller serialized models |

#### Structural Passes

| Pass | What it does |
|---|---|
| `TopologicalSortPass` | Sorts nodes in topological order (stable) |
| `AddDefaultAttributesPass` | Adds missing optional attributes with their ONNX-spec defaults |
| `OutputFixPass` | Inserts Identity nodes for invalid output configurations |
| `NameFixPass` | Ensures all nodes and values have unique names |

#### Advanced Passes

| Pass | What it does |
|---|---|
| `InlinePass` | Inlines local functions into call sites (with cycle detection) |
| `ShapeInferencePass` | Runs ONNX shape inference, merging results back |
| `CheckerPass` | Validates model against ONNX spec (non-modifying) |

### Writing a Custom Pass

```python
class MyFusionPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        graph = model.graph

        for node in graph:  # Safe to mutate during iteration!
            if node.op_type == "MatMul":
                # Check if next consumer is Add (MatMul+Add → Gemm fusion)
                matmul_output = node.outputs[0]
                consumers = matmul_output.consumers()

                if len(consumers) == 1 and consumers[0].op_type == "Add":
                    add_node = consumers[0]

                    # Create fused Gemm node
                    gemm = ir.node(
                        "Gemm",
                        inputs=[node.inputs[0], node.inputs[1], add_node.inputs[1]],
                        attributes={"alpha": 1.0, "beta": 1.0},
                        num_outputs=1,
                    )

                    # Replace in graph
                    ir.convenience.replace_nodes_and_values(
                        graph,
                        insertion_point=add_node,
                        old_nodes=[node, add_node],
                        new_nodes=[gemm],
                        old_values=[add_node.outputs[0]],
                        new_values=[gemm.outputs[0]],
                    )
                    modified = True

        return ir.passes.PassResult(model, modified=modified)
```

### Key Graph Traversal Patterns

```python
import onnx_ir as ir
from onnx_ir.traversal import RecursiveGraphIterator

# Iterate all nodes (including subgraphs in If/Loop)
for node in RecursiveGraphIterator(graph):
    print(node.op_type)

# Find all uses of a value
for usage in value.uses():
    node, input_index = usage
    print(f"Used by {node.op_type} at input {input_index}")

# Walk predecessors
for pred in node.predecessors():
    print(f"Input from {pred.op_type}")

# Walk successors
for succ in node.successors():
    print(f"Output to {succ.op_type}")

# Resolve constant values
tensor = ir.convenience.get_const_tensor(value)
if tensor is not None:
    array = tensor.numpy()
```

---

## Appendix: Class Quick Reference

| Class | File | Purpose |
|---|---|---|
| `Model` | `_core.py` | Top-level container (graph + metadata) |
| `Graph` | `_core.py` | Mutable computation graph |
| `GraphView` | `_core.py` | Read-only graph view |
| `Node` | `_core.py` | Operation in the graph |
| `Value` | `_core.py` | Named data flowing between nodes |
| `Tensor` | `_core.py` | In-memory tensor (numpy/DLPack) |
| `ExternalTensor` | `_core.py` | Disk-backed tensor (mmap) |
| `StringTensor` | `_core.py` | String data tensor |
| `Shape` | `_core.py` | Tensor shape with symbolic dims |
| `Attr` | `_core.py` | Node attribute |
| `TorchTensor` | `tensor_adapters.py` | PyTorch tensor adapter |
| `DoublyLinkedSet` | `_linked_list.py` | O(1) mutable ordered set |
| `GraphInputs` | `_graph_containers.py` | Graph input value list |
| `GraphOutputs` | `_graph_containers.py` | Graph output value list |
| `GraphInitializers` | `_graph_containers.py` | Graph initializer dict |
| `PassBase` | `passes/_pass_infra.py` | Pass base class |
| `InPlacePass` | `passes/_pass_infra.py` | In-place pass base |
| `PassManager` | `passes/_pass_infra.py` | Pass pipeline runner |
