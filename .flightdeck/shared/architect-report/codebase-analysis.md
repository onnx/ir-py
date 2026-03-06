# ONNX IR Codebase Analysis — Graph Editing API Gap Analysis

**Author:** Architect Agent (fae585af)  
**Audience:** All team members  
**Purpose:** Map the existing architecture, identify gaps in graph manipulation APIs, and propose new APIs

---

## 1. Project Structure Overview

```
src/onnx_ir/
├── __init__.py              # Public API (93 exports)
├── _core.py                 # Core IR: Value, Node, Graph, Function, Model, Shape, Attr, Types (~4400 lines)
├── _graph_containers.py     # GraphInputs, GraphOutputs, GraphInitializers, Attributes containers
├── _linked_list.py          # DoublyLinkedSet — O(1) insert/remove backing for Graph node storage
├── _tape.py                 # Tape & Builder — node construction helpers
├── _cloner.py               # Deep clone with value/attr remapping
├── _protocols.py            # Runtime-checkable Protocol interfaces (ValueProtocol, etc.)
├── _name_authority.py       # Unique name generation for nodes/values
├── _enums.py                # DataType, AttributeType enums
├── _display.py              # Pretty-printing
├── _metadata.py             # MetadataStore for analysis passes
├── _convenience/
│   ├── __init__.py           # replace_all_uses_with, replace_nodes_and_values, create_value_mapping, etc.
│   ├── _constructors.py      # ir.node(), ir.tensor(), ir.val() — convenience constructors
│   └── _extractor.py         # extract() — extract subgraph bounded by values
├── convenience.py            # Public re-export of _convenience
├── traversal.py              # RecursiveGraphIterator — depth-first node traversal
├── tape.py                   # Public re-export of Tape/Builder
├── serde.py                  # Serialization/deserialization (protobuf, ONNX text)
├── analysis/
│   └── _implicit_usage.py    # Detect captured variables in subgraphs
├── passes/
│   ├── _pass_infra.py        # PassBase, InPlacePass, FunctionalPass, Sequential, PassManager
│   └── common/               # ~19 concrete passes (CSE, identity elim, unused removal, etc.)
├── journaling/               # Debug/audit system for IR mutations
├── external_data.py          # External tensor data management
├── schemas.py                # ONNX op schema utilities
└── tensor_adapters.py        # Tensor type adapters (numpy, DLPack)
```

---

## 2. Core Data Model

### 2.1 Class Hierarchy

```
Model
├── graph: Graph
├── functions: dict[OperatorIdentifier, Function]
└── opset_imports, ir_version, etc.

Graph (Sequence[Node])
├── _nodes: DoublyLinkedSet[Node]     ← O(1) insert/remove
├── inputs: GraphInputs (UserList[Value])
├── outputs: GraphOutputs (UserList[Value])
├── initializers: GraphInitializers (UserDict[str, Value])
├── opset_imports: dict[str, int]
└── _name_authority: NameAuthority

Function (Sequence[Node])
├── graph: Graph                       ← Delegates to internal Graph
├── domain, name, overload
└── attributes: Attributes

Node
├── domain, op_type, overload, version
├── _inputs: tuple[Value | None, ...]  ← Immutable (use replace_input_with to change)
├── _outputs: tuple[Value, ...]        ← Immutable (use resize_outputs to grow/shrink)
├── attributes: Attributes (UserDict[str, Attr])
└── _graph: Graph | None

Value
├── _producer: Node | None             ← The node that produces this value
├── _index: int | None                 ← Output index in producer
├── _uses: dict[Usage, None]           ← All (consumer_node, input_idx) pairs
├── _name, _shape, _type, _const_value
├── _graph: Graph | None               ← Set if graph input/output/initializer
└── _is_graph_input, _is_graph_output, _is_initializer: bool
```

### 2.2 Data Flow Relationships

```
                     produces
          Node A ─────────────> Value V
                                  │
                    ┌─────────────┼─────────────┐
                    │ uses        │ uses         │ graph.outputs
                    ▼             ▼              ▼
              (Node B, idx=0)  (Node C, idx=1)  graph.outputs[k]
```

- **Forward flow:** `node.outputs` → `value.uses()` → `(consumer_node, idx)`
- **Backward flow:** `node.inputs` → `value.producer()` → `producer_node`
- **Graph navigation:** `node.predecessors()` / `node.successors()`

### 2.3 Ownership Model

| Value Role | producer | graph | Tracked by |
|---|---|---|---|
| Node output | Node | None* | Node._outputs tuple |
| Graph input | None | Graph | GraphInputs list |
| Graph output | varies | Graph | GraphOutputs list |
| Initializer | None | Graph | GraphInitializers dict |
| Detached | None | None | Nothing (GC-eligible) |

\* A node output's `graph` is set to Graph only if it's also a graph output.

---

## 3. Existing Graph Editing Capabilities

### 3.1 Node-Level Operations

| Operation | Method | Complexity | Notes |
|---|---|---|---|
| Rewire one input | `node.replace_input_with(idx, value)` | O(1) | Updates usage tracking |
| Resize inputs | `node.resize_inputs(new_size)` | O(k) | Pads with None or trims |
| Resize outputs | `node.resize_outputs(new_size)` | O(k) | Creates/removes Values |
| Get predecessors | `node.predecessors()` | O(inputs) | Deduplicated |
| Get successors | `node.successors()` | O(outputs × uses) | Deduplicated |
| Insert before self | `node.prepend(nodes)` | O(k) | Delegates to graph |
| Insert after self | `node.append(nodes)` | O(k) | Delegates to graph |

### 3.2 Value-Level Operations

| Operation | Method | Complexity | Notes |
|---|---|---|---|
| Replace all uses | `value.replace_all_uses_with(replacement)` | O(uses) | Can handle graph outputs |
| Get consumers | `value.consumers()` | O(uses) | Deduplicated |
| Get all uses | `value.uses()` | O(1) | Returns dict view |
| Merge shapes | `value.merge_shapes(other)` | O(rank) | In-place |
| Check role | `value.is_graph_input/output/initializer()` | O(1) | |

### 3.3 Graph-Level Operations

| Operation | Method | Complexity | Notes |
|---|---|---|---|
| Append node | `graph.append(node)` | O(1) | Auto-names |
| Extend nodes | `graph.extend(nodes)` | O(n) | |
| Remove node(s) | `graph.remove(nodes, safe=False)` | O(k) | safe=True detaches inputs |
| Insert after | `graph.insert_after(node, new_nodes)` | O(k) | |
| Insert before | `graph.insert_before(node, new_nodes)` | O(k) | |
| Topological sort | `graph.sort()` | O(V+E) | Stable, handles subgraphs |
| Clone | `graph.clone()` | O(V+E) | Shares tensors |
| Register initializer | `graph.register_initializer(value)` | O(1) | |
| Lookup by index/name | `graph.node(idx_or_name)` | O(n) | |

### 3.4 Convenience Utilities

| Operation | Function | Location |
|---|---|---|
| Replace all uses (batch) | `convenience.replace_all_uses_with(values, replacements)` | `_convenience/__init__.py` |
| Replace nodes + values | `convenience.replace_nodes_and_values(graph, insertion_pt, old_nodes, new_nodes, old_vals, new_vals)` | `_convenience/__init__.py` |
| Create value name map | `convenience.create_value_mapping(graph)` | `_convenience/__init__.py` |
| Extract subgraph | `convenience.extract(graph, inputs, outputs)` | `_convenience/_extractor.py` |
| Get constant tensor | `convenience.get_const_tensor(value)` | `_convenience/__init__.py` |
| Node constructor | `ir.node(op_type, inputs, attributes)` | `_convenience/_constructors.py` |
| Tensor constructor | `ir.tensor(value, dtype, name)` | `_convenience/_constructors.py` |
| Value constructor | `ir.val(name, dtype, shape)` | `_convenience/_constructors.py` |

### 3.5 Pass Infrastructure

| Class | Purpose |
|---|---|
| `PassBase` | Abstract base with precondition/postcondition hooks |
| `InPlacePass` | Modifies model in-place |
| `FunctionalPass` | Creates new model without modifying input |
| `Sequential` | Runs multiple passes in order |
| `PassManager` | Runs passes repeatedly with early stopping |
| `functionalize()` | Wraps in-place pass as functional |

### 3.6 Traversal

| Tool | Purpose |
|---|---|
| `RecursiveGraphIterator(graph)` | Depth-first traversal of all nodes including subgraphs |
| `reversed(RecursiveGraphIterator(graph))` | Reverse traversal |
| `graph.all_nodes()` | Iterator over all nodes including subgraphs |
| `graph.subgraphs()` | Iterator over all subgraphs |
| `node.predecessors()` / `node.successors()` | Local neighbors |

---

## 4. Identified Gaps and Improvement Opportunities

### 4.1 CRITICAL GAPS — High Impact, Frequently Needed

#### Gap 1: No `pop` / `remove-and-return` for Graph nodes

**Problem:** `graph.remove()` returns `None`. When you want to extract a node for re-insertion elsewhere, you must hold a reference and call remove separately. There's no atomic "detach this node from its graph and return it."

**Impact:** Every pass that moves nodes has to manually coordinate reference holding + remove + re-insert. Error-prone.

**Suggestion:** Add `graph.pop(node)` → `Node` that removes the node and clears its graph reference, returning it for reuse.

---

#### Gap 2: No node replacement primitive

**Problem:** Replacing one node with another is a multi-step process:
1. Create replacement node
2. `replace_all_uses_with(old_outputs, new_outputs)`
3. `graph.insert_before/after(old_node, new_node)`
4. `graph.remove(old_node, safe=True)`

This 4-step sequence is repeated in almost every pass (CSE, identity elimination, constant manipulation, etc.). The `replace_nodes_and_values` function helps but requires separate old_nodes/new_nodes/old_values/new_values sequences.

**Impact:** This is the #1 most common graph editing pattern, yet requires 4 coordinated steps. Every pass reimplements it slightly differently, leading to subtle bugs (forgotten metadata propagation, name handling, etc.).

**Suggestion:** Add `graph.replace_node(old_node, new_node)` or `node.replace_with(new_node)` that atomically:
- Rewires all downstream consumers
- Handles graph output replacement
- Preserves metadata/names on output values
- Inserts the new node at the old node's position
- Removes the old node

---

#### Gap 3: No subgraph replacement ("pattern matching + rewrite")

**Problem:** There's no way to say "replace this subgraph (set of connected nodes) with this other subgraph." The `replace_nodes_and_values` function is the closest, but it requires the user to manually figure out insertion points, old/new value mappings, etc.

**Impact:** This is the fundamental operation for pattern-based graph rewrites — the bread-and-butter of optimization passes. Currently every rewrite pass hand-rolls this logic.

**Suggestion:** Add a higher-level `graph.replace_subgraph(old_nodes, new_nodes, input_mapping, output_mapping)` that handles all the wiring.

---

#### Gap 4: No topological traversal iterator (forward/backward from a node)

**Problem:** `RecursiveGraphIterator` traverses the entire graph. There's no way to traverse "all nodes reachable from this node" (forward or backward). Passes that need this (e.g., finding all consumers of a value transitively) must hand-roll BFS/DFS.

**Impact:** Forward/backward slicing is fundamental for many analyses and transformations.

**Suggestion:** Add:
- `traversal.forward_slice(node)` → yields all transitively reachable consumer nodes
- `traversal.backward_slice(node)` → yields all transitively reachable producer nodes
- `traversal.between(start_values, end_values)` → yields all nodes in the subgraph between

---

#### Gap 5: No graph splitting / partitioning

**Problem:** The only way to split a graph is `convenience.extract()`, which extracts a single subgraph. There's no way to partition a graph into multiple subgraphs (e.g., for device placement, compilation unit boundaries).

**Impact:** Multi-device deployment, graph partitioning for compilation, and subgraph-based optimization patterns all need this.

**Suggestion:** Add `graph.partition(partition_fn: Callable[[Node], int]) -> list[Graph]` that assigns each node to a partition and creates new subgraphs with proper input/output wiring between them.

---

### 4.2 IMPORTANT GAPS — Meaningful Quality-of-Life Improvements

#### Gap 6: No deep copy for individual nodes

**Problem:** `graph.clone()` copies the whole graph. There's no way to clone a single node (creating a new node with the same op, attributes, and fresh output values). `_cloner.Cloner.clone_node()` exists but it's private and requires setting up a Cloner with value maps.

**Suggestion:** Add `node.clone()` → `Node` that creates a deep copy with fresh output values but shared input references.

---

#### Gap 7: No "splice" operation (insert subgraph inline)

**Problem:** Inserting a chain of nodes between two existing nodes (e.g., adding a quantize-dequantize wrapper around a value) requires:
1. Create all new nodes
2. Rewire the first new node's inputs to point to the original value
3. Rewire all consumers of the original value to point to the last new node's output
4. Insert all nodes at the right position in the graph

**Suggestion:** Add `value.splice_after(new_nodes) -> Value` that inserts a chain of nodes after this value and returns the output value. This enables patterns like:
```python
# Insert Cast after a value
cast_node = ir.node("Cast", [original_value], {"to": DataType.FLOAT16})
new_value = original_value.splice_after(cast_node)
# Now all consumers of original_value use new_value instead
```

---

#### Gap 8: No batch/bulk editing context manager

**Problem:** When making many changes to a graph (e.g., a rewrite pass that touches hundreds of nodes), each individual operation triggers name assignment, ownership checking, etc. There's no way to batch these operations.

**Suggestion:** Add `with graph.bulk_edit():` context manager that defers name assignment and validation until the context exits.

---

#### Gap 9: No node/value search utilities

**Problem:** Finding nodes by op_type, by attribute values, or by patterns requires manual iteration. `graph.node(name)` only searches by name/index.

**Suggestion:** Add:
- `graph.find_nodes(op_type=..., domain=...)` → iterator of matching nodes
- `graph.find_values(name_pattern=...)` → iterator of matching values

---

#### Gap 10: No in-place node mutation helpers

**Problem:** Changing a node's op_type (e.g., fusing Matmul+Add into Gemm) while preserving its position and connections requires creating an entirely new node, rewiring, and removing the old one. You can set `node.op_type` directly, but there's no way to change attributes atomically.

**Suggestion:** Add `node.update(op_type=..., domain=..., attributes=...)` that mutates the node in-place, preserving connections and position.

---

### 4.3 NICE-TO-HAVE — Minor Ergonomic Improvements

#### Gap 11: Weak graph validation / integrity checking

**Problem:** `graph.remove(safe=True)` does some checking, but there's no standalone validation that checks the entire graph for integrity (dangling references, cycles, missing producers, etc.).

**Suggestion:** Add `graph.validate()` that checks all invariants and returns a list of violations.

---

#### Gap 12: No diff/comparison between graphs

**Problem:** `_graph_comparison.py` exists but isn't exposed publicly. Comparing two graphs to see what changed is useful for debugging passes.

**Suggestion:** Expose graph comparison publicly and add a `graph.diff(other)` utility.

---

#### Gap 13: Missing topological order iterator as a first-class citizen

**Problem:** `graph.sort()` sorts in place. There's no way to iterate in topological order without mutating the graph. You have to clone, sort, then iterate.

**Suggestion:** Add `graph.topological_order()` → `Iterator[Node]` that yields nodes in topological order without modifying the graph.

---

## 5. How Existing Passes Modify Graphs (Patterns)

Every existing pass follows one of these patterns:

### Pattern A: "Find and Remove"
```python
for node in reversed(graph):
    if should_remove(node):
        graph.remove(node, safe=True)
```
Used by: `RemoveUnusedNodesPass`, `IdentityEliminationPass`

### Pattern B: "Find, Replace Uses, Remove"
```python
for node in ir.traversal.RecursiveGraphIterator(graph):
    if should_replace(node):
        replacement_value = ...  # existing value or new initializer
        node.outputs[0].replace_all_uses_with(replacement_value)
        graph.remove(node, safe=True)
```
Used by: `LiftConstantsToInitializersPass`, `IdentityEliminationPass`

### Pattern C: "Find, Create New, Replace, Insert, Remove"
```python
for node in graph:
    if should_rewrite(node):
        new_node = ir.node(...)
        ir.convenience.replace_all_uses_with(node.outputs, new_node.outputs)
        graph.insert_before(node, new_node)
        graph.remove(node, safe=True)
```
Used by: `CommonSubexpressionEliminationPass`

### Pattern D: "Batch Replace"
```python
ir.convenience.replace_nodes_and_values(
    graph, insertion_point, old_nodes, new_nodes, old_values, new_values
)
```
Used by: External passes, rewrite rules

---

## 6. Prioritized API Proposals

### Tier 1 — Highest Impact (address most common pain points)

1. **`node.replace_with(new_node)` / `graph.replace_node(old, new)`**  
   Single-call node replacement that handles all wiring. This eliminates Pattern C's 4-step dance.

2. **`graph.replace_subgraph(old_nodes, new_nodes, output_mapping)`**  
   Higher-level subgraph replacement. Eliminates the need to manually coordinate `replace_nodes_and_values`.

3. **`traversal.forward_slice(node)` / `traversal.backward_slice(node)`**  
   Transitive reachability traversal from a starting node.

### Tier 2 — High Impact

4. **`graph.topological_order()` → `Iterator[Node]`**  
   Non-mutating topological iteration.

5. **`node.clone()` → `Node`**  
   Simple deep-copy of a single node.

6. **`graph.find_nodes(op_type=..., domain=...)` → `Iterator[Node]`**  
   Filtered node search.

### Tier 3 — Nice to Have

7. **`graph.validate()` → `list[str]`**  
   Full integrity check.

8. **`value.splice_after(nodes)` → `Value`**  
   Insert a node chain after a value, rewiring consumers.

9. **`graph.partition(fn) → list[Graph]`**  
   Graph partitioning by node classification.

---

## 7. Architecture Observations

### Strengths
- **DoublyLinkedSet** for node storage is excellent — O(1) insert/remove with safe-during-iteration semantics
- **Usage tracking** (Value._uses) enables efficient data flow analysis
- **Protocol-based design** allows custom implementations without inheritance
- **Tape/Builder** provides ergonomic node construction
- **Journaling** infrastructure is forward-looking for debugging

### Design Concerns
- **Node inputs are tuples** (immutable), requiring `replace_input_with` for single changes and tuple reconstruction internally. This is intentional for safety but makes some transformations verbose.
- **No transactional semantics**: if a multi-step transformation fails midway, the graph can be left in an inconsistent state. The `safe=True` parameter on `remove()` helps but isn't comprehensive.
- **Function delegates to Graph internally**: The `Function` class wraps a `Graph`, which is clean but means function-level editing goes through the same Graph APIs. No function-specific editing utilities exist.

### Key Invariants to Preserve
1. Each Value has exactly one producer (or None for inputs/initializers)
2. Usage tracking stays consistent (add/remove_usage called on every rewire)
3. Graph inputs have no producer
4. Node belongs to at most one graph
5. DoublyLinkedSet contains no duplicates

---

## 8. Testing Patterns

- Tests are co-located: `_core.py` → `_core_test.py`, `_tape.py` → `_tape_test.py`
- Passes have their own test files: `common/identity_elimination_test.py`, etc.
- Test infrastructure in `testing.py` provides helpers
- Public API test in `tests/public_api_test.py` validates naming and exports
- Serde roundtrip tests ensure serialization fidelity

New APIs should follow the co-located test pattern and include both:
- Unit tests for the API itself
- Integration tests showing the API used in a realistic pass scenario

---

## Appendix: Key File Locations

| What | Where | Line Range |
|---|---|---|
| Value class | `_core.py` | ~2583-3059 |
| Node class | `_core.py` | ~1915-2393 |
| Graph class | `_core.py` | ~3113-3618 |
| Function class | `_core.py` | ~4002-4200 |
| Model class | `_core.py` | ~3675-3990 |
| DoublyLinkedSet | `_linked_list.py` | ~68-284 |
| GraphInputs/Outputs | `_graph_containers.py` | ~27-216 |
| PassBase/Sequential/PassManager | `passes/_pass_infra.py` | ~73-351 |
| RecursiveGraphIterator | `traversal.py` | ~21-119 |
| replace_all_uses_with | `_convenience/__init__.py` | ~280-360 |
| replace_nodes_and_values | `_convenience/__init__.py` | ~419-455 |
| extract (subgraph) | `_convenience/_extractor.py` | ~129-191 |
| Cloner | `_cloner.py` | (full file) |
| Tape/Builder | `_tape.py` | (full file) |
