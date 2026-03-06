# Synthesis Proposal: ONNX IR Graph Editing API

**Authors:** Architect (fae585af), Radical Thinker (17a6991d)  
**Status:** ✅ FINAL VERSION — Approved by Lead, ready for implementation  
**Audience:** All team members

---

## Executive Summary

Both independent analyses converge on the same core finding: **the ONNX IR has a solid foundation but its graph editing experience is stuck at the "assembly language" level.** The most common graph transformation — replacing a node — requires 4 coordinated steps that every pass reimplements slightly differently, leading to boilerplate and edge-case bugs (especially around graph outputs).

This proposal defines high-level editing "verbs" that encode intent rather than mechanism. It is purely additive — no breaking changes to existing `_core.py` classes or the pass infrastructure.

---

## Areas of Agreement

| Topic | Architect Finding | Radical Thinker Finding | Consensus |
|---|---|---|---|
| #1 pain point | 4-step replace dance | "Mechanism vs intent" gap | **Node replacement must be a single call** |
| Graph output handling | Every pass special-cases `is_graph_output()` | "Recurring tax" | **Editing verbs absorb this complexity** |
| Traversal gaps | No forward/backward slice | Subgraph handles need traversal | **Add transitive reachability iterators** |
| Subgraph operations | `extract()` exists but is copy-only | Need first-class `Subgraph` type | **New `SubgraphHandle` class (GraphView is read-only by design)** |
| Non-mutating topo order | Missing | Missing | **Add lazy topological iterator** |
| Pattern matching | "Greenfield" | DSL proposal | **Start programmatic, defer DSL. Design programmatic API as foundation for future DSL** |
| Function-based passes | Not proposed | Proposed | **Easy win — include it** |
| `node.clone()` | Proposed in gap analysis | Agreed | **Include in Tier 1** |

---

## Areas of Debate (Resolved)

### Transactions / Checkpoints → **Deferred (Tier 3)**
- **Radical Thinker proposed:** `graph.transaction()` context manager with rollback
- **Architect pushback:** True transactions are infeasible in Python due to reference semantics. `Value` objects are shared by identity (`id(value)` is used in `DoublyLinkedSet._value_ids_to_boxes`). A command log with inverse operations would be enormously complex — `replace_all_uses_with` touches N nodes, and undo would need to restore each one.
- **Resolution:** Make the editing verbs correct-by-construction so rollback isn't needed. If atomic operations handle all edge cases (graph outputs, metadata propagation, name preservation), there's nothing to undo. Defer transactions to Tier 3.

### Copy-on-Write Views → **Not feasible**
- **Radical Thinker proposed:** `graph.cow_view()` with lazy materialization
- **Architect pushback:** Python's mutable reference semantics make CoW impractical. Every `Value` and `Node` is identified by `id()`. CoW would require proxy objects that break `isinstance`, `is`, and the protocol system. The existing `FunctionalPass` + `graph.clone()` already serves this use case (clone upfront, modify freely).
- **Resolution:** Not pursuing. The clone-then-mutate pattern is adequate.

### Graph Cursor → **Future exploration (Tier 3)**
- **Radical Thinker proposed:** Cursor-based navigation and editing
- **Architect assessment:** The existing `node.prepend()`/`node.append()` already provide cursor-like local editing. The chainable `.find().next_matching().replace_with()` API conflates navigation with mutation, making error handling unclear.
- **Resolution:** Tier 3. Interesting for interactive tools but not a priority for pass development.

### Pattern Matching: DSL vs Programmatic → **Out of scope (onnxscript overlap)**
- **Radical Thinker proposed:** Declarative DSL (`pat.Op("MatMul") | pat.Op("Add")`)
- **Architect counter:** Start with Python functions as matchers wrapped in `RewriteRule`.
- **User direction:** `onnxscript` already provides pattern matching/rewrite capabilities. Adding a parallel system in `onnx_ir` creates duplication. Deferred to Tier 3 for re-evaluation.
- **Resolution:** Out of scope for now. Focus on the editing verbs (replace_node, replace_subgraph, etc.) which are orthogonal to and composable with any external pattern matching system.

### Module Location: Where do new APIs live?
- **Radical Thinker proposed:** `_convenience/_editing.py` re-exported via `convenience.py`, following existing pattern
- **Architect proposed:** New top-level `editing.py` module
- **Lead directed:** New module, architect makes the call based on codebase conventions
- **Resolution:** **New top-level `onnx_ir/editing.py`** module (like `traversal.py` is top-level). Rationale: editing operations are a distinct concern from the "convenience" grab-bag (which mixes constructors, attribute converters, value mapping, and extraction). A dedicated module signals "this is the primary way to edit graphs" and gives it first-class visibility. Internally uses `_convenience/` utilities. Thin `Node.replace_with()` wrapper on the core class for discoverability.

### SubgraphHandle: New class vs extend GraphView?
- **Architect initially proposed:** Use existing `GraphView`
- **Radical Thinker proposed:** New `SubgraphHandle` class that knows its parent graph
- **Lead directed:** New class if GraphView is read-only by design
- **Resolution:** **New `SubgraphHandle` class.** `GraphView` is intentionally read-only (stores tuples, no mutation methods, no parent reference). `SubgraphHandle` references a parent `Graph` and delegates mutations through it. Placed in `editing.py`.

---

## Prioritized API Proposal

### Tier 1: Implement Now (Highest Impact, Purely Additive)

These APIs live in a new **`onnx_ir/editing.py`** module (standalone functions) plus thin method wrappers on `Node` for discoverability. Together they form a complete toolkit for single-node editing, passthrough elimination, edge insertion, subgraph replacement, topological iteration, easy pass authoring, and node copying.

> **Implementation requirement:** Each API must be implemented as a **separate commit and push** for easy review. All implementations must be **performance-conscious** — this is a critical library. See Performance Requirements section below.

---

#### API 1: `replace_node` — Single-Call Node Replacement

**The #1 most needed API.** Eliminates the 4-step pattern used in every pass.

```python
# onnx_ir/editing.py

def replace_node(
    old_node: ir.Node,
    new_node: ir.Node,
    *,
    output_mapping: dict[ir.Value, ir.Value] | None = None,
    propagate_metadata: bool = True,
) -> None:
    """Replace a node in the graph with another node.

    Handles all rewiring automatically:
    1. Maps old output values to new output values (1:1 by position, or via explicit mapping)
    2. Propagates type, shape, name, and const_value from old outputs to new outputs
       (when propagate_metadata=True)
    3. Redirects all consumers of old outputs to use new outputs
    4. Updates graph outputs if any old outputs were graph outputs
    5. Inserts new_node at old_node's position in the graph
    6. Removes old_node from the graph

    When output counts match, outputs are mapped 1:1 by position (no mapping needed).
    When output counts differ, an explicit output_mapping must be provided.

    Args:
        old_node: The node to replace. Must belong to a graph.
        new_node: The replacement node. Must not belong to any graph.
        output_mapping: Explicit mapping from old output values to new output values.
            Required when the number of outputs differs between old and new nodes.
            When None, outputs are mapped 1:1 by position.
        propagate_metadata: If True (default), propagate type, shape, const_value,
            and name from old outputs to new outputs (old takes precedence over new
            when old is not None). Set to False when the new node intentionally
            changes the output semantics.

    Raises:
        ValueError: If old_node doesn't belong to a graph.
        ValueError: If new_node already belongs to a graph.
        ValueError: If output counts differ and no output_mapping is provided.
        ValueError: If output_mapping doesn't cover all old outputs that have consumers.
    """
```

**Plus a thin method wrapper on Node for discoverability:**

```python
# Added to Node class in _core.py

def replace_with(self, new_node: Node, /, **kwargs) -> None:
    """Replace this node with another node.

    See :func:`onnx_ir.editing.replace_node` for full documentation.
    """
    from onnx_ir import editing
    editing.replace_node(self, new_node, **kwargs)
```

##### Before/After: CSE Pass `_remove_node_and_replace_values`

**BEFORE** (51 lines in `common_subexpression_elimination.py` lines 138-189):
```python
def _remove_node_and_replace_values(graph, /, remove_node, remove_values, new_values):
    # 30 lines to handle graph output edge cases:
    # - Check if any remove_value is a graph output
    # - If new_value is also graph output/input, create Identity node
    # - Otherwise rename new_value to old name and update graph.outputs
    if any(remove_value.is_graph_output() for remove_value in remove_values):
        replacement_mapping = dict(zip(remove_values, new_values))
        for idx, graph_output in enumerate(graph.outputs):
            if graph_output in replacement_mapping:
                new_value = replacement_mapping[graph_output]
                if new_value.is_graph_output() or new_value.is_graph_input():
                    identity_node = ir.node("Identity", inputs=[new_value],
                        outputs=[ir.Value(name=graph_output.name, type=graph_output.type,
                                          shape=graph_output.shape)])
                    graph.outputs[idx] = identity_node.outputs[0]
                    graph.insert_before(remove_node, identity_node)
                else:
                    new_value.name = graph_output.name
                    graph.outputs[idx] = new_value
    ir.convenience.replace_all_uses_with(remove_values, new_values)
    graph.remove(remove_node, safe=True)
```

**AFTER** (1 line):
```python
# In the CSE pass, when a duplicate node is found:
existing_node = existing_node_info_to_the_node[node_info]
editing.replace_node(node, existing_node,
    output_mapping=dict(zip(node.outputs, existing_node.outputs)))
```

##### Before/After: Identity Elimination

**BEFORE** (28 lines in `identity_elimination.py` lines 74-121):
```python
def _try_eliminate_identity_node(self, node):
    if node.op_type != "Identity" or node.domain != "":
        return False
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    input_value = node.inputs[0]
    output_value = node.outputs[0]
    if input_value is None:
        return False
    graph_like = node.graph
    output_is_graph_output = output_value.is_graph_output()
    # Case 3: Both output is graph output AND input is graph input/initializer — keep
    if output_is_graph_output and (input_value.is_graph_input() or input_value.is_initializer()):
        return False
    # Merge shapes manually
    input_value.shape = _merge_shapes(input_value.shape, output_value.shape)
    if input_value.type is None:
        input_value.type = output_value.type
    # Replace uses (must pass replace_graph_outputs=True)
    ir.convenience.replace_all_uses_with(output_value, input_value, replace_graph_outputs=True)
    # Rename if graph output
    if output_is_graph_output:
        input_value.name = output_value.name
    # Remove
    graph_like.remove(node, safe=True)
    return True
```

**AFTER** (using `eliminate_node` — see API 7 and Appendix for full rewrite):
```python
def _try_eliminate_identity_node(self, node):
    if node.op_type != "Identity" or node.domain != "":
        return False
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    if node.inputs[0] is None:
        return False
    if node.outputs[0].is_graph_output() and (
        node.inputs[0].is_graph_input() or node.inputs[0].is_initializer()):
        return False
    # One call: shape merge, type propagation, use replacement,
    # graph output handling, name transfer, node removal
    editing.eliminate_node(node, input_index=0)
    return True
```

The pass-specific decision logic is preserved. All mechanism (5 coordinated steps) is absorbed by `eliminate_node`. See the **Appendix** for the complete pass rewrite including the `@pass_fn` version.

---

#### API 2: `insert_on_edge` — Insert Node on a Value's Edge

The fundamental operation for adding quantization, casts, normalization, etc.

```python
def insert_on_edge(
    value: ir.Value,
    new_node: ir.Node,
    *,
    output_index: int = 0,
) -> ir.Value:
    """Insert a node on all edges from a value to its consumers.

    After this operation:
    - new_node receives ``value`` as input (already wired by the caller)
    - All previous consumers of ``value`` now consume ``new_node.outputs[output_index]``
    - new_node is inserted into the graph at the appropriate position
      (after the producer of ``value``, or at the start if ``value`` is a graph input)

    The caller is responsible for creating new_node with ``value`` as one of its
    inputs. This function handles rewiring the downstream consumers and inserting
    the node into the graph.

    Args:
        value: The value whose consumers should be redirected.
        new_node: The node to insert. Must have ``value`` as one of its inputs.
        output_index: Which output of new_node replaces ``value`` for downstream consumers.

    Returns:
        The new output value (``new_node.outputs[output_index]``) that consumers now use.

    Raises:
        ValueError: If value has no producer and is not a graph input/initializer.
        ValueError: If new_node doesn't have ``value`` as an input.

    Example::

        # Insert a Cast after a value
        x = some_node.outputs[0]  # float32 output
        cast_node = ir.node("Cast", inputs=[x], attributes={"to": ir.DataType.FLOAT16})
        new_x = editing.insert_on_edge(x, cast_node)
        # All consumers of x now consume new_x (float16)

        # Insert Relu after every Conv output
        for node in ir.traversal.find_nodes(graph, op_type="Conv"):
            relu = ir.node("Relu", inputs=[node.outputs[0]])
            editing.insert_on_edge(node.outputs[0], relu)
    """
```

---

#### API 3: `replace_subgraph` — Replace Multiple Connected Nodes

Higher-level subgraph replacement for multi-node rewrites (fusions, pattern rewrites).

```python
def replace_subgraph(
    old_nodes: Sequence[ir.Node],
    new_nodes: Sequence[ir.Node],
    output_mapping: dict[ir.Value, ir.Value],
) -> None:
    """Replace a subgraph (set of connected nodes) with new nodes.

    This is the general form of replace_node for multi-node patterns.
    The caller provides:
    - The old nodes to remove
    - The new nodes to insert (with inputs already wired to values from the graph)
    - A mapping from old output values (consumed outside the subgraph) to new output values

    The function:
    1. Validates that old_nodes form a subgraph within a single graph
    2. Propagates type/shape/name/const_value from old to new values via the mapping
    3. Redirects all external consumers of mapped old values to new values
    4. Handles graph output replacement (with Identity insertion when needed)
    5. Inserts new_nodes at the position of the first old_node
    6. Removes old_nodes from the graph (in safe mode)

    Args:
        old_nodes: Nodes to remove. Must all belong to the same graph.
        new_nodes: Replacement nodes. Must not already belong to a graph.
        output_mapping: Maps old output values (that have external consumers)
            to their replacement values from new_nodes.

    Raises:
        ValueError: If old_nodes is empty.
        ValueError: If old_nodes span multiple graphs.
        ValueError: If any old output with external consumers is not in output_mapping.

    Example::

        # Fuse MatMul + Add into Gemm
        matmul_node = ...  # existing MatMul in graph
        add_node = ...     # existing Add consuming MatMul output
        gemm = ir.node("Gemm",
            inputs=[matmul_node.inputs[0], matmul_node.inputs[1], add_node.inputs[1]],
            attributes={"alpha": 1.0, "beta": 1.0})
        editing.replace_subgraph(
            old_nodes=[matmul_node, add_node],
            new_nodes=[gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]})
    """
```

##### Before/After: MatMul+Add Fusion

**BEFORE** (20+ lines of manual coordination):
```python
for node in ir.traversal.RecursiveGraphIterator(model.graph):
    if node.op_type != "Add":
        continue
    add_node = node
    input0_producer = add_node.inputs[0].producer() if add_node.inputs[0] else None
    if input0_producer is None or input0_producer.op_type != "MatMul":
        continue
    matmul_node = input0_producer
    if len(list(matmul_node.outputs[0].consumers())) != 1:
        continue
    # Create fused node
    gemm = ir.node("Gemm",
        inputs=[matmul_node.inputs[0], matmul_node.inputs[1], add_node.inputs[1]],
        attributes={"alpha": 1.0, "beta": 1.0})
    # Manual 4-step dance
    ir.convenience.replace_all_uses_with(add_node.outputs[0], gemm.outputs[0])
    model.graph.insert_before(matmul_node, gemm)
    model.graph.remove(add_node, safe=True)
    model.graph.remove(matmul_node, safe=True)
```

**AFTER** (3 lines for the edit, same matching logic):
```python
    # ... same matching logic above ...
    gemm = ir.node("Gemm",
        inputs=[matmul_node.inputs[0], matmul_node.inputs[1], add_node.inputs[1]],
        attributes={"alpha": 1.0, "beta": 1.0})
    editing.replace_subgraph(
        [matmul_node, add_node], [gemm],
        output_mapping={add_node.outputs[0]: gemm.outputs[0]})
```

---

#### API 4: `topological_order` — Non-Mutating Topological Iterator

Lazy topological iteration without modifying the graph. Based on the existing Kahn's algorithm in `Graph.sort()` (line 3516 of `_core.py`) but yielding instead of collecting.

```python
# onnx_ir/traversal.py

def topological_order(
    graph_like: GraphLike,
    *,
    recursive: bool = False,
) -> Iterator[Node]:
    """Iterate over nodes in topological order without modifying the graph.

    Uses Kahn's algorithm with a max-heap to produce a stable topological
    ordering. Nodes appearing earlier in the original graph order are yielded
    first among nodes at the same topological level.

    This is a lazy iterator — nodes are yielded one at a time without
    materializing the full sorted list. For huge graphs, this avoids
    allocating the full sorted node list when you only need a single pass.

    Args:
        graph_like: The graph to iterate over.
        recursive: If True, also yields nodes from subgraphs in topological order.

    Yields:
        Nodes in topological order.

    Raises:
        ValueError: If the graph contains a cycle.

    Example::

        for node in ir.traversal.topological_order(graph):
            # Process nodes in dependency order without mutating graph
            process(node)
    """
```

---

#### API 5: `pass_fn` — Function-Based Pass Decorator

Lowers the barrier to writing simple passes. Directly from the radical thinker's proposal.

```python
# onnx_ir/passes/__init__.py

def pass_fn(fn: Callable[[ir.Model], bool]) -> InPlacePass:
    """Decorator that wraps a function into an InPlacePass.

    The decorated function should accept an ir.Model and return True if
    the model was modified, False otherwise.

    Args:
        fn: A function ``(model: ir.Model) -> bool``.

    Returns:
        An InPlacePass instance that can be used in Sequential/PassManager.

    Example::

        @ir.passes.pass_fn
        def eliminate_identity(model: ir.Model) -> bool:
            modified = False
            for node in ir.traversal.RecursiveGraphIterator(model.graph):
                if node.op_type == "Identity" and node.inputs[0] is not None:
                    editing.replace_node(node, node.inputs[0])
                    modified = True
            return modified

        # Use it like any other pass
        result = eliminate_identity(model)
        # Compose with class-based passes
        pipeline = ir.passes.Sequential(eliminate_identity, other_pass)
    """
```

---

#### API 6: `node.clone()` — Single Node Deep Copy

```python
# Added to Node class in _core.py

def clone(self) -> Node:
    """Create a deep copy of this node with fresh output values.

    The cloned node:
    - Has the same op_type, domain, overload, version, and attributes
    - Has the same input references (shared, not copied)
    - Has NEW output Value objects (not shared with the original)
    - Does NOT belong to any graph (must be inserted manually)

    Returns:
        A new Node that is a deep copy of this node.

    Example::

        original = graph.node("my_relu")
        copy = original.clone()
        graph.insert_after(original, copy)
    """
```

---

#### API 7: `eliminate_node` — Bypass a Passthrough Node

Handles the common pattern where a node is a passthrough (Identity, redundant Cast, no-op Reshape) and should be removed by connecting its input directly to its output consumers. This is a distinct pattern from `replace_node` — there is no replacement node, just a value redirect.

```python
# onnx_ir/editing.py

def eliminate_node(
    node: ir.Node,
    /,
    input_index: int = 0,
    *,
    propagate_metadata: bool = True,
) -> None:
    """Eliminate a passthrough node by redirecting its output consumers to its input.

    This is the fundamental operation for identity-like elimination passes.
    It replaces all uses of the node's output(s) with its input at the given
    index, handles graph output replacement and name/type/shape transfer,
    then removes the node.

    Specifically:
    1. Merges shape/type info from the output value into the input value
       (when propagate_metadata=True)
    2. Replaces all uses of outputs[0] with inputs[input_index]
    3. If outputs[0] is a graph output, updates the graph output and transfers the name
    4. Removes the node from the graph (safe mode)

    This operation handles the "graph output tax" automatically — the caller
    never needs to check is_graph_output().

    Args:
        node: The node to eliminate. Must belong to a graph.
        input_index: Which input to redirect consumers to. Default 0.
        propagate_metadata: If True, merge shape/type from output into the input value.

    Raises:
        ValueError: If node doesn't belong to a graph.
        ValueError: If inputs[input_index] is None.
        ValueError: If the node output is a graph output AND the input is already
            a graph input or initializer (creating an invalid graph input→output passthrough
            that must be preserved as-is).

    Example::

        # Eliminate an Identity node
        editing.eliminate_node(identity_node)

        # Eliminate a redundant Cast (input_index=0 is the data input)
        editing.eliminate_node(redundant_cast_node, input_index=0)

        # In a pass:
        @ir.passes.pass_fn
        def eliminate_identity(model: ir.Model) -> bool:
            modified = False
            for node in ir.traversal.RecursiveGraphIterator(model.graph):
                if node.op_type != "Identity" or node.domain != "":
                    continue
                if len(node.inputs) != 1 or len(node.outputs) != 1:
                    continue
                if node.inputs[0] is None:
                    continue
                if node.outputs[0].is_graph_output() and (
                    node.inputs[0].is_graph_input() or node.inputs[0].is_initializer()):
                    continue  # Can't eliminate graph_input→Identity→graph_output
                editing.eliminate_node(node)
                modified = True
            return modified
    """
```

---

### Tier 2: Design Now, Implement Next

---

#### API 8: `forward_slice` / `backward_slice` — Transitive Reachability

```python
# onnx_ir/traversal.py

def forward_slice(
    start: ir.Node | ir.Value,
    *,
    boundary: Callable[[ir.Node], bool] | None = None,
) -> ir.GraphView:
    """Return all nodes transitively reachable from start via consumer edges.

    Performs a BFS/DFS forward from the starting point, following
    Value → consumer_node edges. Returns a GraphView with auto-detected
    boundary inputs and outputs.

    Args:
        start: Starting node or value.
        boundary: Optional predicate — stop traversal at nodes where this returns True.
            Boundary nodes ARE included in the result.

    Returns:
        A GraphView containing the forward slice. The GraphView's inputs are the
        values entering the slice from outside, and outputs are the values leaving.
    """

def backward_slice(
    start: ir.Node | ir.Value,
    *,
    boundary: Callable[[ir.Node], bool] | None = None,
) -> ir.GraphView:
    """Return all nodes transitively reachable from start via producer edges.

    Args:
        start: Starting node or value.
        boundary: Optional predicate — stop traversal at nodes where this returns True.
            Boundary nodes ARE included in the result.

    Returns:
        A GraphView containing the backward slice with auto-detected inputs/outputs.
    """
```

**Composition with editing:**
```python
# Find everything downstream of a node, replace it all
downstream = ir.traversal.forward_slice(some_node)
handle = editing.SubgraphHandle.from_graph_view(graph, downstream)
handle.replace_with(optimized_subgraph)
```

---

#### API 9: `SubgraphHandle` — Mutable Subgraph Reference

A first-class object representing a connected subgraph within a parent graph. Unlike `GraphView` (read-only, no parent reference), `SubgraphHandle` knows its parent and can mutate through it.

```python
# onnx_ir/editing.py

class SubgraphHandle:
    """A mutable handle to a connected subgraph within a parent Graph.

    Unlike GraphView (which is read-only and doesn't reference a parent),
    SubgraphHandle knows its parent graph and supports mutation operations
    that delegate to the parent.

    The handle auto-discovers boundary values:
    - inputs: values consumed by subgraph nodes but produced outside
    - outputs: values produced by subgraph nodes but consumed outside
    - internal: values produced and consumed entirely within the subgraph
    """

    def __init__(self, parent: ir.Graph, nodes: Collection[ir.Node]) -> None: ...

    @classmethod
    def between(
        cls,
        parent: ir.Graph,
        input_values: Sequence[ir.Value],
        output_values: Sequence[ir.Value],
    ) -> SubgraphHandle:
        """Create handle from boundary values (backward traversal from outputs to inputs)."""

    @classmethod
    def from_nodes(cls, parent: ir.Graph, nodes: Collection[ir.Node]) -> SubgraphHandle:
        """Create handle from explicit node set (auto-discovers boundaries)."""

    @classmethod
    def from_graph_view(cls, parent: ir.Graph, view: ir.GraphView) -> SubgraphHandle:
        """Create handle from an existing GraphView."""

    @property
    def nodes(self) -> frozenset[ir.Node]: ...
    @property
    def inputs(self) -> tuple[ir.Value, ...]: ...
    @property
    def outputs(self) -> tuple[ir.Value, ...]: ...
    @property
    def internal_values(self) -> frozenset[ir.Value]: ...

    def replace_with(
        self,
        new_nodes: Sequence[ir.Node],
        output_mapping: dict[ir.Value, ir.Value],
    ) -> None:
        """Replace this subgraph with new nodes. Delegates to replace_subgraph()."""

    def detach(self) -> ir.Graph:
        """Remove from parent and return as standalone Graph (via clone)."""

    def as_graph_view(self) -> ir.GraphView:
        """Return a read-only GraphView of this subgraph."""
```

---

#### ~~API 10: `apply_rewrite` + `RewriteRule`~~ — **OUT OF SCOPE**

> **Deferred.** The `onnxscript` project already provides pattern matching and rewrite rule capabilities. Adding a parallel system in `onnx_ir` would create confusion and duplication. Pattern matching/rewriting is moved to Tier 3 for future re-evaluation of whether tighter integration with the IR editing verbs is warranted.

---

#### API 10: `find_nodes` — Filtered Node Search

```python
# onnx_ir/traversal.py

def find_nodes(
    graph_like: GraphLike,
    *,
    op_type: str | None = None,
    domain: str | None = None,
    predicate: Callable[[ir.Node], bool] | None = None,
    recursive: bool = True,
) -> Iterator[ir.Node]:
    """Find nodes matching the given criteria.

    Convenience wrapper over RecursiveGraphIterator with filtering.

    Args:
        graph_like: The graph to search.
        op_type: Filter by operator type (exact match).
        domain: Filter by domain (exact match, "" matches default ONNX domain).
        predicate: Custom filter function.
        recursive: If True, search subgraphs too.

    Yields:
        Matching nodes in graph order.

    Example::

        # Find all Conv nodes
        for conv in ir.traversal.find_nodes(graph, op_type="Conv"):
            print(conv.name)

        # Find nodes with large kernel sizes
        for node in ir.traversal.find_nodes(graph, op_type="Conv",
            predicate=lambda n: n.attributes.get_ints("kernel_shape", []) and
                                max(n.attributes.get_ints("kernel_shape")) > 5):
            ...
    """
```

---

### Tier 3: Future Research (Bold Ideas That Should Inform Design)

| Idea | Source | Why Defer | Path Forward |
|---|---|---|---|
| **Pattern Matching / RewriteRule** | Both | `onnxscript` already provides pattern matching and rewrite rule capabilities; adding a parallel system creates duplication | Re-evaluate if tighter integration with IR editing verbs is needed after Tier 1/2 are battle-tested |
| **Pattern Matching DSL** | Radical Thinker | Needs DSL design, parser, error reporting; depends on RewriteRule decision | Only relevant if programmatic RewriteRule is added first |
| **Transactions/Checkpoints** | Radical Thinker | Python reference semantics make rollback infeasible | Tier 1 verbs are correct-by-construction, reducing need |
| **Graph Cursor** | Radical Thinker | Novel paradigm, unclear error semantics | Prototype for interactive tooling after core APIs stabilize |
| **Subgraph-as-Function extraction** | Both | Complex wiring semantics | Design after Tier 1/2 battle-tested |
| **`graph.validate()`** | Architect | Useful but not blocking | Add incrementally as common bugs are discovered |
| **`graph.partition()`** | Architect | Complex boundary semantics | Design after `forward_slice`/`backward_slice` prove out |

#### Tier 3 Vision: Declarative Pattern DSL (sketch for future consideration)

> **Note:** Pattern matching is currently out of scope since `onnxscript` provides this capability. This sketch is preserved to show what a tighter integration with the IR editing verbs could look like in the future, if re-evaluation determines it's warranted.

```python
from onnx_ir import patterns as pat

# Pattern definition — declarative, composable
conv_bn_relu = pat.sequence(
    conv := pat.op("Conv", outputs=["conv_out"]),
    bn := pat.op("BatchNormalization", inputs=[conv.out(0), pat.any(), pat.any(), pat.any(), pat.any()]),
    relu := pat.op("Relu", inputs=[bn.out(0)])
).where(
    conv.out(0).has_single_consumer(),
    bn.out(0).has_single_consumer(),
)

# Pattern match + edit using Tier 1 editing verbs
for match in conv_bn_relu.find_all(graph):
    fused = ir.node("FusedConvBnRelu", inputs=[...])
    editing.replace_subgraph(
        match.nodes, [fused],
        output_mapping={match["relu"].outputs[0]: fused.outputs[0]})
```

**Key insight**: Any future pattern matching system should compose with the Tier 1 editing verbs (`replace_node`, `replace_subgraph`, `eliminate_node`), not replace them.

---

## Module Organization

```
src/onnx_ir/
├── editing.py              # NEW — replace_node, eliminate_node, insert_on_edge, replace_subgraph,
│                           #        SubgraphHandle (Tier 2)
├── traversal.py            # EXTENDED — topological_order, forward/backward_slice, find_nodes
├── passes/
│   └── __init__.py         # EXTENDED — pass_fn decorator
├── _core.py                # MINIMAL CHANGE — Node.replace_with() + Node.clone() thin wrappers
└── convenience.py          # UNCHANGED — existing APIs remain, not deprecated
```

**Key principles:**
- `editing.py` is the primary location for all new graph manipulation APIs
- `convenience.py` stays unchanged for backward compatibility — existing passes are NOT broken
- `traversal.py` gets new iterator functions alongside the existing `RecursiveGraphIterator`
- `_core.py` gets only thin method wrappers (`replace_with`, `clone`) that delegate to `editing`
- Over time, users migrate from `convenience.replace_nodes_and_values()` to `editing.replace_node()` / `editing.replace_subgraph()`

---

## Implementation Notes

### Graph Output Handling Strategy

The #1 source of bugs in existing passes. All editing verbs handle this uniformly so pass authors **never** think about `is_graph_output()`:

1. **When old output was a graph output and new output is NOT already a graph output or input:**  
   Replace in `graph.outputs` list, transfer the name from old to new.

2. **When old output was a graph output and new output IS already a graph output or input:**  
   Insert an Identity node to avoid duplicate/conflicting graph outputs. This matches the existing pattern in the CSE pass (lines 163-175 of `common_subexpression_elimination.py`).

3. **When old output was NOT a graph output:**  
   Simple `replace_all_uses_with`, no special handling needed.

### Metadata Propagation Strategy

When `propagate_metadata=True` (the default), propagate in this priority order:
1. `type` — from old value if not None, else keep new value's type
2. `shape` — from old value if not None, else keep new value's shape
3. `const_value` — from old value if not None, else keep new value's const_value
4. `name` — from old value if not None, else keep new value's name
5. `metadata_props` — merged (new takes precedence on conflicts)

This matches the existing behavior in `replace_nodes_and_values()` (lines 438-448 of `_convenience/__init__.py`).

### Backward Compatibility

- **All existing APIs continue to work unchanged**
- `convenience.replace_nodes_and_values()` is NOT deprecated (it serves a lower-level use case)
- `convenience.replace_all_uses_with()` is NOT deprecated (it's used internally by the new APIs)
- Passes can adopt the new APIs incrementally — no forced migration
- No changes to `PassBase`, `InPlacePass`, `FunctionalPass`, or `PassManager`

### Performance Requirements

**Efficiency is critical** — this library is used in performance-sensitive model optimization pipelines. All implementations must be performance-conscious:

| API | Expected Complexity | Notes |
|---|---|---|
| `replace_node` | O(outputs × uses) | Dominated by `replace_all_uses_with` which is O(uses per output). No unnecessary list copies. |
| `insert_on_edge` | O(uses) | Single pass over consumers of the value. Use `replace_all_uses_with` internally. |
| `replace_subgraph` | O(Σ outputs × uses) | Same as N × replace_node. Build `frozenset(old_nodes)` once for O(1) membership checks. |
| `topological_order` | O(V + E) total, lazy | Must be a generator — yield one node at a time. Use heapq like existing `Graph.sort()`. Never materialize full list. |
| `pass_fn` | Zero overhead | Thin wrapper, no per-call allocation beyond PassResult. |
| `node.clone()` | O(inputs + outputs + attributes) | Reuse `_cloner.py` infrastructure. Share tensor data (don't copy). |
| `eliminate_node` | O(uses) | Single `replace_all_uses_with` + O(inputs) for detach. |

**General rules:**
- Avoid materializing lists when iterators suffice
- Use `frozenset` for membership tests over sequences
- Share tensor data (like `graph.clone()` does) — never deep-copy tensor buffers
- Prefer tuple over list for immutable collections (cache-friendly, hashable)
- No unnecessary string formatting in hot paths (use lazy logging: `logger.debug("...", arg)` not `logger.debug(f"...{arg}")`)

### Implementation Workflow

**Each self-contained feature must be a separate commit and push** for easy review:

1. `node.clone()` — standalone, no dependencies on other new APIs
2. `eliminate_node` — depends on existing `replace_all_uses_with` only
3. `replace_node` — can use `eliminate_node` internally for value-redirect cases
4. `insert_on_edge` — uses existing `replace_all_uses_with` + `insert_after`
5. `replace_subgraph` — can use `replace_node`/`replace_all_uses_with` internally
6. `topological_order` — standalone traversal, no editing dependencies
7. `pass_fn` — standalone decorator, no editing dependencies

Commits 1, 6, 7 can be done in any order (no interdependencies). Commits 2-5 have the suggested dependency chain.

---

## Testing Strategy

Each new API gets:
1. **Unit tests** co-located (e.g., `editing_test.py` next to `editing.py`)
2. **Edge case tests**: graph outputs, multi-output nodes, empty graphs, nodes in subgraphs
3. **Integration test**: rewrite an existing pass (e.g., identity elimination) using the new API and verify identical behavior via serde roundtrip comparison
4. **Regression tests**: encode the exact patterns found in existing passes as test cases

---

## Summary Decision Matrix

| # | API | Tier | Impact | Complexity | Breaking Changes | Module |
|---|---|---|---|---|---|---|
| 1 | `replace_node` | 1 | 🔴 Critical | Medium | None | `editing.py` |
| 2 | `insert_on_edge` | 1 | 🟠 High | Low | None | `editing.py` |
| 3 | `replace_subgraph` | 1 | 🟠 High | Medium | None | `editing.py` |
| 4 | `topological_order` | 1 | 🟡 Medium | Low | None | `traversal.py` |
| 5 | `pass_fn` | 1 | 🟡 Medium | Very Low | None | `passes/__init__.py` |
| 6 | `node.clone()` | 1 | 🟡 Medium | Low | None | `_core.py` |
| 7 | `eliminate_node` | 1 | 🟠 High | Low | None | `editing.py` |
| 8 | `forward/backward_slice` | 2 | 🟠 High | Medium | None | `traversal.py` |
| 9 | `SubgraphHandle` | 2 | 🟠 High | Medium | None | `editing.py` |
| 10 | `find_nodes` | 2 | 🟡 Medium | Very Low | None | `traversal.py` |
| — | ~~`RewriteRule`~~ | ~~2~~ | — | — | — | *Out of scope (onnxscript overlap)* |
| — | Pattern DSL | 3 | 🔴 Critical (long-term) | Very High | None | TBD |
| — | Transactions | 3 | 🟡 Medium | Very High | None | TBD |
| — | Graph Cursor | 3 | 🟡 Medium | High | None | TBD |

---

## Appendix: Before/After — Identity Elimination Pass Rewritten

This appendix demonstrates the concrete impact of the new APIs by rewriting the existing `IdentityEliminationPass` (from `passes/common/identity_elimination.py`).

### BEFORE: Current Implementation (lines 74-121)

The pass author must manually handle shape merging, use replacement, graph output detection, name transfer, and node removal — 5 distinct concerns interleaved with the domain logic:

```python
def _merge_shapes(shape1, shape2):
    """18 lines of shape merging logic..."""
    def merge_dims(dim1, dim2):
        if dim1 == dim2: return dim1
        if not isinstance(dim1, ir.SymbolicDim): return dim1
        if not isinstance(dim2, ir.SymbolicDim): return dim2
        if dim1.value is None: return dim2
        return dim1
    if shape1 is None: return shape2
    if shape2 is None: return shape1
    if len(shape1) != len(shape2):
        raise ValueError(f"Shapes must have the same rank, got {len(shape1)} and {len(shape2)}.")
    return ir.Shape([merge_dims(dim1, dim2) for dim1, dim2 in zip(shape1, shape2)])


class IdentityEliminationPass(ir.passes.InPlacePass):
    """Pass for eliminating redundant Identity nodes.

    Rules:
    1. y = Identity(x) where y is NOT a graph output → replace uses of y with x, remove node
    2. y is a graph output, x is NOT a graph input → same + rename x to y
    3. y is graph output AND x is graph input/initializer → KEEP the node
    """

    def call(self, model):
        modified = False
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if self._try_eliminate_identity_node(node):
                modified = True
        for function in model.functions.values():
            for node in ir.traversal.RecursiveGraphIterator(function):
                if self._try_eliminate_identity_node(node):
                    modified = True
        return ir.passes.PassResult(model, modified=modified)

    def _try_eliminate_identity_node(self, node):
        # --- Domain logic: should we eliminate? ---
        if node.op_type != "Identity" or node.domain != "":
            return False
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            return False
        input_value = node.inputs[0]
        output_value = node.outputs[0]
        if input_value is None:
            return False
        graph_like = node.graph
        assert graph_like is not None
        output_is_graph_output = output_value.is_graph_output()
        # Case 3: can't eliminate
        if output_is_graph_output and (
            input_value.is_graph_input() or input_value.is_initializer()
        ):
            return False

        # --- Mechanism: HOW to eliminate (5 concerns mixed together) ---
        # 1. Manually merge shapes
        input_value.shape = _merge_shapes(input_value.shape, output_value.shape)
        # 2. Manually propagate type
        if input_value.type is None:
            input_value.type = output_value.type
        # 3. Manually replace all uses (must remember replace_graph_outputs=True!)
        ir.convenience.replace_all_uses_with(
            output_value, input_value, replace_graph_outputs=True
        )
        # 4. Manually handle name for graph outputs
        if output_is_graph_output:
            input_value.name = output_value.name
        # 5. Manually remove
        graph_like.remove(node, safe=True)
        return True
```

**Total:** ~50 lines for the core method + 18 lines for the shape merging helper = **~68 lines**. Five separate concerns (shape merge, type propagation, use replacement, name transfer, removal) are manually coordinated. Forgetting any step creates a subtle bug.

### AFTER: With `editing.eliminate_node`

The pass author focuses purely on **decision logic** (should this node be eliminated?). All mechanism is delegated to `eliminate_node`:

```python
from onnx_ir import editing


class IdentityEliminationPass(ir.passes.InPlacePass):
    """Pass for eliminating redundant Identity nodes."""

    def call(self, model):
        modified = False
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if self._try_eliminate_identity_node(node):
                modified = True
        for function in model.functions.values():
            for node in ir.traversal.RecursiveGraphIterator(function):
                if self._try_eliminate_identity_node(node):
                    modified = True
        return ir.passes.PassResult(model, modified=modified)

    def _try_eliminate_identity_node(self, node):
        # --- Domain logic ONLY: should we eliminate? ---
        if node.op_type != "Identity" or node.domain != "":
            return False
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            return False
        if node.inputs[0] is None:
            return False
        # Case 3: can't eliminate graph_input → Identity → graph_output
        if node.outputs[0].is_graph_output() and (
            node.inputs[0].is_graph_input() or node.inputs[0].is_initializer()
        ):
            return False
        # --- One call handles ALL mechanism ---
        # Shape merge, type propagation, use replacement, graph output handling,
        # name transfer, and node removal — all absorbed by eliminate_node
        editing.eliminate_node(node, input_index=0)
        return True
```

**Total:** ~30 lines. No shape merging helper needed. No manual use replacement. No graph output special-casing.

### Or, using `@pass_fn` for maximum conciseness:

```python
from onnx_ir import editing

@ir.passes.pass_fn
def eliminate_identity(model: ir.Model) -> bool:
    """Eliminate redundant Identity nodes."""
    modified = False
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        if node.op_type != "Identity" or node.domain != "":
            continue
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            continue
        if node.inputs[0] is None:
            continue
        if node.outputs[0].is_graph_output() and (
            node.inputs[0].is_graph_input() or node.inputs[0].is_initializer()
        ):
            continue
        editing.eliminate_node(node, input_index=0)
        modified = True
    # Note: functions also need processing in the full version
    return modified
```

**Total:** ~18 lines for the entire pass. This is what "encode intent, not mechanism" looks like in practice.

### Impact Summary

| Metric | Before | After | Reduction |
|---|---|---|---|
| Lines of code (core method) | ~50 | ~15 | **70%** |
| Helper functions needed | 1 (`_merge_shapes`) | 0 | **100%** |
| Concerns manually coordinated | 5 | 0 | **100%** |
| Graph output special-casing | Manual (`is_graph_output()` check + name transfer) | Automatic | **Absorbed** |
| Bug surface area | 5 points of failure | 1 call | **80%** |
