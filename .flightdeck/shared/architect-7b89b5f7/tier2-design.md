# Tier 2 Design: SubgraphHandle & GraphCheckpoint

**Author:** Architect (7b89b5f7)
**Status:** REVISED — Incorporates Critical Reviewer feedback (e31069b3)
**Branch:** `justinchu/improvements` (builds on Tier 1)

---

## Problem Statement

Tier 1 gave us four editing verbs (`replace_node`, `eliminate_node`, `insert_on_edge`, `replace_subgraph`) plus supporting APIs. These work well for **individual operations**, but two gaps remain:

1. **Subgraph analysis before mutation.** `replace_subgraph()` requires the caller to manually compute boundary values (which outputs are external, which inputs come from outside). For multi-node patterns, this bookkeeping is tedious and error-prone.

2. **Safe rollback on failure.** If step 3 of 5 in a graph transformation fails or produces an invalid result, the graph is in a partially-modified state. The existing `FunctionalPass` + `graph.clone()` pattern works but forces the pass to operate on a clone upfront — there's no way to try an in-place transformation and roll back on failure.

### Challenging the framing

Before diving in, I challenged whether these are even the right problems:

- **SubgraphHandle vs. just using `replace_subgraph` directly:** `replace_subgraph` already works for simple cases. But when you need to *inspect* a pattern before deciding how to transform it (e.g., "what are this subgraph's external inputs? Does it have side effects? What types flow in?"), there's no structured way to do that. A handle fills this gap — it's the "analyze" step before the "transform" step.

- **Transactions vs. just using `FunctionalPass`:** `FunctionalPass` clones upfront and returns the clone. This works but (a) always pays the O(V+E) clone cost even when the pass succeeds (the common path), and (b) only works within the pass framework — ad hoc code that modifies a graph has no safety net. More importantly, passes that try multiple local transformations (e.g., "try to fuse each MatMul+Add pair, skip if validation fails") need rollback at a **finer granularity** than whole-model clone.

---

## API 1: SubgraphHandle

### Design Philosophy

SubgraphHandle is an **immutable, boundary-annotated node set** — not a live view. You construct it from a set of nodes (or boundary values), it auto-discovers the boundary, and then you can inspect or replace. It's consumed once, not updated incrementally.

This is intentional: a "live" handle that tracks graph mutations would need to intercept every editing operation, adding coupling and overhead with no clear use case. The common pattern is: find pattern → analyze → transform → discard handle.

### Location

`src/onnx_ir/editing.py` — same module as the editing verbs. The handle is a companion to `replace_subgraph()`, not a separate concern.

### Public Interface

```python
# In onnx_ir/editing.py

class SubgraphHandle:
    """An immutable, boundary-annotated handle to a subgraph within a parent Graph.

    Unlike :class:`~onnx_ir.GraphView` (read-only, no parent reference, stores tuples),
    ``SubgraphHandle`` knows its parent graph, auto-discovers boundary values,
    and supports mutation operations that delegate to existing editing functions.

    Terminology:
        - **inputs**: Values consumed by subgraph nodes but produced *outside* the subgraph
          (or are graph inputs/initializers). These are the subgraph's external dependencies.
        - **outputs**: Values produced by subgraph nodes and consumed by at least one node
          *outside* the subgraph, or that appear in the parent graph's output list.
          These are the subgraph's externally-visible results.
        - **internal values**: Values produced and consumed entirely within the subgraph.

    The handle is immutable after construction. Calling :meth:`replace_with` or
    :meth:`detach` consumes the handle (the nodes are removed from the parent graph),
    and using the handle after that raises :class:`RuntimeError`.

    Example::

        import onnx_ir as ir
        from onnx_ir import editing

        # Find a MatMul+Add pattern and analyze it
        handle = editing.SubgraphHandle.from_nodes(graph, [matmul_node, add_node])
        print(handle.inputs)   # e.g., (A, B, bias)
        print(handle.outputs)  # e.g., (add_output,)

        # Replace with fused Gemm
        gemm = ir.node("Gemm", inputs=[*handle.inputs])
        handle.replace_with(
            new_nodes=[gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]},
        )
    """

    def __init__(
        self,
        parent: ir.Graph,
        nodes: Collection[ir.Node],
    ) -> None:
        """Create a SubgraphHandle from a parent graph and a set of nodes.

        Args:
            parent: The graph that contains all the nodes.
            nodes: The nodes in the subgraph. Must all belong to *parent*.

        Raises:
            ValueError: If *nodes* is empty.
            ValueError: If any node does not belong to *parent*.
        """

    # ---- Construction alternatives ----

    @classmethod
    def between(
        cls,
        parent: ir.Graph,
        input_values: Sequence[ir.Value],
        output_values: Sequence[ir.Value],
    ) -> SubgraphHandle:
        """Create a SubgraphHandle from boundary values.

        Performs a backward traversal from *output_values* to *input_values*
        to discover all nodes in between. Uses the same algorithm as
        :func:`onnx_ir.convenience.extract` but without cloning.

        Args:
            parent: The graph containing the subgraph.
            input_values: Values that bound the subgraph's "top" edge.
                Traversal stops at producers of these values.
            output_values: Values that bound the subgraph's "bottom" edge.
                Traversal starts from producers of these values.

        Raises:
            ValueError: If the subgraph is not properly bounded (unreachable
                inputs, missing values).

        Example::

            # Find all nodes between input X and output Y
            handle = SubgraphHandle.between(graph, [x_value], [y_value])
        """

    # ---- Read-only properties ----

    @property
    def parent(self) -> ir.Graph:
        """The parent graph containing this subgraph."""

    @property
    def nodes(self) -> frozenset[ir.Node]:
        """The nodes in this subgraph (unordered set)."""

    @property
    def inputs(self) -> tuple[ir.Value, ...]:
        """Values consumed by subgraph nodes but produced outside.

        Includes graph inputs and initializers consumed by the subgraph.
        Order: deterministic, sorted by first use position in the parent graph.
        """

    @property
    def outputs(self) -> tuple[ir.Value, ...]:
        """Values produced by subgraph nodes and consumed externally.

        Includes values that appear in ``parent.outputs``.
        Order: deterministic, sorted by producer position in the parent graph.
        """

    @property
    def internal_values(self) -> frozenset[ir.Value]:
        """Values produced and consumed entirely within the subgraph."""

    def __len__(self) -> int:
        """Number of nodes in the subgraph."""

    def __contains__(self, node: ir.Node) -> bool:
        """Check if a node is in this subgraph."""

    def __iter__(self) -> Iterator[ir.Node]:
        """Iterate over nodes in parent-graph order."""

    # ---- Mutation operations (consume the handle) ----

    def replace_with(
        self,
        new_nodes: Sequence[ir.Node],
        output_mapping: dict[ir.Value, ir.Value],
        *,
        propagate_metadata: bool = True,
    ) -> None:
        """Replace this subgraph with new nodes.

        Delegates to :func:`replace_subgraph` with the handle's nodes.
        After this call, the handle is **consumed** and must not be reused.

        Before delegating, verifies that all nodes are still in the parent
        graph (guards against stale handles after checkpoint rollback).

        Args:
            new_nodes: Replacement nodes.
            output_mapping: Maps old output values to new output values.
            propagate_metadata: If True, propagate type/shape/name/const_value.

        Raises:
            RuntimeError: If the handle has already been consumed.
            RuntimeError: If any node no longer belongs to the parent graph
                (e.g., after a checkpoint rollback invalidated this handle).
        """

    def as_graph_view(self) -> ir.GraphView:
        """Return a read-only GraphView of this subgraph.

        Does NOT consume the handle. The GraphView references the same
        Node objects (not copies).

        .. warning::
            The GraphView's lifetime is bounded by this handle's. After
            :meth:`replace_with` consumes the handle, any previously-returned
            GraphView is stale (its nodes have been removed from the graph).

        Raises:
            RuntimeError: If the handle has already been consumed.
        """
```

### Boundary Discovery Algorithm

The `__init__` method computes boundaries in O(V + E) where V = len(nodes) and E = total edges of those nodes:

```python
def _compute_boundaries(self) -> None:
    node_set = self._nodes  # frozenset[Node]

    inputs = []  # Values consumed from outside
    outputs = []  # Values visible outside
    internal = set()  # Values produced and consumed internally only
    seen_inputs = set()

    # Discover inputs: any input value whose producer is NOT in the subgraph
    for node in self._ordered_nodes:
        for inp in node.inputs:
            if inp is None or inp in seen_inputs:
                continue
            producer = inp.producer()
            if producer is None or producer not in node_set:
                inputs.append(inp)
                seen_inputs.add(inp)

    # Discover outputs: any output value that has a consumer outside the
    # subgraph, or appears in parent.outputs
    parent_outputs = frozenset(self._parent.outputs)
    for node in self._ordered_nodes:
        for out in node.outputs:
            has_external_consumer = any(
                user not in node_set for user, _ in out.uses()
            )
            if has_external_consumer or out in parent_outputs:
                outputs.append(out)
            else:
                internal.add(out)

    self._inputs = tuple(inputs)
    self._outputs = tuple(outputs)
    self._internal_values = frozenset(internal)
```

### `between()` Implementation Strategy

Reuses the existing `_find_subgraph_bounded_by_values()` from `_convenience/_extractor.py`:

```python
@classmethod
def between(cls, parent, input_values, output_values):
    from onnx_ir._convenience._extractor import _find_subgraph_bounded_by_values
    nodes, _ = _find_subgraph_bounded_by_values(
        parent, input_values, output_values, parent_graph=parent
    )
    return cls(parent, nodes)
```

### Edge Cases

| Case | Behavior |
|------|----------|
| Node not in parent | `ValueError` in `__init__` |
| Empty node set | `ValueError` in `__init__` |
| Node in subgraph has no external consumers | Not in `outputs`, in `internal_values` |
| Graph input consumed by subgraph | Appears in `inputs` |
| Graph output produced by subgraph | Appears in `outputs` |
| Value consumed by multiple subgraph nodes | Appears once in `inputs` |
| `replace_with` on consumed handle | `RuntimeError` |
| `as_graph_view` on consumed handle | `RuntimeError` |
| Single-node subgraph | Works; degenerates to `replace_node` behavior |
| Stale handle after rollback | `RuntimeError` from liveness check in `replace_with()` |

### Integration with Tier 1

- `replace_with()` delegates directly to `editing.replace_subgraph()` — zero new mutation logic.
- `as_graph_view()` creates a `GraphView` with the handle's nodes, inputs, and outputs.
- `between()` reuses the proven `_find_subgraph_bounded_by_values()` from the extractor.
- The handle's `inputs` and `outputs` properties give exactly the information needed to construct `output_mapping` for `replace_subgraph()`.

### Test Cases

1. **Construction from nodes**: Create handle from 2-3 nodes, verify inputs/outputs/internal_values
2. **Construction via `between()`**: Bounded traversal from outputs back to inputs
3. **Single-node handle**: Verify degenerates correctly
4. **Graph inputs/outputs**: Subgraph consuming graph inputs, producing graph outputs
5. **Shared input**: Two subgraph nodes consuming the same external value → appears once in inputs
6. **Internal edges**: Values flowing between subgraph nodes don't appear in inputs/outputs
7. **`replace_with()`**: Replace a 2-node subgraph with a single fused node
8. **`replace_with()` propagate_metadata**: Verify type/shape/name transfer
9. **`replace_with()` graph outputs**: Subgraph outputs that are also graph outputs
10. **`as_graph_view()`**: Verify GraphView has correct nodes/inputs/outputs
11. **Consumed handle — `replace_with()`**: Verify `RuntimeError` after consumption
12. **Consumed handle — `as_graph_view()`**: Verify `RuntimeError` after consumption
13. **Diamond pattern**: Two subgraph nodes sharing an input, merging at an output
14. **Empty external consumers**: All outputs consumed internally → `outputs` is empty (but graph output case still reported)
15. **Iteration order**: `__iter__` yields nodes in parent-graph order
16. **Stale handle after rollback**: Create handle, rollback checkpoint, verify `RuntimeError` on `replace_with()`

---

## API 2: GraphCheckpoint (Transactions)

### Reframing the Problem

The prior session rejected transactions as "infeasible in Python due to reference semantics." I spent significant time re-examining this and arrived at a nuanced conclusion:

**Full command-log transactions (record-and-undo every mutation) are indeed infeasible.** The reasons:
- `Value` objects are identified by `id()` across `DoublyLinkedSet`, use-lists, and graph containers
- `replace_all_uses_with` touches N consumer nodes, each of which modifies its input tuple and the value's use-list — recording and reversing all of this is enormously complex
- Metadata propagation (name/shape/type transfers) is lossy — undoing it requires saving original values
- Graph output slot replacements and Identity node insertions are side effects that compound

**But snapshot-based checkpointing is simple and practical.** Clone the graph upfront (O(V+E), same cost as `FunctionalPass`), operate in-place, restore from clone on failure.

The key insight the prior session missed: **checkpoints add value beyond `FunctionalPass` for two reasons:**

1. **Finer granularity.** A pass can checkpoint before each local transformation attempt, not just once per pass invocation. This enables "try each pattern, skip on failure" loops without corrupting the graph.

2. **Works outside the pass framework.** Ad hoc graph optimization code, notebooks, debugging sessions — anywhere you want a safety net without restructuring code into a Pass subclass.

### Design Philosophy

- **Explicit, not magical.** No implicit interception of editing operations. The checkpoint is a saved state; rollback swaps the graph reference.
- **Model-level.** Works by replacing `model.graph` on rollback. This is the only clean way in Python without fighting reference semantics.
- **Honest about costs.** O(V+E) upfront clone. O(1) rollback (reference swap). Reference invalidation after rollback is clearly documented.
- **Context manager for safety.** Auto-rollback on unhandled exceptions prevents graph corruption.

### Location

`src/onnx_ir/editing.py` — alongside the editing verbs. A checkpoint is an editing concern.

### Public Interface

```python
# In onnx_ir/editing.py

class GraphCheckpoint:
    """A checkpoint for a model's graph, enabling rollback on failure.

    Takes a snapshot of ``model.graph`` at creation time. If the transformation
    fails or produces an invalid result, call :meth:`rollback` to restore the
    graph to its checkpointed state.

    **Cost:** O(V + E) at creation (``graph.clone()``), O(1) at rollback (reference
    swap), O(1) at commit (discard the clone for GC).

    **Reference invalidation:** After :meth:`rollback`, the model's graph is a
    *different object* with different Node and Value instances. Any local
    variables referencing nodes/values from the pre-rollback graph are
    **stale** and must not be used. The context manager pattern naturally
    avoids this issue because rollback exits the ``with`` block.

    Usage as context manager (recommended)::

        with editing.GraphCheckpoint(model) as cp:
            editing.replace_node(old_node, new_node)
            editing.eliminate_node(identity_node)
            if not validate(model):
                cp.rollback()
        # After the block: graph is either transformed (success)
        # or restored (rollback / exception).

    Usage for fine-grained rollback in a loop::

        for matmul, add in find_fusible_pairs(model.graph):
            with editing.GraphCheckpoint(model):
                try:
                    gemm = ir.node("Gemm", inputs=[...])
                    editing.replace_subgraph([matmul, add], [gemm], {...})
                except ValueError:
                    pass  # auto-rollback on exception, try next pair

    Explicit usage (without context manager)::

        cp = editing.GraphCheckpoint(model)
        editing.replace_node(old_node, new_node)
        if not is_valid(model):
            cp.rollback()
        else:
            cp.commit()  # free the snapshot

    Args:
        model: The model whose graph to checkpoint.
    """

    def __init__(self, model: ir.Model) -> None:
        """Create a checkpoint of the model's current graph state.

        Args:
            model: The model to checkpoint. ``model.graph`` is cloned.
        """

    @property
    def is_active(self) -> bool:
        """True if this checkpoint has not been committed or rolled back."""

    def rollback(self) -> None:
        """Restore the model's graph to the checkpointed state.

        After rollback:
        - ``model.graph`` is the cloned graph from checkpoint creation.
        - All Node/Value references from the pre-rollback graph are stale.
        - The checkpoint is consumed (not reusable).

        Nested checkpoints are supported in LIFO order: inner rollback
        first, then outer rollback restores to the outer checkpoint's
        saved state. Each checkpoint unconditionally restores its snapshot.

        Raises:
            RuntimeError: If the checkpoint has already been committed or
                rolled back.
        """

    def commit(self) -> None:
        """Discard the checkpoint snapshot, freeing memory.

        Call this when the transformation succeeded and rollback is no
        longer needed. If using the context manager, commit is called
        automatically on clean exit.

        Raises:
            RuntimeError: If the checkpoint has already been committed or
                rolled back.
        """

    def __enter__(self) -> GraphCheckpoint:
        """Enter the checkpoint context."""

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        """Exit the checkpoint context.

        - On clean exit (no exception, no prior rollback): calls ``commit()``.
        - On exception (no prior rollback): calls ``rollback()``.
        - If already rolled back or committed: no-op.

        Exceptions are never suppressed (returns ``False``).
        """
```

### Implementation

```python
class GraphCheckpoint:
    def __init__(self, model: ir.Model) -> None:
        self._model = model
        self._saved_graph = model.graph.clone()
        self._active = True

    @property
    def is_active(self) -> bool:
        return self._active

    def rollback(self) -> None:
        if not self._active:
            raise RuntimeError(
                "This checkpoint has already been committed or rolled back."
            )
        if self._saved_graph is None:
            raise RuntimeError("Internal error: saved graph is None on active checkpoint.")
        self._model.graph = self._saved_graph
        self._saved_graph = None
        self._active = False

    def commit(self) -> None:
        if not self._active:
            raise RuntimeError(
                "This checkpoint has already been committed or rolled back."
            )
        self._saved_graph = None  # Free the clone for GC
        self._active = False

    def __enter__(self) -> GraphCheckpoint:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self._active:
            if exc_type is not None:
                self.rollback()
            else:
                self.commit()
        return False  # Never suppress exceptions
```

### Design Tradeoffs

| Decision | Alternative | Rationale |
|----------|-------------|-----------|
| **Model-level** (swap `model.graph`) | Graph-level (clear + repopulate) | Graph-level would require removing all nodes and re-inserting cloned nodes — complex, fragile, and still invalidates references. `model.graph = clone` is O(1) and idiomatic. |
| **Clone upfront** | Clone on rollback only | Cloning on rollback would save cost in the success path, but we'd need to clone the *original* state — which is gone by rollback time. The only option is lazy cloning via copy-on-write, which is infeasible in Python (prior session proved this). |
| **Single-use** (checkpoint consumed after rollback/commit) | Reusable (can rollback to same point multiple times) | Reusable checkpoints require keeping the clone alive, which prevents GC. And the use case (rollback then retry from same point) is rare — you'd typically create a new checkpoint for each retry. |
| **Auto-rollback on exception** | Auto-commit on exception | Auto-rollback is safer — an exception usually means the transformation failed, so restoring the graph prevents corruption. |
| **Does NOT suppress exceptions** | Suppress and return True | Suppressing exceptions hides bugs. The caller should catch and handle exceptions explicitly if desired. |

### Edge Cases

| Case | Behavior |
|------|----------|
| Double rollback | `RuntimeError("already committed or rolled back")` |
| Commit then rollback | `RuntimeError` |
| Model with functions | Functions are NOT cloned — see Limitations below |
| Nested checkpoints (normal LIFO order) | Works correctly. Inner rollback restores to inner state; outer rollback restores to outer state. Each checkpoint unconditionally swaps to its snapshot. |
| Nested checkpoints (out-of-order) | Deterministic but confusing. Each checkpoint restores to its own snapshot regardless of order. No error raised — this is by design, since an identity check would also block legitimate LIFO rollback. |
| Rollback then access `model.graph` | Returns the restored graph. Node/Value references from pre-rollback code are stale. |
| No operations between checkpoint and commit | Works fine — commit just frees the unused clone. |

### Limitations (prominently documented in class docstring)

1. **Functions are NOT checkpointed.** Only `model.graph` is cloned. If a pass modifies model functions (adding/removing/editing `ir.Function` objects), those changes survive rollback, creating an inconsistent model state. This is acceptable because existing passes do not modify functions. A future `ModelCheckpoint` could extend coverage if needed.

2. **Reference invalidation after rollback.** After `rollback()`, `model.graph` is a different object. All Node/Value references from the pre-rollback graph are **stale** — their `.graph` attribute will not match `model.graph`. The context manager pattern naturally avoids issues because rollback exits the `with` block. For explicit rollback, callers must not use pre-rollback references after calling `rollback()`.

3. **Memory cost of nested checkpoints.** Each checkpoint clones the full graph. In a loop with fine-grained checkpointing over a large graph (e.g., 100K nodes, 50 attempts), this creates many full-graph clones. Each clone shares tensor data but duplicates all Node/Value/metadata objects. Users doing many fine-grained checkpoints on very large graphs should consider validation-first approaches instead of try-and-rollback.

### Integration with Tier 1 & Pass Infrastructure

The checkpoint is **orthogonal** to the editing verbs — it doesn't intercept or wrap them. You use editing verbs normally between checkpoint/rollback:

```python
@ir.passes.pass_fn
def fuse_matmul_add(model: ir.Model) -> bool:
    modified = False
    # Restart iteration after each successful fusion or rollback,
    # because both operations invalidate node references.
    changed = True
    while changed:
        changed = False
        graph = model.graph
        for node in list(ir.traversal.topological_order(graph)):
            if node.op_type != "Add":
                continue
            matmul = node.inputs[0].producer() if node.inputs[0] else None
            if matmul is None or matmul.op_type != "MatMul":
                continue

            # Checkpoint before each fusion attempt
            with editing.GraphCheckpoint(model) as cp:
                try:
                    handle = editing.SubgraphHandle(graph, [matmul, node])
                    gemm = ir.node("Gemm",
                        inputs=[matmul.inputs[0], matmul.inputs[1], node.inputs[1]],
                        attributes={"alpha": 1.0, "beta": 1.0})
                    handle.replace_with([gemm], {node.outputs[0]: gemm.outputs[0]})
                    modified = True
                    changed = True
                except ValueError:
                    cp.rollback()
            # After any mutation or rollback, break and re-iterate
            # with fresh node references from model.graph
            if changed:
                break
    return modified
```

### Test Cases

1. **Basic commit**: Checkpoint → edit → commit → verify edit persists
2. **Basic rollback**: Checkpoint → edit → rollback → verify original state
3. **Auto-rollback on exception**: Checkpoint in context manager → raise → verify restore
4. **Auto-commit on clean exit**: Checkpoint in context manager → no error → verify edit persists
5. **Double rollback error**: Verify `RuntimeError` on second rollback
6. **Commit then rollback error**: Verify `RuntimeError`
7. **Nested checkpoints — LIFO rollback**: Inner rollback, then outer rollback, verify both restore correct states
8. **Nested checkpoints — LIFO context managers**: Inner exception auto-rolls-back, outer exception auto-rolls-back to original
9. **Exception after inner rollback**: Inner rollback then outer exception → outer auto-rolls-back correctly
9. **Rollback preserves graph integrity**: After rollback, verify graph.sort() works, all edges consistent
10. **Reference invalidation**: After rollback, verify original nodes' `.graph` is not `model.graph`
11. **No-op checkpoint**: Create checkpoint, commit immediately, verify no side effects
12. **Checkpoint with SubgraphHandle**: Use both APIs together (as in integration example)
13. **Auto-rollback with graph already swapped**: Context manager exception after inner rollback → auto-commits (no crash)

---

## Module-Level Changes

### `editing.py` additions to `__all__`:

```python
__all__ = [
    "eliminate_node",
    "insert_on_edge",
    "replace_node",
    "replace_subgraph",
    # Tier 2
    "GraphCheckpoint",
    "SubgraphHandle",
]
```

### `__init__.py` — no changes needed

The `editing` module is already exported. Users access `editing.SubgraphHandle` and `editing.GraphCheckpoint` through the existing `onnx_ir.editing` namespace.

---

## Implementation Plan

### Commit Order (each commit = one atomic, reviewable unit)

| # | Commit | Files | Dependencies |
|---|--------|-------|-------------|
| 1 | `SubgraphHandle` class with boundary discovery + `as_graph_view()` | `editing.py`, `editing_test.py` | None |
| 2 | `SubgraphHandle.between()` class method | `editing.py`, `editing_test.py` | Commit 1 |
| 3 | `SubgraphHandle.replace_with()` with liveness check | `editing.py`, `editing_test.py` | Commit 1 |
| 4 | `GraphCheckpoint` class with nested rollback detection | `editing.py`, `editing_test.py` | None |
| 5 | Integration tests (both APIs together) | `editing_test.py` | Commits 1-4 |

### Estimated Scope

- **SubgraphHandle**: ~120 lines of implementation + ~300 lines of tests
- **GraphCheckpoint**: ~60 lines of implementation + ~200 lines of tests
- **Total**: ~180 lines production code, ~500 lines tests

### Performance Notes

- `SubgraphHandle.__init__`: O(V_sub × max_degree) for boundary discovery, where V_sub = number of nodes in the subgraph. Typically V_sub << V_graph.
- `SubgraphHandle.between()`: O(V_sub + E_sub) backward traversal (same as existing `_find_subgraph_bounded_by_values`).
- `GraphCheckpoint.__init__`: O(V + E) for `graph.clone()`.
- `GraphCheckpoint.rollback()`: O(1) reference swap.
- `GraphCheckpoint.commit()`: O(1) — just drops the reference.

---

## Open Questions

1. ~~**Should `detach()` auto-rewire external consumers?**~~ **Resolved: `detach()` deferred entirely.** Per Critical Reviewer feedback, `detach()` leaves dangling references, contradicting Tier 1's safety guarantees. No motivating use case exists today. If needed later, it should require `output_mapping` to rewire consumers before removal — essentially becoming `replace_with()` + return-extracted-graph.

2. **Should `GraphCheckpoint` also checkpoint functions?** Current design only clones `model.graph`. Documented prominently in Limitations section. A future `ModelCheckpoint` could extend this.

3. ~~**Should `SubgraphHandle` support topological iteration?**~~ **Resolved: No.** Caller can use `topological_order(handle.as_graph_view())`. Keep the handle thin.
