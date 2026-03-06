# Implementation Review: Tier 2 Graph Editing APIs (Second Pass)

**Reviewer:** Critical Reviewer (e31069b3)
**Commit:** `3816f9b` on `justinchu/improvements`
**Files reviewed:** `src/onnx_ir/editing.py` (lines 537–972), `src/onnx_ir/editing_test.py` (lines 1054–1643)
**Verdict:** **Approve with requested changes** — SubgraphHandle is solid. GraphCheckpoint has a correctness bug in nested rollback that needs fixing before merge.

---

## Summary

SubgraphHandle implementation is clean, correct, and well-tested. All my first-pass review items were addressed properly. GraphCheckpoint has one **real bug** (LIFO nested rollback is broken by the identity check) and one **silent safety violation** in `__exit__` that falls out of the same root cause.

---

## SubgraphHandle: ✅ Clean

### Implementation matches design — verified

1. **`from_nodes()` removed** ✓ — only the constructor and `between()` exist.
2. **`detach()` removed** ✓ — only `replace_with()` remains as a mutation operation.
3. **`as_graph_view()` blocked on consumed handles** ✓ — calls `_check_not_consumed()` (line ~810).
4. **Liveness check in `replace_with()`** ✓ — verifies `node.graph is self._parent` for every node before delegating (lines ~775-785).
5. **Consumed-once semantics** ✓ — `_consumed` flag set after `replace_with()`, checked by `_check_not_consumed()`.

### One gap in liveness check (non-blocking, documented)

The liveness check catches nodes *removed* from the parent graph (e.g., by direct `replace_subgraph()`). But it does **not** catch the "handle created before rollback" case:

```python
handle = SubgraphHandle(graph, [node_a, node_b])
cp = GraphCheckpoint(model)
cp.rollback()
# handle.parent is old graph, nodes still have node.graph is old graph
# Liveness check passes — but replace_with mutates the OLD graph, not model.graph
```

After rollback, `model.graph` is a different object, but the old graph (with its nodes) still exists. `node.graph is self._parent` is true (both point to the old graph). So `replace_with()` would silently mutate a graph that's no longer `model.graph`.

This is **documented** in the test (`test_stale_handle_after_rollback_operates_on_wrong_graph`), which correctly states: "SubgraphHandle doesn't hold a model reference, so it cannot detect that model.graph was swapped by rollback." The test verifies `handle.parent is not model.graph` but doesn't test what happens if `replace_with()` is called.

**Severity: Low.** This is a fundamental limitation of the handle not holding a Model reference. Fixing it would require either (a) passing Model to SubgraphHandle (coupling it to Model, which is wrong — subgraphs can exist in functions too), or (b) some kind of graph-epoch counter. Not worth the complexity now. The documentation is honest about it.

**Suggestion:** Add a test that explicitly calls `replace_with()` on a stale handle after rollback to demonstrate/document this edge case. Even if it "succeeds" (mutating the wrong graph), having the test documents the behavior.

### Boundary discovery: correct

The `_compute_boundaries()` method (lines ~630-670) correctly:
- Iterates in parent-graph order (using `_ordered_nodes`)
- Deduplicates inputs via `seen_inputs` set
- Detects external consumers via `any(user not in node_set for user, _ in out.uses())`
- Handles graph outputs via `parent_outputs` frozenset

### Performance: fine

- Construction: O(V_graph) to compute `_ordered_nodes` (iterates parent graph filtering for subgraph nodes), O(V_sub × max_degree) for boundary discovery.
- `__iter__`: O(V_sub) — iterates pre-computed `_ordered_nodes`. ✓ No O(V_graph) per iteration.
- No unexpected quadratic patterns.

---

## GraphCheckpoint: 🔴 Nested rollback bug

### The bug: LIFO nested rollback is broken

The identity check in `rollback()` (lines ~910-916) was intended to catch **out-of-order** rollback (outer before inner — almost certainly a bug). But it also blocks **normal LIFO** rollback (inner then outer — correct and useful).

**Trace through LIFO nested rollback:**

```
State: model.graph = G0

outer = GraphCheckpoint(model)
  outer._original_graph = G0
  outer._saved_graph = clone(G0)

# Edit G0 in place
edit1(model)   # model.graph is still G0, just mutated

inner = GraphCheckpoint(model)
  inner._original_graph = G0   # same object as outer._original_graph!
  inner._saved_graph = clone(G0-with-edit1)

# Edit more
edit2(model)   # model.graph is still G0, mutated further

# Inner rollback (correct LIFO order)
inner.rollback()
  ✓ Check: model.graph (G0) is inner._original_graph (G0) → OK
  → model.graph = inner._saved_graph (clone of G0-with-edit1)
  → model.graph is now a DIFFERENT object from G0

# Outer rollback (correct LIFO order)
outer.rollback()
  ✗ Check: model.graph (clone) is not outer._original_graph (G0)
  → RAISES RuntimeError "out-of-order nested rollback"!
```

**This is wrong.** Inner-then-outer IS the correct LIFO order. The user wants to undo both levels of changes, and the identity check prevents it. The error message even says "out-of-order" which is factually incorrect for this scenario.

### Consequence: silent safety violation in `__exit__`

The `__exit__` handler (lines ~955-970) works around the identity check by falling back to `commit()` when the graph was swapped:

```python
if exc_type is not None:
    if self._model.graph is self._original_graph:
        self.rollback()
    else:
        self.commit()  # ← silently discards the snapshot
```

This means:

```python
with GraphCheckpoint(model):            # outer
    edit(model)                         # edit 1
    with GraphCheckpoint(model) as inner:
        edit_more(model)
        inner.rollback()                # undo edit 2, swap graph
    raise ValueError("outer also fails")
# outer.__exit__: exception, graph swapped → silently COMMITS
# User expects: auto-rollback to pre-edit1 state
# Actual: edit1 persists (graph is at inner's saved state)
```

The user set up auto-rollback for the outer checkpoint specifically to protect against failures. An exception triggers `__exit__`, but instead of rolling back, it silently commits. **The graph is in a partially-modified state that the user explicitly tried to protect against.**

### Fix recommendation

**Remove the identity check from `rollback()`.** Let both LIFO and out-of-order rollback succeed:

```python
def rollback(self) -> None:
    if not self._active:
        raise RuntimeError(
            "This checkpoint has already been committed or rolled back."
        )
    assert self._saved_graph is not None
    self._model.graph = self._saved_graph
    self._saved_graph = None
    self._original_graph = None
    self._active = False
```

**Rationale:**
- LIFO rollback is a legitimate use case that users will need.
- Out-of-order rollback is confusing but deterministic — each checkpoint restores to its snapshot. Trying to detect it creates false positives (the LIFO case).
- If detection of misuse is desired, an epoch-based mechanism would work: increment a counter on the Model or use a stack. But this adds complexity for a rare misuse case. Start simple.

**Simplify `__exit__` accordingly:**

```python
def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
    if self._active:
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
    return False
```

Now auto-rollback always works, regardless of whether an inner checkpoint swapped the graph.

### The test gap

The test `test_nested_checkpoints_inner_rollback_outer_commit` only tests inner-rollback + outer-COMMIT. There is no test for inner-rollback + outer-ROLLBACK, because the current implementation blocks it. After fixing:

Add a test:
```python
def test_nested_checkpoints_lifo_rollback(self):
    """Inner rollback then outer rollback restores to original state."""
    model, node_a, node_b, node_c = _simple_graph_with_chain()
    original_ops = [n.op_type for n in model.graph]

    outer = editing.GraphCheckpoint(model)
    node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
    editing.replace_node(node_b, node_d)

    inner = editing.GraphCheckpoint(model)
    node_e = ir.Node("", "E", inputs=list(node_a.outputs), num_outputs=1)
    editing.replace_node(node_d, node_e)

    inner.rollback()  # restore to [A, D, C]
    self.assertEqual([n.op_type for n in model.graph], ["A", "D", "C"])

    outer.rollback()  # restore to [A, B, C]
    self.assertEqual([n.op_type for n in model.graph], original_ops)
```

### Design doc claim vs implementation

The design doc (line 526) says: "Nested checkpoints (normal order) | Works correctly. Inner rollback restores to inner state; outer rollback restores to outer state."

This claim is **not true** in the implementation. The implementation blocks outer rollback after inner rollback. The design doc and implementation are inconsistent — either fix the implementation (recommended) or update the design doc to say LIFO nested rollback is not supported.

---

## Other observations

### ✅ `__all__` correctly updated
Both `GraphCheckpoint` and `SubgraphHandle` added. Alphabetical within tiers.

### ✅ Import additions are minimal
Only `Collection`, `Iterator`, `Literal` added — all from stdlib.

### ✅ `replace_with` correctly passes `list(self._ordered_nodes)` to `replace_subgraph`
Uses the ordered list, not the frozenset, which is important because `replace_subgraph` uses `old_nodes[0]` to find the graph.

### ✅ `as_graph_view()` correctly constructs GraphView
Passes `list(self._inputs)` and `list(self._outputs)` (converting tuples to lists for the `Sequence` parameters). Passes `self._ordered_nodes` as `nodes=`.

### ✅ `_original_graph` is cleared on both commit and rollback
Prevents retaining a reference to a potentially large graph after the checkpoint is consumed.

### 📝 Minor: `assert self._saved_graph is not None` in rollback()
Line ~920: This is an internal invariant assertion, not a user-facing error. Fine, but it's the only `assert` in the file — all other checks are `raise ValueError/RuntimeError`. Consistency nit, not blocking.

---

## Test coverage assessment

### SubgraphHandle tests: ✅ Thorough
- Construction: 8 tests (two nodes, between, single node, graph I/O, shared input, internal edges, diamond, empty external consumers)
- Validation: 3 tests (empty nodes, node not in parent, iteration order)
- Mutation: 5 tests (replace, metadata propagation, graph outputs, consumed handle, consumed as_graph_view)
- GraphView: 1 test
- Liveness: 2 tests (stale after remove, stale after rollback)

### GraphCheckpoint tests: ⚠️ Missing LIFO test
- Basic: 4 tests (commit, rollback, auto-rollback on exception, auto-commit on clean exit)
- Errors: 3 tests (double rollback, commit-then-rollback, double commit)
- Nested: 2 tests (inner-rollback + outer-commit, out-of-order raises)
- Integrity: 4 tests (edge consistency, reference invalidation, no-op, graph-already-swapped)
- Integration: 1 test (checkpoint + SubgraphHandle)
- **Missing:** inner-rollback + outer-ROLLBACK (LIFO) — blocked by bug
- **Missing:** context manager auto-rollback after inner rollback + exception (the silent commit case)

---

## Summary of findings

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 1 | 🔴 Bug | Identity check in `rollback()` blocks LIFO nested rollback (inner then outer). Design doc claims it works; implementation blocks it. | `editing.py` ~lines 910-916 |
| 2 | 🔴 Bug | `__exit__` silently commits (instead of rolling back) when graph was swapped by inner rollback + outer gets exception. Silent safety violation. | `editing.py` ~lines 960-966 |
| 3 | ⚠️ Gap | No test for LIFO nested rollback (because it's blocked) | `editing_test.py` |
| 4 | ⚠️ Gap | No test for `__exit__` silent commit scenario (inner rollback + outer exception) | `editing_test.py` |
| 5 | 📝 Low | Liveness check doesn't catch stale handle after rollback (documented, acceptable limitation) | `editing.py` ~line 778 |
| 6 | 📝 Low | `assert` vs `raise` inconsistency in rollback() | `editing.py` ~line 920 |

**Fix items 1-2 by removing the identity check and simplifying `__exit__`.** Then add tests for items 3-4.
