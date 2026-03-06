# Tier 2 Implementation Review — SubgraphHandle & GraphCheckpoint

**Reviewer:** Code Reviewer (b14604cc)
**Commit:** `3816f9b` — "Add SubgraphHandle and GraphCheckpoint (Tier 2 graph editing APIs)"
**Files:** `src/onnx_ir/editing.py` (+436 LOC), `src/onnx_ir/editing_test.py` (+614 LOC)
**Tests:** 36 new tests, all passing (90 total)

---

## Overall Assessment: **Approve with minor issues**

This is a high-quality implementation that faithfully translates the design doc into code. Both classes are clean, well-documented, and integrate correctly with the Tier 1 APIs. The test suite is thorough. I have one notable concern about nested rollback semantics and a handful of suggestions.

---

## What's Done Well

1. **SubgraphHandle boundary discovery** (`_compute_boundaries`) — Clean O(V+E) algorithm. The `seen_inputs` dedup set for inputs and the `parent_outputs` frozenset for output detection are both correct and efficient. The traversal order (parent-graph order) ensures deterministic boundary ordering.

2. **Delegation to Tier 1** — `replace_with()` delegates entirely to `replace_subgraph()` with zero new mutation logic. `between()` reuses `_find_subgraph_bounded_by_values()`. This is excellent — no duplicated mutation code means no divergence risk.

3. **Consumed-once semantics** — The `_consumed` flag with `_check_not_consumed()` is simple and correct. The liveness check in `replace_with()` (verifying `node.graph is self._parent`) is a nice defense-in-depth against stale handles.

4. **GraphCheckpoint `__exit__` resilience** — The `__exit__` method's handling of the "graph already swapped" case (line 966) is thoughtful. On exception after inner rollback, it degrades to `commit()` (discard stale snapshot) rather than crashing. This prevents double-faults.

5. **Test quality** — Tests are well-organized into focused classes, each with descriptive names and docstrings. The diamond pattern test, shared input test, and the integration test (`GraphCheckpointWithSubgraphHandleTest`) are particularly good. The `test_stale_handle_after_rollback_operates_on_wrong_graph` test honestly documents a limitation rather than hiding it.

---

## Issues

### Issue 1: Nested rollback is overly restrictive (Notable — not blocking)

**The problem:** The identity check in `rollback()` (line 912: `self._model.graph is not self._original_graph`) blocks BOTH out-of-order rollback AND normal stack-ordered rollback. After inner rollback swaps `model.graph`, the outer checkpoint can no longer rollback because its `_original_graph` no longer matches.

**Trace through:**
```
1. outer = GraphCheckpoint(model)     # outer._original_graph = G0
2. <edit G0 in-place>
3. inner = GraphCheckpoint(model)     # inner._original_graph = G0
4. <edit G0 more>
5. inner.rollback()                   # model.graph = G0_clone; G0 is gone
6. outer.rollback()                   # FAILS: model.graph is G0_clone, not G0
```

**Impact:** The context manager handles this gracefully (`__exit__` falls back to `commit()`), so the common `with` pattern works. But if a user explicitly creates nested checkpoints and tries to rollback both, they get a confusing error message saying "out-of-order nested rollback" when they're actually doing normal LIFO ordering.

**My assessment:** This is **acceptable for v1** but the error message is misleading. The message should distinguish between the two cases, or at minimum say "another checkpoint's rollback has changed model.graph" without calling it "out-of-order."

**Suggested fix:** Update the error message in `rollback()`:
```python
raise RuntimeError(
    "model.graph has been replaced since this checkpoint was created "
    "(likely by another checkpoint's rollback). After a nested checkpoint "
    "rolls back, outer checkpoints cannot rollback because the graph "
    "reference has changed. Use a single checkpoint or create a new "
    "checkpoint after the inner rollback."
)
```

The test `test_nested_checkpoints_inner_rollback_outer_commit` correctly documents the behavior, and the test name accurately describes what happens (outer commits, not rolls back). Good.

### Issue 2: Design doc test case #16 deviation — stale handle after rollback

**Design doc says:** "Stale handle after rollback: Create handle, rollback checkpoint, verify RuntimeError on `replace_with()`"

**Implementation:** `test_stale_handle_after_rollback_operates_on_wrong_graph` does NOT test `RuntimeError` on `replace_with()`. It only verifies `handle.parent is not model.graph`. The docstring honestly explains why: SubgraphHandle doesn't hold a model reference, so it can't detect the rollback. The liveness check passes because the handle's nodes still belong to the handle's parent (the old graph).

**My assessment:** The deviation is correct and the test is better than what the design doc specified. Trying to force a RuntimeError here would require adding a model reference to SubgraphHandle, which is unnecessary coupling. The test's docstring clearly warns callers. **No action needed** — but worth noting for the architect.

### Issue 3: `assert` in production code (Minor)

Line 918: `assert self._saved_graph is not None` — this is inside `rollback()`, which is user-facing. If Python is run with `-O` (optimize), asserts are stripped. This should be a proper check:

```python
if self._saved_graph is None:
    raise RuntimeError("Internal error: saved graph is None despite active checkpoint.")
```

In practice this is unreachable (the `_active` check above guarantees `_saved_graph` is not None), but using `assert` for invariants in user-facing methods is a style concern — it could mask bugs in future refactors.

---

## Suggestions (Non-blocking)

### Suggestion 1: Missing test for `replace_with(propagate_metadata=False)`

`test_replace_with_propagates_metadata` tests the default (True) case. There's no test for the False case. Since this parameter is explicitly in the API, a test confirming metadata is NOT propagated when disabled would increase confidence.

### Suggestion 2: Missing test for `between()` with invalid bounds

The design doc says `between()` raises `ValueError` for improperly bounded subgraphs. There's no test for this error case. A test passing disconnected input/output values would verify the error propagation from `_find_subgraph_bounded_by_values`.

### Suggestion 3: `_ordered_nodes` computation is O(N) where N = all graph nodes

Line 614-616:
```python
self._ordered_nodes: tuple[_core.Node, ...] = tuple(
    n for n in parent if n in self._nodes
)
```

This iterates over ALL nodes in the parent graph, which is O(N) not O(V) where V = len(nodes). For large graphs with small subgraph handles, this could be expensive. Not a problem now, but worth a comment noting the cost.

### Suggestion 4: Consider adding `__repr__` to both classes

Neither `SubgraphHandle` nor `GraphCheckpoint` has a `__repr__`. A simple repr showing node count (for handle) or active status (for checkpoint) would help debugging:

```python
def __repr__(self) -> str:
    status = "consumed" if self._consumed else f"{len(self._nodes)} nodes"
    return f"SubgraphHandle({status})"
```

### Suggestion 5: Loop-based checkpoint docstring example has stale-reference risk

The "Usage for fine-grained rollback in a loop" example in `GraphCheckpoint`'s docstring (line 852):
```python
for matmul, add in find_fusible_pairs(model.graph):
    with editing.GraphCheckpoint(model):
        ...
```

After an auto-rollback (exception caught), the loop continues with the **next** `(matmul, add)` pair from the iterator. But these references were computed from the pre-rollback graph. After rollback, `model.graph` is a different object — the loop's `matmul` and `add` are stale references from the old graph.

The architect's design doc noted this (Critical Reviewer flagged it), but the implementation's docstring doesn't warn about it. Add a note: "**Warning:** After rollback, `model.graph` is a different object. Any iteration over nodes must restart from `model.graph`."

---

## Test Coverage Assessment

### SubgraphHandle (20 tests) — ✅ Thorough

| Design Doc Test Case | Implemented? | Test Name |
|---|---|---|
| 1. Construction from nodes | ✅ | `test_construction_from_two_nodes` |
| 2. Construction via `between()` | ✅ | `test_construction_via_between` |
| 3. Single-node handle | ✅ | `test_single_node_handle` |
| 4. Graph inputs/outputs | ✅ | `test_graph_inputs_and_outputs` |
| 5. Shared input | ✅ | `test_shared_input_appears_once` |
| 6. Internal edges | ✅ | `test_internal_edges_not_in_inputs_or_outputs` |
| 7. `replace_with()` | ✅ | `test_replace_with_fuses_two_nodes` |
| 8. `replace_with()` metadata | ✅ | `test_replace_with_propagates_metadata` |
| 9. `replace_with()` graph outputs | ✅ | `test_replace_with_handles_graph_outputs` |
| 10. `as_graph_view()` | ✅ | `test_as_graph_view_returns_correct_view` |
| 11. Consumed — `replace_with()` | ✅ | `test_consumed_handle_replace_with_raises` |
| 12. Consumed — `as_graph_view()` | ✅ | `test_consumed_handle_as_graph_view_raises` |
| 13. Diamond pattern | ✅ | `test_diamond_pattern` |
| 14. Empty external consumers | ✅ | `test_empty_external_consumers` |
| 15. Iteration order | ✅ | `test_iteration_order` |
| 16. Stale handle after rollback | ⚠️ | `test_stale_handle_after_rollback_operates_on_wrong_graph` (documents limitation, no RuntimeError) |
| — Empty nodes (extra) | ✅ | `test_empty_nodes_raises` |
| — Node not in parent (extra) | ✅ | `test_node_not_in_parent_raises` |
| — len (extra) | ✅ | `test_len` |
| — contains (extra) | ✅ | `test_contains` |

### GraphCheckpoint (14 tests) — ✅ Thorough

All core behaviors tested: commit, rollback, auto-rollback on exception, auto-commit on clean exit, double-commit/rollback errors, nested checkpoints (both directions), graph integrity after rollback, reference invalidation, noop checkpoint, graph-already-swapped edge case, integration with SubgraphHandle.

### Missing tests (non-blocking):
- `replace_with(propagate_metadata=False)` — verify metadata is NOT transferred
- `between()` with invalid/disconnected boundaries — verify ValueError
- `rollback_then_commit_error` — verify RuntimeError when committing after rollback (the reverse of `test_commit_then_rollback_error`)

---

## Conformance to Design Doc

The implementation closely follows the design doc with two notable deviations:

1. **Test #16 (stale handle after rollback):** Design expected RuntimeError; implementation documents the limitation instead. This is the right call — see Issue #2 above.

2. **`detach()` removed:** Correctly omitted per the revised design doc. ✅

3. **`from_nodes()` removed:** The design says "Only constructor + `between()` class method." The implementation uses the constructor directly (not a classmethod), which matches. ✅

4. **Docstring example updated:** Uses `SubgraphHandle(graph, [nodes])` instead of `SubgraphHandle.from_nodes(graph, [nodes])`, matching the revised design. ✅

---

## Summary

Strong implementation. The code is clean, well-tested, and integrates seamlessly with Tier 1. The main concern (nested rollback semantics) is a known limitation that's well-handled by the context manager but needs a better error message. The handful of missing tests are edge cases, not gaps in core functionality.

**Verdict: Approve with the error message improvement (Issue 1) and the docstring warning (Suggestion 5) as recommended follow-ups.**
