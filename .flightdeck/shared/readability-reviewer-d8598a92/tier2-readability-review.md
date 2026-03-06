# Readability Review: Tier 2 Graph Editing APIs

**Reviewer:** Readability Reviewer (d8598a92)
**Branch:** `justinchu/improvements`
**Commit:** `3816f9b` — Add SubgraphHandle and GraphCheckpoint
**Files:** `src/onnx_ir/editing.py`, `src/onnx_ir/editing_test.py`

---

## Overall Assessment: ✅ Very Good

This is well-written, readable code. The naming is clear, the docstrings are thorough with good examples, and the structure follows existing patterns. A new developer could pick up these APIs quickly. I have a few specific improvements below, but nothing blocking.

---

## Findings

### 1. 🟡 Module-Level Docstring is Stale (Important)

**File:** `editing.py`, lines 3–37

The module docstring only mentions the four Tier 1 functions:

```
Key operations:
- :func:`replace_node`: Replace a node with another, rewiring all consumers.
- :func:`eliminate_node`: Remove a passthrough node (Identity, Cast, etc.)
- :func:`insert_on_edge`: Insert a node between a value and all its consumers.
- :func:`replace_subgraph`: Replace multiple connected nodes with new nodes.
```

It does not mention `SubgraphHandle` or `GraphCheckpoint`. This is the first thing a developer reads when they open this module. A new user scanning the module docstring would not know these APIs exist.

**Suggested fix:** Add a second section below "Key operations":

```
Tier 2 (analysis & transactions):

- :class:`SubgraphHandle`: Immutable, boundary-annotated handle to a subgraph.
  Analyze inputs/outputs/internal values, then replace via :meth:`~SubgraphHandle.replace_with`.
- :class:`GraphCheckpoint`: Snapshot-based transaction with rollback support.
  Clone-and-swap pattern for safe in-place transformations.
```

Also update the module example to show a quick SubgraphHandle or GraphCheckpoint usage.

### 2. ✅ Naming Clarity — Excellent

Every name reveals intent without reading the implementation:

- **`SubgraphHandle`** — immediately communicates "a handle to a subgraph." The "handle" metaphor correctly implies it's a reference with operations, not a copy.
- **`GraphCheckpoint`** — clear transaction semantics implied by the checkpoint metaphor.
- **`replace_with`** — clear verb, reads naturally: `handle.replace_with(new_nodes, ...)`.
- **`between`** class method — good name for the boundary-based constructor.
- **`_compute_boundaries`** — clear internal helper name.
- **`_check_not_consumed`** — exact about what it checks.
- **`is_active`** — standard for checkpoint/transaction state.
- **`_original_graph` vs `_saved_graph`** — good distinction. `_original_graph` is the identity reference for safety checks, `_saved_graph` is the clone for restoration.
- **`internal_values`** — clearly distinguishes from `inputs` and `outputs`.

One minor observation: the `inputs`/`outputs` property names on SubgraphHandle could potentially be confused with graph-level inputs/outputs. However, the terminology section in the class docstring addresses this upfront, and the usage is natural in context (e.g., `handle.inputs` clearly means "this subgraph's inputs"). No change needed.

### 3. ✅ Docstrings — Thorough and Well-Structured

**SubgraphHandle class docstring** is excellent:
- Terminology section defining inputs/outputs/internal values — prevents confusion
- Cross-reference to `GraphView` explaining how SubgraphHandle differs
- Complete usage example with `ir.node()` call showing real API
- Clear statement about consumed-once semantics

**GraphCheckpoint class docstring** is outstanding:
- Cost analysis upfront (O(V+E), O(1), O(1)) — sets expectations
- Reference invalidation warning prominently placed
- Three usage examples (context manager, loop, explicit) — covers all patterns
- "Functions are NOT checkpointed" caveat prominently placed

**Method docstrings** consistently include Args, Raises, and behavioral notes. The `replace_with` docstring explaining the liveness check ("guards against stale handles after checkpoint rollback") is especially helpful.

### 4. 🟡 `__exit__` Edge Case Deserves a Docstring Note

**File:** `editing.py`, `GraphCheckpoint.__exit__`

The `__exit__` method has a subtle fourth case not mentioned in its docstring: when there's an exception AND the graph has already been swapped (by an inner rollback), it *commits* instead of rolling back. This is the correct behavior (can't rollback a stale snapshot), and the inline comment explains it well:

```python
# On exception, only rollback if the graph hasn't been swapped
# by another checkpoint. If it has, we can't safely rollback —
# just commit (discard the now-stale snapshot).
```

But the docstring says:
```
- On exception (no prior rollback): calls ``rollback()``.
```

The `(no prior rollback)` parenthetical is easy to miss. Consider making this a separate bullet:

```
- On exception with graph unchanged: calls ``rollback()``.
- On exception after graph swap (e.g., inner rollback): calls ``commit()``
  to discard the stale snapshot instead of crashing.
```

### 5. ✅ Code Organization — Good Fit with Tier 1

- The `# ============================================================================` section banners clearly delineate Tier 2 from Tier 1.
- Placing both classes in `editing.py` is correct — they are editing concerns, not core IR types.
- SubgraphHandle before GraphCheckpoint is the right order (simpler before more complex, and SubgraphHandle can be used with GraphCheckpoint).
- The `__all__` list groups Tier 1 and Tier 2 with a comment — good for discoverability.
- Internal helpers (`_compute_boundaries`, `_check_not_consumed`) are prefixed with `_` — consistent with Tier 1's `_propagate_value_metadata` and `_handle_graph_output_replacement`.

### 6. ✅ Consistency with Existing Patterns

- Uses `_core.Graph`, `_core.Node`, `_core.Value` — same as Tier 1 functions.
- Error messages follow the same pattern: describe what's wrong, then state the requirement (e.g., "Node X does not belong to parent graph Y. All nodes must belong to the parent graph.").
- `replace_with` delegates to `replace_subgraph()` — no new mutation logic, consistent with the principle that Tier 2 builds on Tier 1.
- The deferred import in `between()` (`from onnx_ir._convenience._extractor import ...`) follows the same pattern used elsewhere in the codebase.

### 7. ✅ Test Organization — Well-Structured

Tests follow the existing pattern of `_helper_function()` + `class XxxTest(unittest.TestCase)`:

- `_diamond_graph()` — matches the existing `_simple_graph_with_chain()` pattern with docstring showing the topology.
- Test classes are logically grouped: Construction, Mutation, GraphView, Liveness, Basic, Error, Nested, Integrity, Integration.
- Each test has a descriptive docstring explaining what's being verified.
- The integration test (`GraphCheckpointWithSubgraphHandleTest`) tests the two APIs together — good.
- Access-via-module tests at the end — consistent with existing `EliminateNodeAccessViaModuleTest`.

### 8. 🟢 Minor Nit: `assert` in `rollback()`

**File:** `editing.py`, `GraphCheckpoint.rollback()`

```python
assert self._saved_graph is not None
```

This `assert` is correct — the `if not self._active` guard above guarantees `_saved_graph` is not None. It's a defensive internal check, which is fine. However, `assert` can be disabled with `python -O`. If this is a safety-critical invariant, consider a conditional `raise` instead. Not blocking — the guard above makes this effectively unreachable.

### 9. 🟢 Minor Nit: Ordered Nodes Computation is O(V_parent)

```python
self._ordered_nodes = tuple(n for n in parent if n in self._nodes)
```

This iterates over **all** nodes in the parent graph to extract the subgraph nodes in order. For a 100K-node graph with a 3-node subgraph, this scans 100K nodes. The frozenset lookup makes each check O(1), so it's O(V_parent) overall.

This is acceptable for the initial implementation (boundary computation also needs ordered iteration), but worth noting for future optimization if profiling reveals it as a hotspot on large graphs. No action needed now.

### 10. ✅ API Discoverability — Good

A new user can discover these APIs through:
1. `__all__` export list → shows up in `dir(editing)` and IDE autocomplete
2. Module docstring (once updated per Finding #1)
3. The class docstrings with worked examples
4. Cross-references (`:func:replace_subgraph`, `:class:GraphView`) help users navigate
5. The `between()` class method has an inline example showing the most common use case

The "terminology" section in SubgraphHandle's docstring is particularly valuable for discoverability — it defines the domain vocabulary upfront so users can speak the same language.

---

## Summary

| Category | Rating | Notes |
|----------|--------|-------|
| Naming | ✅ Excellent | All names reveal intent |
| Docstrings | ✅ Excellent | Complete with examples, edge cases, and warnings |
| Organization | ✅ Good | Logical grouping, clear section boundaries |
| Consistency | ✅ Good | Follows Tier 1 patterns |
| Simplicity | ✅ Good | No over-engineering, delegates to Tier 1 |
| Discoverability | 🟡 Needs module docstring update | Otherwise good |

**Action items:**
1. **Update module-level docstring** to mention SubgraphHandle and GraphCheckpoint (Finding #1)
2. **Expand `__exit__` docstring** to cover the graph-already-swapped case (Finding #4)

No blocking readability issues. Good work.
