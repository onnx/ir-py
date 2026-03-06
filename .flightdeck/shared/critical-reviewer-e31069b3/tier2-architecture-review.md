# Architectural Review: Tier 2 Graph Editing API

**Reviewer:** Critical Reviewer (e31069b3)
**Design doc:** `.flightdeck/shared/architect-7b89b5f7/tier2-design.md`
**Tier 1 code reviewed:** `src/onnx_ir/editing.py` on branch `justinchu/improvements` (535 lines)
**Verdict:** **Approve with requested changes** — the overall design is sound and well-reasoned, but there are structural issues that need addressing before implementation.

---

## Executive Summary

Both APIs fill genuine gaps in the editing framework. SubgraphHandle provides the "analyze before transform" step that `replace_subgraph()` lacks. GraphCheckpoint provides fine-grained rollback that `FunctionalPass` can't offer. The design makes good choices: consumed-once handles, snapshot-based checkpoints, delegation to existing Tier 1 verbs. I have one blocking concern (the `detach()` API safety hole), several important design suggestions, and some observations.

---

## SubgraphHandle

### ✅ What's good

1. **Clean separation of analysis and mutation.** The handle provides `inputs`/`outputs`/`internal_values` as read-only properties, letting the caller reason about the subgraph before deciding how to transform it. This is the right abstraction — it's the "analyze" counterpart to `replace_subgraph()`'s "transform".

2. **Consumed-once semantics.** Making the handle single-use avoids an entire class of bugs (stale handles after mutation). The design correctly notes that a "live" handle tracking mutations would add coupling with no clear use case.

3. **`replace_with()` delegates to `replace_subgraph()` — zero new mutation logic.** This is exactly right. The handle doesn't duplicate the Tier 1 verbs; it wraps them with better ergonomics. This means all the edge-case handling in `replace_subgraph()` (graph output handling, Identity insertion, metadata propagation) is inherited for free.

4. **`between()` reuses `_find_subgraph_bounded_by_values()`.** Good reuse of proven code. No duplicated traversal algorithms.

5. **Boundary discovery is O(V_sub × max_degree)**, which is correct for the small subgraphs typical in pattern matching. The design correctly notes V_sub << V_graph.

6. **Deterministic ordering** of `inputs` and `outputs` (sorted by position in parent graph) is a thoughtful touch — it makes the API predictable and testable.

### 🔴 Blocking: `detach()` leaves dangling consumers

**Lines 203-222** of the design:

> The original nodes are removed from the parent graph. External consumers of the subgraph's outputs will need to be rewired by the caller.

This is a **safety hazard** that contradicts the design's own philosophy. The Tier 1 APIs (`replace_node`, `replace_subgraph`) are carefully designed so that they *never* leave dangling references — they always rewire consumers before removing nodes. `detach()` breaks this contract by removing nodes while leaving their output values referenced by consumer nodes still in the graph.

After `detach()`:
- Consumer nodes in the parent graph still reference `Value` objects whose producer nodes have been removed from the graph
- Those `Value.producer().graph` will be `None`
- Any subsequent editing operations on those consumers may fail in confusing ways
- The graph is in an **inconsistent state** until the caller manually rewires

**Recommendation:** Either:
- **(Preferred)** Remove `detach()` from the initial API entirely. It's a speculative feature — the design doc itself says "start with the simple version, add auto-rewire if users need it." Follow your own principle: solve the problem that exists NOW. Nobody has asked for `detach()`.
- **(Alternative)** If `detach()` is kept, require the caller to provide an `output_mapping` (just like `replace_subgraph()`) so that consumers are rewired before the nodes are removed. But this essentially makes `detach()` = `replace_with()` + return a graph, which suggests it should just be a parameter on `replace_with()`.

### ⚠️ Important: `from_nodes` constructor redundancy

**Lines 100-109:** `from_nodes()` is documented as "Equivalent to `SubgraphHandle(parent, nodes)`. Provided for symmetry with `between()`."

This is over-engineering. Two ways to do the exact same thing adds cognitive load for zero benefit. The "symmetry" argument is weak — `SubgraphHandle(graph, nodes)` is already perfectly clear. If `between()` exists as a classmethod because it has different construction logic, that's fine. But `from_nodes()` should be dropped — just use the constructor directly.

### ⚠️ Important: `as_graph_view()` ownership ambiguity

**Lines 224-229:** The returned `GraphView` references the same `Node` objects (not copies). This means:

1. Mutations through the `GraphView` (if someone passes it to code that writes to nodes) affect the parent graph
2. After `replace_with()` consumes the handle, the `GraphView` is stale but there's no way to know

This needs clearer documentation. Also, should `as_graph_view()` be blocked after the handle is consumed? The design says "Does NOT consume the handle" but doesn't address what happens if the handle is consumed *after* getting a GraphView.

**Recommendation:** After the handle is consumed, mark any previously-returned GraphViews as stale by clearing internal references. Or more simply: document that the GraphView's lifetime is bounded by the handle's, and don't try to enforce it. But definitely block `as_graph_view()` on consumed handles.

### 📝 Observation: `__iter__` over parent-graph order requires walking the full graph

If `__iter__` yields nodes in parent-graph order, it needs to iterate the parent graph and filter for nodes in the handle's `frozenset`. This is O(V_graph) per iteration, not O(V_sub). Fine for typical use, but worth noting. The design's `_ordered_nodes` (used in boundary discovery, line 246) suggests this is pre-computed at construction — good.

### 📝 Observation: Thread safety not discussed

Not blocking (the IR doesn't appear to have threading concerns), but if multiple threads create SubgraphHandles over overlapping node sets, consumed-once semantics won't protect against concurrent mutation. This is fine to leave undocumented for now — just noting it.

---

## GraphCheckpoint

### ✅ What's good

1. **Honest about costs.** The design clearly states O(V+E) upfront, O(1) rollback, and explains why alternatives (copy-on-write, command-log undo) are infeasible in Python. This is mature engineering.

2. **Context manager with auto-rollback.** This is the right default. Exceptions mean something went wrong → restore the graph. Never suppressing exceptions (returning `False`) is correct.

3. **Model-level, not Graph-level.** `model.graph = saved_graph` is O(1) and clean. The alternative (clear + repopulate the Graph object) would be fragile and still invalidate references. Good tradeoff analysis.

4. **Single-use checkpoints.** The design correctly reasons that reusable checkpoints keep clones alive and prevent GC, with a rare use case. YAGNI applied well.

5. **`model.graph` is a plain `__slots__` attribute** (verified in `_core.py` line 3991: `self.graph: Graph = graph`), so `model.graph = clone` is a simple attribute assignment. No property setter magic to worry about.

### 🔴 Blocking: Loop + checkpoint interaction is unsafe as documented

**Lines 532-556**, the integration example:

```python
for node in list(ir.traversal.topological_order(graph)):
    ...
    with editing.GraphCheckpoint(model):
        try:
            handle = editing.SubgraphHandle.from_nodes(graph, [matmul, node])
            ...
        except ValueError:
            pass
        graph = model.graph  # re-fetch after potential rollback
```

This has a **critical bug**: the `for` loop iterates over `list(ir.traversal.topological_order(graph))` — a snapshot of the *original* graph's nodes. After rollback, `model.graph` is a *different graph* with *different node objects*. The loop continues iterating stale nodes that no longer belong to `model.graph`. Any attempt to call `node.inputs`, `node.graph`, etc. on these stale nodes will reference the old (discarded) graph.

The `graph = model.graph` re-fetch at line 554 only helps for the *next* checkpoint creation, not for the loop variable `node` which is already stale.

**This is the exact reference invalidation problem the design warns about** (lines 376-378), but the integration example doesn't solve it correctly.

**Recommendation:** The integration example must be fixed to re-start iteration after rollback:

```python
def fuse_matmul_add(model):
    modified = False
    changed = True
    while changed:
        changed = False
        for node in list(ir.traversal.topological_order(model.graph)):
            ...
            with editing.GraphCheckpoint(model):
                try:
                    ...
                    changed = True
                    modified = True
                except ValueError:
                    pass
            if changed:
                break  # re-start iteration with fresh graph
    return modified
```

Or, more importantly, the **documentation must clearly warn** that iteration loops over nodes MUST be restarted after rollback, because all node references from the pre-rollback graph are stale. This isn't just a documentation nit — it's a fundamental API hazard that will bite every user.

### ⚠️ Important: Functions are NOT checkpointed

**Line 521:** "Functions are NOT cloned — only `model.graph`. Functions are typically not modified by passes."

This is fine for now, but it should be **loudly documented** (not just in a table row). If a pass modifies a function and then the checkpoint rolls back `model.graph`, the function modifications persist — creating an inconsistent model state. The class docstring should include a "Limitations" section.

### ⚠️ Important: Nested checkpoint ordering is unchecked

**Line 522:** "Nested checkpoints work correctly. Each checkpoint independently clones `model.graph` at creation time."

This is true in the happy path, but consider:

```python
outer = GraphCheckpoint(model)
# modify graph...
inner = GraphCheckpoint(model)
# modify graph more...
outer.rollback()  # restores to outer state
inner.rollback()  # restores to inner state — WRONG, goes forward in time!
```

Rolling back the outer checkpoint first, then the inner, restores to the *inner* checkpoint's state (which is chronologically later than the outer's). This isn't "incorrect" behavior — each checkpoint restores to its snapshot — but it's deeply confusing and probably indicates a bug in the caller's logic.

**Recommendation:** When `rollback()` is called, check if `model.graph` is still the same object as when the checkpoint was created. If not (another checkpoint already swapped it), log a warning or raise. This catches out-of-order rollback, which is almost certainly a bug.

### 📝 Observation: Memory pressure with nested checkpoints

Each nested checkpoint clones the full graph. In a loop with fine-grained checkpointing over a large graph (e.g., 100K nodes, trying 50 fusion patterns), this creates 50 full-graph clones. Each clone shares tensors but duplicates all node/value objects. For large graphs, this could be significant memory pressure.

Not blocking — the design is honest about the O(V+E) cost — but worth noting in documentation. Users doing many fine-grained checkpoints on large graphs should consider alternative approaches (e.g., validation instead of rollback).

### 📝 Observation: `commit()` in context manager on clean exit is implicit

The context manager auto-commits on clean exit (line 500-501). This means:

```python
with GraphCheckpoint(model):
    editing.replace_node(old, new)
    if not validate(model):
        cp.rollback()  # ← but cp isn't bound! Missing `as cp`
```

If the user forgets `as cp`, they can't call `rollback()` explicitly inside the block. The auto-commit on clean exit then persists the invalid state. The documentation shows the correct `as cp` pattern, but the API allows the footgun.

Not blocking — this is a Python context manager convention issue, not a design flaw.

---

## Cross-API Composition

### ✅ SubgraphHandle + GraphCheckpoint compose well

The two APIs are orthogonal: SubgraphHandle provides structured analysis/mutation, GraphCheckpoint provides rollback. Neither depends on the other. They can be used independently or together. This is good separation of concerns.

### ⚠️ But: SubgraphHandle created before rollback is stale

If you create a `SubgraphHandle`, then a checkpoint rolls back the graph, the handle's nodes are now in the old (discarded) graph. The handle's `replace_with()` would operate on a graph that is no longer `model.graph`. The consumed-once check won't catch this because the handle isn't consumed — it's just stale.

**Recommendation:** `SubgraphHandle.replace_with()` should verify that its nodes are still in `self.parent` and that `self.parent` is still the active graph. At minimum, check `node.graph is self.parent` for any node before delegating to `replace_subgraph()`.

---

## Module Placement

### ✅ `editing.py` is the right location

Both APIs are companions to the editing verbs. SubgraphHandle wraps `replace_subgraph()`; GraphCheckpoint wraps `graph.clone()` for editing safety. Putting them in `editing.py` keeps the module cohesive around "graph mutation" operations.

### 📝 File size concern

`editing.py` is currently 535 lines. Adding ~200 lines of production code brings it to ~735 lines. This is still manageable, but it's approaching the point where splitting into submodules (`editing/_subgraph.py`, `editing/_checkpoint.py`) might be warranted. Not blocking for this iteration.

---

## Summary of Requested Changes

| # | Severity | Issue | Section |
|---|----------|-------|---------|
| 1 | 🔴 Blocking | `detach()` leaves dangling consumer references — remove from initial API or require output_mapping | SubgraphHandle |
| 2 | 🔴 Blocking | Integration example has stale-reference bug after rollback in loop | GraphCheckpoint |
| 3 | ⚠️ Important | Remove redundant `from_nodes()` classmethod | SubgraphHandle |
| 4 | ⚠️ Important | Block `as_graph_view()` on consumed handles | SubgraphHandle |
| 5 | ⚠️ Important | Functions-not-checkpointed must be documented prominently | GraphCheckpoint |
| 6 | ⚠️ Important | Detect out-of-order nested rollback (check graph identity) | GraphCheckpoint |
| 7 | ⚠️ Important | SubgraphHandle should verify node liveness before mutation | Cross-API |
| 8 | 📝 Note | `__iter__` is O(V_graph), document or pre-compute | SubgraphHandle |
| 9 | 📝 Note | Memory pressure from nested checkpoints on large graphs | GraphCheckpoint |

---

## Final Assessment

The design demonstrates strong architectural thinking: consumed-once semantics, delegation to existing primitives, honest cost analysis, and clear tradeoff documentation. The decision to use snapshot-based checkpointing instead of command-log transactions is well-reasoned and correctly identifies the infeasibility of the alternative in Python's reference-based object model.

The two blocking issues are both about **reference safety after graph replacement** — the fundamental hazard of snapshot-based approaches. The design acknowledges this hazard in documentation but doesn't fully address it in the API surface or the integration examples. Fixing these will make the API much safer for users.

Scope is appropriate — two focused APIs that fill real gaps, no speculative features (except `detach()`, which should be deferred).
