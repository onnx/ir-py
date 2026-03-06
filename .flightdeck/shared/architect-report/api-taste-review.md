# API Taste Review — Tier 1 Implementations

**Reviewer:** Architect (fae585af)
**Date:** 2026-03-05
**Standard:** The user demands ELEGANT, TASTEFUL APIs — the face of the library for pass developers. Judged against pathlib/itertools/dataclasses quality bar and consistency with existing `_core.py` conventions.

---

## 1. `topological_order` — `src/onnx_ir/traversal.py:123-235`

### GOOD
- **Signature is clean**: `topological_order(graph_like, *, recursive=False)` — positional graph, keyword-only option. Matches the library's `GraphLike` type alias convention.
- **Naming is excellent**: `topological_order` reads naturally as a noun phrase (what you get) rather than an imperative (what to do). Consistent with Python stdlib naming: `itertools.permutations`, not `itertools.permute`.
- **Lazy generator**: Yields nodes one at a time via `yield`, not returning a list. This is the right design for composition (`islice`, `takewhile`, etc.).
- **Stable ordering**: Min-heap keyed by original position. This is a thoughtful detail — deterministic output matters for reproducible passes.
- **Cycle detection**: Raises `ValueError` with clear message. Correct exception type for invalid input data.
- **`versionadded` annotation**: Good practice.

### IMPROVE

1. **Docstring inaccuracy (MEDIUM)**: Lines 134-136 claim "this avoids allocating the full sorted node list" but line 162 does `list(RecursiveGraphIterator(graph_like))`. The sorted list isn't materialized (nodes yield from the heap), but the *input* node list IS fully materialized. This is architecturally necessary for Kahn's algorithm (need complete in-degree map before yielding), but the docstring misleads. Fix: change to "nodes are yielded incrementally from the heap without materializing the full sorted result, though the initial dependency graph requires a full pass over all nodes."

2. **`recursive` parameter overloading (LOW)**: In `RecursiveGraphIterator`, `recursive` is a `Callable[[Node], bool]` (a filter predicate). In `topological_order`, it's a `bool` flag. Same name, different semantics. This is mildly confusing but acceptable — the bool is the right choice here for simplicity, and the two APIs serve different audiences. **No change needed**, but worth a note in the docstring.

3. **Missing example for `recursive=True` (LOW)**: The docstring example only shows the default case. A second example showing `recursive=True` with subgraph-bearing nodes (If/Loop) would teach users when to use it.

### CRITICAL

4. **No test coverage**: Zero tests for `topological_order`. This is a foundational API — passes will depend on its correctness. Required tests:
   - Linear chain ordering
   - Diamond dependency (A→B, A→C, B→D, C→D)
   - Disconnected components
   - Cycle detection (ValueError)
   - Empty graph (yields nothing)
   - `recursive=True` vs `False` with If/Loop subgraphs
   - Stable ordering (same-level nodes in original order)

**Verdict: GOOD design, needs docstring fix and tests.**

---

## 2. `pass_fn` — `src/onnx_ir/passes/_pass_infra.py:355-398`

### GOOD
- **Naming is perfect**: `pass_fn` — short, memorable, says exactly what it does. Parallel to `dataclasses.dataclass` as a decorator that wraps a plain function into a richer object.
- **Decorator UX is Pythonic**: `@ir.passes.pass_fn` reads naturally. No parentheses needed (not a decorator factory), which is the right call — there are no configuration options.
- **Return type is `InPlacePass`**: Smart. This means the result works with `Sequential`, `PassManager`, and `functionalize()` without any special casing. Composition for free.
- **Function contract is simple**: `(ir.Model) -> bool`. Can't get simpler. The bool return meaning "was the model modified" is the exact convention `PassResult.modified` uses.
- **Name/qualname/doc/module transfer**: Lines 393-396 preserve the function's identity so `repr()`, `help()`, and `__module__` all work correctly. This is the kind of attention to detail that makes APIs feel polished.
- **Repr is informative**: `_FnPass(my_pass_name)` — tells you what's inside.
- **Excellent test coverage**: 10 tests covering type, result, modified flag, name preservation, docstring preservation, Sequential integration, PassManager integration, PassResult acceptance, functionalization, and repr. Thorough.

### IMPROVE

1. **Docstring example could show pipeline composition (LOW)**: The example shows standalone use and Sequential, which is good. Could also show `functionalize(my_pass)` in the example to demonstrate the composition story.

2. **Inner class naming (NITPICK)**: `_FnPass` is fine internally, but when a pass errors out, the traceback will show `_FnPass`. The `__name__` override (line 393) mitigates this for repr, but Python exception tracebacks use `__class__.__name__` which won't be overridden. Consider `_FnPass.__name__ = fn.__name__` → this already exists. Actually, this is fine — the `__qualname__` is set too. Exception messages will show the function name. **No change needed.**

### CRITICAL

None. This is the most polished of the three APIs.

**Verdict: EXCELLENT. Ship as-is.**

---

## 3. `node.clone()` — `src/onnx_ir/_core.py:2383-2481`

### GOOD
- **Method name is perfect**: `.clone()` is the standard Python/Java convention for deep-ish copy. It's what users expect. Not `.copy()` (which implies shallow in Python), not `.deep_copy()` (verbose).
- **Zero parameters**: The right call. There's no configuration needed — clone always does the same thing. This is the pathlib philosophy: one obvious way to do it.
- **Semantics are well-chosen**: Shared inputs (references to existing values), fresh outputs (new identity). This matches the mental model of "I want another node that does the same thing, inserted into the same graph."
- **Fast path / slow path split**: Lines 2458-2464 check for graph attributes before invoking the heavier `_cloner` machinery. Performance-conscious.
- **Output properties preserved**: Name, type, shape, const_value, doc_string, metadata_props, meta — comprehensive.
- **Docstring teaches the mental model**: "Inputs are shared by reference... the clone is meant to be inserted into the same (or a related) graph." This note prevents the most common misunderstanding.
- **Excellent test coverage**: 59 clone-related tests. Thorough.

### IMPROVE

1. **Example could be more realistic (LOW)**: The example `graph.node("my_relu")` uses a lookup-by-name API. A more realistic example would show the common pattern:
   ```python
   for node in graph:
       if some_condition(node):
           copy = node.clone()
           graph.insert_after(node, copy)
   ```
   This teaches the typical use case (clone during iteration + insert).

2. **`shape.copy()` vs `Shape(...)` (LOW)**: Line 2436 does `original_output.shape.copy()`. This is correct (shapes are mutable, must not share), but it relies on `Shape` having a `.copy()` method. If `Shape` ever changes to not have `.copy()`, this breaks silently. Consider adding a brief comment: `# Deep copy to avoid shared mutation`. Actually, the code is clear enough. **No change needed.**

3. **`_metadata` access pattern (NITPICK)**: Lines 2442-2449 access `_metadata_props` and `_metadata` with `# pylint: disable=protected-access`. This is unavoidable since `clone()` lives on the class itself. The pylint suppression is the right approach. **No change needed.**

### CRITICAL

None.

**Verdict: EXCELLENT. Well-designed, well-tested, well-documented.**

---

## 4. `replace_node` — `src/onnx_ir/editing.py:102-207` (BONUS — just shipped)

### GOOD
- **Signature follows the proposal**: `replace_node(old_node, new_node, *, output_mapping=None, propagate_metadata=True)`. Clean separation of required positional args and optional keyword args.
- **Simple case is simple**: When output counts match (the 90% case), no `output_mapping` needed. When they differ, the explicit mapping is required — not guessed.
- **Error messages are helpful**: "old_node does not belong to a graph. It must be part of a graph to be replaced." — tells you what's wrong AND what the requirement is. This is teaching-quality error messaging.
- **Graph output handling is extracted**: `_handle_graph_output_replacement` and `_propagate_value_metadata` are reusable helpers for `eliminate_node` and `replace_subgraph`. Good factoring.
- **`frozenset(graph.outputs)` for membership**: Line 187 — avoids O(n) list scan per output. Performance-conscious as required.
- **Ordering is correct**: Metadata propagation → graph output handling → use replacement → insert → remove. This order is critical — graph output handling must happen before `replace_all_uses_with` because the latter would clobber graph output references.

### IMPROVE

1. **Double iteration over mapping (LOW)**: Lines 188-192 and 195-201 iterate over `mapping.items()` twice. Could be fused into a single loop with deferred `replace_all_uses_with` calls. In practice this is O(outputs) which is tiny, so not a real issue. **No change needed for correctness, but worth noting for `replace_subgraph` where output counts could be larger.**

2. **The `replace_graph_outputs=True` fallback (MEDIUM)**: Lines 198-201: after `_handle_graph_output_replacement` already swapped graph outputs, the code checks `if old_output.is_graph_output()` again and calls `replace_all_uses_with(..., replace_graph_outputs=True)`. This implies the graph output handling might not have fully replaced — which can happen if `_handle_graph_output_replacement` only handles one occurrence while the value appears multiple times. The logic is correct but the comment "We already handled graph outputs above, so check if old_output is still listed" is confusing. Suggest: clarify that this handles the case where the same value appears in graph outputs multiple times (which is unusual but possible with suboptimal graphs).

3. **`new_node.graph is not None` check (LOW)**: Line 159 rejects new_node if it already belongs to a graph. This is the right guard, but the error message could suggest the fix: "Create a new node or use node.clone() to get a copy." Add that guidance.

### CRITICAL

4. **Identity node insertion position (MEDIUM)**: In `_handle_graph_output_replacement` line 95: `graph.insert_before(insertion_point, identity_node)`. The `insertion_point` is `old_node`. If `old_node` is about to be removed (which it is, in `replace_node`), the Identity is inserted before it, which is correct positionally. But if someone calls `_handle_graph_output_replacement` from a different context where the insertion point is wrong, the Identity could end up in a bad position. The helper is currently private so this is acceptable, but add a docstring note about the insertion_point contract.

**Verdict: GOOD. Minor documentation improvements. Solid implementation.**

---

## Summary

| API | Design | Tests | Docs | Overall |
|-----|--------|-------|------|---------|
| `topological_order` | ⭐⭐⭐⭐ | ❌ Missing | ⚠️ Inaccurate claim | GOOD — needs tests |
| `pass_fn` | ⭐⭐⭐⭐⭐ | ✅ 10 tests | ✅ Clean | EXCELLENT |
| `node.clone()` | ⭐⭐⭐⭐⭐ | ✅ 59 tests | ✅ Teaches | EXCELLENT |
| `replace_node` | ⭐⭐⭐⭐ | ✅ 19 tests | ✅ Helpful errors | GOOD |

### Action Items (Priority Order)

1. **CRITICAL**: Add tests for `topological_order` — at least 7 test cases covering the scenarios listed above.
2. **MEDIUM**: Fix `topological_order` docstring inaccuracy about lazy allocation.
3. **MEDIUM**: Clarify the graph-output double-replacement logic comment in `replace_node`.
4. **LOW**: Add `"Create a new node or use node.clone()"` guidance to `replace_node`'s error message for `new_node.graph is not None`.
5. **LOW**: Add `recursive=True` example to `topological_order` docstring.

### Taste Assessment

The APIs feel consistent with each other and with the existing library. Key patterns are maintained:
- Positional-only for primary arguments, keyword-only for options (matches `graph.remove(nodes, /, safe=False)`)
- `propagate_metadata=True` as a default (optimistic, batteries-included)
- Helpful error messages that teach rather than blame
- Docstrings with `versionadded`, `Args`, `Returns`, `Raises`, `Example` sections

The naming is strong across the board: `topological_order`, `pass_fn`, `clone`, `replace_node`, `eliminate_node` — each name tells you what it does without reading the docstring. This is the hallmark of a well-designed API.

**Overall: This is a library I'd want to use. The APIs have good taste. 👍**
