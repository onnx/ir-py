# Slide Review: Technical Accuracy

Reviewer: Architect (1ce04b33)
Source: `/presentations/slides.md`
Cross-referenced against: actual source code in `src/onnx_ir/`

---

## 🔴 CRITICAL — Wrong Class Names (Slide: "Built-in Passes", lines 537–557)

**6 of 11 pass names in the table are wrong.** An audience member who tries to use them will get ImportError.

| Slide Name (WRONG) | Actual Name |
|---|---|
| `UnusedRemovalPass` | `RemoveUnusedNodesPass` |
| `InitializerDeduplicationPass` | `DeduplicateInitializersPass` |
| `ConstantManipulationPass` | `LiftConstantsToInitializersPass` (not a single pass — there are 4 in this module) |
| `OnnxCheckerPass` | `CheckerPass` |
| `DefaultAttributesPass` | `AddDefaultAttributesPass` |
| `InlinerPass` | `InlinePass` |

**Also wrong on line 519** in the PassManager example: `UnusedRemovalPass()` should be `RemoveUnusedNodesPass()`.

**Fix:** Replace all 7 occurrences with the actual class names from `passes/common/`.

---

## 🔴 CRITICAL — Buggy Custom Pass Example (Slide: "Writing Your Own Pass", lines 565–592)

The `FuseAddReluPass` example has **two bugs** that would crash at runtime:

### Bug 1: `insert_before` called with swapped arguments AND on a removed node

```python
# Line 589 (WRONG):
model.graph.insert_before(fused, node)
```

The actual signature is `graph.insert_before(reference_node, new_nodes)` — so this tries to insert `node` before `fused`. But worse, `node` was already removed on line 587, so `fused` isn't in the graph either.

### Bug 2: Incorrect operation order

The code removes both nodes (lines 587-588) before inserting the replacement (line 589). After removal, neither node is in the graph, so `insert_before` cannot find a reference point.

**Fix:** Use `replace_nodes_and_values` (which is already shown in other slides) or fix the ordering:

```python
# Correct approach using convenience API:
ir.convenience.replace_nodes_and_values(
    model.graph,
    insertion_point=node,
    old_nodes=[producer, node],
    new_nodes=[fused],
    old_values=[node.outputs[0]],
    new_values=[fused.outputs[0]],
)
```

Or if writing manually, insert first, then replace uses, then remove:

```python
model.graph.insert_before(node, fused)
ir.convenience.replace_all_uses_with(node.outputs[0], fused.outputs[0])
model.graph.remove(node, safe=True)
model.graph.remove(producer, safe=True)
```

---

## 🟡 MODERATE — Wrong API Signatures (Slide: "Convenience APIs", lines 439–484)

### Issue 1: `replace_nodes_and_values` shown with wrong signature (line 451-453)

```python
# Slide (WRONG):
ir.convenience.replace_nodes_and_values(mapping)
```

The actual signature is:
```python
replace_nodes_and_values(
    graph_or_function,   # positional-only
    insertion_point,
    old_nodes, new_nodes,
    old_values, new_values,
)
```

This is **not** a mapping-based API. It takes 6 explicit arguments.

### Issue 2: `convert_attribute` shown with wrong signature (line 473-474)

```python
# Slide (WRONG):
attr = ir.convenience.convert_attribute(value, dtype)
```

Actual signature: `convert_attribute(name: str, attr, attr_type=None)`. First arg is the attribute *name*, second is the value.

**Fix:** Show correct signatures or remove these if they clutter the slide:

```python
ir.convenience.replace_nodes_and_values(
    graph, insertion_point=old_node,
    old_nodes=[old], new_nodes=[new],
    old_values=[old.outputs[0]], new_values=[new.outputs[0]],
)

attr = ir.convenience.convert_attribute("alpha", 1.0)
```

---

## 🟡 MODERATE — ExternalTensor Inheritance Wrong (Slide: "ExternalTensor", line 285)

```python
# Slide (WRONG):
class ExternalTensor(TensorProtocol):
```

Actual code (line 605 of `_core.py`):
```python
class ExternalTensor(TensorBase, _protocols.TensorProtocol):
```

`ExternalTensor` inherits from `TensorBase` (the ABC), not directly from `TensorProtocol`. This matters because it shows the layered design: TensorBase provides common implementations, TensorProtocol defines the interface.

**Fix:** Change to `class ExternalTensor(TensorBase):` (simpler) or the full actual signature.

---

## 🟡 MODERATE — Fabricated `_array` Attribute in ExternalTensor (Slide lines 290-296)

The slide shows:
```python
self._array: np.ndarray | None = None  # Lazily populated

def numpy(self) -> np.ndarray:
    if self._array is None:
        self._array = np.memmap(self.path, offset=self._offset, ...)
    return self._array
```

The actual implementation doesn't have a `_array` attribute. The lazy loading mechanism works differently — it memory-maps on each call or caches via different internal mechanisms. The concept is correct (lazy/mmap), but the specific code shown is fabricated.

**Suggestion:** Either show the real implementation or clearly label as "simplified pseudocode."

---

## 🟡 MODERATE — Erased-Flag Description Subtlety (Slide: "Safe Iteration", lines 369-398)

The slide's `erase()` code is accurate, but the narrative is subtly misleading:

> "removed nodes stay linked, but marked as erased" (line 371)

After `erase()`, the erased box is **unlinked from the chain** (neighbors skip over it), but its **own `next`/`prev` pointers still point to the original neighbors**. So the box doesn't "stay linked" — rather, its outgoing pointers remain valid while incoming pointers are removed.

The `__iter__` code shown on the slide is also **missing the `owning_list` check** that exists in the actual code (line 118):
```python
if box.owning_list is not self:
    raise RuntimeError(...)
```

This check is important — it detects when a node has been moved to a different graph during iteration.

**Fix:** Change the description to: "removed nodes' forward pointers remain valid, so active iterators can follow the chain." Add the `owning_list` check to the shown iterator code.

---

## 🟢 MINOR — Value Class Missing `shape` Property (Slide: "The Value Object", line 207)

The Value class snippet shows `name`, `type`, `const_value`, `producer()`, `uses()`, `index()` but omits `shape: Shape | None`. Shape is one of the most frequently accessed properties on Value.

**Fix:** Add `shape: Shape | None` to the snippet.

---

## 🟢 MINOR — `uses()` Return Type (Slide line 213)

```python
def uses(self) -> Collection[Use]:
```

The actual return type is `Collection[Usage]` where `Usage` is a `NamedTuple` with `(node, idx)` fields. `Use` is not a defined type.

**Fix:** `Collection[Usage]` or `Collection[tuple[Node, int]]`.

---

## 🟢 MINOR — `extract` Operates on Graph, Not Model (Slide line 468)

```python
submodel = ir.convenience.extract(model, inputs, outputs)
```

The actual function takes `Graph | Function | GraphView`, not `Model`. And it returns `Graph`, not a model.

**Fix:**
```python
subgraph = ir.convenience.extract(graph, inputs, outputs)
```

---

## 🟢 MINOR — Missing Topics for a 20-min Talk

The slides cover the core well. A few topics from the codebase that could strengthen the talk:

1. **GraphView** — mentioned in findings but not in slides. The read-only graph view is a nice design pattern worth a quick mention (type system enforces immutability).

2. **`meta` vs `metadata_props`** — the dual metadata system (serializable vs. pass-only) is an elegant design decision not covered.

3. **Function support and overloads** — `model.functions` as a dict keyed by `(domain, name, overload)` is a significant improvement over protobuf's flat list.

4. **`graph.sort()`** — the built-in topological sort is worth mentioning as it's a common operation.

5. **`Value.consumers()`** — in addition to `uses()`, there's `consumers()` which returns deduplicated nodes. This is the more commonly used method in passes.

---

## ✅ What's Accurate and Well-Done

- The protobuf vs. IR motivation slide is excellent and accurate
- The initializer comparison (protobuf vs. IR) is spot-on
- The Value object slide with the three-card layout (input/initializer/computed) is a great teaching tool
- The TensorProtocol slide with three implementations is accurate
- The DoublyLinkedSet architecture (sentinel, _LinkBox, _value_ids_to_boxes) is correct
- The comparison table on the summary slide is accurate
- The speaker notes are well-written and add appropriate context
- The `ir.load` / `ir.save` roundtrip example is a clean finish

---

## Summary: Priority Fixes

| Priority | Issue | Slide |
|---|---|---|
| 🔴 Critical | 6 wrong pass class names | "Built-in Passes" |
| 🔴 Critical | Buggy custom pass example (crash at runtime) | "Writing Your Own Pass" |
| 🟡 Moderate | Wrong `replace_nodes_and_values` signature | "Convenience APIs" |
| 🟡 Moderate | Wrong `convert_attribute` signature | "Convenience APIs" |
| 🟡 Moderate | ExternalTensor wrong parent class | "ExternalTensor" |
| 🟡 Moderate | Fabricated `_array` attribute | "ExternalTensor" |
| 🟡 Moderate | Erased-flag description subtlety + missing owning_list check | "Safe Iteration" |
| 🟢 Minor | Missing `shape` on Value | "The Value Object" |
| 🟢 Minor | `Use` → `Usage` type name | "The Value Object" |
| 🟢 Minor | `extract` takes graph not model | "Convenience APIs" |
