# 🔥 Radical Thinker Report: ONNX IR Graph Editing — First Principles Analysis

## Current State: What We Have

The ONNX IR (`onnx_ir`) is a well-engineered in-memory representation for ONNX models. The core design is solid:

- **Doubly-linked list** of `Node`s with safe mutation during iteration
- **Use-def chains** via `Value.uses()` / `Value.producer()` / `Value.consumers()`
- **Tape/Builder** pattern for graph construction
- **Pass infrastructure** with `InPlacePass`/`FunctionalPass`, `PassManager`, `Sequential`
- **RecursiveGraphIterator** for subgraph-aware traversal
- **Convenience** module with `replace_all_uses_with`, `extract`, `get_const_tensor`

The existing passes (14 built-in) demonstrate standard compiler optimization patterns: identity elimination, DCE, CSE, constant lifting, topological sort, inlining, deduplication.

### What Works Well
- Mutation-safe iteration (DoublyLinkedSet) is a genuine differentiator
- Use-def chains are bidirectional and well-maintained
- The pass infrastructure is clean: preconditions, postconditions, composition
- Journaling for debugging is a nice touch
- Zero-copy tensor handling with mmap is excellent engineering

### Where It Gets Painful

After studying every built-in pass and the convenience APIs, here are the friction points:

1. **Graph output handling is a recurring tax** — Every pass must check `is_graph_output()` and special-case. Identity elimination alone has 3 branches for this.
2. **Subgraph recursion is manual boilerplate** — Each pass re-implements "find GRAPH/GRAPHS attributes and recurse."
3. **No pattern matching** — Passes use raw `if node.op_type == "Identity"` checks. Multi-node patterns require hand-written state machines.
4. **No atomic "replace subgraph" operation** — Replacing a sequence of nodes requires multiple coordinated calls (create new nodes, rewire, remove old) with no transactional guarantee.
5. **No graph diff/undo** — Once you mutate, you can't go back except by cloning the entire model upfront.
6. **Building and editing are separate worlds** — The `Tape`/`Builder` pattern works for construction, but editing an existing graph means dropping to low-level insert/remove/rewire.

---

## Bold Proposals

### Proposal 1: Pattern Matching & Rewriting DSL

**The Problem**: Every optimization pass is essentially "find a pattern in the graph, replace it with something." But there's no first-class support for this. Each pass hand-codes its own pattern detection.

**Inspiration**: MLIR's [DRR (Declarative Rewrite Rules)](https://mlir.llvm.org/docs/DeclarativeRewrites/) and TVM's [pattern matching](https://tvm.apache.org/docs/reference/langref/relay_pattern.html) make this declarative.

**The Proposal**: A composable pattern-matching API that lets you express graph patterns as Python objects:

```python
import onnx_ir as ir
from onnx_ir.patterns import pat, rewrite

# Define a pattern: MatMul followed by Add (a.k.a. linear layer / Gemm)
matmul_add = pat.Op("MatMul", inputs=[pat.Var("A"), pat.Var("B")]) \
           | pat.Op("Add", inputs=[pat.PREV, pat.Var("C")])

# Or a more Pythonic "builder" style:
with pat.Pattern() as p:
    mm = p.op("MatMul", p.var("A"), p.var("B"))
    result = p.op("Add", mm, p.var("C"))

# Define the replacement
def fuse_to_gemm(match):
    return ir.node("Gemm", inputs=[match["A"], match["B"], match["C"]],
                   attributes={"alpha": 1.0, "beta": 1.0})

# Apply it as a rewrite rule
rule = rewrite.Rule(pattern=matmul_add, replacement=fuse_to_gemm)
rule.apply(graph)  # Finds all matches and replaces them

# Or use it within a pass
class GemmFusionPass(ir.passes.InPlacePass):
    def call(self, model):
        modified = rule.apply(model.graph)
        return ir.passes.PassResult(model, modified=modified)
```

**Key design choices:**
- `pat.Var("name")` — Wildcard that captures any single value
- `pat.Op("Type", inputs=[...])` — Matches a specific op with specific input pattern
- `pat.PREV` — Refers to the output of the previous matched node (for chain patterns)
- `pat.AnyOf(...)` — Matches any of several patterns (for op_type variants)
- `pat.Const()` — Matches values backed by constant tensors
- `pat.Predicate(lambda v: ...)` — Custom matching conditions
- Replacements can be functions (for computed attributes) or static templates

**Why this is 10x, not 10%**: It eliminates an entire category of boilerplate. A typical optimization pass body shrinks from 50-100 lines to 5-10 lines. It also makes patterns *composable* — you can build a library of common patterns and combine them.

**Tradeoffs:**
- (+) Massive productivity boost for pass authors
- (+) Patterns are self-documenting — you can read the pattern to understand what a pass does
- (+) Patterns can be analyzed statically (conflict detection, coverage analysis)
- (-) Complex patterns (conditional logic, cross-subgraph) may not fit the declarative model
- (-) Performance: naive pattern matching is O(nodes × pattern_size); needs indexing
- (-) Learning curve for the pattern DSL itself

**Mitigation**: The imperative `call()` escape hatch always exists. The pattern API is additive, not a replacement.

---

### Proposal 2: Graph Rewriting Transactions

**The Problem**: Graph mutations are immediately visible. If step 3 of a 5-step rewrite fails, the graph is in an inconsistent state. The only defense is cloning the whole model upfront (expensive) or writing bulletproof code (hard).

**Inspiration**: Database transactions (ACID), Git's staging area, MLIR's `PatternRewriter` (which batches operations).

**The Proposal**: A lightweight transaction/context manager for graph edits:

```python
with graph.transaction() as txn:
    # All edits are staged, not yet applied
    new_node = txn.create_node("Relu", inputs=[x])
    txn.replace_all_uses_with(old_value, new_node.outputs[0])
    txn.remove(old_node)
    
    # If anything raises, all edits are rolled back
    # On success, all edits are applied atomically

# Explicit commit/rollback
txn = graph.begin_transaction()
try:
    txn.create_node(...)
    txn.replace_all_uses_with(...)
    txn.commit()  # Apply all at once
except SomeError:
    txn.rollback()  # Discard everything
```

**Implementation sketch**: The transaction proxies all mutation methods, recording them as a list of `Edit` objects. On `commit()`, it applies them in order. On `rollback()` (or exception), it discards them. Optionally, it can validate the resulting graph before committing.

**Why this matters**:
- **Safety**: No more half-mutated graphs from bugs in passes
- **Composability**: Multiple rewrite rules can be tried speculatively
- **Debugging**: You can inspect the edit list before committing
- **Undo/Redo**: Transactions naturally give you undo (rollback) and redo (replay)

**Tradeoffs:**
- (+) Fundamentally safer graph mutation
- (+) Enables speculative optimization (try a rewrite, check if it improves, keep or discard)
- (+) Natural undo/redo for interactive tools
- (-) Memory overhead: must store edit log (but typically small relative to model size)
- (-) Complexity: proxy objects, deferred application, conflict detection
- (-) Performance: extra indirection for every mutation

**Alternative lightweight approach**: Instead of full transactions, provide a `graph.checkpoint()` / `graph.restore()` mechanism using copy-on-write snapshots:

```python
checkpoint = graph.checkpoint()
try:
    # ... risky edits ...
except:
    graph.restore(checkpoint)
```

---

### Proposal 3: Graph Editing "Verbs" — High-Level Composable Operations

**The Problem**: Today's editing operations are low-level primitives (insert node, remove node, rewire value). Common compound operations require 5-15 lines of careful coordination. This is like editing text by manipulating a linked list of characters instead of using find-and-replace.

**Inspiration**: Text editors (find/replace, cut/paste, undo), Unix philosophy (composable tools), NetworkX's graph operations (union, compose, intersection).

**The Proposal**: A library of high-level, composable graph editing "verbs":

```python
from onnx_ir import editing

# === Structural Operations ===

# Insert a node "inline" on an edge (splits the edge)
editing.insert_on_edge(source_value, new_node)
# Before: A --x--> B
# After:  A --x--> NewNode --y--> B

# Replace a single node with another (auto-rewires inputs/outputs)
editing.replace_node(old_node, new_node)

# Replace a subgraph (multiple connected nodes) with a single node or subgraph
editing.replace_subgraph(
    old_nodes=[matmul, bias_add, relu],
    new_nodes=[fused_conv_relu],
    input_mapping={matmul.inputs[0]: fused.inputs[0], ...},
    output_mapping={relu.outputs[0]: fused.outputs[0]}
)

# Fuse a sequence of nodes into a Function (extract to subprogram)
editing.extract_to_function(
    nodes=[conv, bn, relu],
    name="ConvBnRelu",
    graph=model.graph,
    model=model
)

# Inline a Function call back into the graph
editing.inline_function(call_node, model)

# === Value Operations ===

# Fork a value (insert a copy/identity and give each consumer its own value)
editing.fork_value(value)  
# Before: A --x--> [B, C, D]
# After:  A --x--> Identity1 --x1--> B
#                  Identity2 --x2--> C  
#                  Identity3 --x3--> D

# Merge values (opposite of fork — route multiple values through a single node)
editing.merge_values([v1, v2, v3], merge_op="Concat", axis=0)

# === Graph Composition ===

# Concatenate two graphs (output of G1 feeds into input of G2)
editing.chain(graph1, graph2)

# Run two subgraphs in parallel (independent inputs/outputs)
editing.parallel(subgraph1, subgraph2)

# Wrap a subgraph in a conditional (If node)
editing.wrap_in_conditional(nodes, condition_value)

# Wrap a subgraph in a loop
editing.wrap_in_loop(nodes, trip_count, loop_condition)
```

**Why this is transformative**: These operations encode *intent* rather than mechanism. `replace_subgraph` says "I want to swap these nodes for those nodes" — the library handles all the rewiring, output mapping, cleanup, and edge cases. This is the difference between assembly and a high-level language.

**Tradeoffs:**
- (+) Dramatically reduces pass complexity and bug surface
- (+) Self-documenting: the operation names describe what's happening
- (+) Can be combined with transactions for safety
- (+) Can be combined with pattern matching for powerful rewrite rules
- (-) Must handle many edge cases correctly (graph outputs, subgraphs, initializers)
- (-) May not cover all possible graph edits (need escape hatch to primitives)
- (-) Some operations have ambiguous semantics (what does "replace_subgraph" do with side outputs?)

---

### Proposal 4: Immutable Graph Views with Lazy Materialization

**The Problem**: FunctionalPass requires cloning the entire model upfront. InPlacePass is efficient but unsafe (mutations are immediate). There's no middle ground.

**Inspiration**: Persistent data structures (Clojure, Haskell), copy-on-write (CoW) in operating systems, Git's content-addressed storage.

**The Proposal**: Copy-on-write graph views that share structure with the original until modified:

```python
# Create a CoW view — no copy happens yet
view = graph.cow_view()

# Reads go through to the original (zero cost)
for node in view:
    print(node.op_type)

# First write triggers a lazy copy of ONLY the affected structure
view.remove(some_node)  # Only copies the linked list spine, not node data

# Original is untouched
assert some_node in graph  # Still there!

# Materialize the view into a real graph when ready
new_graph = view.materialize()

# Or discard it cheaply
del view  # No cleanup needed
```

**Why this is interesting**: It gives you the safety of FunctionalPass with near the performance of InPlacePass. You can speculatively try multiple optimization strategies in parallel, each on their own CoW view, and pick the best result.

**Tradeoffs:**
- (+) Best of both worlds: safety + performance
- (+) Enables parallel/speculative optimization
- (+) Natural snapshotting for undo
- (-) Complex implementation (structural sharing, reference counting)
- (-) Debugging complexity (which version of a node am I looking at?)
- (-) Python's reference semantics make true CoW tricky (need proxy objects)

---

### Proposal 5: The "Graph Cursor" — A Fundamentally Different Editing Paradigm

**The Problem**: All current editing is "outside-in" — you get a reference to the graph, then reach in and modify nodes. This requires global knowledge of the graph structure.

**Inspiration**: Text editor cursors, Haskell's Zippers, tree-sitter's TreeCursor, Clojure's zippers.

**The Proposal**: A cursor that "sits on" a node and provides local editing operations:

```python
from onnx_ir.cursor import Cursor

# Create a cursor pointing at a specific node
cursor = Cursor(graph, target_node)

# Navigate
cursor.next()           # Move to next node in topological order
cursor.prev()           # Move to previous node
cursor.into_subgraph()  # Descend into a subgraph attribute
cursor.up()             # Return to parent graph
cursor.goto(some_node)  # Jump to a specific node

# Local operations (relative to cursor position)
cursor.insert_before(new_node)   # Insert before current node
cursor.insert_after(new_node)    # Insert after current node
cursor.replace_with(new_node)    # Replace current node, auto-rewire
cursor.remove()                  # Remove current node, advance cursor
cursor.wrap_with("Relu")         # Wrap current node's output through a new op

# Query local context
cursor.node                     # Current node
cursor.predecessors()            # Nodes that feed into this one
cursor.successors()              # Nodes that consume this one's outputs
cursor.neighborhood(depth=2)     # All nodes within 2 hops

# Chainable cursor operations (like jQuery for graphs!)
cursor.find("Conv") \
      .next_matching("BatchNormalization") \
      .replace_with(fused_conv_bn_node)
```

**Why this is radical**: It makes graph editing feel like navigating and editing text. You don't need to hold the entire graph in your head — you just need local context. This is incredibly powerful for:
- Interactive graph editors (cursor IS the UI selection)
- Peephole optimizations (look at a small window, make local changes)
- Streaming transformations (process the graph node-by-node)

**Tradeoffs:**
- (+) Extremely intuitive for local transformations
- (+) Natural fit for interactive tools
- (+) Can be layered on top of existing primitives
- (-) Poorly suited for global transformations (CSE, DCE)
- (-) Topological "next" is not always unique (multiple valid orderings)
- (-) Overhead per operation (cursor bookkeeping)

---

### Proposal 6: First-Class Subgraph Handles

**The Problem**: Today, a "subgraph" is just an informal concept — a set of nodes you're thinking about together. There's no object representing a contiguous subgraph that you can manipulate as a unit.

**Inspiration**: MLIR regions, TVM's relay patterns, database query plan fragments.

**The Proposal**: A `Subgraph` value type that represents a connected portion of a graph:

```python
from onnx_ir import Subgraph

# Define a subgraph by boundary values
sub = Subgraph.between(input_values=[x, w], output_values=[y])
# Automatically captures all nodes on paths from inputs to outputs

# Or by explicit node set
sub = Subgraph.from_nodes([conv, bn, relu])
# Auto-discovers boundary values

# Properties
sub.nodes          # The nodes in the subgraph
sub.inputs         # Values entering from outside
sub.outputs        # Values leaving to outside
sub.internal       # Values that are purely internal
sub.is_connected   # Whether the subgraph is a connected component
sub.is_acyclic     # Whether the subgraph is a DAG

# Operations
sub.clone()                    # Deep copy the subgraph
sub.detach()                   # Remove from parent graph, return standalone Graph
sub.replace_with(other_sub)    # Swap this subgraph for another one
sub.wrap_as_function("name")   # Extract into a Function, leave a call node
sub.fuse_into(single_node)     # Replace with a single fused op node

# Composition
combined = sub1.compose(sub2)  # Chain two subgraphs
parallel = sub1.parallel(sub2) # Side-by-side subgraphs

# Pattern matching returns subgraphs
matches = pattern.find_all(graph)  # Returns list of Subgraph
for match in matches:
    match.replace_with(optimized_version)
```

**Why this is powerful**: It gives names and operations to something that's currently implicit. Every pass that does "find a group of nodes and do something with them" would benefit. It also bridges pattern matching (which *finds* subgraphs) and graph editing (which *modifies* them).

**Tradeoffs:**
- (+) Unifies pattern matching results with editing operations
- (+) Makes subgraph-level operations first-class
- (+) Enables subgraph-level analysis (cost modeling, memory estimation)
- (-) Boundary definition can be ambiguous (multiple valid decompositions)
- (-) Ownership semantics: does the subgraph own its nodes or just reference them?
- (-) Complexity: maintaining subgraph consistency as the parent graph changes

---

## Comparisons with Other Systems

### MLIR (Multi-Level IR)
- **Regions and Blocks**: MLIR has a hierarchical structure (Region > Block > Operation). ONNX IR's flat graph is simpler but less expressive.
- **DRR**: Declarative rewrite rules are a standout feature. ONNX IR has nothing comparable.
- **Operation interfaces**: Allow generic code to work on operations with specific properties. ONNX IR uses protocols but could go further.
- **What to steal**: Pattern matching infrastructure, the concept of "rewrite drivers" that manage pattern application order.

### TVM Relay/TIR
- **Pattern language**: TVM's pattern matching is well-designed for DNN-specific patterns (conv+bn fusion, etc.).
- **Relay passes**: Very similar to ONNX IR's pass infrastructure, but with richer pass-level composition.
- **Mutator/Visitor**: Explicit visitor pattern with mutation support.
- **What to steal**: The pattern language design, DNN-specific pattern primitives.

### LLVM
- **RAUW (Replace All Uses With)**: ONNX IR already has this — good!
- **InstCombine**: A giant peephole optimizer that uses pattern matching. Shows the power of local pattern-based optimization.
- **Dominance and SSA**: LLVM's SSA form makes many analyses trivial. ONNX IR values are SSA (single producer), but the graph structure isn't enforced.
- **What to steal**: The `IRBuilder` pattern (insert point + create operations), which is similar to ONNX IR's Tape but more ergonomic.

### NetworkX
- **Rich graph operations**: union, intersection, compose, product, subgraph views.
- **Algorithms library**: Hundreds of built-in algorithms (shortest path, centrality, flow, etc.).
- **What to steal**: The idea that graph operations should be as rich and composable as set operations. Also: `SubGraph` views (non-copying views into a graph).

### PyTorch FX
- **Graph Module**: Python functions as IR. Very accessible to ML engineers.
- **Subgraph rewriting**: `replace_pattern()` function that does pattern matching + replacement.
- **What to steal**: `replace_pattern()` — it's simple, it's Python, and it works. Also the idea that the graph should be inspectable as Python code.

---

## Priority Ranking & Recommendations

### 🏆 Highest Impact: Graph Editing "Verbs" (Proposal 3)

**Why first**: This is the most immediately useful proposal with the lowest risk. It's purely additive (new module, no changes to existing code), solves the most common pain points, and can be implemented incrementally. Start with `insert_on_edge`, `replace_node`, and `replace_subgraph`.

### 🥈 Second: Pattern Matching (Proposal 1) + Subgraph Handles (Proposal 6)

**Why together**: These are two sides of the same coin. Pattern matching *finds* subgraphs; subgraph handles let you *do things* with them. Implement Subgraph handles first (they're useful standalone), then build pattern matching on top.

### 🥉 Third: Transactions/Checkpoints (Proposal 2)

**Why third**: Important for safety and interactive tools, but the current "clone before mutating" approach works. Implement the lightweight `checkpoint()/restore()` variant first.

### 🔮 Future: CoW Views (Proposal 4) and Graph Cursor (Proposal 5)

**Why later**: These are architecturally significant changes. CoW requires deep changes to the data structures. The cursor is a new paradigm that would be most useful once the other pieces are in place. Worth prototyping but not the first priority.

---

## Concrete API Sketch: What a "v2 Editing API" Could Look Like

Here's how all these proposals fit together in a unified vision:

```python
import onnx_ir as ir
from onnx_ir import editing, patterns

# === Load a model ===
model = ir.load("model.onnx")

# === Pattern matching ===
# Find all Conv+BN sequences
conv_bn = patterns.chain(
    patterns.op("Conv", inputs=[patterns.var("x"), patterns.var("w"), patterns.var("b")]),
    patterns.op("BatchNormalization")
)

matches = conv_bn.find_all(model.graph)
print(f"Found {len(matches)} Conv+BN patterns")

# === Subgraph-level editing ===
for match in matches:
    sub = match.as_subgraph()
    fused_node = create_fused_conv_bn(match)
    sub.replace_with(fused_node)  # All rewiring handled automatically

# === High-level verbs ===
# Insert a Relu after every Conv
for node in model.graph:
    if node.op_type == "Conv":
        editing.insert_after_node(node, ir.node("Relu", inputs=[node.outputs[0]]))

# === Transaction safety ===
with model.graph.transaction() as txn:
    txn.remove(dead_nodes)
    txn.sort()
    # If any exception, all changes roll back

# === Build a pass using all of the above ===
class ConvBnFusion(ir.passes.InPlacePass):
    pattern = patterns.chain(
        patterns.op("Conv"), 
        patterns.op("BatchNormalization")
    )
    
    def call(self, model):
        count = 0
        for match in self.pattern.find_all(model.graph):
            sub = match.as_subgraph()
            sub.replace_with(self._fuse(match))
            count += 1
        return ir.passes.PassResult(model, modified=count > 0)
```

---

## The Radical Question: What If Passes Were Just Functions?

One final thought experiment. Today, a pass is a class:

```python
class MyPass(ir.passes.InPlacePass):
    def call(self, model):
        ...
```

**What if it were just a decorated function?**

```python
@ir.pass_fn
def my_pass(model: ir.Model) -> bool:
    """Returns True if modified."""
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        if node.op_type == "Identity":
            ir.convenience.replace_all_uses_with(node.outputs[0], node.inputs[0])
            node.graph.remove(node, safe=True)
            return True
    return False

# Compose them like functions
pipeline = ir.passes.compose(my_pass, other_pass, third_pass)
result = pipeline(model)
```

This would dramatically lower the barrier to writing passes. The class-based approach is more structured, but the function-based approach is more Pythonic and accessible. Both could coexist.

**Even more radical**: What about a **declarative pass DSL**?

```python
@ir.rewrite_rule
def eliminate_identity(node: ir.Node):
    """Eliminate identity nodes."""
    if node.op_type == "Identity":
        yield ir.Rewrite(
            remove=node,
            replace={node.outputs[0]: node.inputs[0]}
        )
```

This is essentially what `InstCombine` does in LLVM — a collection of local rewrite rules applied greedily. It's the most productive pattern for peephole optimizations.

---

## Summary

The ONNX IR has a strong foundation. The proposals above aren't about fixing what's broken — they're about making the leap from "a good IR library" to "the definitive tool for ONNX model transformation." The biggest wins come from:

1. **Raising the abstraction level** (Editing Verbs, Subgraph Handles)
2. **Making patterns first-class** (Pattern Matching DSL)
3. **Adding safety rails** (Transactions, Checkpoints)
4. **Lowering the barrier to pass creation** (Function-based passes, Rewrite Rules)

The common thread: **encode intent, not mechanism**. The library should understand *what* the user wants to do (replace a pattern, fuse nodes, insert an operation) and handle the *how* (rewiring, cleanup, validation) automatically.
