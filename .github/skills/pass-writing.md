# ONNX IR Pass Writing Skill

This skill provides comprehensive guidance for creating transformation passes in the ONNX IR project. It encapsulates best practices, conventions, and patterns derived from existing passes in `onnx_ir/passes/common/`.

## Overview

The ONNX IR pass infrastructure is designed for graph construction, analysis, and transformation. Passes are composable units that transform ONNX models in a well-defined way.

### Key Concepts

- **Pass**: A transformation that takes an `ir.Model` and returns a `PassResult` containing the transformed model and a boolean indicating if modifications were made
- **InPlacePass**: A pass that modifies the input model directly and returns it (most efficient)
- **FunctionalPass**: A pass that returns a new model without modifying the input
- **PassManager/Sequential**: Composes multiple passes to run in sequence

## Pass Infrastructure

### Base Classes

All passes inherit from one of these base classes defined in `onnx_ir.passes`:

#### `InPlacePass`
```python
class MyPass(ir.passes.InPlacePass):
    """Most common pass type - modifies model in place."""
    
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        # Transform the model
        return ir.passes.PassResult(model, modified=modified)
```

**Use when**: You want efficient in-place mutation (recommended for most passes)

**Properties**:
- `in_place = True` (automatically set)
- `changes_input = True` (automatically set)

#### `FunctionalPass`
```python
class MyPass(ir.passes.FunctionalPass):
    """Pure functional pass - does not modify input."""
    
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Must return a different model object
        new_model = model.clone()
        # Transform new_model
        return ir.passes.PassResult(new_model, modified=True)
```

**Use when**: You need to preserve the original model unchanged

**Properties**:
- `in_place = False` (automatically set)
- `changes_input = False` (automatically set)

### Pass Lifecycle

1. **Preconditions** (`requires` method): Check input model validity (optional)
2. **Transformation** (`call` method): Apply the transformation
3. **Postconditions** (`ensures` method): Validate output model (optional)

```python
class MyPass(ir.passes.InPlacePass):
    def requires(self, model: ir.Model) -> None:
        """Validate preconditions. Raise PreconditionError if violated."""
        # Example: Ensure specific opset version
        if model.graph.opset_imports.get("", 0) < 13:
            raise ir.passes.PreconditionError("Requires opset >= 13")
    
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Main transformation logic."""
        modified = False
        # ... transformation code ...
        return ir.passes.PassResult(model, modified=modified)
    
    def ensures(self, model: ir.Model) -> None:
        """Validate postconditions. Raise PostconditionError if violated."""
        # Example: Check model validity
        pass
```

## Best Practices for Pass Implementation

### 1. Graph Traversal

#### Traverse All Nodes (Including Subgraphs)
```python
import onnx_ir as ir

# Use RecursiveGraphIterator to process all nodes including subgraphs
for node in ir.traversal.RecursiveGraphIterator(model.graph):
    # Process node
    if node.op_type == "Identity":
        # ... handle identity node ...
        pass
```

#### Process Functions
```python
# Don't forget to process functions in the model
for function in model.functions.values():
    for node in ir.traversal.RecursiveGraphIterator(function):
        # Process node in function
        pass
```

#### Simple Graph Iteration
```python
# For non-recursive iteration of the main graph
for node in model.graph:
    # Process only direct nodes (no subgraphs)
    pass

# Reverse iteration (useful for removal)
for node in reversed(model.graph):
    # Process in reverse topological order
    pass
```

### 2. Node Manipulation

#### Safely Remove Nodes
```python
# Always use safe=True to ensure proper cleanup
graph.remove(node, safe=True)
```

#### Create New Nodes
```python
# Create a node with the ir.node() helper
new_node = ir.node(
    "Identity",
    inputs=[input_value],
    outputs=[
        ir.Value(
            name="output_name",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 3, 224, 224]),
        )
    ],
)

# Insert the node at a specific position
graph.insert_before(reference_node, new_node)
graph.insert_after(reference_node, new_node)

# Or append to the end
graph.append(new_node)
```

#### Modify Node Attributes
```python
# Access attributes as a dictionary
if "training_mode" in node.attributes:
    node.attributes.pop("training_mode")

# Add new attributes
node.attributes["new_attr"] = ir.Attr("new_attr", ir.AttributeType.STRING, "value")
```

### 3. Value Manipulation

#### Replace All Uses of a Value
```python
import onnx_ir.convenience as convenience

# Replace all uses of old_value with new_value
convenience.replace_all_uses_with(
    old_value,
    new_value,
    replace_graph_outputs=True  # Also replace in graph outputs if present
)

# Replace multiple values at once
convenience.replace_all_uses_with(
    [old_value1, old_value2],
    [new_value1, new_value2],
)
```

#### Check Value Usage
```python
# Check if a value is used
if output_value.uses():
    # Value has consumers
    pass

# Check if value is a graph output
if value.is_graph_output():
    # Special handling for outputs
    pass

# Check if value is a graph input
if value.is_graph_input():
    # Special handling for inputs
    pass
```

#### Merge Value Information
```python
# When eliminating nodes, preserve shape/type information
def merge_shapes(shape1: ir.Shape | None, shape2: ir.Shape | None) -> ir.Shape | None:
    if shape1 is None:
        return shape2
    if shape2 is None:
        return shape1
    # More sophisticated merging logic...
    return shape1

# Copy shape and type information
input_value.shape = merge_shapes(input_value.shape, output_value.shape)
if input_value.type is None:
    input_value.type = output_value.type
```

### 4. Initializers and Constants

#### Work with Initializers
```python
# Access initializers by name
initializers = graph.initializers
if "weight" in initializers:
    weight_initializer = initializers["weight"]

# Register a new initializer
new_initializer = ir.Value(
    name="new_weight",
    shape=ir.Shape([3, 3, 64, 64]),
    type=ir.TensorType(ir.DataType.FLOAT),
    const_value=tensor_data,
)
graph.register_initializer(new_initializer)

# Remove unused initializers
graph_outputs = frozenset(graph.outputs)
graph_inputs = frozenset(graph.inputs)
for init in list(initializers.values()):
    if not (init.uses() or init in graph_outputs or init in graph_inputs):
        assert init.name is not None
        del initializers[init.name]
```

#### Lift Constants to Initializers
```python
# Check if node is a Constant
if node.op_type == "Constant" and node.domain in ("", "onnx.ai"):
    # Get the tensor from the value attribute
    attr_value = node.attributes.get("value")
    if attr_value:
        tensor = attr_value.as_tensor()
        # Create initializer
        initializer = ir.Value(
            name=node.outputs[0].name,
            shape=tensor.shape,
            type=ir.TensorType(tensor.dtype),
            const_value=tensor,
        )
        graph.register_initializer(initializer)
        # Replace uses and remove node
        node.outputs[0].replace_all_uses_with(initializer)
        graph.remove(node, safe=True)
```

### 5. Logging and Debugging

```python
import logging

logger = logging.getLogger(__name__)

class MyPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = 0
        for node in model.graph:
            # Use debug for detailed information
            logger.debug("Processing node: %s", node)
            
            # Use info for important changes
            logger.info("Removed node: %s", node.name)
            count += 1
        
        # Summarize at the end
        if count:
            logger.info("MyPass removed %s nodes", count)
        
        return ir.passes.PassResult(model, modified=bool(count))
```

### 6. Handling Subgraphs

```python
# Process attributes that may contain subgraphs
for attr in node.attributes.values():
    if not isinstance(attr, ir.Attr):
        continue
    
    if attr.type == ir.AttributeType.GRAPH:
        subgraph = attr.as_graph()
        # Process the subgraph recursively
        modified |= self._process_graph(subgraph)
    
    elif attr.type == ir.AttributeType.GRAPHS:
        for subgraph in attr.as_graphs():
            # Process each subgraph
            modified |= self._process_graph(subgraph)
```

### 7. Opset Version Handling

```python
# Get opset version for the graph
onnx_opset_version = model.graph.opset_imports.get("", None)

# Check if a specific opset is available
if onnx_opset_version is not None and onnx_opset_version >= 13:
    # Use features from opset 13+
    pass

# Get schema information for a node
import onnx

try:
    op_schema = onnx.defs.get_schema(
        node.op_type,
        onnx_opset_version,
        domain=node.domain
    )
    # Use schema information
except Exception:
    logger.info("Failed to get schema for %s", node)
```

## Common Pass Patterns

### Pattern 1: Node Elimination

Eliminate nodes that match certain criteria (e.g., Identity, unused nodes).

```python
class NodeEliminationPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if self._should_eliminate(node):
                if self._try_eliminate_node(node):
                    modified = True
        
        return ir.passes.PassResult(model, modified=modified)
    
    def _should_eliminate(self, node: ir.Node) -> bool:
        """Check if node should be eliminated."""
        return node.op_type == "Identity" and node.domain == ""
    
    def _try_eliminate_node(self, node: ir.Node) -> bool:
        """Try to eliminate node. Returns True if successful."""
        # Validate node structure
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            return False
        
        input_value = node.inputs[0]
        output_value = node.outputs[0]
        
        if input_value is None:
            return False
        
        # Replace uses
        ir.convenience.replace_all_uses_with(
            output_value, input_value, replace_graph_outputs=True
        )
        
        # Remove node
        assert node.graph is not None
        node.graph.remove(node, safe=True)
        return True
```

### Pattern 2: Dead Code Elimination

Remove unused nodes and values.

```python
class DeadCodeEliminationPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = self._remove_unused_nodes(model.graph)
        
        for function in model.functions.values():
            count += self._remove_unused_nodes(function)
        
        return ir.passes.PassResult(model, modified=bool(count))
    
    def _remove_unused_nodes(self, graph_like: ir.Graph | ir.Function) -> int:
        """Remove nodes that produce no used outputs."""
        graph_outputs = frozenset(graph_like.outputs)
        count = 0
        
        # Iterate in reverse to handle dependencies
        for node in reversed(graph_like):
            removable = True
            for output in node.outputs:
                if output in graph_outputs or output.uses():
                    removable = False
                    break
            
            if removable:
                graph_like.remove(node, safe=True)
                count += 1
        
        return count
```

### Pattern 3: Common Subexpression Elimination

Eliminate duplicate computations.

```python
class CSEPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = self._eliminate_cse(model.graph)
        return ir.passes.PassResult(model, modified=modified)
    
    def _eliminate_cse(self, graph: ir.Graph) -> bool:
        modified = False
        # Map from (op_identifier, inputs, attributes) to node
        existing_nodes: dict[tuple, ir.Node] = {}
        
        for node in graph:
            # Skip non-deterministic ops
            if self._is_non_deterministic(node):
                continue
            
            # Create a hashable key for the node
            node_key = (
                node.op_identifier(),
                tuple(id(inp) for inp in node.inputs),
                tuple(sorted(node.attributes.items())),
            )
            
            if node_key in existing_nodes:
                # Found duplicate - replace with existing
                existing_node = existing_nodes[node_key]
                ir.convenience.replace_all_uses_with(
                    node.outputs,
                    existing_node.outputs
                )
                graph.remove(node, safe=True)
                modified = True
            else:
                existing_nodes[node_key] = node
        
        return modified
    
    def _is_non_deterministic(self, node: ir.Node) -> bool:
        """Check if node is non-deterministic."""
        non_deterministic_ops = frozenset({
            "RandomUniform", "RandomNormal",
            "RandomUniformLike", "RandomNormalLike",
            "Multinomial"
        })
        return node.op_type in non_deterministic_ops and node.domain == ""
```

### Pattern 4: Graph Normalization

Ensure graph is in a canonical form (e.g., topological sort, name fixing).

```python
class TopologicalSortPass(ir.passes.InPlacePass):
    """Sort nodes in topological order."""
    
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        original_nodes = list(model.graph)
        model.graph.sort()  # Built-in method
        sorted_nodes = list(model.graph)
        
        # Check if order changed
        modified = False
        for node, new_node in zip(original_nodes, sorted_nodes):
            if node is not new_node:
                modified = True
                break
        
        # Also sort functions
        for function in model.functions.values():
            function.sort()
        
        return ir.passes.PassResult(model, modified=modified)
```

### Pattern 5: Attribute/Metadata Manipulation

Modify node attributes or clear metadata.

```python
class ClearMetadataPass(ir.passes.InPlacePass):
    """Clear metadata and doc strings from the model."""
    
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        
        # Clear model metadata
        if model.doc_string or model.metadata_props:
            model.doc_string = ""
            model.metadata_props.clear()
            modified = True
        
        # Clear graph metadata
        modified |= self._clear_graph_metadata(model.graph)
        
        # Clear function metadata
        for function in model.functions.values():
            modified |= self._clear_graph_metadata(function)
        
        return ir.passes.PassResult(model, modified=modified)
    
    def _clear_graph_metadata(self, graph_like: ir.Graph | ir.Function) -> bool:
        modified = False
        
        if graph_like.doc_string:
            graph_like.doc_string = ""
            modified = True
        
        for node in ir.traversal.RecursiveGraphIterator(graph_like):
            if node.doc_string or node.metadata_props:
                node.doc_string = ""
                node.metadata_props.clear()
                modified = True
        
        return modified
```

## Testing Your Pass

### Basic Test Structure

```python
import onnx_ir as ir

def test_my_pass():
    # Create a test model
    model = create_test_model()
    
    # Apply the pass
    pass_instance = MyPass()
    result = pass_instance(model)
    
    # Verify the result
    assert result.modified == True
    assert len(result.model.graph) == expected_node_count
    
    # Verify specific transformations
    # ...
```

### Using Test Utilities

```python
# Compare graphs
from onnx_ir._graph_comparison import GraphComparer

expected_graph = create_expected_graph()
comparer = GraphComparer()
assert comparer.compare(result.model.graph, expected_graph)
```

## Common Pitfalls to Avoid

1. **Modifying while iterating**: Use list copy or reversed iteration when removing nodes
   ```python
   # Bad: Can cause iteration issues
   for node in graph:
       graph.remove(node)
   
   # Good: Iterate in reverse
   for node in reversed(graph):
       if should_remove(node):
           graph.remove(node, safe=True)
   ```

2. **Forgetting subgraphs**: Always use `RecursiveGraphIterator` or manually process subgraphs

3. **Not checking for None inputs**: Nodes can have optional None inputs
   ```python
   for input_value in node.inputs:
       if input_value is not None:  # Always check
           # Process input
           pass
   ```

4. **Modifying graph outputs incorrectly**: Be careful when replacing graph output values
   ```python
   # Update graph outputs properly
   if output_value.is_graph_output():
       # Find and update in graph.outputs
       for idx, graph_output in enumerate(graph.outputs):
           if graph_output is output_value:
               graph.outputs[idx] = new_value
   ```

5. **Not handling edge cases**: Check for empty inputs/outputs, graph boundaries

6. **Forgetting to process functions**: Many passes should also process `model.functions`

## Performance Considerations

1. **Use in-place passes** when possible (most efficient)
2. **Minimize graph traversals**: Combine multiple checks in one traversal
3. **Use frozenset for lookups**: When checking membership in graph inputs/outputs
   ```python
   graph_outputs = frozenset(graph.outputs)
   if value in graph_outputs:  # O(1) lookup
       pass
   ```
4. **Batch operations**: Remove multiple nodes in one traversal rather than multiple passes
5. **Early exit**: Return early if no modifications are needed

## Pass Composition

### Sequential Pass Execution

```python
# Combine multiple passes
passes = ir.passes.Sequential(
    RemoveUnusedNodesPass(),
    IdentityEliminationPass(),
    TopologicalSortPass(),
)

result = passes(model)
```

### Pass Manager with Iteration

```python
# Run passes multiple times until convergence
passes = ir.passes.PassManager(
    [
        CommonSubexpressionEliminationPass(),
        RemoveUnusedNodesPass(),
    ],
    steps=5,  # Maximum iterations
    early_stop=True,  # Stop if no changes
)

result = passes(model)
```

## Security and Error Handling

1. **Validate inputs**: Check node structure, input/output counts
2. **Handle exceptions gracefully**: Catch and log errors in schema lookups
3. **Use safe removal**: Always call `graph.remove(node, safe=True)`
4. **Avoid hardcoding assumptions**: Check opset versions, schema availability
5. **Log warnings**: When skipping nodes due to unexpected conditions

```python
try:
    schema = onnx.defs.get_schema(node.op_type, opset_version, domain=node.domain)
except Exception:
    logger.warning("Could not get schema for %s, skipping", node, exc_info=True)
    continue
```

## Import Conventions

```python
# Standard imports for pass files
from __future__ import annotations

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)
```

## Summary

When creating a new pass:

1. **Choose the right base class**: `InPlacePass` (most common) or `FunctionalPass`
2. **Implement `call` method**: Main transformation logic
3. **Return `PassResult`**: With model and modified flag
4. **Use `RecursiveGraphIterator`**: To process all nodes including subgraphs
5. **Process functions**: Don't forget `model.functions`
6. **Use convenience functions**: `replace_all_uses_with`, etc.
7. **Log appropriately**: Use debug/info logging levels
8. **Handle edge cases**: None inputs, graph boundaries, subgraphs
9. **Test thoroughly**: Write tests for various scenarios
10. **Document clearly**: Explain what the pass does and any parameters

## References

- Pass Infrastructure: `src/onnx_ir/passes/_pass_infra.py`
- Example Passes: `src/onnx_ir/passes/common/`
- Convenience Functions: `src/onnx_ir/convenience.py`
- Traversal Utilities: `src/onnx_ir/traversal.py`
