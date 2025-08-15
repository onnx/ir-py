# Symbolic Shape and Type Inference

This module provides symbolic shape and type inference for ONNX IR models, enabling compile-time analysis of tensor shapes and types with support for symbolic dimensions.

## Overview

The inference engine performs forward propagation through ONNX models to determine output shapes and types based on input specifications. It supports symbolic dimensions using SymPy expressions, allowing for dynamic shape analysis.

## Key Components

### Core Classes

- **`SymbolicInferenceEngine`**: Main orchestrator that processes models and applies inference
- **`NodeInferrer`**: Base class for operation-specific inference logic
- **`InferenceResult`**: Container for inference results with status and optional message
- **`InferenceStatus`**: Enum for inference operation status (SUCCESS, PARTIAL, MISSING_INFO, INVALID_NODE)

### Reconciliation Policies

The engine supports different strategies for handling conflicts between inferred and existing values:

- **`OVERWRITE`**: Always use inferred values
- **`IGNORE`**: Keep existing values if they exist
- **`RECONCILE`**: Merge inferred and existing values intelligently
- **`STRICT`**: Fail if inferred values don't match existing ones

### Inference Status System

The `InferenceResult` uses a status-based approach for granular error handling:

- **`SUCCESS`**: Complete inference successful with full shape/type information
- **`PARTIAL`**: Partial information available (e.g., type only, rank only)
- **`MISSING_INFO`**: Missing required input information (shapes, types)
- **`INVALID_NODE`**: Node is invalid or malformed

```python
# Example usage in inferrers
def infer(self, node: ir.Node) -> InferenceResult:
    if node.inputs[0].shape is None:
        return InferenceResult(
            status="missing_info",
            msg="Input shape is required"
        )

    # Partial inference - only type available
    if can_infer_type_only():
        return InferenceResult(
            values=[ir.Value(type=inferred_type)],
            status="partial",
            msg="Shape unavailable, type only"
        )

    # Full inference (status defaults to "success")
    return InferenceResult(values=[full_value])
```

## Architecture

```text
SymbolicInferenceEngine
├── NodeInferrer Registry (by op_type + domain)
├── Opset Version Matching
└── Reconciliation Logic

NodeInferrer Implementations
├── ElementwiseInferrer (unary operations)
├── BinaryInferrer (broadcasting operations)
└── Specialized Inferrers (50+ operations)
```

## Inferrer Selection

The engine selects the appropriate inferrer using a two-stage process:

1. **Registry Lookup**: Inferrers are registered by `(op_type, domain)` key
2. **Opset Matching**: Among matching inferrers, select those supporting the model's opset version

```python
# Example: For a Squeeze node with opset 14
# - Multiple Squeeze inferrers may be registered
# - Engine selects Squeeze13Inferrer (supports opset 13-23)
# - Ignores Squeeze12Inferrer (supports opset 1-12)
```

## Symbolic Dimensions

The system stores symbolic expressions in `ir.SymbolicDim` objects:

```python
class SymbolicDim:
    value: str | None      # String identifier (e.g., "N", "batch_size")
    expr: sympy.Expr | None  # SymPy expression for computed dimensions
```

Dimensions are accessed via `get_expr()` which converts to SymPy expressions:

- `SymbolicDim(value="N")` → `sympy.Symbol("N")`
- `SymbolicDim(expr=N*2)` → `N*2` (SymPy expression)
- Integer dimensions → `sympy.Integer(value)`

## NodeInferrer Design Decisions

### Base Class Structure
The `NodeInferrer` abstract base class enforces a consistent interface:

```python
class NodeInferrer(abc.ABC):
    def __init__(self, op_type: str, opsets: Collection[int], domain: str = ""):
        # Store operation metadata for registry matching

    @abc.abstractmethod
    def infer(self, node: ir.Node) -> InferenceResult:
        # Operation-specific inference logic
```

### Design Rationale

1. **Single Responsibility**: Each inferrer handles exactly one operation type
2. **Opset Awareness**: Inferrers declare supported ONNX opset versions for compatibility
3. **Domain Support**: Enables custom domains beyond standard ONNX operators
4. **Validation Decorators**: `@requires_non_none_inputs(n)` and `@requires_outputs(n)` provide consistent input validation
5. **Failure Handling**: Return `InferenceResult` with either `values` or `failure` for graceful error handling

### Inheritance Patterns

- **ElementwiseInferrer**: Template for unary operations that preserve input shape/type
- **BinaryInferrer**: Template for binary operations with broadcasting logic
- **Specialized Inferrers**: Custom logic for complex operations (Conv, Reshape, etc.)

## Usage

### Basic Usage

```python
from onnx_ir._shape_type_inference.factory import create_standard_inference_engine
from onnx_ir._shape_type_inference import ReconciliationPolicy

# Create engine with all standard operations
engine = create_standard_inference_engine(ReconciliationPolicy.RECONCILE)

# Perform inference on a model
engine.infer_model(model)
```

### Custom Engine

```python
from onnx_ir._shape_type_inference import SymbolicInferenceEngine
from onnx_ir._shape_type_inference.ops.matmul import MatMulInferrer
from onnx_ir._shape_type_inference.ops.standard_ops import BinaryInferrer

# Create custom engine with specific operations
inferrers = [
    MatMulInferrer(),
    BinaryInferrer("Add"),
    BinaryInferrer("Mul"),
]

engine = SymbolicInferenceEngine(inferrers, ReconciliationPolicy.STRICT)
```

## Opset Version Support

Each inferrer specifies supported ONNX opset versions to handle API changes:

```python
class Squeeze12Inferrer(NodeInferrer):
    def __init__(self):
        super().__init__("Squeeze", opsets=range(1, 13))

class Squeeze13Inferrer(NodeInferrer):
    def __init__(self):
        super().__init__("Squeeze", opsets=range(13, 24))
```

## Error Handling

The engine provides comprehensive error handling:

- **Validation Errors**: Invalid input/output counts, missing shapes
- **Type Mismatches**: Incompatible input types for binary operations
- **Inference Failures**: Operation-specific inference errors
- **Reconciliation Conflicts**: Value mismatches in strict mode

## Factory Functions

Pre-configured engines for common use cases:

- **`create_standard_inference_engine()`**: Full operation coverage (50+ ops)
- **`create_minimal_inference_engine()`**: Essential operations only

## Subgraphs and ONNX Functions

### Design Approach

#### Subgraph Pre-Processing Strategy

The engine uses a **subgraph-first** approach for cleaner separation of concerns:

1. **Pre-Processing Phase**: Before running node inference, detect and recursively process all subgraphs
2. **Bottom-Up Inference**: Subgraphs are fully inferred before their parent nodes
3. **Simplified Node Logic**: Control flow inferrers (If, Loop, Scan) can assume subgraph shapes are already available

```python
class SymbolicInferenceEngine:
    def _infer_node(self, node: ir.Node, model: ir.Model) -> None:
        # First: recursively infer any subgraphs
        for attr in node.attributes:
            if isinstance(attr.value, ir.Graph):
                self._infer_subgraph(attr.value, model)

        # Then: run node-specific inference with subgraphs already processed
        inferrer = self._find_inferrer(node, model)
        result = inferrer.infer(node)  # Subgraph shapes already available
```

#### ONNX Function Support

Functions are handled through **automatic expansion** without custom inferrer logic:

1. **Function Context**: Engine maintains intermediate value mappings during function execution
2. **Transparent Expansion**: Function calls are expanded inline and processed like regular subgraphs
3. **No Custom Logic**: Users don't implement function-specific inferrers - the engine handles it automatically

```python
class SymbolicInferenceEngine:
    def _infer_function_call(self, node: ir.Node, function: ir.Function) -> InferenceResult:
        # Create isolated context for function execution
        function_context = self._create_function_context(node.inputs, function)

        # Process function body as a subgraph
        for func_node in function.nodes:
            self._infer_node_in_context(func_node, function_context)

        # Map function outputs back to caller node
        return self._extract_function_outputs(function_context, function.outputs)
```

### Key Benefits

1. **Cleaner Separation**: Subgraph inference is handled by the engine, not individual inferrers
2. **Automatic Function Support**: No need to implement custom logic for each function
3. **Simplified Debugging**: Each phase (subgraphs → nodes) can be debugged independently
4. **Consistent Context**: Function calls maintain proper variable scoping and type consistency

## Extension Points

To add support for new operations:

1. Create a new inferrer class inheriting from `NodeInferrer`
2. Implement the `infer()` method with operation-specific logic
3. Register with the engine or add to factory functions

```python
class CustomOpInferrer(NodeInferrer):
    def __init__(self):
        super().__init__("CustomOp", opsets=range(1, 24), domain="custom_domain")

    def infer(self, node: ir.Node) -> InferenceResult:
        # Custom inference logic
        return InferenceResult(values=[result_value])
```
