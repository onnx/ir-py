# onnx_ir.schemas

```{eval-rst}
.. automodule:: onnx_ir.schemas
.. currentmodule:: onnx_ir.schemas
```

The `onnx_ir.schemas` module provides classes for representing operator signatures and their parameters. These classes are useful for introspecting operator definitions and validating operator usage.

## OpSignature

The `OpSignature` class represents the schema for an ONNX operator, including its inputs, outputs, and attributes.

```{eval-rst}
.. autoclass:: OpSignature
   :members:
   :undoc-members:
```

## Parameter

The `Parameter` class represents a formal input parameter of an operator.

```{eval-rst}
.. autoclass:: Parameter
   :members:
   :undoc-members:
```

## AttributeParameter

The `AttributeParameter` class represents an attribute parameter in the operator signature.

```{eval-rst}
.. autoclass:: AttributeParameter
   :members:
   :undoc-members:
```

## TypeConstraintParam

The `TypeConstraintParam` class represents type constraints for parameters, specifying which types are allowed.

```{eval-rst}
.. autoclass:: TypeConstraintParam
   :members:
   :undoc-members:
```
