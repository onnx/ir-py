"""Factory functions for creating inference engines with standard inferrers."""

from __future__ import annotations

from onnx_ir._shape_type_inference._engine import ReconciliationPolicy, SymbolicInferenceEngine
from onnx_ir._shape_type_inference.ops.concat import ConcatInferrer
from onnx_ir._shape_type_inference.ops.constant import ConstantInferrer
from onnx_ir._shape_type_inference.ops.matmul import MatMulInferrer
from onnx_ir._shape_type_inference.ops.reshape import ReshapeInferrer
from onnx_ir._shape_type_inference.ops.squeeze import Squeeze12Inferrer, Squeeze13Inferrer
from onnx_ir._shape_type_inference.ops.standard_ops import BinaryInferrer, ElementwiseInferrer
from onnx_ir._shape_type_inference.ops.transpose import TransposeInferrer
from onnx_ir._shape_type_inference.ops.unsqueeze import (
    Unsqueeze12Inferrer,
    Unsqueeze13Inferrer,
)


def create_standard_inference_engine(
    reconciliation_policy: ReconciliationPolicy = ReconciliationPolicy.RECONCILE,
) -> SymbolicInferenceEngine:
    """Create a SymbolicInferenceEngine with all standard operation inferrers.

    Args:
        reconciliation_policy: Policy for handling conflicts between inferred and existing values.

    Returns:
        A configured SymbolicInferenceEngine.
    """
    inferrers = []

    # Core tensor operations
    inferrers.extend(
        [
            ConstantInferrer(),
            ReshapeInferrer(),
            TransposeInferrer(),
            # Squeeze/Unsqueeze with opset versions
            Squeeze12Inferrer(),
            Squeeze13Inferrer(),
            Unsqueeze12Inferrer(),
            Unsqueeze13Inferrer(),
        ]
    )

    # Tensor manipulation
    inferrers.extend(
        [
            # GatherInferrer(),
            # GatherElementsInferrer(),
            # GatherNDInferrer(),
            # ScatterElementsInferrer(),
            # ExpandInferrer(),
            # SliceInferrer(),
            # SplitInferrer(),
            ConcatInferrer(),
            # PadInferrer(),
            # TileInferrer(),
            # WhereInferrer(),
            # OneHotInferrer(),
            # CompressInferrer(),
        ]
    )

    # Mathematical operations
    inferrers.extend(
        [
            MatMulInferrer(),
            # EinsumInferrer(),
            # ReduceSumInferrer(),
            # ReduceProdInferrer(),
        ]
    )

    # Generation operations
    inferrers.extend(
        [
            # RangeInferrer(),
            # ConstantOfShapeInferrer(),
            # NonZeroInferrer(),
        ]
    )

    # Pooling and convolution
    inferrers.extend(
        [
            # ConvInferrer(),
            # AveragePoolInferrer(),
            # MaxPoolInferrer(),
            # BatchNormalizationInferrer(),
        ]
    )

    # Sequence operations
    inferrers.extend(
        [
            # ConcatFromSequenceInferrer(),
            # SplitToSequenceInferrer(),
            # SequenceAtInferrer(),
            # SequenceInsertInferrer(),
        ]
    )

    # Control flow
    inferrers.extend(
        [
            # IfInferrer(),
            # LoopInferrer(),
            # ScanInferrer(),
        ]
    )

    # ML-specific operations
    inferrers.extend(
        [
            # TopKInferrer(),
            # NonMaxSuppressionInferrer(),
            # SoftmaxCrossEntropyLossInferrer(),
            # GroupNormInferrer(),
            # GeluInferrer(),
        ]
    )

    # Utility operations
    inferrers.extend(
        [
            # ArrayFeatureExtractorInferrer(),
            # CategoryMapperInferrer(),
            # ZipMapInferrer(),
            # CumSumInferrer(),
            # ResizeInferrer(),
        ]
    )

    # Elementwise operations (covers many unary ops)
    elementwise_ops = [
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Ceil",
        "Cos",
        "Cosh",
        "Erf",
        "Exp",
        "Floor",
        "Log",
        "Neg",
        "Reciprocal",
        "Relu",
        "Round",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Sqrt",
        "Tan",
        "Tanh",
        "Identity",
        "IsInf",
        "IsNaN",
    ]
    for op_type in elementwise_ops:
        inferrers.append(ElementwiseInferrer(op_type))

    # Binary operations (covers broadcasting ops)
    binary_ops = [
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Max",
        "Min",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "And",
        "Or",
        "Xor",
    ]
    for op_type in binary_ops:
        inferrers.append(BinaryInferrer(op_type))

    return SymbolicInferenceEngine(inferrers, reconciliation_policy)


def create_minimal_inference_engine(
    reconciliation_policy: ReconciliationPolicy = ReconciliationPolicy.RECONCILE,
) -> SymbolicInferenceEngine:
    """Create a minimal SymbolicInferenceEngine with only essential inferrers.

    Args:
        reconciliation_policy: Policy for handling conflicts between inferred and existing values.

    Returns:
        A minimal SymbolicInferenceEngine.
    """
    inferrers = [
        # Core essentials
        ConstantInferrer(),
        ReshapeInferrer(),
        TransposeInferrer(),
        MatMulInferrer(),
        ConcatInferrer(),
        # Basic elementwise and binary
        ElementwiseInferrer("Identity"),
        BinaryInferrer("Add"),
        BinaryInferrer("Mul"),
    ]

    return SymbolicInferenceEngine(inferrers, reconciliation_policy)
