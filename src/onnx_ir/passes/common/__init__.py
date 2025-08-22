# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "AddInitializersToInputsPass",
    "CheckerPass",
    "ClearMetadataAndDocStringPass",
    "CommonSubexpressionEliminationPass",
    "DeduplicateHashedInitializersPass",
    "DeduplicateInitializersPass",
    "IdentityEliminationPass",
    "InlinePass",
    "LiftConstantsToInitializersPass",
    "LiftSubgraphInitializersToMainGraphPass",
    "NameFixPass",
    "RemoveInitializersFromInputsPass",
    "RemoveUnusedFunctionsPass",
    "RemoveUnusedNodesPass",
    "RemoveUnusedOpsetsPass",
    "ShapeInferencePass",
    "TopologicalSortPass",
]

from onnx_ir.passes.common.clear_metadata_and_docstring import (
    ClearMetadataAndDocStringPass,
)
from onnx_ir.passes.common.common_subexpression_elimination import (
    CommonSubexpressionEliminationPass,
)
from onnx_ir.passes.common.constant_manipulation import (
    AddInitializersToInputsPass,
    LiftConstantsToInitializersPass,
    LiftSubgraphInitializersToMainGraphPass,
    RemoveInitializersFromInputsPass,
)
from onnx_ir.passes.common.identity_elimination import (
    IdentityEliminationPass,
)
from onnx_ir.passes.common.initializer_deduplication import (
    DeduplicateHashedInitializersPass,
    DeduplicateInitializersPass,
)
from onnx_ir.passes.common.inliner import InlinePass
from onnx_ir.passes.common.naming import NameFixPass
from onnx_ir.passes.common.onnx_checker import CheckerPass
from onnx_ir.passes.common.shape_inference import ShapeInferencePass
from onnx_ir.passes.common.topological_sort import TopologicalSortPass
from onnx_ir.passes.common.unused_removal import (
    RemoveUnusedFunctionsPass,
    RemoveUnusedNodesPass,
    RemoveUnusedOpsetsPass,
)
