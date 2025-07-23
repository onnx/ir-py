# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes import _pass_infra


class PassBaseTest(unittest.TestCase):
    def test_pass_results_can_be_used_as_pass_input(self):
        class TestPass(_pass_infra.PassBase):
            @property
            def in_place(self) -> bool:
                return True

            @property
            def changes_input(self) -> bool:
                return False

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # This is a no-op pass
                return _pass_infra.PassResult(model=model, modified=False)

        pass_ = TestPass()
        model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result = pass_(model)
        self.assertIsInstance(result, _pass_infra.PassResult)
        # pass can take the result of another pass as input
        result_1 = pass_(result)
        # It can also take the model as input
        result_2 = pass_(result.model)
        self.assertIs(result_1.model, result_2.model)


class PostconditionTest(unittest.TestCase):
    """Test that postconditions are checked on the result model, not the input model."""

    def test_ensures_called_with_result_model_not_input_model(self):
        """Test that ensures() is called with result.model, not the input model."""

        class TestPass(_pass_infra.PassBase):
            def __init__(self):
                self.ensures_called_with = None

            @property
            def in_place(self) -> bool:
                return False  # Not in-place to create a new model

            @property
            def changes_input(self) -> bool:
                return False

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # Create a new model (different object)
                new_model = ir.Model(
                    graph=ir.Graph([], [], nodes=[]), ir_version=model.ir_version
                )
                return _pass_infra.PassResult(model=new_model, modified=True)

            def ensures(self, model: ir.Model) -> None:
                # Record which model ensures was called with
                self.ensures_called_with = model

        pass_ = TestPass()
        input_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result = pass_(input_model)

        # Verify that ensures was called with the result model, not the input model
        self.assertIs(pass_.ensures_called_with, result.model)
        self.assertIsNot(pass_.ensures_called_with, input_model)

    def test_ensures_called_with_result_model_in_place_pass(self):
        """Test that ensures() is called with result.model for in-place passes."""

        class TestInPlacePass(_pass_infra.InPlacePass):
            def __init__(self):
                self.ensures_called_with = None

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # In-place pass returns the same model
                return _pass_infra.PassResult(model=model, modified=True)

            def ensures(self, model: ir.Model) -> None:
                # Record which model ensures was called with
                self.ensures_called_with = model

        pass_ = TestInPlacePass()
        input_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result = pass_(input_model)

        # For in-place passes, result.model should be the same as input_model
        self.assertIs(result.model, input_model)
        # Verify that ensures was called with the result model (which is the same as input)
        self.assertIs(pass_.ensures_called_with, result.model)
        self.assertIs(pass_.ensures_called_with, input_model)

    def test_ensures_called_with_result_model_functional_pass(self):
        """Test that ensures() is called with result.model for functional passes."""

        class TestPass(_pass_infra.FunctionalPass):
            def __init__(self):
                self.ensures_called_with = None

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # Create a new model (different object)
                new_model = ir.Model(
                    graph=ir.Graph([], [], nodes=[]), ir_version=model.ir_version
                )
                return _pass_infra.PassResult(model=new_model, modified=True)

            def ensures(self, model: ir.Model) -> None:
                # Record which model ensures was called with
                self.ensures_called_with = model

        pass_ = TestPass()
        input_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result = pass_(input_model)

        # Verify that ensures was called with the result model, not the input model
        self.assertIs(pass_.ensures_called_with, result.model)
        self.assertIsNot(pass_.ensures_called_with, input_model)

    def test_postcondition_error_raised_when_ensures_fails(self):
        """Test that PostconditionError is raised when ensures() raises an exception."""

        class TestPass(_pass_infra.PassBase):
            @property
            def in_place(self) -> bool:
                return True

            @property
            def changes_input(self) -> bool:
                return True

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model=model, modified=False)

            def ensures(self, model: ir.Model) -> None:
                # Simulate a postcondition failure
                raise ValueError("Postcondition failed")

        pass_ = TestPass()
        model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)

        with self.assertRaisesRegex(
            ir.passes.PostconditionError, "Post-condition for pass 'TestPass' failed"
        ) as cm:
            pass_(model)

        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_postcondition_error_raised_when_ensures_raises_postcondition_error(self):
        """Test that PostconditionError is re-raised when ensures() raises PostconditionError."""

        class TestPass(_pass_infra.PassBase):
            @property
            def in_place(self) -> bool:
                return True

            @property
            def changes_input(self) -> bool:
                return True

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model=model, modified=False)

            def ensures(self, model: ir.Model) -> None:
                # Directly raise PostconditionError
                raise ir.passes.PostconditionError("Direct postcondition error")

        pass_ = TestPass()
        model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)

        with self.assertRaisesRegex(
            ir.passes.PostconditionError, "Direct postcondition error"
        ):
            pass_(model)


if __name__ == "__main__":
    unittest.main()
