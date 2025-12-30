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


class FunctionalizeTest(unittest.TestCase):
    """Test the functionalize function that converts passes to functional passes."""

    def test_functionalize_converts_in_place_pass_to_functional_pass(self):
        """Test that functionalize converts an in-place pass to a functional pass."""

        class TestInPlacePass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # Modify the model in place
                model.ir_version = 42
                return _pass_infra.PassResult(model=model, modified=True)

        original_pass = TestInPlacePass()
        functional_pass = _pass_infra.functionalize(original_pass)

        # Verify the returned pass is a FunctionalPass
        self.assertIsInstance(functional_pass, _pass_infra.FunctionalPass)

        # Verify it has functional pass properties
        self.assertFalse(functional_pass.in_place)
        self.assertFalse(functional_pass.changes_input)

    def test_functionalize_does_not_modify_input_model(self):
        """Test that the functionalized pass does not modify the input model."""

        class TestInPlacePass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # Modify the model in place
                model.ir_version = 42
                return _pass_infra.PassResult(model=model, modified=True)

        original_pass = TestInPlacePass()
        functional_pass = _pass_infra.functionalize(original_pass)

        # Create an input model
        input_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        original_ir_version = input_model.ir_version

        # Run the functionalized pass
        result = functional_pass(input_model)

        # Verify the input model was not modified
        self.assertEqual(input_model.ir_version, original_ir_version)
        # Verify the result model was modified
        self.assertEqual(result.model.ir_version, 42)
        # Verify they are different objects
        self.assertIsNot(result.model, input_model)

    def test_functionalize_returns_new_model_object(self):
        """Test that the functionalized pass returns a new model object."""

        class TestInPlacePass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model=model, modified=False)

        original_pass = TestInPlacePass()
        functional_pass = _pass_infra.functionalize(original_pass)

        input_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result = functional_pass(input_model)

        # Verify a new model object is returned
        self.assertIsNot(result.model, input_model)

    def test_functionalize_preserves_modified_flag(self):
        """Test that the functionalized pass preserves the modified flag from the inner pass."""

        class TestInPlacePassModified(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                model.ir_version = 42
                return _pass_infra.PassResult(model=model, modified=True)

        class TestInPlacePassNotModified(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model=model, modified=False)

        # Test when the inner pass reports modified=True
        functional_pass_modified = _pass_infra.functionalize(TestInPlacePassModified())
        input_model_1 = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result_1 = functional_pass_modified(input_model_1)
        self.assertTrue(result_1.modified)

        # Test when the inner pass reports modified=False
        functional_pass_not_modified = _pass_infra.functionalize(TestInPlacePassNotModified())
        input_model_2 = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result_2 = functional_pass_not_modified(input_model_2)
        self.assertFalse(result_2.modified)

    def test_functionalize_works_with_functional_pass(self):
        """Test that functionalize can be applied to already functional passes."""

        class TestFunctionalPass(_pass_infra.FunctionalPass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # Create a new model
                new_model = ir.Model(
                    graph=ir.Graph([], [], nodes=[]), ir_version=model.ir_version + 1
                )
                return _pass_infra.PassResult(model=new_model, modified=True)

        original_pass = TestFunctionalPass()
        functional_pass = _pass_infra.functionalize(original_pass)

        input_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result = functional_pass(input_model)

        # Verify the input model was not modified
        self.assertEqual(input_model.ir_version, 10)
        # Verify the result model has the expected value
        self.assertEqual(result.model.ir_version, 11)
        # Verify they are different objects
        self.assertIsNot(result.model, input_model)

    def test_functionalize_passes_through_preconditions(self):
        """Test that functionalize preserves the inner pass's preconditions."""

        class TestInPlacePassWithPrecondition(_pass_infra.InPlacePass):
            def requires(self, model: ir.Model) -> None:
                if model.ir_version < 10:
                    raise _pass_infra.PreconditionError("IR version must be >= 10")

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model=model, modified=False)

        original_pass = TestInPlacePassWithPrecondition()
        functional_pass = _pass_infra.functionalize(original_pass)

        # Model that fails precondition
        bad_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=9)
        with self.assertRaises(_pass_infra.PreconditionError):
            functional_pass(bad_model)

        # Model that satisfies precondition should work
        good_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result = functional_pass(good_model)
        self.assertIsNotNone(result)

    def test_functionalize_passes_through_postconditions(self):
        """Test that functionalize preserves the inner pass's postconditions."""

        class TestInPlacePassWithPostcondition(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                model.ir_version = 5
                return _pass_infra.PassResult(model=model, modified=True)

            def ensures(self, model: ir.Model) -> None:
                if model.ir_version < 10:
                    raise _pass_infra.PostconditionError("IR version must be >= 10")

        original_pass = TestInPlacePassWithPostcondition()
        functional_pass = _pass_infra.functionalize(original_pass)

        input_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        with self.assertRaises(_pass_infra.PostconditionError):
            functional_pass(input_model)

    def test_functionalize_with_destructive_pass(self):
        """Test that functionalize works with destructive passes."""

        class TestDestructivePass(_pass_infra.PassBase):
            @property
            def in_place(self) -> bool:
                return False  # Not in-place

            @property
            def changes_input(self) -> bool:
                return True  # But changes input (destructive)

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # Modify the input model
                model.ir_version = 42
                # Return a new model
                new_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=888)
                return _pass_infra.PassResult(model=new_model, modified=True)

        original_pass = TestDestructivePass()
        self.assertTrue(original_pass.destructive)

        functional_pass = _pass_infra.functionalize(original_pass)

        input_model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        original_ir_version = input_model.ir_version

        result = functional_pass(input_model)

        # Verify the input model was not modified (functionalize protects it)
        self.assertEqual(input_model.ir_version, original_ir_version)
        # Verify the result model has the expected value
        self.assertEqual(result.model.ir_version, 888)

    def test_functionalize_clones_complex_model(self):
        """Test that functionalize properly clones complex models."""

        class TestInPlacePass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # Modify graph inputs
                if model.graph.inputs:
                    model.graph.inputs[0].name = "modified_name"
                return _pass_infra.PassResult(model=model, modified=True)

        # Create a model with inputs
        input_value = ir.Value(name="original_name", shape=ir.Shape([1, 2, 3]))
        graph = ir.Graph(inputs=[input_value], outputs=[], nodes=[])
        input_model = ir.Model(graph=graph, ir_version=10)

        functional_pass = _pass_infra.functionalize(TestInPlacePass())
        result = functional_pass(input_model)

        # Verify the input model's graph inputs were not modified
        self.assertEqual(input_model.graph.inputs[0].name, "original_name")
        # Verify the result model's graph inputs were modified
        self.assertEqual(result.model.graph.inputs[0].name, "modified_name")


if __name__ == "__main__":
    unittest.main()
