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


def _empty_model(ir_version: int = 10) -> ir.Model:
    return ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=ir_version)


class PassBaseErrorHandlingTest(unittest.TestCase):
    def test_precondition_wraps_generic_exception(self):
        """Non-PreconditionError from requires() is wrapped in PreconditionError."""

        class BadPrecondition(_pass_infra.InPlacePass):
            def requires(self, model: ir.Model) -> None:
                raise RuntimeError("something went wrong")

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model, modified=False)

        with self.assertRaises(_pass_infra.PreconditionError) as cm:
            BadPrecondition()(_empty_model())
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)

    def test_precondition_reraises_precondition_error(self):
        """PreconditionError from requires() is re-raised directly."""

        class DirectPrecondition(_pass_infra.InPlacePass):
            def requires(self, model: ir.Model) -> None:
                raise _pass_infra.PreconditionError("direct")

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model, modified=False)

        with self.assertRaisesRegex(_pass_infra.PreconditionError, "direct"):
            DirectPrecondition()(_empty_model())

    def test_call_returning_non_pass_result_raises_post_or_type_error(self):
        """call() returning something other than PassResult raises error.

        PostconditionError is raised because ensures is called on result.model
        which fails first, or TypeError.
        """

        class BadReturn(_pass_infra.InPlacePass):
            def call(self, model: ir.Model):
                return model  # Wrong: should return PassResult

        with self.assertRaises((_pass_infra.PostconditionError, TypeError)):
            BadReturn()(_empty_model())

    def test_in_place_pass_returning_different_model_raises(self):
        """In-place pass returning a different model object raises PassError."""

        class BadInPlace(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                new_model = _empty_model()
                return _pass_infra.PassResult(new_model, modified=True)

        with self.assertRaises(_pass_infra.PassError):
            BadInPlace()(_empty_model())

    def test_not_in_place_pass_returning_same_model_raises(self):
        """Not-in-place pass returning the same model object raises PassError."""

        class BadFunctional(_pass_infra.FunctionalPass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model, modified=False)

        with self.assertRaises(_pass_infra.PassError):
            BadFunctional()(_empty_model())

    def test_destructive_property(self):
        """A pass that is not in_place but changes_input is destructive."""

        class Destructive(_pass_infra.PassBase):
            @property
            def in_place(self) -> bool:
                return False

            @property
            def changes_input(self) -> bool:
                return True

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(_empty_model(), modified=True)

        p = Destructive()
        self.assertTrue(p.destructive)

    def test_non_destructive_in_place(self):
        class ConcreteInPlace(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model, modified=False)

        p = ConcreteInPlace()
        self.assertFalse(p.destructive)


class SequentialTest(unittest.TestCase):
    def test_sequential_empty_raises(self):
        with self.assertRaises(ValueError):
            _pass_infra.Sequential()

    def test_sequential_runs_passes_in_order(self):
        call_order = []

        class Pass1(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                call_order.append(1)
                return _pass_infra.PassResult(model, modified=True)

        class Pass2(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                call_order.append(2)
                return _pass_infra.PassResult(model, modified=False)

        seq = _pass_infra.Sequential(Pass1(), Pass2())
        result = seq(_empty_model())
        self.assertEqual(call_order, [1, 2])
        self.assertTrue(result.modified)

    def test_sequential_propagates_errors(self):
        """Error in a pass within Sequential is wrapped in PassError."""

        class GoodPass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model, modified=False)

        class BadPass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                raise RuntimeError("pass failed")

        seq = _pass_infra.Sequential(GoodPass(), BadPass())
        with self.assertRaises(_pass_infra.PassError):
            seq(_empty_model())

    def test_sequential_in_place_when_all_in_place(self):
        class P(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model, modified=False)

        seq = _pass_infra.Sequential(P(), P())
        self.assertTrue(seq.in_place)
        self.assertTrue(seq.changes_input)

    def test_sequential_not_in_place_when_mixed(self):
        class InP(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model, modified=False)

        class FunP(_pass_infra.FunctionalPass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(_empty_model(), modified=False)

        seq = _pass_infra.Sequential(InP(), FunP())
        self.assertFalse(seq.in_place)

    def test_sequential_changes_input_follows_first_pass(self):
        class FunP(_pass_infra.FunctionalPass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(_empty_model(), modified=False)

        class InP(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                return _pass_infra.PassResult(model, modified=False)

        # Functional first => doesn't change input
        seq = _pass_infra.Sequential(FunP(), InP())
        self.assertFalse(seq.changes_input)


class PassManagerTest(unittest.TestCase):
    def test_pass_manager_runs_multiple_steps(self):
        call_count = 0

        class CountPass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                nonlocal call_count
                call_count += 1
                return _pass_infra.PassResult(model, modified=True)

        pm = _pass_infra.PassManager([CountPass()], steps=3, early_stop=False)
        pm(_empty_model())
        self.assertEqual(call_count, 3)

    def test_pass_manager_early_stop_when_no_modification(self):
        call_count = 0

        class OncePass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                nonlocal call_count
                call_count += 1
                # Only modify on first call
                return _pass_infra.PassResult(model, modified=(call_count == 1))

        pm = _pass_infra.PassManager([OncePass()], steps=10, early_stop=True)
        result = pm(_empty_model())
        self.assertEqual(
            call_count, 2
        )  # Ran twice: first modified, second not modified => stop
        self.assertTrue(result.modified)

    def test_pass_manager_wraps_error_with_step_info(self):
        class FailPass(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                raise RuntimeError("boom")

        pm = _pass_infra.PassManager([FailPass()], steps=2)
        with self.assertRaises(_pass_infra.PassError) as cm:
            pm(_empty_model())
        self.assertIn("step", str(cm.exception).lower())

    def test_pass_manager_overall_modified_is_true_if_any_step_modifies(self):
        step = 0

        class SometimesModify(_pass_infra.InPlacePass):
            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                nonlocal step
                step += 1
                return _pass_infra.PassResult(model, modified=(step == 1))

        pm = _pass_infra.PassManager([SometimesModify()], steps=3, early_stop=False)
        result = pm(_empty_model())
        self.assertTrue(result.modified)


if __name__ == "__main__":
    unittest.main()
