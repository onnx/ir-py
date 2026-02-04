# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the symbolic expression parser."""

from __future__ import annotations

import unittest

import sympy

from onnx_ir._symbolic_shapes import parse_symbolic_expression


class ParseSymbolicExpressionTest(unittest.TestCase):
    """Tests for the parse_symbolic_expression function."""

    def test_simple_identifier(self):
        """Test parsing a simple identifier."""
        result = parse_symbolic_expression("batch")
        self.assertEqual(result, sympy.Symbol("batch", integer=True, positive=True))

    def test_identifier_with_underscore(self):
        """Test parsing an identifier with underscore."""
        result = parse_symbolic_expression("seq_len")
        self.assertEqual(result, sympy.Symbol("seq_len", integer=True, positive=True))

    def test_identifier_with_number(self):
        """Test parsing an identifier with number."""
        result = parse_symbolic_expression("dim0")
        self.assertEqual(result, sympy.Symbol("dim0", integer=True, positive=True))

    def test_integer_literal(self):
        """Test parsing an integer literal."""
        result = parse_symbolic_expression("42")
        self.assertEqual(result, sympy.Integer(42))

    def test_addition(self):
        """Test parsing addition."""
        result = parse_symbolic_expression("N + 1")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, N + 1)

    def test_subtraction(self):
        """Test parsing subtraction."""
        result = parse_symbolic_expression("N - 1")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, N - 1)

    def test_multiplication(self):
        """Test parsing multiplication."""
        result = parse_symbolic_expression("2 * N")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, 2 * N)

    def test_division(self):
        """Test parsing division."""
        result = parse_symbolic_expression("N / 2")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, N / 2)

    def test_floor_division(self):
        """Test parsing floor division."""
        result = parse_symbolic_expression("N // 2")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, sympy.floor(N / 2))

    def test_power(self):
        """Test parsing exponentiation."""
        result = parse_symbolic_expression("N ** 2")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, N**2)

    def test_power_right_associative(self):
        """Test that power is right-associative (2**3**2 = 2^(3^2) = 512)."""
        result = parse_symbolic_expression("2 ** 3 ** 2")
        self.assertEqual(result, sympy.Integer(512))

    def test_unary_minus(self):
        """Test parsing unary minus."""
        result = parse_symbolic_expression("-N")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, -N)

    def test_unary_minus_in_expression(self):
        """Test parsing unary minus in an expression."""
        result = parse_symbolic_expression("a + -b")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        self.assertEqual(result, a - b)

    def test_parentheses(self):
        """Test parsing parenthesized expression."""
        result = parse_symbolic_expression("(a + b) * c")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        c = sympy.Symbol("c", integer=True, positive=True)
        self.assertEqual(result, (a + b) * c)

    def test_nested_parentheses(self):
        """Test parsing nested parentheses."""
        result = parse_symbolic_expression("((a + b))")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        self.assertEqual(result, a + b)

    def test_operator_precedence_mul_over_add(self):
        """Test that multiplication has higher precedence than addition."""
        result = parse_symbolic_expression("a + b * c")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        c = sympy.Symbol("c", integer=True, positive=True)
        self.assertEqual(result, a + b * c)

    def test_operator_precedence_power_over_mul(self):
        """Test that power has higher precedence than multiplication."""
        result = parse_symbolic_expression("a * b ** 2")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        self.assertEqual(result, a * b**2)

    def test_complex_expression(self):
        """Test parsing a complex expression."""
        result = parse_symbolic_expression("a * b + c // 2")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        c = sympy.Symbol("c", integer=True, positive=True)
        self.assertEqual(result, a * b + sympy.floor(c / 2))

    def test_function_max_two_args(self):
        """Test parsing max function with two arguments."""
        result = parse_symbolic_expression("max(a, b)")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        self.assertEqual(result, sympy.Max(a, b))

    def test_function_max_three_args(self):
        """Test parsing max function with three arguments."""
        result = parse_symbolic_expression("max(a, b, c)")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        c = sympy.Symbol("c", integer=True, positive=True)
        self.assertEqual(result, sympy.Max(a, b, c))

    def test_function_min_two_args(self):
        """Test parsing min function with two arguments."""
        result = parse_symbolic_expression("min(a, b)")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        self.assertEqual(result, sympy.Min(a, b))

    def test_function_min_three_args(self):
        """Test parsing min function with three arguments."""
        result = parse_symbolic_expression("min(a, b, c)")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        c = sympy.Symbol("c", integer=True, positive=True)
        self.assertEqual(result, sympy.Min(a, b, c))

    def test_function_floor(self):
        """Test parsing floor function."""
        result = parse_symbolic_expression("floor(N / 2)")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, sympy.floor(N / 2))

    def test_function_sqrt(self):
        """Test parsing sqrt function."""
        result = parse_symbolic_expression("sqrt(N)")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, sympy.sqrt(N))

    def test_nested_function_calls(self):
        """Test parsing nested function calls."""
        result = parse_symbolic_expression("max(floor(a / 2), b)")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        self.assertEqual(result, sympy.Max(sympy.floor(a / 2), b))

    def test_function_with_expression_args(self):
        """Test parsing function with expression arguments."""
        result = parse_symbolic_expression("max(a + 1, b - 1)")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        self.assertEqual(result, sympy.Max(a + 1, b - 1))

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        result1 = parse_symbolic_expression("a+b")
        result2 = parse_symbolic_expression("a + b")
        result3 = parse_symbolic_expression("  a  +  b  ")
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)

    def test_multiple_same_variables(self):
        """Test expression with same variable multiple times."""
        result = parse_symbolic_expression("N + N")
        N = sympy.Symbol("N", integer=True, positive=True)
        self.assertEqual(result, 2 * N)


class ParseSymbolicExpressionErrorTest(unittest.TestCase):
    """Tests for error handling in parse_symbolic_expression."""

    def test_unknown_function_raises(self):
        """Test that unknown function raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("unknown_func(a)")
        self.assertIn("Unknown function", str(context.exception))
        self.assertIn("unknown_func", str(context.exception))

    def test_unexpected_character_raises(self):
        """Test that unexpected character raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("a @ b")
        self.assertIn("Unexpected character", str(context.exception))

    def test_unclosed_parenthesis_raises(self):
        """Test that unclosed parenthesis raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("(a + b")
        self.assertIn("Expected", str(context.exception))

    def test_extra_closing_parenthesis_raises(self):
        """Test that extra closing parenthesis raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("a + b)")
        self.assertIn("Unexpected token", str(context.exception))

    def test_empty_expression_raises(self):
        """Test that empty expression raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("")
        self.assertIn("Unexpected end of expression", str(context.exception))

    def test_whitespace_only_raises(self):
        """Test that whitespace-only expression raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("   ")
        self.assertIn("Unexpected end of expression", str(context.exception))

    def test_trailing_operator_raises(self):
        """Test that trailing operator raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("a +")
        self.assertIn("Unexpected end of expression", str(context.exception))

    def test_double_operator_raises(self):
        """Test that double operator raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("a + + b")
        # The second + is parsed as unary +, which is not supported
        self.assertIn("Unexpected", str(context.exception))

    def test_function_with_no_arguments(self):
        """Test that function with no arguments works (sympy allows it)."""
        # sympy.Max() with no arguments returns -oo (negative infinity)
        # This is valid sympy behavior, so we don't raise an error
        result = parse_symbolic_expression("max()")
        self.assertEqual(result, sympy.Max())


class ParseSymbolicExpressionSecurityTest(unittest.TestCase):
    """Security tests to ensure the parser is safe from code injection."""

    def test_code_is_not_executed(self):
        """Test that code in expressions is not executed."""
        # Create a mutable object to detect if code runs
        execution_tracker = {"executed": False}

        # Define a function that would set the flag if called
        def malicious_function():
            execution_tracker["executed"] = True
            return 1

        # Try various ways to inject code - none should execute
        malicious_expressions = [
            "malicious_function()",
            "__import__('os').system('echo pwned')",
            "exec('execution_tracker[\"executed\"] = True')",
            "eval('1+1')",
            "(lambda: execution_tracker.__setitem__('executed', True))()",
        ]

        for expr in malicious_expressions:
            execution_tracker["executed"] = False
            with self.assertRaises(ValueError):
                parse_symbolic_expression(expr)
            self.assertFalse(
                execution_tracker["executed"],
                f"Code was executed for expression: {expr}",
            )

    def test_rejects_dunder_import(self):
        """Test that __import__ is rejected."""
        with self.assertRaises(ValueError):
            parse_symbolic_expression('__import__("os")')

    def test_rejects_eval(self):
        """Test that eval is rejected as unknown function."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression('eval("1+1")')
        self.assertIn("Unknown function", str(context.exception))

    def test_rejects_exec(self):
        """Test that exec is rejected as unknown function."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression('exec("print(1)")')
        self.assertIn("Unknown function", str(context.exception))

    def test_rejects_open(self):
        """Test that open is rejected as unknown function."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression('open("file.txt")')
        self.assertIn("Unknown function", str(context.exception))

    def test_rejects_getattr(self):
        """Test that getattr is rejected as unknown function."""
        with self.assertRaises(ValueError) as context:
            parse_symbolic_expression("getattr(a, b)")
        self.assertIn("Unknown function", str(context.exception))

    def test_rejects_lambda(self):
        """Test that lambda-like syntax is rejected."""
        with self.assertRaises(ValueError):
            parse_symbolic_expression("lambda: 1")

    def test_rejects_semicolon(self):
        """Test that semicolon (statement separator) is rejected."""
        with self.assertRaises(ValueError):
            parse_symbolic_expression("a; b")

    def test_rejects_equals(self):
        """Test that equals sign is rejected."""
        with self.assertRaises(ValueError):
            parse_symbolic_expression("a = 1")

    def test_rejects_brackets(self):
        """Test that square brackets are rejected."""
        with self.assertRaises(ValueError):
            parse_symbolic_expression("a[0]")

    def test_rejects_braces(self):
        """Test that curly braces are rejected."""
        with self.assertRaises(ValueError):
            parse_symbolic_expression("{a: 1}")

    def test_rejects_dot_access(self):
        """Test that dot access is rejected."""
        with self.assertRaises(ValueError):
            parse_symbolic_expression("a.b")

    def test_rejects_string_literal(self):
        """Test that string literals are rejected."""
        with self.assertRaises(ValueError):
            parse_symbolic_expression('"hello"')


if __name__ == "__main__":
    unittest.main()
