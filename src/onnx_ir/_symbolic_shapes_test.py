# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the symbolic expression parser."""

from __future__ import annotations

import unittest

import sympy

import onnx_ir as ir
from onnx_ir._symbolic_shapes import parse_symbolic_expression


class SymbolicDimTest(unittest.TestCase):
    """Tests for SymbolicDim class."""

    def test_string_value(self):
        dim = ir.SymbolicDim("batch")
        self.assertEqual(dim.value, "batch")
        self.assertIsNone(dim._expr_cache)  # Lazy - not created yet

    def test_none_value(self):
        dim = ir.SymbolicDim(None)
        self.assertIsNone(dim.value)
        self.assertIsNone(dim._expr)

    def test_sympy_expr_value(self):
        expr = sympy.Symbol("N") + 1
        dim = ir.SymbolicDim(expr)
        self.assertEqual(dim.value, "N + 1")
        self.assertEqual(dim._expr, expr)

    def test_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            ir.SymbolicDim(42)

    def test_equality_with_string(self):
        dim = ir.SymbolicDim("N")
        self.assertEqual(dim, "N")
        self.assertNotEqual(dim, "M")

    def test_equality_with_none(self):
        dim_none = ir.SymbolicDim(None)
        dim_named = ir.SymbolicDim("N")
        self.assertEqual(dim_none, None)
        self.assertNotEqual(dim_named, None)

    def test_equality_with_symbolic_dim(self):
        dim1 = ir.SymbolicDim("N")
        dim2 = ir.SymbolicDim("N")
        dim3 = ir.SymbolicDim("M")
        self.assertEqual(dim1, dim2)
        self.assertNotEqual(dim1, dim3)

    def test_hash_consistency(self):
        dim1 = ir.SymbolicDim("batch")
        dim2 = ir.SymbolicDim("batch")
        self.assertEqual(hash(dim1), hash(dim2))

    def test_add_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim + 1
        self.assertEqual(result.value, "N + 1")

    def test_add_with_symbolic_dim(self):
        dim1 = ir.SymbolicDim("N")
        dim2 = ir.SymbolicDim("M")
        result = dim1 + dim2
        self.assertEqual(result.value, "M + N")

    def test_add_with_none_returns_none(self):
        dim = ir.SymbolicDim(None)
        result = dim + 1
        self.assertIsNone(result.value)

    def test_sub_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim - 1
        self.assertEqual(result.value, "N - 1")

    def test_mul_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim * 2
        self.assertEqual(result.value, "2*N")

    def test_floordiv_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim // 2
        self.assertEqual(result.value, "floor(N/2)")

    def test_mod_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim % 2
        self.assertEqual(result.value, "Mod(N, 2)")

    def test_radd(self):
        dim = ir.SymbolicDim("N")
        result = 1 + dim
        self.assertEqual(result.value, "N + 1")

    def test_rsub(self):
        dim = ir.SymbolicDim("N")
        result = 10 - dim
        self.assertEqual(result.value, "10 - N")

    def test_rmul(self):
        dim = ir.SymbolicDim("N")
        result = 2 * dim
        self.assertEqual(result.value, "2*N")

    def test_truediv_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim / 3
        self.assertEqual(result.value, "N/3")

    def test_rtruediv(self):
        dim = ir.SymbolicDim("N")
        result = 6 / dim
        self.assertEqual(result.value, "6/N")

    def test_ceil(self):
        import math

        dim = ir.SymbolicDim("N")
        result = math.ceil(dim / 3)
        self.assertEqual(result.value, "ceiling(N/3)")

    def test_floor(self):
        import math

        dim = ir.SymbolicDim("N")
        result = math.floor(dim / 3)
        self.assertEqual(result.value, "floor(N/3)")

    def test_trunc(self):
        import math

        dim = ir.SymbolicDim("N")
        result = math.trunc(dim)
        self.assertEqual(result.value, "N")

    def test_unsupported_operand_raises_type_error(self):
        dim = ir.SymbolicDim("N")
        with self.assertRaises(TypeError) as ctx:
            _ = dim + "string"
        self.assertIn("unsupported operand type", str(ctx.exception))

    def test_simplify(self):
        dim = ir.SymbolicDim("N") + 0
        simplified = dim.simplify()
        self.assertEqual(simplified.value, "N")

    def test_evaluate(self):
        dim = ir.SymbolicDim("N") * 2 + 1
        result = dim.evaluate({"N": 5})
        self.assertEqual(result, 11)

    def test_evaluate_none_returns_symbolic_dim_none(self):
        dim = ir.SymbolicDim(None)
        result = dim.evaluate({"N": 5})
        self.assertIsInstance(result, ir.SymbolicDim)
        self.assertIsNone(result.value)

    def test_evaluate_incomplete_bindings_returns_symbolic_dim(self):
        dim = ir.SymbolicDim("N") + ir.SymbolicDim("M")
        result = dim.evaluate({"N": 5})  # M not provided
        self.assertIsInstance(result, ir.SymbolicDim)
        # The partially evaluated expression can be further evaluated
        self.assertEqual(result.evaluate({"M": 3}), 8)

    def test_free_symbols(self):
        dim = ir.SymbolicDim("N") + ir.SymbolicDim("M")
        symbols = dim.free_symbols()
        self.assertEqual(symbols, frozenset({"N", "M"}))

    def test_free_symbols_none(self):
        dim = ir.SymbolicDim(None)
        symbols = dim.free_symbols()
        self.assertEqual(symbols, frozenset())


class ShapeEvaluateTest(unittest.TestCase):
    """Tests for Shape.evaluate() and Shape.simplify()."""

    def test_evaluate_static_shape(self):
        shape = ir.Shape([1, 2, 3])
        result = shape.evaluate({})
        self.assertIsInstance(result, ir.Shape)
        self.assertEqual(result, (1, 2, 3))

    def test_evaluate_symbolic_shape(self):
        shape = ir.Shape(["batch", 256, ir.SymbolicDim("seq") + 1])
        result = shape.evaluate({"batch": 32, "seq": 128})
        self.assertIsInstance(result, ir.Shape)
        self.assertEqual(result, (32, 256, 129))

    def test_evaluate_incomplete_returns_shape_with_symbolic_dims(self):
        shape = ir.Shape(["batch", "seq"])
        result = shape.evaluate({"batch": 32})  # seq not provided
        self.assertIsInstance(result, ir.Shape)
        self.assertEqual(result[0], 32)
        self.assertIsInstance(result[1], ir.SymbolicDim)

    def test_simplify(self):
        shape = ir.Shape([ir.SymbolicDim("N") + 0, ir.SymbolicDim("M") * 1])
        simplified = shape.simplify()
        self.assertEqual(simplified[0].value, "N")
        self.assertEqual(simplified[1].value, "M")

    def test_free_symbols(self):
        shape = ir.Shape(["batch", 256, "seq_len"])
        symbols = shape.free_symbols()
        self.assertEqual(symbols, frozenset({"batch", "seq_len"}))


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
        result = parse_symbolic_expression("n + 1")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, n + 1)

    def test_subtraction(self):
        """Test parsing subtraction."""
        result = parse_symbolic_expression("n - 1")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, n - 1)

    def test_multiplication(self):
        """Test parsing multiplication."""
        result = parse_symbolic_expression("2 * n")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, 2 * n)

    def test_division(self):
        """Test parsing division."""
        result = parse_symbolic_expression("n / 2")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, n / 2)

    def test_floor_division(self):
        """Test parsing floor division."""
        result = parse_symbolic_expression("n // 2")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, sympy.floor(n / 2))

    def test_power(self):
        """Test parsing exponentiation."""
        result = parse_symbolic_expression("n ** 2")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, n**2)

    def test_power_right_associative(self):
        """Test that power is right-associative (2**3**2 = 2^(3^2) = 512)."""
        result = parse_symbolic_expression("2 ** 3 ** 2")
        self.assertEqual(result, sympy.Integer(512))

    def test_unary_minus(self):
        """Test parsing unary minus."""
        result = parse_symbolic_expression("-n")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, -n)

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
        result = parse_symbolic_expression("floor(n / 2)")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, sympy.floor(n / 2))

    def test_function_sqrt(self):
        """Test parsing sqrt function."""
        result = parse_symbolic_expression("sqrt(n)")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, sympy.sqrt(n))

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
        result = parse_symbolic_expression("n + n")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, 2 * n)

    def test_modulo_with_integer(self):
        """Test parsing modulo operator with integer."""
        result = parse_symbolic_expression("n % 2")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, sympy.Mod(n, 2))

    def test_modulo_with_symbols(self):
        """Test parsing modulo operator with two symbols."""
        result = parse_symbolic_expression("a % b")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        self.assertEqual(result, sympy.Mod(a, b))

    def test_modulo_integer_values(self):
        """Test that modulo evaluates correctly for integer values."""
        result = parse_symbolic_expression("10 % 3")
        self.assertEqual(result, sympy.Integer(1))

    def test_modulo_precedence_same_as_mul_div(self):
        """Test that % has the same precedence as * and /."""
        # a + b % c should be parsed as a + (b % c)
        result = parse_symbolic_expression("a + b % c")
        a = sympy.Symbol("a", integer=True, positive=True)
        b = sympy.Symbol("b", integer=True, positive=True)
        c = sympy.Symbol("c", integer=True, positive=True)
        self.assertEqual(result, a + sympy.Mod(b, c))

    def test_modulo_with_floor_division(self):
        """Test parsing expression combining modulo and floor division."""
        result = parse_symbolic_expression("n // 4 + n % 4")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, sympy.floor(n / 4) + sympy.Mod(n, 4))

    def test_modulo_in_function_arg(self):
        """Test parsing modulo inside a function argument."""
        result = parse_symbolic_expression("max(n % 2, 1)")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, sympy.Max(sympy.Mod(n, 2), 1))

    def test_mod_function_call(self):
        """Test parsing mod() as a function call (already supported)."""
        result = parse_symbolic_expression("mod(n, 2)")
        n = sympy.Symbol("n", integer=True, positive=True)
        self.assertEqual(result, sympy.Mod(n, 2))

    def test_modulo_operator_and_mod_function_equivalent(self):
        """Test that % operator and mod() function produce the same result."""
        result_op = parse_symbolic_expression("n % 3")
        result_fn = parse_symbolic_expression("mod(n, 3)")
        self.assertEqual(result_op, result_fn)


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

        # Just mark the function as used so linters do not complain
        self.assertTrue(callable(malicious_function))

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
