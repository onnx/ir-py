# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Safe expression parser for symbolic dimension expressions.

This module provides a safe recursive descent parser for parsing symbolic
dimension expressions without using eval().
"""

from __future__ import annotations

from collections.abc import Callable

import sympy

__all__ = [
    "parse_symbolic_expression",
]

# Allowed functions for parsing symbolic dimension expressions
_ALLOWED_FUNCTIONS: dict[str, Callable[..., sympy.Expr]] = {
    "max": sympy.Max,
    "Max": sympy.Max,
    "min": sympy.Min,
    "Min": sympy.Min,
    "floor": sympy.floor,
    "sqrt": sympy.sqrt,
    "mod": sympy.Mod,
    "Mod": sympy.Mod,
}


class _ExpressionTokenizer:
    """Tokenizer for symbolic dimension expressions.

    This is a safe tokenizer that does not use eval().
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.length = len(text)

    def peek(self) -> str | None:
        """Peek at the current character without consuming it."""
        self._skip_whitespace()
        if self.pos >= self.length:
            return None
        return self.text[self.pos]

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < self.length and self.text[self.pos].isspace():
            self.pos += 1

    def get_token(self) -> tuple[str, str | int] | None:
        """Get the next token.

        Returns:
            A tuple of (token_type, token_value) or None if end of input.
            Token types: 'NUMBER', 'IDENT', 'OP', 'LPAREN', 'RPAREN', 'COMMA'
        """
        self._skip_whitespace()
        if self.pos >= self.length:
            return None

        char = self.text[self.pos]

        # Number
        if char.isdigit():
            start = self.pos
            while self.pos < self.length and self.text[self.pos].isdigit():
                self.pos += 1
            return ("NUMBER", int(self.text[start : self.pos]))

        # Identifier (dots allowed for names like 'decoder_input_ids.45_dim_1')
        if char.isalpha() or char == "_":
            start = self.pos
            while self.pos < self.length and (
                self.text[self.pos].isalnum() or self.text[self.pos] in "_."
            ):
                self.pos += 1
            return ("IDENT", self.text[start : self.pos])

        # Two-character operators
        if self.pos + 1 < self.length:
            two_char = self.text[self.pos : self.pos + 2]
            if two_char in ("//", "**"):
                self.pos += 2
                return ("OP", two_char)

        # Single-character tokens
        if char in "+-*/%":
            self.pos += 1
            return ("OP", char)
        if char == "(":
            self.pos += 1
            return ("LPAREN", "(")
        if char == ")":
            self.pos += 1
            return ("RPAREN", ")")
        if char == ",":
            self.pos += 1
            return ("COMMA", ",")

        raise ValueError(
            f"Unexpected character '{char}' at position {self.pos} in expression '{self.text}'"
        )


class _ExpressionParser:
    """Safe recursive descent parser for symbolic dimension expressions.

    Supports:
        - Basic arithmetic: +, -, *, /, //, %, **
        - Functions: max(), min(), floor(), sqrt()
        - Symbolic variables (identifiers)
        - Integer literals
        - Parentheses for grouping

    Grammar:
        expr       -> term (('+' | '-') term)*
        term       -> power (('*' | '/' | '//' | '%') power)*
        power      -> unary ('**' power)?
        unary      -> '-' unary | primary
        primary    -> NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
        args       -> expr (',' expr)*
    """

    def __init__(self, text: str) -> None:
        self.tokenizer = _ExpressionTokenizer(text)
        self.text = text
        self.current_token: tuple[str, str | int] | None = None
        self._advance()

    def _advance(self) -> None:
        """Advance to the next token."""
        self.current_token = self.tokenizer.get_token()

    def _expect(self, token_type: str) -> str | int:
        """Expect a specific token type and consume it."""
        if self.current_token is None or self.current_token[0] != token_type:
            raise ValueError(
                f"Expected {token_type} but got {self.current_token} in expression '{self.text}'"
            )
        value = self.current_token[1]
        self._advance()
        return value

    def parse(self) -> sympy.Expr:
        """Parse the expression and return a SymPy expression."""
        result = self._parse_expr()
        if self.current_token is not None:
            raise ValueError(
                f"Unexpected token {self.current_token} at end of expression '{self.text}'"
            )
        return result

    def _parse_expr(self) -> sympy.Expr:
        """Parse an expression (handles + and -)."""
        left = self._parse_term()

        while (
            self.current_token is not None
            and self.current_token[0] == "OP"
            and self.current_token[1] in ("+", "-")
        ):
            op = self.current_token[1]
            self._advance()
            right = self._parse_term()
            if op == "+":
                left = left + right
            else:
                left = left - right

        return left

    def _parse_term(self) -> sympy.Expr:
        """Parse a term (handles *, /, //, %)."""
        left = self._parse_power()

        while (
            self.current_token is not None
            and self.current_token[0] == "OP"
            and self.current_token[1] in {"*", "/", "//", "%"}
        ):
            op = self.current_token[1]
            self._advance()
            right = self._parse_power()
            if op == "*":
                left = left * right
            elif op == "/":
                left = left / right
            elif op == "//":
                left = sympy.floor(left / right)
            else:  # %
                left = sympy.Mod(left, right)

        return left

    def _parse_power(self) -> sympy.Expr:
        """Parse a power expression (handles **)."""
        base = self._parse_unary()

        if (
            self.current_token is not None
            and self.current_token[0] == "OP"
            and self.current_token[1] == "**"
        ):
            self._advance()
            # Right-associative
            exponent = self._parse_power()
            return base**exponent

        return base

    def _parse_unary(self) -> sympy.Expr:
        """Parse a unary expression (handles unary -)."""
        if (
            self.current_token is not None
            and self.current_token[0] == "OP"
            and self.current_token[1] == "-"
        ):
            self._advance()
            return -self._parse_unary()

        return self._parse_primary()

    def _parse_primary(self) -> sympy.Expr:
        """Parse a primary expression (number, identifier, function call, or parenthesized expression)."""
        if self.current_token is None:
            raise ValueError(f"Unexpected end of expression '{self.text}'")

        token_type, token_value = self.current_token

        # Number
        if token_type == "NUMBER":
            self._advance()
            return sympy.Integer(token_value)

        # Identifier or function call
        if token_type == "IDENT":
            name = token_value
            assert isinstance(name, str)
            self._advance()

            # Check for function call
            if self.current_token is not None and self.current_token[0] == "LPAREN":
                return self._parse_function_call(name)

            # Simple identifier - create a symbol
            return sympy.Symbol(name, integer=True, positive=True)

        # Parenthesized expression
        if token_type == "LPAREN":
            self._advance()
            expr = self._parse_expr()
            self._expect("RPAREN")
            return expr

        raise ValueError(f"Unexpected token {self.current_token} in expression '{self.text}'")

    def _parse_function_call(self, name: str) -> sympy.Expr:
        """Parse a function call."""
        if name not in _ALLOWED_FUNCTIONS:
            raise ValueError(
                f"Unknown function '{name}' in expression '{self.text}'. "
                f"Allowed functions: {', '.join(sorted(_ALLOWED_FUNCTIONS.keys()))}"
            )

        self._expect("LPAREN")
        args: list[sympy.Expr] = []

        # Parse arguments
        if self.current_token is not None and self.current_token[0] != "RPAREN":
            args.append(self._parse_expr())
            while self.current_token is not None and self.current_token[0] == "COMMA":
                self._advance()
                args.append(self._parse_expr())

        self._expect("RPAREN")

        func = _ALLOWED_FUNCTIONS[name]
        return func(*args)


def parse_symbolic_expression(value: str) -> sympy.Expr:
    """Parse a string into a SymPy expression for symbolic dimensions.

    This parser is safe and does not use eval().

    Supports:
        - Basic arithmetic: +, -, *, /, //, %, **
        - Functions: max(), min(), floor(), sqrt()
        - Symbolic variables (identifiers)
        - Integer literals
        - Parentheses for grouping

    Args:
        value: The string to parse.

    Returns:
        A SymPy expression representing the parsed string.

    Raises:
        ValueError: If the expression contains unsupported constructs.
    """
    # Check if it's a simple identifier (no operators or function calls)
    # This is the common case and avoids the overhead of parsing
    if value.isidentifier():
        return sympy.Symbol(value, integer=True, positive=True)

    parser = _ExpressionParser(value)
    return parser.parse()
