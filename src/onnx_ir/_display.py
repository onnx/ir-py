# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Internal utilities for displaying the intermediate representation of a model.

NOTE: All third-party imports should be scoped and imported only when used to avoid
importing unnecessary dependencies.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations


class PrettyPrintable:
    def display(self, *, page: bool = False) -> None:
        """Pretty print the object.

        Args:
            page: Whether to page the output.
        """
        text = str(self)

        if page:
            import rich.console

            console = rich.console.Console()
            with console.pager():
                console.print(text)
        else:
            print(text)
