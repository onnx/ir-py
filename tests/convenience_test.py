# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest

import onnx_ir as ir


class ReplaceAllUsesWithTest(unittest.TestCase):
    def test_replace_all_uses_with_updates_graph_outputs(self) -> None:
        x = ir.val("x", ir.DataType.FLOAT, (2,))
        y = ir.val("y", ir.DataType.FLOAT, (2,))
        node = ir.Node("", "Identity", [x], outputs=[y], name="Id")
        graph = ir.Graph(inputs=[x], outputs=[y], nodes=[node], name="g")

        ir.convenience.replace_all_uses_with(y, x, replace_graph_outputs=True)

        self.assertIs(graph.outputs[0], x)
        self.assertFalse(y.is_graph_output())
        self.assertTrue(x.is_graph_output())


if __name__ == "__main__":
    unittest.main()
