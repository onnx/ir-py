"""Tests for graph composition functionality."""
import unittest

import numpy as np

import onnx_ir as ir


class GraphCompositionTest(unittest.TestCase):
    """Test cases for Graph.__call__ method and graph composition."""

    def test_basic_composition(self):
        """Test basic graph composition with two inputs."""
        # Create a reusable graph that adds two inputs
        input1 = ir.Value(name="input1", type=ir.TensorType(ir.DataType.FLOAT))
        input2 = ir.Value(name="input2", type=ir.TensorType(ir.DataType.FLOAT))
        
        add_node = ir.Node(
            domain="", 
            op_type="Add", 
            inputs=[input1, input2],
            num_outputs=1
        )
        output = add_node.outputs[0]
        output.name = "add_output"
        
        # Create the reusable graph
        add_graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[output],
            nodes=[add_node]
        )
        
        # Create a target graph that will use the add graph
        new_input1 = ir.Value(name="new_input1", type=ir.TensorType(ir.DataType.FLOAT))
        new_input2 = ir.Value(name="new_input2", type=ir.TensorType(ir.DataType.FLOAT))
        
        target_graph = ir.Graph(
            inputs=[new_input1, new_input2],
            outputs=[],
            nodes=[]
        )
        
        # Test the composition
        composed_outputs = add_graph(new_input1, new_input2)
        
        # Verify the results
        self.assertEqual(len(composed_outputs), 1)
        self.assertEqual(composed_outputs[0].name, "add_output")
        self.assertEqual(len(target_graph), 1)  # One node added
        self.assertIn(composed_outputs[0].producer(), target_graph)

    def test_composition_with_initializers(self):
        """Test graph composition with initializers."""
        # Create a graph with an initializer
        input_val = ir.Value(name="input", type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create a constant initializer
        const_tensor = ir.Tensor(np.array([1.0], dtype=np.float32))
        const_value = ir.Value(
            name="constant", 
            type=ir.TensorType(ir.DataType.FLOAT),
            const_value=const_tensor
        )
        
        add_node = ir.Node(
            domain="", 
            op_type="Add", 
            inputs=[input_val, const_value],
            num_outputs=1
        )
        output = add_node.outputs[0]
        output.name = "add_const_output"
        
        # Create the graph with initializer
        add_const_graph = ir.Graph(
            inputs=[input_val],
            outputs=[output],
            nodes=[add_node],
            initializers=[const_value]
        )
        
        # Create target graph
        new_input = ir.Value(name="new_input", type=ir.TensorType(ir.DataType.FLOAT))
        target_graph = ir.Graph(
            inputs=[new_input],
            outputs=[],
            nodes=[]
        )
        
        # Test composition
        composed_outputs = add_const_graph(new_input)
        
        # Verify initializers were copied
        self.assertIn("constant", target_graph.initializers)
        self.assertEqual(len(composed_outputs), 1)
        self.assertEqual(len(target_graph), 1)

    def test_wrong_number_of_arguments_raises_error(self):
        """Test that wrong number of arguments raises ValueError."""
        # Create a graph with two inputs
        input1 = ir.Value(name="input1", type=ir.TensorType(ir.DataType.FLOAT))
        input2 = ir.Value(name="input2", type=ir.TensorType(ir.DataType.FLOAT))
        
        add_node = ir.Node(
            domain="", 
            op_type="Add", 
            inputs=[input1, input2],
            num_outputs=1
        )
        
        test_graph = ir.Graph(
            inputs=[input1, input2],
            outputs=add_node.outputs,
            nodes=[add_node]
        )
        
        # Create target value
        new_input = ir.Value(name="new_input", type=ir.TensorType(ir.DataType.FLOAT))
        target_graph = ir.Graph(
            inputs=[new_input],
            outputs=[],
            nodes=[]
        )
        
        # Test wrong number of arguments
        with self.assertRaises(ValueError) as cm:
            test_graph(new_input)  # Should fail - need 2 inputs
        
        self.assertIn("Expected 2 input arguments, got 1", str(cm.exception))

    def test_orphan_value_raises_error(self):
        """Test that values not belonging to a graph raise ValueError."""
        # Create a graph
        input1 = ir.Value(name="input1", type=ir.TensorType(ir.DataType.FLOAT))
        input2 = ir.Value(name="input2", type=ir.TensorType(ir.DataType.FLOAT))
        
        add_node = ir.Node(
            domain="", 
            op_type="Add", 
            inputs=[input1, input2],
            num_outputs=1
        )
        
        test_graph = ir.Graph(
            inputs=[input1, input2],
            outputs=add_node.outputs,
            nodes=[add_node]
        )
        
        # Create target values
        new_input1 = ir.Value(name="new_input1", type=ir.TensorType(ir.DataType.FLOAT))
        target_graph = ir.Graph(
            inputs=[new_input1],
            outputs=[],
            nodes=[]
        )
        
        # Create orphan value
        orphan_value = ir.Value(name="orphan", type=ir.TensorType(ir.DataType.FLOAT))
        
        # Test orphan value
        with self.assertRaises(ValueError) as cm:
            test_graph(orphan_value, new_input1)
        
        self.assertIn("does not belong to any graph", str(cm.exception))

    def test_multiple_compositions(self):
        """Test composing the same graph multiple times."""
        # Create a simple graph
        input1 = ir.Value(name="input1", type=ir.TensorType(ir.DataType.FLOAT))
        input2 = ir.Value(name="input2", type=ir.TensorType(ir.DataType.FLOAT))
        
        add_node = ir.Node(
            domain="", 
            op_type="Add", 
            inputs=[input1, input2],
            num_outputs=1
        )
        output = add_node.outputs[0]
        output.name = "output"
        
        add_graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[output],
            nodes=[add_node]
        )
        
        # Create target graph with multiple inputs
        inputs = []
        for i in range(4):
            val = ir.Value(name=f"input_{i}", type=ir.TensorType(ir.DataType.FLOAT))
            inputs.append(val)
        
        target_graph = ir.Graph(
            inputs=inputs,
            outputs=[],
            nodes=[]
        )
        
        # Compose the add graph twice
        output1 = add_graph(inputs[0], inputs[1])
        output2 = add_graph(inputs[2], inputs[3])
        
        # Verify both compositions worked
        self.assertEqual(len(output1), 1)
        self.assertEqual(len(output2), 1)
        self.assertEqual(len(target_graph), 2)  # Two add nodes

    def test_empty_graph_composition(self):
        """Test composing a graph with no inputs."""
        # Create a graph with no inputs (just a constant)
        const_tensor = ir.Tensor(np.array([42.0], dtype=np.float32))
        const_value = ir.Value(
            name="constant", 
            type=ir.TensorType(ir.DataType.FLOAT),
            const_value=const_tensor
        )
        
        identity_node = ir.Node(
            domain="", 
            op_type="Identity", 
            inputs=[const_value],
            num_outputs=1
        )
        output = identity_node.outputs[0]
        
        const_graph = ir.Graph(
            inputs=[],  # No inputs
            outputs=[output],
            nodes=[identity_node],
            initializers=[const_value]
        )
        
        # Compose with no arguments
        composed_outputs = const_graph()
        
        # Verify the composition
        self.assertEqual(len(composed_outputs), 1)
        
    def test_value_properties_preserved(self):
        """Test that value properties are preserved during composition."""
        # Create a graph
        input_val = ir.Value(
            name="input", 
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 2, 3])
        )
        
        identity_node = ir.Node(
            domain="", 
            op_type="Identity", 
            inputs=[input_val],
            num_outputs=1
        )
        output = identity_node.outputs[0]
        output.name = "output"
        output.type = ir.TensorType(ir.DataType.FLOAT)
        output.shape = ir.Shape([1, 2, 3])
        
        test_graph = ir.Graph(
            inputs=[input_val],
            outputs=[output],
            nodes=[identity_node]
        )
        
        # Create target
        new_input = ir.Value(
            name="new_input", 
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 2, 3])
        )
        target_graph = ir.Graph(
            inputs=[new_input],
            outputs=[],
            nodes=[]
        )
        
        # Compose
        composed_outputs = test_graph(new_input)
        
        # Verify properties are preserved
        self.assertEqual(composed_outputs[0].name, "output")
        self.assertEqual(composed_outputs[0].type, ir.TensorType(ir.DataType.FLOAT))
        self.assertEqual(composed_outputs[0].shape, ir.Shape([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()