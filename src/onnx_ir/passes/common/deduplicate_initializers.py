from onnx_ir._core import Node, Graph
from onnx_ir.traversal import RecursiveGraphIterator


class DeduplicateInitializersPass:
    def apply(self, graph: Graph) -> Graph:
        seen = {}      # (dtype, shape) → {tobytes: name}
        name_map = {}  # Duplicate name → canonical name

        for initializer in list(graph.initializers.values()):
            dtype = initializer.const_value.dtype
            shape = tuple(initializer.const_value.shape)
            content = initializer.const_value.tobytes()

            if (dtype, shape) not in seen:
                seen[(dtype, shape)] = {}

            group = seen[(dtype, shape)]
            if content in group:
                canonical_name = group[content]
                name_map[initializer.name] = canonical_name
                graph.initializers.pop(initializer.name)
            else:
                group[content] = initializer.name

        for node in RecursiveGraphIterator(graph):
            for i, input_val in enumerate(node.inputs):
                if input_val and input_val.name in name_map:
                    canonical_name = name_map[input_val.name]
                    replacement = graph.initializers[canonical_name]
                    node.replace_input_with(i, replacement)

        return graph



