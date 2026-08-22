# FAQ

## Is ONNX IR a runtime?

No. ONNX IR is a library for in-memory model representation, analysis, and
transformation. It does not execute models like an inference runtime.

## Does ONNX IR support the full ONNX spec?

It is designed to represent all valid ONNX protobuf models, plus a subset of
invalid models to support repair workflows.

## Should I use ONNX protobuf helpers directly?

Prefer ONNX IR APIs for model manipulation and conversions. This generally gives
better ergonomics and avoids protobuf-heavy workflows in transformation code.

## How should I handle large external tensors safely?

Use external tensor support with a configured `base_dir` when loading model
artifacts, especially from untrusted sources. This enables containment checks.

## Is zero-copy always guaranteed?

Not always. ONNX IR is designed to minimize copies where possible, but some
operations (such as certain conversions or materializations) may copy data.

## Can I mutate a graph while iterating nodes?

Yes. ONNX IR is designed for robust mutation workflows and supports safe
iteration patterns during graph edits. Nodes inserted after the current node are
visited; nodes inserted before it are not. If the current node is removed or
moved, iteration continues from the node that followed its original location.

See [Graph transformation patterns](graph_transformations.md).

## How do I infer symbolic shapes?

Use the built-in `ShapeInferencePass` for ONNX shape inference. For richer SymPy
expressions, shape-data propagation, and custom operator inference, use the
optional [`onnx-shape-inference`](https://pypi.org/project/onnx-shape-inference/)
package.

## Where is the API reference?

See [API Reference](api/index.md).
