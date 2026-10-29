# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the runnable examples, including saved models and dynamic batches."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import onnx


def test_vmap_examples(tmp_path):
    script = pathlib.Path(__file__).resolve().parents[1] / "examples" / "vmap.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for name in ("image_classifier", "recommendation"):
        for batch_size in (1, 3, 7):
            assert (
                f"{name}: batch_size={batch_size}, independent execution matches"
                in result.stdout
            )
        for kind in ("single", "batched"):
            path = tmp_path / f"{name}.{kind}.onnx"
            model = onnx.load(path)
            onnx.checker.check_model(model, full_check=True)
            assert all(node.op_type not in ("Loop", "Scan") for node in model.graph.node)
            text = path.with_suffix(".txt").read_text(encoding="utf-8")
            assert text in result.stdout
            assert onnx.parser.parse_model(text).SerializeToString()
            snapshot_path = (
                script.parents[1] / "docs" / "_static" / "vmap" / f"{name}.{kind}.txt"
            )
            snapshot = onnx.parser.parse_model(snapshot_path.read_text(encoding="utf-8"))
            assert [
                (node.op_type, list(node.input), list(node.output))
                for node in snapshot.graph.node
            ] == [
                (node.op_type, list(node.input), list(node.output))
                for node in model.graph.node
            ]
