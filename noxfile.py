# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test with different environment configuration with nox.

Documentation:
    https://nox.thea.codes/
"""

import nox

nox.options.error_on_missing_interpreters = False
nox.options.default_venv_backend = "uv"


COMMON_TEST_DEPENDENCIES = (
    "expecttest>0.1.6",
    "hypothesis",
    "numpy",
    "packaging",
    "parameterized",
    'psutil; sys_platform != "win32"',
    "pytest-cov",
    "pytest-randomly",
    "pytest-subtests",
    "pytest-xdist",
    "pytest>7.1.0",
    "pyyaml",
    "types-PyYAML",
    "typing_extensions>=4.10",
    "ml-dtypes",
    "onnxruntime",
    "safetensors",
)
ONNX = "onnx==1.18"
ONNXSCRIPT = "onnxscript"
ONNX_RUNTIME = "onnxruntime==1.20.1"
PYTORCH = "torch==2.7.0"
TORCHVISON = "torchvision==0.22.0"
TRANSFORMERS = "transformers==4.37.2"


@nox.session(tags=["build"])
def build(session):
    """Build package."""
    session.install("build", "wheel")
    session.run("python", "-m", "build")


@nox.session(tags=["test"])
def test(session):
    """Test onnx_ir and documentation."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        ONNX,
        PYTORCH,
    )
    session.install(ONNXSCRIPT, "--no-deps")
    session.install(".", "--no-deps")
    session.run("uv", "pip", "list")
    session.run("pytest", "src", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)


@nox.session(tags=["test-onnx-weekly"])
def test_onnx_weekly(session):
    """Test with ONNX weekly (preview) build."""
    session.install(*COMMON_TEST_DEPENDENCIES, PYTORCH)
    session.install("-r", "requirements/ci/requirements-onnx-weekly.txt")
    session.install(ONNXSCRIPT, "--no-deps")
    session.install(".", "--no-deps")
    session.run("uv", "pip", "list")
    session.run("pytest", "src", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)


@nox.session(tags=["test-torch-nightly"])
def test_torch_nightly(session):
    """Test with PyTorch nightly (preview) build."""
    session.install(*COMMON_TEST_DEPENDENCIES)
    session.install("-r", "requirements/ci/requirements-onnx-weekly.txt")
    session.install("-r", "requirements/ci/requirements-pytorch-nightly.txt")
    session.install(ONNXSCRIPT, "--no-deps")
    session.install(".", "--no-deps")
    session.run("uv", "pip", "list")
    session.run("pytest", "src", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)


# TODO(justinchuby): Integration test with ONNX Script
