# Contributing to ONNX IR

Welcome to the ONNX IR project! We appreciate your interest in contributing.

## Code Style and Design Principles

Before contributing, please familiarize yourself with our development guidelines:

- **[Coding Style](https://github.com/onnx/ir-py/wiki/Coding-style)**: Our coding conventions and style guidelines
- **[Design Principles](https://github.com/onnx/ir-py/wiki/Design-Principles)**: The core principles guiding the design of ONNX IR

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Setting up the Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/onnx/ir-py.git
   cd ir-py
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## Code Quality and Linting

We use [lintrunner](https://github.com/suo/lintrunner) for code quality checks. The project includes several linters:

- **RUFF**: Python linter and code formatter
- **MYPY**: Static type checker
- **EDITORCONFIG-CHECKER**: EditorConfig compliance checker

### Setting up lintrunner

1. **Initialize lintrunner** (this installs the required linting tools):
   ```bash
   lintrunner init
   ```

2. **Run all linters**:
   ```bash
   lintrunner
   ```

3. **Apply automatic fixes** where possible:
   ```bash
   lintrunner -a --output oneline
   ```

4. **Format code only**:
   ```bash
   lintrunner f --output oneline
   ```

### Linting specific files

You can lint specific files or directories:
```bash
lintrunner src/onnx_ir/
lintrunner path/to/specific/file.py
```

## Testing

The project uses [nox](https://nox.thea.codes/) for testing across different environments and [pytest](https://pytest.org/) as the test runner.

### Running tests with nox

```bash
# Run all tests
nox -s test

# Run tests with specific Python version (if available)
nox -s test --python 3.11

# Run tests with ONNX weekly build
nox -s test-onnx-weekly

# Run tests with PyTorch nightly
nox -s test-torch-nightly
```

### Running tests directly with pytest

If you prefer to run tests directly:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=onnx_ir

# Run specific test file
pytest tests/test_specific.py

# Run doctests
pytest src --doctest-modules
```

## Submitting Contributions

### Before submitting

1. **Ensure your code passes all linting checks**:
   ```bash
   lintrunner
   ```

2. **Run the test suite**:
   ```bash
   pytest path/to/test.py
   ```

### Pull Request Guidelines

1. **Fork the repository** and create a feature branch from `main`
2. **Write clear, descriptive commit messages**. Be sure to [signoff](https://github.com/apps/dco) your commits.
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Ensure all CI checks pass**
6. **Request review** from maintainers

### Pull Request Description

Use clear and descriptive PR description:
```
[component] brief description of change

More detailed explanation if needed, including:
- What was changed
- Why it was changed
- Any breaking changes
```

## Development Workflow

1. **Create an issue** or comment on an existing one to discuss your proposed changes
2. **Fork the repository** and create a feature branch
3. **Make your changes** following our coding style and design principles
4. **Add or update tests** as appropriate
5. **Run linting and tests** locally
6. **Submit a pull request** with a clear description of your changes

## Getting Help

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/onnx/ir-py/issues)
- **Discussions**: For questions and discussions, use [GitHub Discussions](https://github.com/onnx/ir-py/discussions)
- **Documentation**: Visit the [official documentation](https://onnx.ai/ir-py/)

## License

By contributing to ONNX IR, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

Thank you for contributing to ONNX IR! 🎉
