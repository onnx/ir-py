# Agent Skills

This directory contains specialized skills for AI agents working with the ONNX IR project.

## Available Skills

### [Pass Writing](./pass-writing.md)

Comprehensive guide for creating transformation passes in ONNX IR. This skill covers:

- Pass infrastructure (InPlacePass, FunctionalPass, PassManager)
- Best practices for graph traversal and manipulation
- Node and value manipulation patterns
- Common pass patterns (elimination, dead code removal, CSE, normalization)
- Testing and error handling conventions
- Performance considerations

Use this skill when developing new graph transformation passes or modifying existing ones.

### [Shape Inference Ops](./shape-inference-ops.md)

Guide for implementing and testing shape inference operators. This skill covers:

- Op implementation patterns (registration, preconditions, graceful degradation)
- The `ts()` test helper for concise type+shape assertions
- Auto-generated symbolic dim naming (`_d0`, `_d1`, …) and counter reset behavior
- Parameterized test patterns and assertion style
- Which ops propagate named dims vs create new `_d` dims

Use this skill when adding new shape inference operators or writing/modifying their tests.

## Using These Skills

These skills are designed to be used by AI agents to improve code quality and consistency when working with the ONNX IR codebase. Each skill document contains:

- Conceptual overviews
- Practical code examples
- Best practices and conventions
- Common pitfalls to avoid
- References to relevant source files

## Contributing

When adding new skills:

1. Create a descriptive markdown file in this directory
2. Update this README with a brief description of the new skill
3. Include comprehensive examples from the codebase
4. Document both what to do and what to avoid
