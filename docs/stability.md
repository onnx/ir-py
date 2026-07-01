# Stability and Versioning Policy

This page defines the intended stability contract for ONNX IR 1.x.

## Semantic versioning

Starting with 1.0, ONNX IR follows semantic versioning:

- **MAJOR**: breaking API or behavior changes
- **MINOR**: backward-compatible feature additions
- **PATCH**: backward-compatible fixes

## API stability levels

### Stable surface

The following are expected to remain backward-compatible within a major release:

- Public top-level APIs documented in the docsite API reference.
- Core model/graph/value/node mutation and traversal interfaces.
- Serialization/deserialization entry points intended for users.

### Evolving surface

The following may change more frequently and should be version-pinned by advanced
users:

- Experimental helpers not explicitly documented as stable.
- Performance-oriented internals and private modules (names prefixed with `_`).
- Emerging pass/helper utilities that are not yet designated stable.

## Deprecation policy

When a stable API must change:

1. It is first marked deprecated in documentation and/or code comments.
2. A migration path is provided.
3. Removal occurs in a later major release, except for urgent security reasons.

## Behavior and correctness expectations

- ONNX spec coverage and serialization correctness are prioritized over convenience.
- Security-sensitive behavior (for example external tensor path containment) is
  fail-closed where applicable.
- Performance improvements should not silently change public semantics.

## What this means for adopters

- Pin to `<2.0` if you want 1.x compatibility.
- Treat private modules as implementation details.
- Follow release notes and migration docs before upgrading major versions.
