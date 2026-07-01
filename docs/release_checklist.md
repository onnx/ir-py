# Release Checklist

This checklist is intended for maintainers preparing a release.

## Scope and compatibility

- [ ] Freeze the intended 1.0 feature set.
- [ ] Confirm public API surface and stability guarantees.
- [ ] Validate supported Python/ONNX dependency matrix.

## Quality gates

- [ ] Run linters and type checks in CI.
- [ ] Run full test suite and confirm green status.
- [ ] Execute model round-trip tests (`deserialize -> transform -> serialize`).
- [ ] Validate external tensor security behavior on representative artifacts.

## Documentation gates

- [ ] Verify all "Getting Started" examples run.
- [ ] Ensure migration guidance covers the most common upgrade paths.
- [ ] Confirm troubleshooting and FAQ reflect observed user issues.
- [ ] Rebuild docsite and check for warnings/errors.

## Release operations

- [ ] Tag release candidate and smoke test install from package index.
- [ ] Publish final version.
- [ ] Announce release notes and upgrade guidance.
- [ ] Monitor issues for post-release regressions.

## Post-release

- [ ] Triage and prioritize 1.0.x patch candidates.
- [ ] Document any urgent compatibility notes.
- [ ] Keep migration and troubleshooting docs updated with field feedback.
