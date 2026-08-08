# Release checklist

This checklist describes the release process implemented by
`.github/workflows/main.yml`. Pushing a `v*` tag runs the full test matrix,
builds the distribution, and publishes it to PyPI through the `release`
environment.

## Prepare the release

- [ ] Choose a [PEP 440](https://peps.python.org/pep-0440/) version and identify
      the previous release tag.
- [ ] Review merged changes since the previous tag for compatibility risks,
      deprecations, and migration notes.
- [ ] Update `onnx_ir.__version__` in `src/onnx_ir/__init__.py` in a dedicated
      pull request.
- [ ] Confirm that the supported Python and ONNX versions in `pyproject.toml`
      and `noxfile.py` agree.
- [ ] Draft release notes grouped by user-visible changes. Call out breaking
      changes, deprecations, security fixes, and contributor credits.

## Validate the release commit

- [ ] Merge the version-bump pull request and use its commit on `main` as the
      release commit.
- [ ] Confirm all required checks pass, including lint, the supported Python
      and operating-system test matrix, ONNX weekly, package build, and
      documentation builds.
- [ ] Build the distribution from the release commit with `python -m build`
      and install the resulting wheel in a clean environment.
- [ ] Verify that the installed version matches the intended release:

  ```bash
  python -c "import onnx_ir; print(onnx_ir.__version__)"
  ```

- [ ] Smoke test loading and saving a representative model with the installed
      package.

## Publish

- [ ] Create and push a tag named `v<version>` from the verified release commit.
- [ ] Approve the protected `release` environment when prompted.
- [ ] Confirm the tag workflow publishes the distribution successfully.
- [ ] Verify the new files and version on
      [PyPI](https://pypi.org/project/onnx-ir/).
- [ ] Create the GitHub release from the same tag and publish the prepared
      release notes.

## Verify and monitor

- [ ] Install `onnx-ir==<version>` from PyPI in a clean environment and repeat
      the import and model round-trip smoke tests.
- [ ] Confirm the GitHub release, PyPI project, and documentation site link to
      the expected version and guidance.
- [ ] Monitor CI, packaging reports, and new issues for regressions. Add urgent
      compatibility notes to the release and documentation when needed.
