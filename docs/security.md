# Security Model for External Data in onnx-ir

This document describes the threat model, implemented defenses, known
limitations, and design rationale for how **onnx-ir** handles external tensor
data (the `ExternalTensor` class).

## Threat model

ONNX models can reference external data files via relative paths stored in the
`location` field of a `TensorProto`. A malicious model could abuse this to
read arbitrary files on the host. The main attack vectors are:

| Vector | Example |
|---|---|
| **Path traversal** | `../../etc/passwd` |
| **Absolute path** | `/etc/passwd` |
| **Symlink escape** | `data.bin → /etc/passwd` |
| **Hardlink smuggling** | Hard-linking a sensitive file into the model directory |

## Implemented defenses

`ExternalTensor._check_path_containment()` enforces a two-layer check
whenever a non-empty `base_dir` is set:

| Layer | What it does |
|---|---|
| **1. String-based containment** | Normalizes the path (without resolving symlinks) and verifies it stays within `base_dir`. Catches `../` traversal and absolute paths without requiring the file to exist. |
| **2. Realpath containment** | Resolves all symlinks via `os.path.realpath()` and re-checks containment. Catches symlinks whose target is outside `base_dir`. Symlinks that resolve *within* `base_dir` are allowed. |
| **3. Hardlink detection** | Rejects files with more than one hard link (`st_nlink > 1`). Prevents an attacker from hard-linking a sensitive file into the model directory to bypass the containment boundary. |

All three checks run at **load time** — when `numpy()`, `tobytes()`, or
`tofile()` is called — not at construction time. This allows safe
deserialization of untrusted protos without triggering I/O.

## `base_dir=""` bypass (by design)

When `base_dir` is empty (the default), **all security checks are skipped**.
This is intentional for two reasons:

1. **Programmatic construction** — when a developer creates an `ExternalTensor`
   in code, they control the paths directly and do not need containment checks.
2. **Deserialization safety** — the IR deserializer may create `ExternalTensor`
   objects before a `base_dir` is known. Containment is only meaningful when
   the caller sets `base_dir` to the model's directory.

If you are loading an untrusted model, always set `base_dir` to the directory
containing the model file.

## Divergence from onnx/onnx

The reference ONNX runtime ([onnx/onnx#7717](https://github.com/onnx/onnx/pull/7717))
implements a four-layer defense:

| Layer | onnx/onnx | onnx-ir | Notes |
|---|---|---|---|
| 1. String-based containment | ✅ | ✅ | Equivalent |
| 2. Realpath containment | ✅ | ✅ | Equivalent |
| 3. `O_NOFOLLOW` on open | ✅ | ❌ | See *Future hardening* |
| 4. Allowlist for filenames | ✅ | ❌ | Not applicable — ir-py is a library, not a runtime |

### Why the difference?

* **onnx-ir is a library**, not a runtime. It focuses on safe loading of model
  data for inspection and transformation, not sandboxed execution.
* **`O_NOFOLLOW`** closes a TOCTOU (time-of-check-to-time-of-use) race between
  the containment check and the `open()` call. This is a valuable defense-in-depth
  measure but requires platform-specific code (`os.O_NOFOLLOW` is not available
  on Windows). It is planned for a future release.
* **Filename allowlisting** is a policy decision better left to the application
  layer that consumes onnx-ir.

## Known limitations

* **TOCTOU window** — A small race exists between `_check_path_containment()`
  and the subsequent `open()`. An attacker who can modify the filesystem
  concurrently could swap a safe file for a symlink after the check passes.
  Mitigation: use `O_NOFOLLOW` (planned).
* **`base_dir=""` bypass** — As described above, an empty `base_dir` disables
  all checks. Callers loading untrusted models must set `base_dir`.
* **Hardlink detection is best-effort** — The `st_nlink` check only detects
  hard links at the time of the check. It cannot prevent hard links created
  after the check. On some filesystems or operating systems, `st_nlink` may
  not accurately reflect the number of hard links.

## Future hardening

* **`O_NOFOLLOW` on file open** — Use `os.open()` with `O_NOFOLLOW` to close
  the TOCTOU window at the kernel level (Linux/macOS).
* **`_open_validated()` wrapper** — Centralize file-open + security checks so
  future code paths cannot accidentally bypass containment.
