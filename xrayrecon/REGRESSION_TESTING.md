# Regression Testing

The repository uses `pyproject.toml` as the canonical package and pytest
configuration. From a clean Python 3.10–3.12 environment:

```bash
python -m pip install -e ".[test]"
python -m pytest -m "not slow"
```

## Test layers

- **Unit:** pure registration math, scoring, session state, spacing, materials,
  layer-range construction, and deterministic synthetic generation.
- **Integration:** registration acceptance/invalidation and explicit label
  transfer across the application model and registration session.
- **GUI:** offscreen interaction with the embedded registration panel and main
  workspace. These tests assert state and signals rather than pixels or GPU
  output.
- **Slow:** reserved for larger datasets or performance checks that should not
  block the normal development loop.

The fast suite is expected to remain under 60 seconds on a normal development
machine and is run on Windows with Python 3.12 in CI.

## Synthetic registration contract

`xrayrecon.synthetic_registration` owns the deterministic test and demo cases.
Every case records its seed, physical spacing, known transform, geometry type,
identifiability, and minimum expected score gain. The canonical matrix covers:

- identity and translation on each axis;
- in-plane rotation and uniform scale;
- combined 5-DOF transforms;
- different grid shapes and voxel spacing;
- intensity remapping, noise, and partial cropping;
- constant low-information data; and
- intentionally ambiguous symmetric geometry.

Known-transform tests use asymmetric, feature-rich geometry. Constant and
symmetric cases must not be used to claim transform recovery.

## Reliability rules

- Use fixed seeds for all random data.
- Never import fixtures from the test package into application code.
- Core algorithm tests must not require Qt or OpenGL.
- GUI tests set `QT_QPA_PLATFORM=offscreen` before importing Qt.
- Do not depend on test execution order or persistent files.
- Use pytest temporary directories for generated files.
- Do not use pixel-perfect screenshots as the primary GUI oracle.
- Project-owned `RuntimeWarning` and `FutureWarning` messages are test errors.
- External dependency deprecations are reported but do not fail the suite.
- A suggested transform is never equivalent to acceptance or label transfer.
