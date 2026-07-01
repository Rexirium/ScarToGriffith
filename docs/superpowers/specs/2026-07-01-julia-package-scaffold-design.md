# Julia Package Scaffold Design

## Goal

Initialize ScarToGriffith as a conventional Julia package without guessing at
the eventual physics-domain architecture.

## Scientific Framing

ScarToGriffith investigates the relationship between quantum many-body scars
(QMBS) and the quantum Griffith phase (QGP). One hypothesis is that QMBS may
contribute to, or provide a mechanism for, QGP behavior, but the project does
not assume that this causal relationship is established. Future numerical work
should distinguish evidence of correlation from evidence of causation.

## Structure

- `src/ScarToGriffith.jl` defines the top-level `ScarToGriffith` module. It
  initially exposes no public API.
- `test/runtests.jl` loads the package and provides a minimal smoke test that
  confirms the module is available.
- `AGENTS.md` records repository-specific setup, testing, layout, and
  contribution guidance for future coding agents.
- `Project.toml` gains standard package identity metadata and Julia's `Test`
  standard library under test extras and targets. Existing dependencies remain
  unchanged.

## Verification

Instantiate the environment and run the package test suite with:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The scaffold is complete when Julia can load `ScarToGriffith` and the smoke
test passes.

## Scope

This initialization does not introduce domain modules, numerical algorithms,
data formats, plotting APIs, or simulation workflows. Those boundaries should
be designed when their requirements are known. In particular, the scaffold
does not encode a presumed direction of causation between QMBS and QGP.
