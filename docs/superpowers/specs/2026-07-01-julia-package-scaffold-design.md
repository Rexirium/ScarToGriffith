# Julia Package Scaffold Design

## Goal

Initialize ScarToGriffith as a conventional Julia package without guessing at
the eventual physics-domain architecture.

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
be designed when their requirements are known.
