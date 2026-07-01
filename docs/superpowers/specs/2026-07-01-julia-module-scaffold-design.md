# Julia Module Scaffold Design

## Goal

Initialize ScarToGriffith as an exploratory Julia workspace with a plain module,
without turning the repository into a package or guessing at the eventual
research architecture.

## Scientific Framing

ScarToGriffith investigates the relationship between quantum many-body scars
(QMBS) and the quantum Griffith phase (QGP). One hypothesis is that QMBS may
contribute to, or provide a mechanism for, QGP behavior, but the project does
not assume that this causal relationship is established. Future numerical work
should distinguish evidence of correlation from evidence of causation.

## Structure

- `src/ScarToGriffith.jl` defines the top-level `ScarToGriffith` module. It
  initially exposes no public API.
- `AGENTS.md` records repository-specific setup, module loading, layout, and
  contribution guidance for future coding agents.
- `Project.toml` remains a dependency-only environment. It does not gain package
  identity metadata, extras, or targets.
- No domain modules, package API, or permanent test structure are introduced
  before the research direction becomes clearer.

## Module Loading

Commands and exploratory scripts run from the repository root and guard module
loading so rerunning them in an interactive session does not redefine
`ScarToGriffith`:

```julia
if !isdefined(@__MODULE__, :ScarToGriffith)
    include("src/ScarToGriffith.jl")
    using .ScarToGriffith
end
```

## Verification

Instantiate the environment and load the module from the repository root:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'if !isdefined(@__MODULE__, :ScarToGriffith); include("src/ScarToGriffith.jl"); using .ScarToGriffith; end'
```

The scaffold is complete when Julia can load `ScarToGriffith` without treating
the workspace as a package.

## Scope

This initialization does not introduce domain modules, numerical algorithms,
data formats, plotting APIs, simulation workflows, or a package test harness.
Those boundaries should be designed when their requirements are known. In
particular, the scaffold does not encode a presumed direction of causation
between QMBS and QGP.
