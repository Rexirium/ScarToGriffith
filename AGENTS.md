# Repository Guidelines

## Research Context

ScarToGriffith investigates the relationship between quantum many-body scars
(QMBS) and the quantum Griffith phase (QGP). QMBS may contribute to or provide
a mechanism for QGP behavior, but that causal relationship is a hypothesis,
not an assumption. Distinguish evidence of correlation from evidence of
causation in code, documentation, and analysis.

The research direction is still exploratory. Avoid imposing domain boundaries,
public APIs, or a permanent workflow before concrete research requirements make
them useful.

## Current Layout

- `Project.toml` and `Manifest.toml` define the Julia dependency environment;
  this repository is not currently a Julia package.
- `src/ScarToGriffith.jl` defines the plain `ScarToGriffith` module.
- Add focused files under `src/` only when shared research code emerges.
- Keep generated datasets, figures, and machine-specific configuration out of
  version control.

## Commands

Run commands from the repository root.

Instantiate dependencies:

`julia --project=. -e 'using Pkg; Pkg.instantiate()'`

Load the module without redefining it:

```julia
if !isdefined(@__MODULE__, :ScarToGriffith)
    include("src/ScarToGriffith.jl")
    using .ScarToGriffith
end
```

## Skill Development

- Create skills needed specifically by this repository directly under
  `.agents/skills/<skill-name>`, and update those repository skills in place.
- For repository-specific skills, run the initializer with
  `--path .agents/skills` and do not stage them in `.skill-build`.
- Create cross-project, globally reusable skills under
  `%USERPROFILE%\.codex\skills\<skill-name>` unless another location is
  explicitly requested.

## Development Conventions

- Use four-space indentation and standard Julia naming conventions.
- Prefer small, type-stable functions with explicit inputs and outputs.
- Keep exploratory assumptions visible and document the physical interpretation
  of parameters, observables, and numerical tolerances.
- Seed randomized computations when reproducibility matters.
- Add focused tests when stable behavior appears; no permanent test harness is
  required at this exploratory stage.
- Do not edit `Manifest.toml` manually; update it through Julia's package
  manager.
- Do not commit generated plots, PDFs, archives, or local editor settings.
