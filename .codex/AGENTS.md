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

## Knowledge Base Use

Treat `.knowledge/` as an on-demand reference, not as default session context, a
curriculum, or a task list. Never recursively read, concatenate, or summarize the
whole directory.

- Start with the repository skill that matches the task. Let `quantum-model`,
  `physics`, `method-*`, `scaling-fit`, or a software skill route the work to the
  relevant knowledge card.
- Form a narrow question before opening knowledge files. Search filenames and
  indexes first, then read at most three `.knowledge/` files initially unless the
  active skill explicitly requires more. If a concrete question remains, expand
  one file at a time.
- For a named model, prefer `models/<name>/MODEL.md`. For a phase, mechanism, or
  diagnostic, prefer `physics/<topic>/PHYSICS.md`. Read linked cards only when
  the first card identifies a dependency relevant to the task.
- For method selection, start with `method-property-map.md` or `methods/INDEX.md`,
  then open only the selected `methods/<name>/METHOD.md`. Use `conventions.md`,
  `limits.md`, and `symmetry-cheatsheet.md` only when the task needs those checks.
- For literature evidence, read the relevant local `INDEX.md` before opening a
  specific rendered reference. Do not scan an entire literature family or read
  `ref.bib` unless the task needs bibliography metadata or a collection-wide
  search.
- Prefer the most specific structured card and the sources it cites. Do not load
  adjacent cards merely because they share a directory.
- Treat local cards as curated project context, not proof that a claim is current.
  Verify version-sensitive, recent, contested, or source-critical claims against
  primary sources when the task requires them.
- When a result depends materially on a knowledge file, name that file in the
  analysis or report so the user can inspect the basis of the claim.

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
    include("../src/ScarToGriffith.jl")
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
