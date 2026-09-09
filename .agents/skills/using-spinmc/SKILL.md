---
name: using-spinmc
description: Use when choosing, running, or validating SpinMonteCarlo.jl for finite-temperature classical lattice-spin Monte Carlo or the supported QuantumXXZ loop algorithm, including temperature scans, critical observables, custom lattices, autocorrelation, checkpoints, and reproducible sampling.
---

# Using SpinMonteCarlo.jl

## Overview

Use this skill for the `yomichi/SpinMonteCarlo.jl` package. Do not confuse it with `SpinMC.jl` or `ClassicalSpinMC.jl`, whose APIs are unrelated.

**Core principle:** map the Hamiltonian to package conventions first. Accept a result only after its statistics converge.

## Sources

- Stack contract: `stack.toml`
- Current package details and parameter tables: `references/spinmontecarlo-api.md`
- Runnable square-lattice Ising example: `scripts/ising_square.jl`
- Official manual: `https://yomichi.github.io/SpinMonteCarlo.jl/latest/`
- Official repository: `https://github.com/yomichi/SpinMonteCarlo.jl`

Read the API reference before writing a new simulation or using custom models, lattices, estimators, snapshots, or checkpoints.

## Fit

Use built-ins for `Ising`, `Potts`, `Clock`, `XY`, `AshkinTeller`, or `QuantumXXZ` on a supported lattice. Route generic fermions, classical Heisenberg spins, real-time dynamics, or unsupported sign-problem cases to another method or tool.

## Workflow

1. Record `pkgversion(SpinMonteCarlo)` and verify version-sensitive calls against the official manual.
2. Pin the Hamiltonian, sign convention, lattice name, dimensions, boundary conditions, temperature `T`, coupling arrays by bond type or site type, observable normalization, and seed policy.
3. Choose `local_update!` as a classical baseline. Use `SW_update!` or `Wolff_update!` for supported classical cluster sampling. Use `loop_update!` only for `QuantumXXZ`. For frustrated Ising couplings, cluster updates remain unbiased but may not accelerate mixing. Compare them with local updates.
4. Run a small seeded pilot. Inspect returned keys, limiting cases, and agreement between update algorithms before a production scan.
5. Increase thermalization and measurement MCS independently. Use binned jackknife errors and multiple independent seeds. Use autocorrelation extrapolation when the installed version supports it. Never replace these estimates with an IID standard error over raw correlated samples.
6. For phase or critical claims, scan temperature and several sizes. Check Binder crossings, susceptibility and specific heat, histogram or metastability signals, and finite-size drift.
7. Report the package version and full `Parameter`. Include the lattice and boundary convention, seeds and IDs, thermalization, MCS, binning, update method, uncertainty, autocorrelation evidence, and convergence checks.

## Quick Reference

| Task | API or choice |
|---|---|
| Install | `Pkg.add("SpinMonteCarlo")` |
| Input | `Parameter`, an alias of `Dict{String,Any}` |
| Run one or many | `runMC(param)` or `runMC(params; parallel=..., autoID=...)` |
| Statistics | `mean(jk)`, `stderror(jk)`, `confidence_interval(jk, p)` |
| Classical updates | `local_update!`, `SW_update!`, `Wolff_update!` |
| Quantum XXZ | `QuantumXXZ` with `loop_update!` |
| Geometry | `generatelattice(param)` and custom lattice, Bravais, and unit-cell dictionaries |

## Common Mistakes

- Use exact string keys: `"Thermalization"`, not `"Therm"`; `"T"` is temperature, not inverse temperature.
- Give each batch member a distinct mutable `Parameter` and reproducible stream via integer `"Seed"` plus `"ID"` or `autoID=true`.
- Do not infer a thermodynamic transition from one size, one seed, or one short chain.
- Do not treat checkpoint restart as portable across Julia or package system-image changes.
- Before using snapshot or RNG options from the latest docs, confirm that the installed version exposes them.
