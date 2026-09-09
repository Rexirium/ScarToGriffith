# SpinMonteCarlo.jl API and Physics Checklist

## Contents

- Package identity and version gate
- Core driver parameters
- Built-in models, lattices, and updates
- Results and statistical analysis
- Parallel scans, checkpoints, and snapshots
- Custom extensions
- Physics validation checklist
- Official sources

## Package Identity and Version Gate

This reference covers `SpinMonteCarlo` by Yuichi Motoyama (`yomichi/SpinMonteCarlo.jl`), not the separate packages `SpinMC.jl` or `ClassicalSpinMC.jl`.

Check the loaded version and source before using optional APIs:

```julia
using SpinMonteCarlo
@show pkgversion(SpinMonteCarlo)
@show pathof(SpinMonteCarlo)
```

The repository environment had SpinMonteCarlo v1.2.2 when this skill was authored. The latest online manual already documents later additions such as explicit `"RNG"`, snapshot controls, deterministic child-seed derivation, and removed pre-v1.3 snapshot APIs. Treat the online manual for the installed/tagged version as authoritative.

## Core Driver Parameters

`Parameter` is `Dict{String,Any}`. `runMC` requires:

| Key | Meaning | Example |
|---|---|---|
| `"Model"` | Model type, not an instance | `Ising` |
| `"Lattice"` | Exact lattice name | `"square lattice"` |
| `"Update Method"` | In-place update function | `SW_update!` |

Common optional keys:

| Key | Meaning / default |
|---|---|
| `"MCS"` | Measurement steps; default `8192` |
| `"Thermalization"` | Discarded equilibration steps; default `MCS >> 3` |
| `"Binning Size"` | Measurements per bin; automatic when both bin controls are absent |
| `"Number of Bins"` | Alternative bin control |
| `"Seed"` | Integer seed for reproducibility |
| `"ID"` | Job/checkpoint identity; with `Seed`, selects a distinct child stream in current releases |
| `"Checkpoint Interval"` | Seconds between checkpoints; `0.0` disables restart I/O |
| `"Checkpoint Filename Prefix"` | Prefix used with `ID` |
| `"Verbose"` | Print parameters and progress |

Current manuals also document `"RNG"`, `"Snapshot Interval"`, and `"Snapshot Filename Prefix"`. Gate these on the installed version.

Model-specific keys are expanded by site or bond type when necessary:

| Model | Hamiltonian parameters | Required model key |
|---|---|---|
| `Ising` | `"J"` | none |
| `Potts` | `"J"` | `"Q"` |
| `Clock` | `"J"` | `"Q"` |
| `XY` | `"J"` | none |
| `AshkinTeller` | `"Jsigma"`, `"Jtau"`, `"K"` | none |
| `QuantumXXZ` | `"Jz"`, `"Jxy"`, `"Gamma"` | `"S"` |

Use a scalar for one bond/site type or a vector matching `numbondtypes(model)` / `numsitetypes(model)`. Do not silently substitute `"J"` for the distinct `QuantumXXZ` couplings when anisotropy matters.

## Built-in Models, Lattices, and Updates

Built-in lattice names are exact strings:

- `"chain lattice"`
- `"bond-alternating chain lattice"`
- `"square lattice"`
- `"J1J2 square lattice"`
- `"triangular lattice"`
- `"honeycomb lattice"`
- `"ladder"`
- `"cubic lattice"`
- `"fully connected graph"`

`"L"` sets the main extent; rectangular/3D definitions may also use their lattice-specific width/height parameters. Confirm with `generatelattice(param)` and inspect `size`, `numsites`, `numbonds`, `bondtype`, and boundary conventions before production.

Installed v1.2.2 provides `local_update!`, `SW_update!`, and `Wolff_update!` methods for every built-in classical model. `loop_update!` is the built-in `QuantumXXZ` update.

Algorithm choice:

| Situation | Starting choice | Required check |
|---|---|---|
| Classical smoke/baseline | `local_update!` | acceptance/mixing and agreement with cluster run |
| Classical critical slowing down | `SW_update!` or `Wolff_update!` | supported method and decorrelation improvement |
| Frustrated Ising | cluster or local | cluster result remains unbiased, but may not speed relaxation |
| `QuantumXXZ` | `loop_update!` | sign, low-temperature convergence, coupling convention |

## Results and Statistical Analysis

`runMC(param)` returns a dictionary of jackknife observables. Available keys depend on the model and estimator, so inspect them:

```julia
result = runMC(param)
println(sort!(collect(keys(result))))
```

Use the package statistics:

```julia
using Statistics

jk = result["Specific Heat"]
value = mean(jk)
error = stderror(jk)
ci95 = confidence_interval(jk, 0.95)
```

Typical classical outputs include energy, magnetization moments, Binder ratio, susceptibility, connected susceptibility, specific heat, and timing. XY/Clock results also include vector components and helicity moduli. `QuantumXXZ` uses sign-aware improved estimators internally; inspect the output keys rather than assuming the classical set.

In releases that expose binning observables, use `binning`, `tau`, `extrapolate_tau`, and `extrapolate_stderror` to test bin-size saturation. An uncertainty is acceptable only after increasing bin size no longer materially inflates it and independent chains agree.

For nonlinear observables such as Binder ratios and fluctuation-derived response functions, keep the package jackknife result. Do not recompute an error by inserting independently averaged moments into naive propagation.

## Parallel Scans, Checkpoints, and Snapshots

Build a fresh parameter dictionary for every point:

```julia
params = [
    Parameter(
        "Model" => Ising,
        "Lattice" => "square lattice",
        "L" => L,
        "T" => T,
        "J" => 1.0,
        "Update Method" => SW_update!,
        "MCS" => 8192,
        "Thermalization" => 2048,
        "Seed" => 20260908,
    )
    for L in (8, 16, 32), T in range(2.1, 2.4; length=13)
]

results = runMC(params; parallel=false, autoID=true)
```

Set `parallel=true` only after adding Julia workers and confirming the environment on each worker. Treat independent parameter points and independent chains as the primary parallelization axis.

Checkpoint files use the prefix and `ID`. Avoid collisions between concurrent runs. Serialized restarts may fail after Julia version or system-image changes; a restart also does not excuse rechecking the chain parameters.

Current snapshot files contain flattened configurations without embedded model/lattice metadata. Store that metadata next to the output. Julia column-major flattening means Ashkin-Teller rows interleave the two spin fields.

## Custom Extensions

For a custom lattice, define lattice, Bravais, and unit-cell dictionaries and pass them through `"LatticeDict"`, `"BravaisDict"`, and `"UnitcellDict"`. Verify site coordinates, periodic flags, bond endpoints, and bond-type counts on a tiny lattice.

A custom model must contain `lat::Lattice` and an RNG field, provide a `Model(param::Parameter)` constructor, and define `convert_parameter`. Use `@gen_convert_parameter` when its scalar/vector expansion matches the model. Custom workflows may also define an update, estimator, `default_estimator`, `postproc`, and `snapshot`.

An estimator returns a `Dict{String,Any}` for one configuration. `postproc` should use `jackknife(obs)` before forming nonlinear functions of expectation values.

## Physics Validation Checklist

- Write the simulated Hamiltonian and compare its signs/normalization with the package convention.
- Verify lattice site/bond counts and boundaries on a small size.
- Check high-temperature and low-temperature limiting behavior.
- Compare at least two valid updates or an exact small-system result.
- Increase thermalization until means lose dependence on initialization.
- Increase MCS, bins, and independent chains until errors stabilize.
- For ground-state claims from finite temperature, demonstrate convergence as `T -> 0` (or `beta -> infinity`).
- For critical claims, use several sizes and finite-size scaling; a peak or crossing at one size is not a thermodynamic transition.
- For `QuantumXXZ`, monitor the sign and refuse an uncontrolled sign-problem interpretation.

## Official Sources

- Repository and README: https://github.com/yomichi/SpinMonteCarlo.jl
- Manual home and basic example: https://yomichi.github.io/SpinMonteCarlo.jl/latest/
- Driver, parameters, output, parallelism, restart: https://yomichi.github.io/SpinMonteCarlo.jl/latest/runmc/
- Lattice definitions: https://yomichi.github.io/SpinMonteCarlo.jl/latest/lattice/
- Custom model/update/estimator/postprocess: https://yomichi.github.io/SpinMonteCarlo.jl/latest/develop/
- Exported public API: https://yomichi.github.io/SpinMonteCarlo.jl/latest/lib/public/

Sources were checked on 2026-09-08. Recheck upstream for version-sensitive work.
