---
name: using-itensors
description: Use when choosing or running ITensors.jl or ITensorMPS.jl for DMRG, TEBD, MPS calculations, tensor-network checks, or ITensors setup failures.
---

# ITensors

Software-stack skill for ITensors.jl / ITensorMPS.jl — the harness's canonical 1D and quasi-1D tensor-network workflows. It owns the **software layer**: run mechanics, **software parameters (step 3)**, and the **time estimate (feeds step 4)**. It is the **step-2 handoff target** from `/method-mps` (DMRG/TEBD) and `/method-ltrg`.

It does **not** own method selection or the method algorithm / "why" — the DMRG, TEBD, and LTRG algorithms and their convergence criteria live in `/method-mps` and `/method-ltrg` (`## Details`); model choice → `.knowledge/models/`; paper figure facts → `/reproduce-paper`. This card carries the ITensors API surface and parameter values to *express* those methods, not the methods themselves.

## Sources

- Stack contract: `skills/using-itensors/stack.toml`
- Method cards: `skills/method-mps/SKILL.md`, `skills/method-ltrg/SKILL.md`
- Install target: `make install itensors`
- Smoke test: `julia --project=julia-env -e 'using ITensors, ITensorMPS, KrylovKit, MPSKit'`
- Official docs (verify the current API here — there is no in-repo software paper): `https://docs.itensor.org/ITensors/dev/`, ITensorMPS.jl `https://github.com/ITensor/ITensorMPS.jl`
- Local API reference (key API + worked examples, with links to upstream docs): `references/itensors-api.md`

## What ITensors is — step 2 (the handoff target)

What `/method-mps` and `/method-ltrg` route here for, and what to confirm before running.

- **The library.** ITensors.jl with ITensorMPS.jl — a Julia tensor-network library (the ITensor collaboration; Fishman, White, Stoudenmire). Typed indices with automatic contraction matching, block-sparse storage when quantum numbers are conserved, BLAS-backed dense contraction. ITensorMPS.jl carries the MPS/MPO/DMRG/TEBD layer.
- **Canonical for** DMRG ground states, imaginary-/real-time TEBD, and MPS measurements; the official docs ship a large worked-example ecosystem (DMRG, TEBD, DMRG-X, quantum-number conservation).
- **Efficiency.** Dense contraction via BLAS; large speedups from block-sparse tensors when `conserve_qns` is on; first-run Julia precompilation is setup time, not physics time.
- **Features to confirm fit the target** before routing here: a built-in site type (`S=1/2`, `Electron`, …), quantum-number conservation, `OpSum` → `MPO` Hamiltonian build, `dmrg` / `apply`, and `svd` with `maxdim` / `cutoff`. Confirm the current spelling against the official docs in *Sources* — the ITensors / ITensorMPS split moved several names.

## Run mechanics

1. Consult `stack.toml` before setup and run `/setup-julia` first when Julia is not usable.
2. Pin lattice, boundary, conserved quantum numbers, bond dimension, sweeps, cutoff, initialization, and the convergence observable (the values come from *Parameters*; the convergence *criteria* are the method card's).
3. Record energy, variance or residual proxy, discarded weight, and bond-dimension convergence.
4. Use cluster execution (`/using-slurm`) when bond dimension, cylinder width, or scans exceed the local threshold.

### Read for the selected workflow

Use the API reference sections needed for the current task:

- DMRG: [solver API](references/itensors-api.md#dmrg) and worked examples in section 11.
- TEBD: [gate application](references/itensors-api.md#tebd) and the real-time example in section 11.3; that section also gives the imaginary-time substitution.
- LTRG: [tensor primitives and normalization](references/itensors-api.md#ltrg).
- Parameter values: [starting points](references/itensors-api.md#starting-points).

## Parameters — step 3 (software)

The source for ITensors / MPS-specific reproduction knobs unless the paper or official code fixes a value. Starting points are software practice, not paper-anchored: begin from each, then converge it — the convergence check (`maxdim`/bond dimension, `cutoff`, sweep count), not the starting number, is what makes the result trustworthy.

What to pin:

- **System / operator:** site type, length / width, boundary, conserved quantum numbers, MPO convention, long-range terms, and whether PBC forces a larger bond dimension.
- **Algorithm:** DMRG for ground states; imaginary-time TEBD for a preparation / evolution route; real-time TEBD only for dynamics.
- **Accuracy:** `maxdim` schedule, sweep count, `cutoff`, noise schedule if needed, TEBD time step and total time, Krylov/Trotter settings.
- **Initialization:** paper-stated state, product state in the target sector, random MPS, warm start, seed policy.
- **Measurements:** observable, normalization, correlation range, cadence, and whether edge effects require a bulk window.
- **Convergence diagnostics the tool exposes:** energy vs sweep and `chi`, variance / residual proxy, discarded weight, `tau` extrapolation for TEBD. The *criteria* for "converged" are the method card's.

Read [parameter starting points](references/itensors-api.md#starting-points) for the selected algorithm; these are starting values to converge, not fixed requirements.

## Caller Contract

The scientific values — model, lattice, sectors, observable, bond-dimension target, convergence criteria, validation target — are caller-supplied; resolve open ones via the step-4 brainstorm, deferring the method algorithm / "why" and the convergence criteria to `/method-mps` or `/method-ltrg` and model physics to the model card. This skill turns agreed values into a runnable ITensors script; it does not originate them.

## Time estimate — feeds step 4

Estimate from length `L`, local dimension `d`, bond dimension `chi`, sweeps / time steps, and whether symmetries are used; the result feeds `/reproduce-paper`'s step-4 resource confirmation.

- DMRG wall time scales roughly as `sweeps · L · chi^3` times the local MPO/site factor; memory roughly `L · chi^2` tensors, with a dtype and conserved-sector factor.
- TEBD wall time scales as `time_steps · gates · chi^3`; memory follows the same `L · chi^2` pattern.
- First-run Julia precompilation is setup time, not physics time; report it separately.
- For uncertain cases, a tiny probe may time a few low-`chi` sweeps or TEBD steps, then extrapolate to the paper `chi` and the largest local-PC-in-15-min setting.
