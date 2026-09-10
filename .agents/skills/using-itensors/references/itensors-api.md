# ITensors.jl + ITensorMPS.jl — API & Examples Reference

Single-file API + worked-examples reference for the harness's canonical tensor-network
stack. Covers >90% of real DMRG / TEBD / MPS usage. Code blocks are copied verbatim from
the official docs where possible; constructed snippets (e.g. the Hubbard build) are labeled.

**Official docs** (verify the live API here — the ITensors / ITensorMPS split moved many names):
- ITensors.jl (core: `Index`, `ITensor`, contraction, `svd`/`factorize`, `QN`): https://docs.itensor.org/ITensors/dev/
- ITensorMPS.jl (MPS / MPO / DMRG / TEBD / OpSum / observers): https://docs.itensor.org/ITensorMPS/dev/
- Repo / examples: https://github.com/ITensor/ITensorMPS.jl

**Package split (read first).** As of ITensors.jl v0.8.3, *all* MPS/MPO/DMRG/OpSum/SiteType
functionality lives in **ITensorMPS.jl**. The core **ITensors.jl** package now owns only
`Index`, `ITensor`, `QN`, contraction, and tensor factorizations (`svd`, `factorize`, `eigen`,
`qr`). Every script needs `using ITensors, ITensorMPS`. See *Pitfalls*.

---

## 1. Core: Index

A typed tensor leg. Contraction matches indices by identity (id + tags + prime level), so
index *order* never matters.

```julia
Index(dim::Int; tags::Union{AbstractString, TagSet} = "", plev::Int = 0)
Index(dim::Integer, tags::Union{AbstractString, TagSet}; plev::Int = 0)
# QN (block-sparse) index — pairs of QN => block-size:
Index(qnblocks::Pair{QN, Int64}...; tags = "", plev::Integer = 0)
Index(qnblocks::Vector{Pair{QN, Int64}}; tags = "", plev::Integer = 0)
Index(qnblocks::Vector{Pair{QN, Int64}}, tags; plev::Integer = 0)
```

Example:

```julia
i = Index(2; tags = "l", plev = 1)   # (dim=2|id=818|"l")'
```

Prime / tag / direction manipulation (all out-of-place — return a copy):

| Function | Purpose |
|---|---|
| `prime(i, plinc=1)` / `i'` / `i^pl` | increment prime level |
| `setprime(i, plev)` | set prime level to `plev` |
| `noprime(i)` | set prime level to 0 |
| `settags(i, ts)` | replace tags entirely |
| `addtags(i, ts)` | add tags |
| `removetags(i, ts)` | remove tags |
| `replacetags(i, old, new)` | swap tags |
| `dag(i)` | reverse Index direction (needed for QN tensors) |
| `dim(i)` | dimension |

---

## 2. Core: ITensor

A tensor whose legs are `Index` objects.

**Construction**

```julia
ITensor(inds...)              # zero-filled with given indices
ITensor(undef, inds...)       # uninitialized (faster)
ITensor(x::Number, inds...)   # all elements = x
ITensor(A::AbstractArray, inds...)  # from array data
random_itensor(inds...)       # i.i.d. normal random elements
onehot(i => n)                # single element = 1 (rest 0)
delta(inds...)                # diagonal "copy"/identity tensor of ones
```

**Element access** (by `Index => value` pairs; order-independent):

```julia
A[i => 1, j => 2]          # get
A[i => 1, j => 2] = x      # set
A[i => 2, j => :]          # slice
```

**Contraction** — the `*` operator contracts *all* indices common to both tensors and sums
over them:

```julia
C = A * B
```

Helpers: `inds(T)`, `ind(T, n)`, `dim(T)`, `order(T)` (number of indices), `norm(T)`,
`hasinds(T, is...)`, `commoninds(A, B)`, `uniqueinds(A, B)`, `replaceprime`, `prime(T, ...)`.

**Factorizations** (the workhorses for truncation):

```julia
U, S, V = svd(A::ITensor, left_inds...; kwargs...)
```
Key kwargs: `maxdim::Int` (cap on retained singular values / bond dim), `cutoff::Float64`
(discard squared-singular-value weight below this), `mindim::Int`, `lefttags`, `righttags`
(tags for the new bond index), `use_absolute_cutoff::Bool`, `use_relative_cutoff::Bool`,
`alg ∈ ("divide_and_conquer", "qr_iteration", "recursive")`. Truncation error available as
`S` spectrum / via the returned spectrum object.

```julia
L, R = factorize(A::ITensor, Linds...; ortho="left", which_decomp=nothing, maxdim, cutoff, ...)
# A ≈ L * R ; ortho ∈ ("left", "right", "none")
D, U = eigen(A::ITensor, Linds, Rinds; ishermitian=false, kwargs...)
Q, R = qr(A::ITensor, Linds...)
```

`maxdim` vs `cutoff`: `cutoff` is accuracy-driven (keep enough to lose < cutoff weight);
`maxdim` is a hard cost cap. In production set both — `cutoff` decides, `maxdim` bounds.

---

## 3. Core: QN (quantum numbers)

Holds up to four named integer values; abelian, with optional modular addition.

```julia
QN(qvs...)                          # tuple form
QN(name, val::Int, modulus::Int=1)  # single named value
QN(val::Int, modulus::Int=1)        # single unnamed value
```

Examples:

```julia
q = QN(("Sz", 1))
q = QN(("N", 1), ("Sz", -1))
q = QN(("P", 0, 2), ("Sz", 0))      # P is mod-2 (parity)
```

Adding/subtracting QNs acts element-wise per name (missing name ⇒ 0). Accessors: `val(q, name)`,
`modulus(q, name)`, `zero(q)`. In MPS work you rarely build `QN` by hand — `siteinds(...;
conserve_qns=true)` attaches them automatically (see §4).

---

## 4. SiteType and `siteinds`

`siteinds("Type", N; kwargs...)` builds a length-`N` `Vector{Index}` of the named local
Hilbert space; the kwargs turn on quantum-number conservation. Built-in types:

| SiteType | local dim | typical use |
|---|---|---|
| `"S=1/2"` (= `"S=½"`), `"Qubit"` | 2 | spin-½ / qubits |
| `"S=1"` | 3 | spin-1 chains |
| `"Fermion"` | 2 | spinless fermions |
| `"Electron"` | 4 | Hubbard (spinful fermions) |
| `"tJ"` | 3 | t-J (no double occupancy) |
| `"Boson"` / `"Qudit"` | `dim` kwarg | bosons / d-level |

**Conservation kwargs** (passed through `siteinds`):

```julia
# S=1/2 / S=1:  conserve_qns, conserve_sz, conserve_szparity, qnname_sz="Sz"
# Electron:     conserve_qns, conserve_sz, conserve_nf, conserve_nfparity
#               qnname_sz="Sz", qnname_nf="Nf", qnname_nfparity="NfParity"
# Fermion:      conserve_qns, conserve_nf, conserve_nfparity, conserve_sz
# tJ:           conserve_qns, conserve_sz, conserve_nf, conserve_nfparity
# Qubit:        conserve_qns, conserve_parity, conserve_number
# Boson/Qudit:  dim=2, conserve_qns, conserve_number
sites = siteinds("S=1/2", N; conserve_qns=true)        # conserves total S^z
sites = siteinds("Electron", N; conserve_qns=true)     # conserves N_f and S^z
```

`conserve_qns=true` is the umbrella switch; the per-symmetry flags default to it. Turning it on
enables **block-sparse** storage → large speedups, but pins the calculation to the sector of the
initial state.

**Operators per site type** (names passed to `op`, `OpSum`, `expect`):

- `S=1/2`, `S=1`: `"Sz"`, `"Sx"`, `"Sy"`, `"S+"`, `"S-"`, `"Id"`
- `Electron`: `"Nup"`, `"Ndn"`, `"Nupdn"`, `"Ntot"`, `"Cup"`, `"Cdagup"`, `"Cdn"`, `"Cdagdn"`,
  `"Sz"`, `"Sx"`, `"Sy"`, `"S+"`, `"S-"`
- `tJ`: same fermion/spin operator names as `Electron`, no double occupancy
- `Boson`/`Qudit`: `"N"`, `"A"`, `"Adag"` (creation/annihilation)

**States** (names passed to `MPS(sites, states)` / `state`): `"Up"`, `"Dn"` (spins);
`"Emp"`, `"Up"`, `"Dn"`, `"UpDn"` (Electron); `"0"`,`"1"` (Fermion/Qubit).

**`op` and `state`:**

```julia
op(opname::String, s::Index; kwargs...)
op(opname::String, sites::Vector{<:Index}, n::Int; kwargs...)
op(M::Matrix, s::Index...)               # custom operator from a matrix
state(s::Index, name::String; kwargs...)

s  = siteind("S=1/2")
Sz = op("Sz", s)
sup = state(s, "Up"); sdn = state(s, "Dn")
```

---

## 5. OpSum → MPO (Hamiltonian builder)

`OpSum` is a symbolic sum of operator strings. Build it, then convert to an MPO; the conversion
does automatic bond-dimension compression.

```julia
opsum = OpSum()
# += with a tuple (coef optional, defaults to 1):
opsum += "Sz", 2, "Sz", 3
opsum += 0.5, "S+", 4, "S-", 5
opsum -= 4, "Sz", j, "Sz", j+1          # subtraction works
# equivalent add! API:
add!(opsum, "Sz", 2, "Sz", 3)
add!(opsum, 0.5, "S+", 4, "S-", 5)
# broadcasted .+= avoids reallocation in tight loops:
opsum .+= (0.5, "S+", 5, "S-", 6)
```

Convert to MPO:

```julia
MPO(os::OpSum, sites::Vector{<:Index}; splitblocks=true, kwargs...)
MPO(eltype::Type{<:Number}, os::OpSum, sites::Vector{<:Index}; splitblocks=true, kwargs...)

H = MPO(os, sites)
H = MPO(Float32, os, sites)          # control element type
H = MPO(os, sites; splitblocks=false)
```

`splitblocks=true` (default) increases sparsity → faster DMRG with conserved QNs.

---

## 6. MPS / MPO

**MPS constructors**

```julia
MPS(N::Int)                                  # N default tensors
MPS(sites::Vector{<:Index}; linkdims=1)      # empty tensors from site indices
MPS(sites::Vector{<:Index}, states)          # product state from state names
MPS(sites, n -> isodd(n) ? "Up" : "Dn")      # product state from a function

random_mps(sites; linkdims=1)
random_mps(eltype::Type, sites; linkdims=1)
random_mps(sites, state; linkdims=1)         # random around a product state
```

(Older spellings `randomMPS` / `productMPS` are deprecated → `random_mps` / `MPS(sites, states)`.)

**MPO constructors**

```julia
MPO(N::Int)
MPO(sites, ops::Vector{String})   # different op per site
MPO(sites, op::String)            # same op on all sites
# plus MPO(os::OpSum, sites)  — see §5
```

**Properties & indices**

```julia
length(M); maxlinkdim(M); norm(M)
flux(M); totalqn(M)                  # total QN (QN-conserving states)
siteind(M, j); siteinds(M); linkind(M, j)
findsite(M, is); findsites(M, is)
```

**Orthogonalization / truncation**

```julia
orthogonalize!(M, j)                 # gauge center to site j (in place)
orthogonalize(M, j)                  # out-of-place
truncate!(M; cutoff, maxdim, site_range)
```

**Algebra**

```julia
inner(A::MPS, B::MPS)                # ⟨A|B⟩
inner(A', H::MPO, B)                 # ⟨A|H|B⟩  (note the prime on A)
dot(A, B); loginner(A, B)            # loginner for tiny/huge overlaps
norm(A); normalize(A); normalize!(A)
+(A::MPS...; cutoff=1e-15, maxdim, alg="densitymatrix")   # MPS sum
contract(A::MPO, ψ::MPS; cutoff, maxdim, alg="densitymatrix")
apply(A::MPO, ψ::MPS)               # = replaceprime(contract(...), 2=>1)
outer(x::MPS, y::MPS)               # |x⟩⟨y| as MPO
projector(x::MPS; normalize=true)  # |x⟩⟨x|
```

`copy(M)` is shallow (shares tensor data); use `deepcopy(M)` for an independent copy.

---

<a id="dmrg"></a>

## 7. DMRG (ground / excited states)

```julia
dmrg(H::MPO, psi0::MPS; kwargs...)
dmrg(Hs::Vector{MPO}, psi0::MPS; kwargs...)               # H = sum of the MPOs
dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS; weight=1.0, kwargs...)  # excited states
# Returns: (energy::Number, psi::MPS)
```

| kwarg | default | meaning |
|---|---|---|
| `nsweeps::Int` | (required) | number of sweeps |
| `maxdim` | — | max bond dim; scalar or per-sweep array |
| `cutoff` | — | SVD truncation weight; scalar or per-sweep array |
| `mindim` | — | min bond dim where feasible |
| `noise` | — | noise term to escape stuck states; usually a decaying array |
| `observer` | — | `AbstractObserver` for per-sweep measurement / stopping |
| `outputlevel` | 1 | 0 = silent, 1 = per-sweep, 2 = per-bond |
| `weight` | 1.0 | penalty weight for orthogonality to `Ms` (excited states) |
| `ishermitian` | true | whether `H` is Hermitian |
| `eigsolve_krylovdim` | 3 | local Krylov subspace dim |
| `eigsolve_tol` | 1e-14 | local eigensolver tolerance |
| `eigsolve_maxiter` | 1 | Krylov restarts |
| `write_when_maxdim_exceeds` | — | offload to disk above this bond dim |
| `write_path` | `tempdir()` | disk path for offloading |

Per-sweep arrays: `maxdim = [10,20,100,100,200]` grows the bond dimension one sweep at a time;
a length-1 array like `cutoff = [1E-10]` is reused for every sweep.

---

## 8. Measurement

```julia
expect(psi::MPS, op::String; sites=1:length(psi))   # ⟨ψ|Oⱼ|ψ⟩ per site
expect(psi, ["Sx", "Sz"])                           # several operators at once
correlation_matrix(psi::MPS, A::String, B::String;
                   sites=1:length(psi), ishermitian=false)  # ⟨Aᵢ Bⱼ⟩ matrix
sample(m::MPS)                                       # one sample from |ψ|²
sample!(m::MPS)                                      # samples + re-orthogonalizes
```

Examples:

```julia
Z  = expect(psi, "Sz")            # vector over all sites
Z  = expect(psi, "Sz"; sites=2:4)
XZ = expect(psi, ["Sx", "Sz"])
C  = correlation_matrix(psi, "Sz", "Sz")
```

`expect` / `correlation_matrix` assume `psi` is normalized; call `normalize!(psi)` first if a
gate sequence (TEBD) has changed the norm.

---

<a id="tebd"></a>

## 9. TEBD (apply gates / time evolution)

Build two-site Trotter gates with `op`/`exp`, then `apply` them to the MPS. `apply` truncates
with `cutoff`/`maxdim` per gate. Use `-im*τ` for real-time, `-τ` for imaginary-time evolution.

```julia
apply(o::ITensor, ψ::MPS; cutoff, maxdim, move_sites_back=true)
apply(gates::Vector{ITensor}, ψ::MPS; cutoff, maxdim, ...)
```

See the full worked example in §11.

---

## 10. Observers (convergence tracking & custom measurement)

Subtype `AbstractObserver` and define `ITensorMPS.measure!`; pass via `observer=` to `dmrg`.
Available keywords inside `measure!`: `bond`, `sweep`, `half_sweep`, `psi`, `projected_operator`.
There is also a built-in `DMRGObserver` for energy/observable convergence and early stopping.

Entanglement-entropy observer (verbatim from docs):

```julia
using ITensors, ITensorMPS

mutable struct EntanglementObserver <: AbstractObserver
end

function ITensorMPS.measure!(o::EntanglementObserver; bond, psi, half_sweep, kwargs...)
  wf_center, other = half_sweep==1 ? (psi[bond+1],psi[bond]) : (psi[bond],psi[bond+1])
  U,S,V = svd(wf_center, uniqueinds(wf_center,other))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
  println("  Entanglement across bond $bond = $SvN")
end
```

---

## 11. Worked examples (verbatim from the official docs)

### 11.1 Ground-state DMRG (Heisenberg, S=1, N=100)

```julia
using ITensors, ITensorMPS
let
  N = 100
  sites = siteinds("S=1",N)

  os = OpSum()
  for j=1:N-1
    os += "Sz",j,"Sz",j+1
    os += 1/2,"S+",j,"S-",j+1
    os += 1/2,"S-",j,"S+",j+1
  end
  H = MPO(os,sites)

  psi0 = random_mps(sites;linkdims=10)

  nsweeps = 5
  maxdim = [10,20,100,100,200]
  cutoff = [1E-10]

  energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

  return
end
```

### 11.2 QN-conserving DMRG (fixed total S^z sector)

The total QN is fixed by the *initial state* — the alternating Up/Dn product state has total
S^z = 0, and DMRG stays in that sector for the whole run.

```julia
using ITensors, ITensorMPS
let
  N = 100
  sites = siteinds("S=1",N;conserve_qns=true)

  os = OpSum()
  for j=1:N-1
    os += "Sz",j,"Sz",j+1
    os += 1/2,"S+",j,"S-",j+1
    os += 1/2,"S-",j,"S+",j+1
  end
  H = MPO(os,sites)

  state = [isodd(n) ? "Up" : "Dn" for n=1:N]
  psi0 = MPS(sites,state)
  @show flux(psi0)

  nsweeps = 5
  maxdim = [10,20,100,100,200]
  cutoff = [1E-10]

  energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)

  return
end
```

### 11.3 TEBD real-time evolution (transverse measurement of ⟨Sz⟩)

```julia
using ITensors, ITensorMPS

let
  N = 100
  cutoff = 1E-8
  tau = 0.1
  ttotal = 5.0

  # Make an array of 'site' indices
  s = siteinds("S=1/2", N; conserve_qns=true)

  # Make gates (1,2),(2,3),(3,4),...
  gates = ITensor[]
  for j in 1:(N - 1)
    s1 = s[j]
    s2 = s[j + 1]
    hj =
      op("Sz", s1) * op("Sz", s2) +
      1 / 2 * op("S+", s1) * op("S-", s2) +
      1 / 2 * op("S-", s1) * op("S+", s2)
    Gj = exp(-im * tau / 2 * hj)
    push!(gates, Gj)
  end
  # Include gates in reverse order too
  # (N,N-1),(N-1,N-2),...
  append!(gates, reverse(gates))

  # Initialize psi to be a product state (alternating up and down)
  psi = MPS(s, n -> isodd(n) ? "Up" : "Dn")

  c = div(N, 2) # center site

  # Compute and print <Sz> at each time step
  # then apply the gates to go to the next time
  for t in 0.0:tau:ttotal
    Sz = expect(psi, "Sz"; sites=c)
    println("$t $Sz")

    t≈ttotal && break

    psi = apply(gates, psi; cutoff)
    normalize!(psi)
  end

  return
end
```

The half-step `exp(-im*tau/2*hj)` plus the appended reverse list realizes a 2nd-order Trotter
step (error O(τ²)). For *imaginary*-time (ground-state cooling) use `exp(-tau/2*hj)`.

### 11.4 Excited states via the weight-penalty method (verbatim)

`dmrg(H, [psi0], psi1_init; ..., weight)` finds the first excited state by penalizing overlap
with the already-found state(s).

```julia
using ITensors, ITensorMPS

let
  N = 20
  sites = siteinds("S=1/2",N)
  h = 4.0
  weight = 20*h

  os = OpSum()
  for j=1:N-1
    os -= 4,"Sz",j,"Sz",j+1
  end
  for j=1:N
    os -= 2*h,"Sx",j;
  end
  H = MPO(os,sites)

  nsweeps = 30
  maxdim = [10,10,10,20,20,40,80,100,200,200]
  cutoff = [1E-8]
  noise = [1E-6]

  psi0_init = random_mps(sites;linkdims=2)
  energy0,psi0 = dmrg(H,psi0_init;nsweeps,maxdim,cutoff,noise)

  psi1_init = random_mps(sites;linkdims=2)
  energy1,psi1 = dmrg(H,[psi0],psi1_init;nsweeps,maxdim,cutoff,noise,weight)

  @show inner(psi1,psi0)
  println("DMRG energy gap = ",energy1-energy0);

  psi2_init = random_mps(sites;linkdims=2)
  energy2,psi2 = dmrg(H,[psi0,psi1],psi2_init;nsweeps,maxdim,cutoff,noise,weight)

  @show inner(psi2,psi0)
  @show inner(psi2,psi1)
end
```

### 11.5 Hubbard model on Electron sites (constructed from documented operators)

Not a verbatim docs example — built from the documented `Electron` operators and the OpSum/DMRG
API above. Hopping `-t Σ (c†ᵢcⱼ + h.c.)` per spin + on-site `U Σ nᵢ↑nᵢ↓`, conserving both N and S^z.

```julia
using ITensors, ITensorMPS
let
  N = 20; t = 1.0; U = 8.0; Npart = N   # half filling
  sites = siteinds("Electron", N; conserve_qns=true)

  os = OpSum()
  for j in 1:N-1
    os += -t, "Cdagup", j, "Cup", j+1
    os += -t, "Cdagup", j+1, "Cup", j
    os += -t, "Cdagdn", j, "Cdn", j+1
    os += -t, "Cdagdn", j+1, "Cdn", j
  end
  for j in 1:N
    os += U, "Nupdn", j
  end
  H = MPO(os, sites)

  # Product state at the target filling / Sz sector (alternating Up/Dn → N=N, Sz=0):
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  psi0 = MPS(sites, state)
  @show flux(psi0)

  nsweeps = 10
  maxdim  = [50,100,200,400,800]
  cutoff  = [1E-10]
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

  nup = expect(psi, "Nup"); ndn = expect(psi, "Ndn")
  return
end
```

---

<a id="starting-points"></a>

### 11.6 Parameter starting points

Unless the paper or official code fixes a value, start from these software-practice values and converge them against the requested accuracy.

Concrete starting points (DMRG and TEBD share the bond-dimension / cutoff controls):

#### DMRG

| Knob | Effect | Starting point |
|---|---|---|
| `maxdim` schedule | Maximum bond dimension per sweep. Drives accuracy and cost. | Grow gently from ~10. Targets: 50–200 (1D chains), 200–1000 (cylinders), 1000+ (frustrated 2D). |
| `cutoff` | SVD truncation threshold. | `1e-10` for entry/medium accuracy; tighten for critical points or if variance is non-zero at convergence. |
| `nsweeps` | Number of sweeps. | 10–30; stop when the energy stops changing within the accuracy goal. |
| Initial state | Random MPS or a product state in the target sector. | Product state for sectors (e.g. Néel for `S^z = 0`); random when there is no clear product-state representative. |
| `noise` | Adds noise to break stuck states (older API). | Use only if convergence stalls. |

#### TEBD

| Knob | Effect | Starting point |
|---|---|---|
| `τ` (Trotter step) | Trotter error scales as `τ^2` (2nd order). Smaller is more accurate but slower. | `0.05–0.1` for entry; reduce if energy not converged. |
| `T_total` | Imaginary-time evolution length. Need `T_total ≫ 1/Δ`. | Start at 10–20 (units where the largest coupling = 1); extend until energy stops dropping. |
| `maxdim` | MPS bond dimension cap. | 50–200 for 1D entry-level. |
| `cutoff` | SVD truncation per gate application. | `1e-10`. |

<a id="ltrg"></a>

### 11.7 LTRG primitives

The LTRG algorithm — build local transfer tensors, then repeatedly absorb a layer into the boundary, SVD-truncate to `Dc`, normalize, and accumulate log scale factors — is owned by `/method-ltrg` (`## Details`). ITensors supplies the primitives to express it; keep index tags explicit, write convergence data incrementally, and record the normalization convention with the output.

```julia
using ITensors

s1, s2 = Index(q, "site1"), Index(q, "site2")           # typed local-basis indices, primed for adjacent layers
gate   = exp(-tau * h)                                   # imaginary-time gate from local h(s1,s2,s1',s2')
U, S, V = svd(T, (s1, s2); maxdim = Dc, cutoff = 1e-12)  # truncate the boundary to Dc
```

## 12. Common pitfalls / gotchas

- **`using ITensors, ITensorMPS` always.** Post-v0.8.3 all MPS/MPO/DMRG/OpSum/SiteType names
  live in ITensorMPS.jl. `MPS`, `dmrg`, `OpSum`, `siteinds`, `expect` are *not* in ITensors.jl.
- **Renamed functions.** `randomMPS → random_mps`, `productMPS → MPS(sites, states)`,
  `randomITensor → random_itensor`, `add!`/`OpSum` is the current name for the old `AutoMPO`.
  When old tutorials fail to resolve a name, check for the new spelling.
- **Prime the bra in `inner(A', H, B)`.** Computing ⟨A|H|B⟩ needs `inner(A', H, B)` — without the
  prime on `A` the site indices do not match the (primed) MPO row indices.
- **Conserved QN is set by the initial state, not a keyword.** `conserve_qns=true` only *enables*
  conservation; the sector is whatever `flux(psi0)` is at construction. Build `psi0` as a product
  state in the target sector and `@show flux(psi0)` to confirm before running.
- **Normalize after TEBD / gate application.** `apply(gates, psi; cutoff)` does not preserve
  norm; call `normalize!(psi)` each step before measuring or `expect` will be off.
- **`cutoff` vs `maxdim`.** `cutoff` is the accuracy knob (discarded weight); `maxdim` is the
  cost ceiling (hard bond-dim cap). Set both: `cutoff` decides retention, `maxdim` bounds it.
  Relying on `maxdim` alone hides under-convergence; relying on `cutoff` alone can blow up cost.
- **Per-sweep arrays are read positionally and the last entry repeats.** `maxdim=[10,20,100]`
  with `nsweeps=5` uses 10, 20, 100, 100, 100. Make the schedule at least as long as your real
  growth phase.
- **`copy` is shallow.** `copy(M::MPS)` shares the underlying tensors; mutate a copy and you
  mutate the original. Use `deepcopy` for true independence.
- **`splitblocks`/QN sparsity.** With `conserve_qns=true`, keep `MPO(os, sites)` default
  (`splitblocks=true`) for the block-sparse speedup; turning it off slows DMRG.
- **First run is slow.** Julia precompilation dominates the first invocation — that is setup
  time, not physics time; report it separately.

---

## 13. Source links

| Section | URL |
|---|---|
| Package split, ITensors index | https://docs.itensor.org/ITensors/dev/ |
| Index (§1) | https://docs.itensor.org/ITensors/dev/IndexType.html |
| ITensor + svd/factorize/eigen (§2) | https://docs.itensor.org/ITensors/dev/ITensorType.html |
| QN (§3) | https://docs.itensor.org/ITensors/dev/QN.html |
| SiteType / siteinds / op / state (§4) | https://docs.itensor.org/ITensorMPS/dev/SiteType.html |
| OpSum → MPO (§5) | https://docs.itensor.org/ITensorMPS/dev/OpSum.html |
| MPS / MPO / inner / expect / correlation_matrix / apply (§6,§8,§9) | https://docs.itensor.org/ITensorMPS/dev/MPSandMPO.html |
| DMRG kwargs (§7) | https://docs.itensor.org/ITensorMPS/dev/DMRG.html |
| Observers (§10) | https://docs.itensor.org/ITensorMPS/dev/examples/DMRG.html |
| Ground-state DMRG example (§11.1) | https://docs.itensor.org/ITensorMPS/dev/tutorials/DMRG.html |
| QN DMRG example (§11.2) | https://docs.itensor.org/ITensorMPS/dev/tutorials/QN_DMRG.html |
| TEBD example (§11.3) | https://docs.itensor.org/ITensorMPS/dev/tutorials/MPSTimeEvolution.html |
| Excited states + observers (§11.4) | https://docs.itensor.org/ITensorMPS/dev/examples/DMRG.html |
| ITensorMPS.jl repo / examples | https://github.com/ITensor/ITensorMPS.jl |
