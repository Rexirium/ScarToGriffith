module RandomTFIM

using LinearAlgebra
using Random
using Statistics

export sample_disorder, energy_gap, ground_state, correlations, disorder_ensemble
export autocorrelation

function check_boundary(boundary)
    boundary in (:open, :periodic) || throw(ArgumentError("boundary must be :open or :periodic"))
    return nothing
end

function check_chain(J, h, boundary=:periodic)
    check_boundary(boundary)
    L = length(h)

    L >= 2 && iseven(L) || throw(ArgumentError("require even L >= 2"))
    length(J) == (boundary == :open ? L-1 : L) ||
        throw(DimensionMismatch("J must have L-1 bonds for :open, L bonds for :periodic"))
    all(x -> isfinite(x) && x >= 0, J) || throw(ArgumentError("J must be finite and nonnegative"))
    all(x -> isfinite(x) && x > 0, h) || throw(ArgumentError("h must be finite and positive"))
    return nothing
end

# Pivoted skew-symmetric elimination, using transpose (never adjoint).
# Internal input is an even-dimensional complex skew-symmetric matrix.
function pfaffian!(A::Matrix{ComplexF64})
    n = size(A, 1)
    value = one(ComplexF64)

    @inbounds for k in 1:2:n-1
        _, offset = findmax(abs, @view A[k, k+1:n])
        p = k + offset
        iszero(A[k, p]) && return zero(ComplexF64)

        if p != k+1
            for q in 1:n
                A[k+1, q], A[p, q] = A[p, q], A[k+1, q]
            end
            for q in 1:n
                A[q, k+1], A[q, p] = A[q, p], A[q, k+1]
            end
            value = -value
        end

        pivot = A[k, k+1]
        value *= pivot
        # Pivoting bounds these ratios by one, even for subnormal pivots.
        for a in k+2:n
            A[k, a] /= pivot
        end
        for b in k+2:n
            x, y = A[k, b], A[k+1, b]
            for a in k+2:b-1
                A[a, b] -= A[k, a]*y - x*A[k+1, a]
                A[b, a] = -A[a, b]
            end
        end
    end

    return value
end

@inline function time_scale(times, time_domain)
    time_domain in (:imaginary, :real) ||
        throw(ArgumentError("time_domain must be :imaginary or :real"))
    all(isfinite, times) || throw(ArgumentError("times must be finite"))
    time_domain == :imaginary && any(t -> t < 0, times) &&
        throw(ArgumentError("imaginary times must be nonnegative"))
    return time_domain == :imaginary ? 1.0 : 1.0im
end

"""Zero-temperature longitudinal autocorrelation; boundary=:open or :periodic.

Returns `(C,)`, with C[n] = <sigma_z(j,times[n]) sigma_z(j,0)>.
Default time_domain=:imaginary uses sigma_z(j,tau)=exp(tau*H)*sigma_z(j)*exp(-tau*H)
and returns Vector{Float64}, with tau >= 0. time_domain=:real uses
sigma_z(j,t)=exp(im*t*H)*sigma_z(j)*exp(-im*t*H) and returns Vector{ComplexF64}.
Real times may be negative; this is the unsymmetrized correlation, retaining its phase.
This also equals the connected correlation: the finite-chain parity ground
state has <sigma_z(j)> = 0. Thermal and nonstationary initial states are not supported.
J has L-1 (:open) or L (:periodic) bonds, h has L fields; require even L >= 2.
Default j=L/2 is the left midpoint. Gaussian evolution is evaluated by an
L-dimensional scaled determinant without square roots or growing exponentials.
Scaling avoids overflow but does not remove long-time loss of relative precision
near singularity; tiny tails can reach the floating-point noise floor.
Periodic chains switch fermion parity sectors. Open chains near an endpoint
use a shorter JW-string Pfaffian instead.
"""
Base.@constprop :aggressive function autocorrelation(J::AbstractVector{<:Real}, h::AbstractVector{<:Real},
        times::AbstractVector{<:Real}; j::Int=length(h) ÷ 2, boundary::Symbol=:open,
        time_domain::Symbol=:imaginary)
    L = length(h)
    check_chain(J, h, boundary)
    1 <= j <= L || throw(ArgumentError("require 1 <= j <= L"))
    scale = time_scale(times, time_domain)
    isempty(times) && return (C=typeof(scale)[],)

    initial = svd!(fermion_matrix(J, h, Val(boundary)))
    evolution = boundary == :open ? initial : svd!(fermion_matrix(J, h, 1))
    return autocorrelation(initial, evolution, times, j, Val(boundary), scale)
end

# At the right endpoint, reflecting the chain makes the JW string short too.
function autocorrelation(initial, evolution, times, j, ::Val{:open}, scale=1.0)
    L = length(initial.S)
    depth = min(j, L+1-j)

    # Keep the scalar Pfaffian path only for genuinely short strings.
    if 4depth-2 <= L ÷ 4
        return string_autocorrelation(initial, times, depth, j > L ÷ 2, scale)
    end
    return determinant_autocorrelation(initial, evolution, times, j, scale)
end

autocorrelation(initial, evolution, times, j, ::Val{:periodic}, scale=1.0) =
    determinant_autocorrelation(initial, evolution, times, j, scale)

function string_autocorrelation(F, times, j, reflected, scale=1.0)
    L = length(F.S)
    N = 2j-1
    U, V = F.U, F.V
    M = Matrix{ComplexF64}(undef, N, L)

    # K of the reflected chain is reverse(K', dims=(1,2)), so U and V swap.
    @inbounds for mu in 1:L
        for a in 1:j
            M[2a-1, mu] = reflected ? V[L+1-a, mu] : U[a, mu]
        end
        for a in 1:j-1
            M[2a, mu] = 1im * (reflected ? U[L+1-a, mu] : V[a, mu])
        end
    end

    equal_time = M * M'
    # Same-sublattice contractions vanish exactly; avoid SVD roundoff there.
    for b in 1:N, a in 1:N
        iseven(a-b) && (equal_time[a, b] = 0)
    end

    phased = similar(M)
    Q = Matrix{ComplexF64}(undef, N, N)
    W = Matrix{ComplexF64}(undef, 2N, 2N)
    C = Vector{typeof(scale)}(undef, length(times))

    for (n, t) in enumerate(times)
        for mu in 1:L
            angle = 2 * F.S[mu] * t
            isfinite(angle) || throw(ArgumentError("time-frequency product overflows Float64"))
            phase = scale isa Real ? exp(-angle) : cis(-angle)
            @inbounds for a in 1:N
                phased[a, mu] = M[a, mu] * phase
            end
        end

        mul!(Q, phased, M')
        @inbounds for b in 1:N, a in 1:N
            W[a, b] = W[N+a, N+b] = equal_time[a, b]
            W[a, N+b] = Q[a, b]
            W[N+b, a] = -Q[a, b]
        end
        value = (-1)^(j-1) * pfaffian!(W)
        # Only imaginary-time correlations are real.
        C[n] = scale isa Real ? real(value) : value
    end

    return (; C)
end

# sigma_z(j) is a product of the first 2j-1 Majoranas. Conjugation changes
# the odd/even covariance to D_odd * Gamma_oe * D_even. Its state is Gaussian
# with odd parity; no cyclic relabeling or extra SVD is needed.
function determinant_autocorrelation(initial, evolution, times, j, scale=1.0)
    L = length(initial.S)
    Gamma_oe = initial.U * initial.V'
    @views Gamma_oe[1:j, :] .*= -1
    @views Gamma_oe[:, 1:j-1] .*= -1
    B = evolution.U' * Gamma_oe * evolution.V

    diagonal = diag(B)
    B ./= 2

    block = Matrix{typeof(scale)}(undef, L, L)
    decays = Vector{typeof(scale)}(undef, L)
    row_factors = similar(decays)
    C = Vector{typeof(scale)}(undef, length(times))
    # E0 + sum(evolution.S); exactly zero for OBC. Sum differences for PBC.
    energy_shift = initial === evolution ? 0.0 : sum(evolution.S .- initial.S)

    for (n, t) in enumerate(times)
        isfinite(energy_shift*t) || throw(ArgumentError("time-energy product overflows Float64"))
        for a in 1:L
            angle = 2 * evolution.S[a] * t
            isfinite(angle) || throw(ArgumentError("time-frequency product overflows Float64"))
            decays[a] = scale isa Real ? exp(-angle) : cis(-angle)
            row_factors[a] = 1 - decays[a]
        end

        # R = (I+B_original)/2 + diag(decays)*(I-B_original)/2.
        # Keep diagonal terms separate to retain tiny decays when B_original[b,b] == -1.
        @inbounds for b in 1:L
            @simd for a in 1:L
                block[a, b] = row_factors[a]*B[a, b]
            end
            block[b, b] = (1 + diagonal[b])/2 + decays[b]*((1 - diagonal[b])/2)
        end
        logabs, phase = logabsdet(lu!(block; check=false))
        C[n] = phase * exp(logabs + scale*(energy_shift*t))
    end

    return (; C)
end

"""Draw box disorder; J has L (:periodic, default) or L-1 (:open) bonds.

The same RNG seed yields the same fields and interior bonds for both boundaries.
"""
function sample_disorder(rng::AbstractRNG, L::Int, h0::Real; boundary::Symbol=:periodic)
    check_boundary(boundary)
    L >= 2 && iseven(L) || throw(ArgumentError("require even L >= 2"))
    isfinite(h0) && h0 > 0 || throw(ArgumentError("h0 must be finite and positive"))
    # 1-rand excludes zero, which would give an exactly degenerate chain.
    J, h = 1 .- rand(rng, L), Float64(h0) .* (1 .- rand(rng, L))
    # Draw the seam even for OBC to keep all physical fields/bonds paired across boundaries.
    return (J=boundary == :open ? J[1:end-1] : J, h=h)
end

# K=A+B, with eigenvalues of the 2L BdG matrix equal to ±svdvals(K).
# Add bonds rather than assign: L=2 has two physical bonds between the sites.
function fermion_matrix(J, h, boundary::Int)
    L = length(h)
    K = Matrix(Diagonal(Float64.(h)))
    for i in 1:L-1
        K[i+1, i] -= J[i]
    end
    boundary == 0 || (K[1, L] -= boundary * J[L])
    return K
end

# Ground-state fermion sector for each spin boundary; the integer method also
# supports the opposite (periodic fermion) sector needed by spin dynamics/gaps.
fermion_matrix(J, h, ::Val{:open}) = Bidiagonal(Float64.(h), -Float64.(J), :L)
fermion_matrix(J, h, ::Val{:periodic}) = fermion_matrix(J, h, -1)

function gap_result(gap, energy_scale)
    # A conservative numerical resolution indicator, not a rigorous error bound.
    resolved = gap > 64eps(Float64) * max(energy_scale, 1.0)
    return (; gap, resolved)
end

gap_result(J, h, epsilon, ::Val{:open}) = gap_result(2minimum(epsilon), sum(epsilon))

function gap_result(J, h, epsilon_ap, ::Val{:periodic})
    epsilon_p = svdvals!(fermion_matrix(J, h, 1))
    return gap_result(J, h, epsilon_ap, epsilon_p)
end

function gap_result(J, h, epsilon_ap, epsilon_p::AbstractVector)
    # For even L, det(G_p) has sign prod(h)-prod(J). Logs avoid overflow.
    # At equality a zero mode makes either occupancy have the same energy.
    vacuum_even = sum(log, h) >= sum(log, J)
    correction = vacuum_even ? 2minimum(epsilon_p) : 0.0
    E0 = -sum(epsilon_ap)
    E1 = -sum(epsilon_p) + correction
    gap = sum(epsilon_ap .- epsilon_p) + correction

    return gap_result(gap, max(abs(E0), abs(E1)))
end

"""Lowest spin gap E1-E0; boundary=:periodic (default) or :open.

Periodic chains use both fermion sectors (Eqs. 43-45); open chains use 2minimum(epsilon).
J has L or L-1 bonds respectively. `gap` retains the raw
Float64 result; `resolved` indicates whether it exceeds the numerical threshold.
Returns `(gap, resolved)` as a named tuple.
Inputs are not mutated. Positive fields and nonnegative bonds are required.
"""
function energy_gap(J::AbstractVector{<:Real}, h::AbstractVector{<:Real}; boundary::Symbol=:periodic)
    check_chain(J, h, boundary)
    bc = Val(boundary)
    return gap_result(J, h, svdvals!(fermion_matrix(J, h, bc)), bc)
end

"""Return the spin gap and G[i,j]=⟨(c†ᵢ-cᵢ)(c†ⱼ+cⱼ)⟩.

Select boundary=:periodic (default, AP vacuum) or :open (open-chain vacuum).
Pass the same boundary to `correlations` when using the returned G.
For K=U*S*V', phi=U' and psi=V', hence G=-V*U' (Eq. 56).
SVD avoids squaring K and the resulting loss of small singular values.
Returns `(gap, resolved, G)` as a named tuple.
"""
function ground_state(J::AbstractVector{<:Real}, h::AbstractVector{<:Real}; boundary::Symbol=:periodic)
    check_chain(J, h, boundary)
    bc = Val(boundary)
    F = svd!(fermion_matrix(J, h, bc))
    G = -(F.V * F.U')
    result = gap_result(J, h, F.S, bc)
    return (; result..., G)
end

"""C(i,i+r) and log C for distances 0:rmax (default rmax=L/2).

boundary=:periodic (default) allows rmax<=L/2; :open allows rmax<=L-1.
For :open, only origins 1:L-r exist; other entries are NaN, not wrapped pairs.
Rows label origins, columns label distance r+1. AP fermions acquire a minus
sign on crossing the seam, while the spin correlations remain periodic.
Use incremental orthogonal QR updates and log determinants to evaluate all
distances per origin in O(rmax^3). Suspect prefixes use a triangular condition
estimate, without refactorization. Returns `(C, logC, resolved)`; signed C is
retained, while unresolved or nonpositive determinants give NaN in logC.
logtol (default 1e-6) is the target absolute error in log|C|. The diagnostic
uses relative roundoff scale 64*r*eps and r*scale/rcond, not a rigorous bound
or certification of the upstream G. Unscreened prefixes are provisionally resolved.
"""
function correlations(G::AbstractMatrix{Float64}; boundary::Symbol=:periodic,
        rmax::Int=size(G, 1) ÷ 2, logtol::Real=1e-6)
    check_boundary(boundary)
    L = size(G, 1)
    size(G, 2) == L || throw(DimensionMismatch("G must be square"))
    0 <= rmax <= (boundary == :open ? L-1 : L ÷ 2) || throw(ArgumentError("invalid rmax for boundary"))
    isfinite(logtol) && logtol > 0 || throw(ArgumentError("logtol must be finite and positive"))
    all(isfinite, G) || throw(ArgumentError("G must be finite"))

    C, logC = fill(NaN, L, rmax+1), fill(NaN, L, rmax+1)
    resolved = fill(false, L, rmax+1)
    C[:, 1] .= 1
    logC[:, 1] .= 0
    resolved[:, 1] .= true
    rmax == 0 && return (; C, logC, resolved)

    work = Matrix{Float64}(undef, rmax, rmax)
    scales = Vector{Float64}(undef, rmax)

    for i in 1:L
        count = boundary == :open ? min(rmax, L-i) : rmax
        count == 0 && continue

        # Transposed prefix; only periodic pairs can cross the AP seam.
        @inbounds for b in 1:count, a in 1:count
            row, col = i+b-1, i+a
            seam = xor(row > L, col > L) ? -1.0 : 1.0
            work[a, b] = seam * G[row > L ? row-L : row, col > L ? col-L : col]
        end

        leading_correlations!(C, logC, resolved, i, work, scales, count, logtol)
    end

    return (; C, logC, resolved)
end

# Store the transpose so rotations touch contiguous columns. At step r,
# rotations only mix columns 1:r and have determinant +1; thus the product
# of the first r diagonals is the original r-th leading principal minor.
# Unlike a Schur-complement update this also survives a singular earlier prefix.
function leading_correlations!(C, logC, resolved, i, A, scales, n, logtol)
    # Save each original prefix's scale before rotations overwrite A.
    magnitude = 0.0
    @inbounds for r in 1:n
        for k in 1:r
            magnitude = max(magnitude, abs(A[k, r]), abs(A[r, k]))
        end
        scales[r] = magnitude
    end

    @inbounds for r in 1:n
        for k in 1:r-1
            iszero(A[k, r]) && continue
            rotation, diagonal = givens(A[k, k], A[k, r], k, r)
            A[k, k], A[k, r] = diagonal, 0.0
            for b in k+1:n
                x, y = A[b, k], A[b, r]
                A[b, k] = rotation.c*x + rotation.s*y
                A[b, r] = -rotation.s*x + rotation.c*y
            end
        end

        error_scale = r * (64r * eps(Float64))
        tolerance = max(sqrt(eps(Float64)), error_scale/logtol) * scales[r]
        logabs, phase, small = 0.0, 1.0, false
        for k in 1:r
            diagonal = A[k, k]
            logabs += log(abs(diagonal))
            phase *= sign(diagonal)
            small |= abs(diagonal) <= tolerance
        end

        ok = isfinite(logabs)
        if small && ok
            # Column rotations leave a LOWER triangular factor. TRCON reads
            # it without modifying the QR state needed by later distances.
            rcond = LAPACK.trcon!('I', 'L', 'N', @view A[1:r, 1:r])
            ok = rcond > 0 && error_scale <= logtol*rcond
        end

        C[i, r+1] = phase * exp(logabs)
        resolved[i, r+1] = ok
        logC[i, r+1] = ok && phase > 0 ? logabs : NaN
    end

    return nothing
end

function mean_sem(samples::AbstractMatrix)
    n = size(samples, 2)
    average = vec(mean(samples; dims=2))
    sem = n > 1 ? vec(std(samples; dims=2, mean=reshape(average, :, 1))) / sqrt(n) :
        fill(NaN, size(samples, 1))
    return (; average, sem)
end

"""Seeded disorder ensemble; columns of per-sample means label realizations.

Returns `(gaps, resolved, C_mean, C_sem, logC_mean, logC_sem, Ct_mean, Ct_sem, pair_logC)`.
Means and SEM are vectors over distance or time, computed across independent
realizations (corrected sample variance); SEM is NaN for one realization.
For complex real-time correlations, Ct_mean is complex and Ct_sem is real,
using squared absolute deviations. Empty times yields empty Ct_mean/Ct_sem.
Set keep_samples=true to also return sample_C, sample_logC and sample_Ct;
these per-realization matrices are omitted by default.
Each realization is drawn once; gaps, spatial pairs and time correlations
are evaluated in the same sample loop. sample_Ct[tau, realization] is a
Matrix{Float64} for time_domain=:imaginary (default), or Matrix{ComplexF64}
for time_domain=:real, retaining the unsymmetrized correlation's phase.
Realizations are drawn serially, then evaluated with Threads.@threads;
the seed and sample order are independent of the number of Julia threads.
Use julia --threads=N and a single BLAS thread for parallel sample evaluation.
Set times to a finite grid at j (default L/2); imaginary times must be nonnegative.
Empty times (default) skips dynamics; rmax=0 skips nontrivial spatial pairs.
Use both for gap-only runs. `pair_logC` is empty unless keep_pairs=true.
Sites within a sample are correlated, so SEM uses sample columns, not sites.
Average pair logarithms before exponentiating to get the typical correlation.
Select boundary=:periodic (default) or :open. Open-chain sample means use
only the L-r valid origins; invalid entries of pair_logC are NaN.
logtol is passed to correlations; unresolved spatial logs propagate as NaN
through sample and ensemble means, without dropping sites or realizations.
"""
Base.@constprop :aggressive function disorder_ensemble(L::Int, h0::Real; nsamples::Int=100, seed::Int=1996,
        rmax::Int=L ÷ 2, keep_pairs::Bool=false, keep_samples::Bool=false, boundary::Symbol=:periodic,
        times::AbstractVector{<:Real}=Float64[], j::Int=L ÷ 2,
        time_domain::Symbol=:imaginary, logtol::Real=1e-6)
    check_boundary(boundary)
    nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    isfinite(logtol) && logtol > 0 || throw(ArgumentError("logtol must be finite and positive"))
    1 <= j <= L || throw(ArgumentError("require 1 <= j <= L"))
    scale = time_scale(times, time_domain)
    0 <= rmax <= (boundary == :open ? L-1 : L ÷ 2) || throw(ArgumentError("invalid rmax"))

    bc = Val(boundary)
    dynamics = !isempty(times)
    rng = Xoshiro(seed)
    # Preserve the serial RNG stream; worker threads never share a mutable RNG.
    samples = [sample_disorder(rng, L, h0; boundary) for _ in 1:nsamples]

    gaps = zeros(nsamples)
    resolved = fill(false, nsamples)
    sample_C, sample_logC = zeros(rmax+1, nsamples), zeros(rmax+1, nsamples)
    sample_Ct = Matrix{typeof(scale)}(undef, length(times), nsamples)
    pair_logC = Array{Float64}(undef, keep_pairs ? (L, rmax+1, nsamples) : (0, 0, 0))

    # Equal-site spatial correlations are known without a factorization.
    sample_C[1, :] .= 1.0
    keep_pairs && (pair_logC[:, 1, :] .= 0.0)

    # Each iteration owns its workspaces and writes only sample n's output.
    Threads.@threads for n in 1:nsamples
        J, h = samples[n]
        if rmax == 0 && !dynamics
            result = energy_gap(J, h; boundary)
            gaps[n], resolved[n] = result.gap, result.resolved
            continue
        end

        # Share the SVDs across the gap, spatial correlations and dynamics.
        initial = svd!(fermion_matrix(J, h, bc))
        if boundary == :periodic && dynamics
            evolution = svd!(fermion_matrix(J, h, 1))
            result = gap_result(J, h, initial.S, evolution.S)
        else
            evolution = initial
            result = gap_result(J, h, initial.S, bc)
        end
        gaps[n], resolved[n] = result.gap, result.resolved

        if rmax > 0
            G = -(initial.V * initial.U')
            pair = correlations(G; rmax, boundary, logtol)
            for r in 1:rmax
                origins = 1:(boundary == :open ? L-r : L)
                sample_C[r+1, n] = mean(@view pair.C[origins, r+1])
                sample_logC[r+1, n] = mean(@view pair.logC[origins, r+1])
            end
            keep_pairs && (pair_logC[:, :, n] = pair.logC)
        end

        dynamics && (sample_Ct[:, n] = autocorrelation(initial, evolution, times, j, bc, scale).C)
    end

    spatial, log_spatial, temporal = mean_sem(sample_C), mean_sem(sample_logC), mean_sem(sample_Ct)
    summary = (; gaps, resolved, C_mean=spatial.average, C_sem=spatial.sem,
        logC_mean=log_spatial.average, logC_sem=log_spatial.sem,
        Ct_mean=temporal.average, Ct_sem=temporal.sem, pair_logC)
    return keep_samples ? (; summary..., sample_C, sample_logC, sample_Ct) : summary
end

end
