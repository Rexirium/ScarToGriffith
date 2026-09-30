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
# Real workspaces for imaginary time; complex workspaces for real time.
function logabspfaffian!(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    n = size(A, 1)
    logabs, phase = 0.0, one(T)

    @inbounds for k in 1:2:n-1
        _, offset = findmax(abs, @view A[k, k+1:n])
        p = k + offset
        iszero(A[k, p]) && return (-Inf, zero(T))

        if p != k+1
            for q in 1:n
                A[k+1, q], A[p, q] = A[p, q], A[k+1, q]
            end
            for q in 1:n
                A[q, k+1], A[q, p] = A[q, p], A[q, k+1]
            end
            phase = -phase
        end

        pivot = A[k, k+1]
        magnitude = abs(pivot)
        logabs += log(magnitude)
        phase *= pivot / magnitude
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

    return logabs, phase
end

function check_rcond_tol(rcond_tol)
    isfinite(rcond_tol) && 0 <= rcond_tol <= 1 ||
        throw(ArgumentError("rcond_tol must be finite and in [0, 1]"))
    return nothing
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

Returns C::Vector{Float64} containing the real part of
<sigma_z(j,times[n]) sigma_z(j,0)>.
Default time_domain=:imaginary uses sigma_z(j,tau)=exp(tau*H)*sigma_z(j)*exp(-tau*H)
with tau >= 0. time_domain=:real uses
sigma_z(j,t)=exp(im*t*H)*sigma_z(j)*exp(-im*t*H) and returns only the real part,
equivalently <{sigma_z(j,t), sigma_z(j,0)}> / 2. Real times may be negative.
This also equals the connected correlation: the finite-chain parity ground
state has <sigma_z(j)> = 0. Thermal and nonstationary initial states are not supported.
J has L-1 (:open) or L (:periodic) bonds, h has L fields; require even L >= 2.
Default j=L/2 is the left midpoint. For both boundaries and time domains, estimate
the reciprocal 1-norm condition number of the L-dimensional scaled matrix R
using its LU factors. Use the determinant when rcond(R) > rcond_tol. Otherwise,
open chains use a JW-string Pfaffian of dimension 4min(j,L+1-j)-2 (reflecting
right-half sites); periodic chains use a 2L-dimensional Gaussian-overlap
Pfaffian preserving the change of fermion parity sector.
rcond_tol defaults to sqrt(eps(Float64)); 0 forces the determinant and 1 forces
the Pfaffian. The threshold is a numerical heuristic, not an error guarantee.
Periodic chains switch fermion parity sectors in both algorithms. Extremely
small tails can still underflow or require higher-precision validation.
"""
Base.@constprop :aggressive function autocorrelation(J::AbstractVector{<:Real}, h::AbstractVector{<:Real},
        times::AbstractVector{<:Real}; j::Int=length(h) ÷ 2, boundary::Symbol=:open,
        time_domain::Symbol=:imaginary, rcond_tol::Real=sqrt(eps(Float64)))
    L = length(h)
    check_chain(J, h, boundary)
    1 <= j <= L || throw(ArgumentError("require 1 <= j <= L"))
    scale = time_scale(times, time_domain)
    check_rcond_tol(rcond_tol)
    isempty(times) && return Float64[]

    initial = svd!(fermion_matrix(J, h, Val(boundary)))
    evolution = boundary == :open ? initial : svd!(fermion_matrix(J, h, 1))
    return autocorrelation(initial, evolution, times, j, scale; rcond_tol)
end

# OBC fallback: evaluate all selected times together to reuse the JW workspace.
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
    C = Vector{Float64}(undef, length(times))

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
        logabs, phase = logabspfaffian!(W)
        C[n] = real((-1)^(j-1) * phase * exp(logabs))
    end

    return C
end

# Choose a nonzero Fock reference by sequentially conditioning occupations.
# At each mode, choose the outcome with probability >= 1/2; the update's
# denominator has magnitude >= 1. Particle-hole flips then give an even
# Thouless chart Z=(I-S*B)/(I+S*B), with real skew-symmetric Z.
function pfaffian_chart(B, ::Type{T}) where {T}
    L = size(B, 1)
    conditional = copy(B)
    signs = ones(L)
    for k in 1:L
        signs[k] = conditional[k, k] >= 0 ? 1.0 : -1.0
        denominator = conditional[k, k] + signs[k]
        for b in k+1:L, a in k+1:L
            conditional[a, b] -= conditional[a, k]*conditional[k, b]/denominator
        end
    end
    transformed = signs .* B
    Z = (I - transformed) / (I + transformed)
    # Enforce the exact antisymmetry: forbidden-parity weights must stay zero.
    Z = (Z - transpose(Z)) / 2
    lognormalization = -first(logabsdet(I + Z))
    return (; Z, occupied=signs .< 0, lognormalization,
        work=zeros(T, 2L, 2L), q=Vector{T}(undef, L))
end

# Signed Gaussian overlap, not a Pfaffian embedding of the ill-conditioned R.
# For reference occupations r, let k_i=d_i if r_i=1 else 1, and
# q_i=1 if r_i=1 else d_i. Row/column scaling of the Thouless-overlap Pfaffian
# gives W=[Z -diag(k); diag(k) -diag(q)*Z*diag(q)], using only decays/phases.
# C = (-1)^(L(L+1)/2) pf(W) / det(I+Z) * exp(energy_shift*scale*t).
function gaussian_pfaffian!(chart, decays, exponent)
    Z, occupied, W, q = chart.Z, chart.occupied, chart.work, chart.q
    L = length(decays)
    fill!(W, 0)
    q .= ifelse.(occupied, one(eltype(decays)), decays)
    @inbounds for b in 1:L
        for a in 1:b-1
            W[a, b] = Z[a, b]
            W[b, a] = -W[a, b]
            W[L+a, L+b] = -q[a] * Z[a, b] * q[b]
            W[L+b, L+a] = -W[L+a, L+b]
        end
        W[b, L+b] = occupied[b] ? -decays[b] : -one(eltype(decays))
        W[L+b, b] = -W[b, L+b]
    end
    logabs, phase = logabspfaffian!(W)
    return real((-1)^(L*(L+1) ÷ 2) * phase *
        exp(logabs + chart.lognormalization + exponent))
end

# sigma_z(j) is a product of the first 2j-1 Majoranas. Conjugation changes
# the odd/even covariance to D_odd * Gamma_oe * D_even. Its state is Gaussian
# with odd parity; no cyclic relabeling or extra SVD is needed.
function autocorrelation(initial, evolution, times, j, scale;
        rcond_tol::Real=sqrt(eps(Float64)))
    isempty(times) && return Float64[]
    L = length(initial.S)
    Gamma_oe = initial.U * initial.V'
    @views Gamma_oe[1:j, :] .*= -1
    @views Gamma_oe[:, 1:j-1] .*= -1
    B = evolution.U' * Gamma_oe * evolution.V

    diagonal = diag(B)
    half_B = B / 2
    chart = nothing
    jw_indices = Int[]

    block = Matrix{typeof(scale)}(undef, L, L)
    decays = Vector{typeof(scale)}(undef, L)
    row_factors = similar(decays)
    C = Vector{Float64}(undef, length(times))
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

        # Forced Pfaffian needs neither R nor its LU/condition estimate.
        if rcond_tol != 1
            # Keep diagonal terms separate to retain tiny decoupled-spin tails.
            @inbounds for b in 1:L
                @simd for a in 1:L
                    block[a, b] = row_factors[a]*half_B[a, b]
                end
                block[b, b] = (1 + diagonal[b])/2 + decays[b]*((1 - diagonal[b])/2)
            end
            matrix_norm = rcond_tol == 0 ? 0.0 : opnorm(block, 1)
            factors = lu!(block; check=false)
            reciprocal_condition = rcond_tol == 0 ? 1.0 :
                issuccess(factors) ? LAPACK.gecon!('1', factors.factors, matrix_norm) : 0.0
            if isfinite(reciprocal_condition) && reciprocal_condition > rcond_tol
                logabs, phase = logabsdet(factors)
                C[n] = real(phase * exp(logabs + scale*(energy_shift*t)))
                continue
            end
        end
        if initial === evolution # OBC shares the initial/evolution SVD.
            push!(jw_indices, n)
        else
            isnothing(chart) && (chart = pfaffian_chart(B, typeof(scale)))
            C[n] = gaussian_pfaffian!(chart, decays, scale*(energy_shift*t))
        end
    end
    if !isempty(jw_indices)
        # Batch only the selected points: build the short-string workspace once,
        # and preserve arbitrary time ordering and repeated times.
        depth = min(j, L+1-j)
        C[jw_indices] = string_autocorrelation(initial, times[jw_indices], depth, j > L ÷ 2, scale)
    end

    return C
end

function check_field_distribution(field_distribution::Symbol)
    field_distribution in (:uniform, :fixed) ||
        throw(ArgumentError("field_distribution must be :uniform or :fixed"))
    return field_distribution
end

"""Draw box disorder; J has L (:periodic, default) or L-1 (:open) bonds.

Select field_distribution=:uniform for h ~ U(0,h0), or :fixed for h=h0/e.
J ~ U(0,1) in both modes; both consume the same RNG draws.
The same RNG seed yields the same fields and interior bonds for both boundaries.
"""
function sample_disorder(rng::AbstractRNG, L::Int, h0::Real;
        boundary::Symbol=:periodic, field_distribution::Symbol=:uniform)
    check_boundary(boundary)
    L >= 2 && iseven(L) || throw(ArgumentError("require even L >= 2"))
    isfinite(h0) && h0 > 0 || throw(ArgumentError("h0 must be finite and positive"))
    check_field_distribution(field_distribution)
    # 1-rand excludes zero, which would give an exactly degenerate chain.
    J, h = 1 .- rand(rng, L), Float64(h0) .* (1 .- rand(rng, L))
    field_distribution == :fixed && fill!(h, Float64(h0) / exp(1))
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

"""C(i,i+r) for distances 0:rmax (default rmax=L/2).

boundary=:periodic (default) allows rmax<=L/2; :open allows rmax<=L-1.
For :open, only origins 1:L-r exist; other entries are NaN, not wrapped pairs.
Rows label origins, columns label distance r+1. AP fermions acquire a minus
sign on crossing the seam, while the spin correlations remain periodic.
Use incremental orthogonal QR updates and log determinants to evaluate all
distances per origin in O(rmax^3). Returns C::Matrix{Float64}, retaining signed values
and floating-point underflow. Logarithms and statistics belong to disorder_ensemble.
"""
function correlations(G::AbstractMatrix{Float64}; boundary::Symbol=:periodic,
        rmax::Int=size(G, 1) ÷ 2)
    check_boundary(boundary)
    L = size(G, 1)
    size(G, 2) == L || throw(DimensionMismatch("G must be square"))
    0 <= rmax <= (boundary == :open ? L-1 : L ÷ 2) || throw(ArgumentError("invalid rmax for boundary"))
    all(isfinite, G) || throw(ArgumentError("G must be finite"))

    C = fill(NaN, L, rmax+1)
    C[:, 1] .= 1
    rmax == 0 && return C

    work = Matrix{Float64}(undef, rmax, rmax)

    for i in 1:L
        count = boundary == :open ? min(rmax, L-i) : rmax
        count == 0 && continue

        # Transposed prefix; only periodic pairs can cross the AP seam.
        @inbounds for b in 1:count, a in 1:count
            row, col = i+b-1, i+a
            seam = xor(row > L, col > L) ? -1.0 : 1.0
            work[a, b] = seam * G[row > L ? row-L : row, col > L ? col-L : col]
        end

        leading_correlations!(C, i, work, count)
    end

    return C
end

# Store the transpose so rotations touch contiguous columns. At step r,
# rotations only mix columns 1:r and have determinant +1; thus the product
# of the first r diagonals is the original r-th leading principal minor.
# Unlike a Schur-complement update this also survives a singular earlier prefix.
function leading_correlations!(C, i, A, n)
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

        logabs, phase = 0.0, 1.0
        for k in 1:r
            diagonal = A[k, k]
            logabs += log(abs(diagonal))
            phase *= sign(diagonal)
        end

        C[i, r+1] = phase * exp(logabs)
    end

    return nothing
end

# Preserve invalid samples in log statistics; never take abs or drop samples.
correlation_log(x::Real) = isfinite(x) && x > 0 ? log(x) : NaN

function mean_sem(samples::AbstractVector)
    n = length(samples)
    return (average=mean(samples), sem=n > 1 ? std(samples) / sqrt(n) : NaN)
end

function mean_sem(samples::AbstractMatrix)
    n = size(samples, 2)
    average = vec(mean(samples; dims=2))
    sem = n > 1 ? vec(std(samples; dims=2, mean=reshape(average, :, 1))) / sqrt(n) :
        fill(NaN, size(samples, 1))
    return (; average, sem)
end

"""Seeded disorder ensemble; columns of per-sample means label realizations.

Returns scalar gap_mean/gap_sem and log_gap_mean/log_gap_sem,
spatial statistics C_mean/C_sem and logC_mean/logC_sem,
imaginary-time statistics Ct_mean/Ct_sem and logCt_mean/logCt_sem, and
real-time statistics real_Ct_mean/real_Ct_sem and real_logCt_mean/real_logCt_sem.
Unresolved gaps give NaN logarithms, which propagate through log-gap statistics.
Correlation means and SEM are vectors over distance or time. All statistics use
independent realizations (corrected sample variance); SEM is NaN for one realization.
All correlation samples and statistics are real; real-time statistics describe
only the real part. Nonpositive real-time values give NaN logarithms.
Set keep_samples=true to also return gaps and resolved vectors, and sample_C,
sample_Ct and sample_real_Ct matrices (columns are samples). Log samples are not returned.
Each realization is drawn once; gaps, spatial pairs and both time domains share
SVDs in the same sample loop. Both time domains use the required positional times:
a finite, nonnegative grid at j (default L/2). Empty times yields empty temporal
statistics; rmax=0 skips nontrivial spatial pairs.
Realizations are drawn serially, then evaluated with Threads.@threads;
the seed and sample order are independent of the number of Julia threads.
Use julia --threads=N and a single BLAS thread for parallel sample evaluation.
Sites within a sample are correlated, so SEM uses sample columns, not sites.
Average pair logarithms before exponentiating to get the typical correlation.
Select boundary=:periodic (default) or :open. Open-chain sample means use
only the L-r valid origins.
Select field_distribution=:uniform (default) or :fixed (h=h0/e); J remains uniform.
The rcond_tol keyword is forwarded to both imaginary- and real-time
autocorrelations; it has the same meaning and default as in autocorrelation.
Logarithms are taken per pair/time before averaging. Nonpositive or nonfinite
values give NaN, which propagates through
sample and ensemble statistics without dropping sites or realizations.
"""
Base.@constprop :aggressive function disorder_ensemble(L::Int, h0::Real,
        times::AbstractVector{<:Real}; nsamples::Int=100, seed::Int=1996,
        rmax::Int=L ÷ 2, keep_samples::Bool=false, boundary::Symbol=:periodic,
        j::Int=L ÷ 2, rcond_tol::Real=sqrt(eps(Float64)),
        field_distribution::Symbol=:uniform)
    check_boundary(boundary)
    nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    1 <= j <= L || throw(ArgumentError("require 1 <= j <= L"))
    time_scale(times, :imaginary)
    check_rcond_tol(rcond_tol)
    0 <= rmax <= (boundary == :open ? L-1 : L ÷ 2) || throw(ArgumentError("invalid rmax"))

    bc = Val(boundary)
    rng = Xoshiro(seed)
    # Preserve the serial RNG stream; worker threads never share a mutable RNG.
    samples = [sample_disorder(rng, L, h0; boundary, field_distribution) for _ in 1:nsamples]

    gaps = zeros(nsamples)
    resolved = fill(false, nsamples)
    sample_C, sample_logC = zeros(rmax+1, nsamples), zeros(rmax+1, nsamples)
    sample_Ct = Matrix{Float64}(undef, length(times), nsamples)
    sample_real_Ct = similar(sample_Ct)

    # Equal-site spatial correlations are known without a factorization.
    sample_C[1, :] .= 1.0

    # Each iteration owns its workspaces and writes only sample n's output.
    Threads.@threads for n in 1:nsamples
        J, h = samples[n]
        # Share the SVDs across the gap, spatial correlations and dynamics.
        initial = svd!(fermion_matrix(J, h, bc))
        if boundary == :periodic
            evolution = svd!(fermion_matrix(J, h, 1))
            result = gap_result(J, h, initial.S, evolution.S)
        else
            evolution = initial
            result = gap_result(J, h, initial.S, bc)
        end
        gaps[n], resolved[n] = result.gap, result.resolved

        if rmax > 0
            G = -(initial.V * initial.U')
            pair = correlations(G; rmax, boundary)
            logs = correlation_log.(pair)
            for r in 1:rmax
                origins = 1:(boundary == :open ? L-r : L)
                sample_C[r+1, n] = mean(@view pair[origins, r+1])
                sample_logC[r+1, n] = mean(@view logs[origins, r+1])
            end
        end

        sample_Ct[:, n] = autocorrelation(initial, evolution, times, j, 1.0; rcond_tol)
        sample_real_Ct[:, n] = autocorrelation(initial, evolution, times, j, 1.0im; rcond_tol)
    end

    sample_logCt = correlation_log.(sample_Ct)
    sample_real_logCt = correlation_log.(sample_real_Ct)
    spatial, log_spatial = mean_sem(sample_C), mean_sem(sample_logC)
    temporal, log_temporal = mean_sem(sample_Ct), mean_sem(sample_logCt)
    real_temporal, real_log_temporal = mean_sem(sample_real_Ct), mean_sem(sample_real_logCt)
    gap_stats = mean_sem(gaps)
    log_gap_stats = mean_sem(map((gap, ok) -> ok ? log(gap) : NaN, gaps, resolved))
    summary = (; gap_mean=gap_stats.average, gap_sem=gap_stats.sem,
        log_gap_mean=log_gap_stats.average, log_gap_sem=log_gap_stats.sem,
        C_mean=spatial.average, C_sem=spatial.sem,
        logC_mean=log_spatial.average, logC_sem=log_spatial.sem,
        Ct_mean=temporal.average, Ct_sem=temporal.sem,
        logCt_mean=log_temporal.average, logCt_sem=log_temporal.sem,
        real_Ct_mean=real_temporal.average, real_Ct_sem=real_temporal.sem,
        real_logCt_mean=real_log_temporal.average, real_logCt_sem=real_log_temporal.sem)
    return keep_samples ? (; summary..., gaps, resolved, sample_C, sample_Ct, sample_real_Ct) : summary
end

end
