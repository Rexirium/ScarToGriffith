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
