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

"""Zero-temperature longitudinal autocorrelation of an open chain.

Returns C::Vector{Float64} containing the real part of
<sigma_z(j,times[n]) sigma_z(j,0)>.
Default time_domain=:imaginary uses sigma_z(j,tau)=exp(tau*H)*sigma_z(j)*exp(-tau*H)
with tau >= 0. time_domain=:real uses
sigma_z(j,t)=exp(im*t*H)*sigma_z(j)*exp(-im*t*H) and returns only the real part,
equivalently <{sigma_z(j,t), sigma_z(j,0)}> / 2. Real times may be negative.
With offset=false (default), this also equals the connected correlation: the finite-chain parity ground
state has <sigma_z(j)> = 0. Thermal and nonstationary initial states are not supported.
offset=true subtracts |<GS_-|sigma_z(j)|GS_+>|^2 * exp(-Delta*z),
where GS_+ is the even ground state, GS_- the lowest odd state,
Delta=E_--E_+=2minimum(F.S), and z=tau or im*t (only the real part is returned).
The matrix element uses the lowest SVD mode; if it is degenerate, this selects
one odd ground state, not the whole degenerate subspace. Subtraction can lose
relative precision when the remaining correlation is small; values are not clipped.
J has L-1 bonds, h has L fields; require L >= 2.
Default j=cld(L,2) is the midpoint (left midpoint for even L).
Endpoints use a direct spectral sum. At interior sites, estimate
the reciprocal 1-norm condition number of the L-dimensional scaled matrix R
using its LU factors. Use the determinant when rcond(R) > rcond_tol. Otherwise,
use a JW-string Pfaffian of dimension 4min(j,L+1-j)-2 (reflecting
right-half sites).
rcond_tol defaults to sqrt(eps(Float64)); at interior sites, 0 forces the determinant
and 1 forces the Pfaffian. Endpoints always use the spectral sum.
The threshold is a numerical heuristic, not an error guarantee.
Extremely small tails can still underflow or require higher-precision validation.
"""
Base.@constprop :aggressive function autocorrelation(J::AbstractVector{<:Real}, h::AbstractVector{<:Real},
        times::AbstractVector{<:Real}; j::Int=cld(length(h), 2),
        time_domain::Symbol=:imaginary, rcond_tol::Real=sqrt(eps(Float64)),
        offset::Bool=false)
    L = length(h)
    check_chain(J, h)
    1 <= j <= L || throw(ArgumentError("require 1 <= j <= L"))
    scale = time_scale(times, time_domain)
    check_rcond_tol(rcond_tol)
    isempty(times) && return Float64[]

    F = svd!(fermion_matrix(J, h))
    return autocorrelation(F, times, j, scale; rcond_tol, offset)
end

# OBC fallback: evaluate all selected times together to reuse the JW workspace.
function string_autocorrelation(F, times, j, reflected, scale=1.0)
    L = length(F.S)
    N = 2j-1
    U, V = F.U, F.V
    M = Matrix{typeof(scale)}(undef, N, L)

    # K of the reflected chain is reverse(K', dims=(1,2)), so U and V swap.
    @inbounds for mu in 1:L
        for a in 1:j
            M[2a-1, mu] = reflected ? V[L+1-a, mu] : U[a, mu]
        end
        for a in 1:j-1
            M[2a, mu] = reflected ? U[L+1-a, mu] : V[a, mu]
        end
    end

    equal_time = M * M'
    # Remove the fixed i on even Majoranas; its Pfaffian phase cancels the JW sign.
    for b in 1:N, a in 1:N
        equal_time[a, b] = iseven(a-b) ? 0.0 : (isodd(b) ? 1 : -1)*equal_time[a, b]
    end

    phased = Matrix{typeof(scale)}(undef, N, L)
    Q = Matrix{typeof(scale)}(undef, N, N)
    W = Matrix{typeof(scale)}(undef, 2N, 2N)
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
            W[a, N+b] = (isodd(b) ? 1 : -1)*Q[a, b]
            W[N+b, a] = -W[a, N+b]
        end
        logabs, phase = logabspfaffian!(W)
        C[n] = real(phase * exp(logabs))
    end

    return C
end

function endpoint_autocorrelation(F, times, j, scale)
    modes = j == 1 ? F.U : F.V
    weights = abs2.(@view modes[j, :])
    C = Vector{Float64}(undef, length(times))
    for (n, t) in enumerate(times)
        value = 0.0
        for mu in eachindex(F.S)
            angle = 2 * F.S[mu] * t
            isfinite(angle) || throw(ArgumentError("time-frequency product overflows Float64"))
            value += weights[mu] * (scale isa Real ? exp(-angle) : cos(angle))
        end
        C[n] = value
    end
    return C
end

# In one OBC mode basis, B=(I-2u'u)(I-2v'v); no site-basis covariance is needed.
function spin_flip_covariance(F, j)
    L = length(F.S)
    depth = min(j, L+1-j)
    u = j <= cld(L, 2) ? view(F.U, 1:depth, :) : view(F.V, L-depth+1:L, :)
    v = j <= cld(L, 2) ? view(F.V, 1:depth-1, :) : view(F.U, L-depth+2:L, :)
    B = u' * u
    B .*= -2
    for a in 1:L
        B[a, a] += 1
    end
    mul!(B, B * v', v, -2.0, 1.0)
    return B
end

# Wick contractions of the short JW string with the lowest quasiparticle.
function ground_transition_weight(F, j)
    L = length(F.S)
    mu = argmin(F.S)
    j == 1 && return abs2(F.U[1, mu])
    j == L && return abs2(F.V[L, mu])
    depth = min(j, L+1-j)
    u = j <= cld(L, 2) ? view(F.U, 1:depth, :) : view(F.V, L-depth+1:L, :)
    v = j <= cld(L, 2) ? view(F.V, 1:depth-1, :) : view(F.U, L-depth+2:L, :)
    logabs, _ = logabsdet(hcat(u * v', u[:, mu]))
    return exp(2logabs)
end

function autocorrelation(F, times, j, scale;
        rcond_tol::Real=sqrt(eps(Float64)), B=nothing, offset::Bool=false, transition=nothing)
    isempty(times) && return Float64[]
    if offset
        C = autocorrelation(F, times, j, scale; rcond_tol, B)
        # Internal callers can share the same sample/site transition across time domains.
        (; weight, gap) = isnothing(transition) ?
            (; weight=ground_transition_weight(F, j), gap=2minimum(F.S)) : transition
        for (n, t) in enumerate(times)
            C[n] -= weight * (scale isa Real ? exp(-gap*t) : cos(gap*t))
        end
        return C
    end
    L = length(F.S)
    (j == 1 || j == L) && return endpoint_autocorrelation(F, times, j, scale)
    depth = min(j, L+1-j)
    rcond_tol == 1 && return string_autocorrelation(F, times, depth, j > cld(L, 2), scale)
    isnothing(B) && (B = spin_flip_covariance(F, j))

    jw_indices = Int[]

    block = Matrix{typeof(scale)}(undef, L, L)
    decays = Vector{typeof(scale)}(undef, L)
    row_factors = similar(decays)
    C = Vector{Float64}(undef, length(times))

    for (n, t) in enumerate(times)
        for a in 1:L
            angle = 2 * F.S[a] * t
            isfinite(angle) || throw(ArgumentError("time-frequency product overflows Float64"))
            decays[a] = scale isa Real ? exp(-angle) : cis(-angle)
            row_factors[a] = (1 - decays[a])/2
        end

        # Keep diagonal terms separate to retain tiny decoupled-spin tails.
        @inbounds for b in 1:L
            @simd for a in 1:L
                block[a, b] = row_factors[a]*B[a, b]
            end
            diagonal = B[b, b]
            block[b, b] = (1 + diagonal)/2 + decays[b]*((1 - diagonal)/2)
        end
        matrix_norm = rcond_tol == 0 ? 0.0 : opnorm(block, 1)
        factors = lu!(block; check=false)
        reciprocal_condition = rcond_tol == 0 ? 1.0 :
            issuccess(factors) ? LAPACK.gecon!('1', factors.factors, matrix_norm) : 0.0
        if isfinite(reciprocal_condition) && reciprocal_condition > rcond_tol
            logabs, phase = logabsdet(factors)
            C[n] = real(phase * exp(logabs))
            continue
        end
        push!(jw_indices, n)
    end
    if !isempty(jw_indices)
        # Batch only the selected points: build the short-string workspace once,
        # and preserve arbitrary time ordering and repeated times.
        C[jw_indices] = string_autocorrelation(F, times[jw_indices], depth, j > cld(L, 2), scale)
    end

    return C
end
