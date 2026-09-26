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
    for k in 1:2:n-1
        p = k + argmax(abs.(@view A[k, k+1:n]))
        iszero(A[k, p]) && return zero(ComplexF64)
        if p != k+1
            A[[k+1, p], :] = A[[p, k+1], :]
            A[:, [k+1, p]] = A[:, [p, k+1]]
            value = -value
        end
        pivot = A[k, k+1]
        value *= pivot
        for b in k+2:n, a in k+2:b-1
            A[a, b] -= (A[k, a]*A[k+1, b] - A[k, b]*A[k+1, a]) / pivot
            A[b, a] = -A[a, b]
        end
    end
    return value
end

"""Zero-temperature longitudinal real-time autocorrelation; boundary=:open or :periodic.

Returns `(C,)`, with C[n] = <sigma_z(j,times[n]) sigma_z(j,0)> (complex).
This also equals the connected correlation: the finite-chain parity ground
state has <sigma_z(j)> = 0. Thermal and nonstationary initial states are not supported.
J has L-1 (:open) or L (:periodic) bonds, h has L fields; require even L >= 2.
Default j=L/2 is the left midpoint. Open chains use the full JW string;
periodic chains switch parity sectors and evaluate the Gaussian evolution
operator by a Pfaffian, retaining its complex phase without square roots.
"""
function autocorrelation(J::AbstractVector{<:Real}, h::AbstractVector{<:Real},
        times::AbstractVector{<:Real}; j::Int=length(h) ÷ 2, boundary::Symbol=:open)
    L = length(h)
    check_chain(J, h, boundary)
    1 <= j <= L || throw(ArgumentError("require 1 <= j <= L"))
    all(isfinite, times) || throw(ArgumentError("times must be finite"))
    return autocorrelation(J, h, times, j, Val(boundary))
end

# Boundary-specific kernels receive validated inputs from the public method.
function autocorrelation(J, h, times, j, boundary::Val{:open})
    L = length(h)
    # In (all odd, all even) order, A = [0 -2K; 2K' 0]. The SVD of K
    # diagonalizes this Majorana generator, whose frequencies are ±2F.S.
    F = svd!(fermion_matrix(J, h, boundary))
    U, V = F.U, F.V
    odd, even = 1:2:2L, 2:2:2L
    Gamma = zeros(2L, 2L)
    Gamma[odd, even] = U * V'
    Gamma[even, odd] = -transpose(Gamma[odd, even])
    contractions = I - 1im * Gamma
    N = 2j-1
    S = 1:N
    equal_time = -1im * Gamma[S, S]
    R = zeros(2L, 2L)
    C = Vector{ComplexF64}(undef, length(times))
    for (n, t) in enumerate(times)
        angles = 2 .* F.S .* t
        all(isfinite, angles) || throw(ArgumentError("time-frequency product overflows Float64"))
        c, s = Diagonal(cos.(angles)), Diagonal(sin.(angles))
        R[odd, odd] = U * c * U'
        R[odd, even] = -U * s * V'
        R[even, odd] = V * s * U'
        R[even, even] = V * c * V'
        # Multiply over ALL Majorana modes before restricting both endpoints.
        Q = R[S, :] * contractions[:, S]
        W = [equal_time Q; -transpose(Q) equal_time]
        C[n] = (-1)^(j-1) * pfaffian!(W)
    end
    return (; C)
end

# After cyclic relabeling sigma_z(j)=gamma_1. The state gamma_1|GS,+>
# is Gaussian with odd parity and evolves under the periodic fermion Hamiltonian.
function autocorrelation(J, h, times, j, boundary::Val{:periodic})
    L = length(h)
    order = mod1.(j:j+L-1, L)
    bonds, fields = J[order], h[order]
    initial = svd!(fermion_matrix(bonds, fields, boundary))
    evolution = svd!(fermion_matrix(bonds, fields, 1))
    Gamma_oe = initial.U * initial.V'
    Gamma_oe[1, :] .*= -1 # conjugation by gamma_1
    B = evolution.U' * Gamma_oe * evolution.V
    odd, even = 1:2:2L, 2:2:2L
    C = Vector{ComplexF64}(undef, length(times))
    E0 = -sum(initial.S)
    for (n, t) in enumerate(times)
        angles = evolution.S .* t
        all(isfinite, angles) && isfinite(E0*t) ||
            throw(ArgumentError("time-energy product overflows Float64"))
        # exp(-i H_- t) = product_mu (cos(eps_mu*t) - sin(eps_mu*t)*alpha_mu*beta_mu).
        # Wick expansion: Pf(D_cos + T*(-i Gamma_modes)*T^T),
        # T_(alpha, beta)=(-sin, 1). Here Gamma_modes has only odd/even blocks.
        block = Diagonal(cos.(angles)) + 1im * Diagonal(sin.(angles)) * B
        W = zeros(ComplexF64, 2L, 2L)
        W[odd, even] = block
        W[even, odd] = -transpose(block)
        C[n] = cis(E0*t) * pfaffian!(W)
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
fermion_matrix(J, h, ::Val{:open}) = fermion_matrix(J, h, 0)
fermion_matrix(J, h, ::Val{:periodic}) = fermion_matrix(J, h, -1)

function gap_result(gap, energy_scale)
    # A conservative numerical resolution indicator, not a rigorous error bound.
    resolved = gap > 64eps(Float64) * max(energy_scale, 1.0)
    return (; gap, resolved)
end

gap_result(J, h, epsilon, ::Val{:open}) = gap_result(2minimum(epsilon), sum(epsilon))

function gap_result(J, h, epsilon_ap, ::Val{:periodic})
    epsilon_p = svdvals!(fermion_matrix(J, h, 1))

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
Use logabsdet to avoid determinant underflow; a nonpositive determinant is
flagged by NaN in logC (zero gives -Inf). Signed C is retained for diagnosis.
"""
function correlations(G::AbstractMatrix{Float64}; boundary::Symbol=:periodic,
        rmax::Int=size(G, 1) ÷ 2)
    check_boundary(boundary)
    L = size(G, 1)
    size(G, 2) == L || throw(DimensionMismatch("G must be square"))
    0 <= rmax <= (boundary == :open ? L-1 : L ÷ 2) || throw(ArgumentError("invalid rmax for boundary"))
    C, logC = ones(L, rmax+1), zeros(L, rmax+1)

    for r in 1:rmax
        minor = Matrix{Float64}(undef, r, r)
        for i in 1:L
            if boundary == :open && i+r > L
                C[i, r+1] = logC[i, r+1] = NaN
                continue
            end
            for b in 1:r, a in 1:r
                row, col = i+a-1, i+b
                seam = xor(row > L, col > L) ? -1.0 : 1.0
                minor[a, b] = seam * G[mod1(row, L), mod1(col, L)]
            end
            logabs, sign = logabsdet(lu!(minor; check=false))
            C[i, r+1] = sign * exp(logabs)
            logC[i, r+1] = sign > 0 ? logabs : sign == 0 ? -Inf : NaN
        end
    end
    return (; C, logC)
end

"""Seeded disorder ensemble; columns of per-sample means label realizations.

Returns `(gaps, resolved, sample_C, sample_logC, pair_logC, sample_Ct)`.
Each realization is drawn once; gaps, spatial pairs and time correlations
are evaluated in the same sample loop. sample_Ct[time, realization] is complex.
Set times to a real-time grid to compute dynamics at j (default L/2).
Empty times (default) skips dynamics; rmax=0 skips nontrivial spatial pairs.
Use both for gap-only runs. `pair_logC` is empty unless keep_pairs=true.
Compute means and SEM across sample columns; sites within a sample are correlated.
Average pair logarithms before exponentiating to get the typical correlation.
Select boundary=:periodic (default) or :open. Open-chain sample means use
only the L-r valid origins; invalid entries of pair_logC are NaN.
"""
function disorder_ensemble(L::Int, h0::Real; nsamples::Int=100, seed::Int=1996,
        rmax::Int=L ÷ 2, keep_pairs::Bool=false, boundary::Symbol=:periodic,
        times::AbstractVector{<:Real}=Float64[], j::Int=L ÷ 2)
    check_boundary(boundary)
    nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    1 <= j <= L || throw(ArgumentError("require 1 <= j <= L"))
    all(isfinite, times) || throw(ArgumentError("times must be finite"))
    0 <= rmax <= (boundary == :open ? L-1 : L ÷ 2) || throw(ArgumentError("invalid rmax"))
    rng = Xoshiro(seed)
    gaps = zeros(nsamples)
    resolved = fill(false, nsamples)
    sample_C, sample_logC = zeros(rmax+1, nsamples), zeros(rmax+1, nsamples)
    sample_Ct = Matrix{ComplexF64}(undef, length(times), nsamples)
    pair_logC = Array{Float64}(undef, keep_pairs ? (L, rmax+1, nsamples) : (0, 0, 0))

    for n in 1:nsamples
        J, h = sample_disorder(rng, L, h0; boundary)
        if rmax == 0
            result = energy_gap(J, h; boundary)
            sample_C[1, n] = 1.0
            keep_pairs && (pair_logC[:, :, n] .= 0.0)
        else
            result = ground_state(J, h; boundary)
            pair = correlations(result.G; rmax, boundary)
            for r in 0:rmax
                origins = 1:(boundary == :open ? L-r : L)
                sample_C[r+1, n] = mean(@view pair.C[origins, r+1])
                sample_logC[r+1, n] = mean(@view pair.logC[origins, r+1])
            end
            keep_pairs && (pair_logC[:, :, n] = pair.logC)
        end
        gaps[n], resolved[n] = result.gap, result.resolved
        isempty(times) || (sample_Ct[:, n] = autocorrelation(J, h, times; j, boundary).C)
    end

    return (; gaps, resolved, sample_C, sample_logC, pair_logC, sample_Ct)
end

end
