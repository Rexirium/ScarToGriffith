module RandomTFIM

using LinearAlgebra
using Random
using Statistics

export sample_disorder, energy_gap, ground_state, correlations, disorder_ensemble

function check_chain(J, h)
    L = length(J)
    L >= 2 && iseven(L) || throw(ArgumentError("require even L >= 2"))
    length(h) == L || throw(DimensionMismatch("J and h must have length L"))
    all(x -> isfinite(x) && x >= 0, J) || throw(ArgumentError("J must be finite and nonnegative"))
    all(x -> isfinite(x) && x > 0, h) || throw(ArgumentError("h must be finite and positive"))
    return nothing
end

"""Draw independent box (Eq. 2) or bimodal (Eq. 61) disorder using an explicit RNG."""
function sample_disorder(rng::AbstractRNG, L::Int, h0::Real; distribution::Symbol=:box)
    L >= 2 && iseven(L) || throw(ArgumentError("require even L >= 2"))
    isfinite(h0) && h0 > 0 || throw(ArgumentError("h0 must be finite and positive"))
    if distribution == :box
        # 1-rand excludes zero, which would give an exactly degenerate chain.
        return (J=1 .- rand(rng, L), h=Float64(h0) .* (1 .- rand(rng, L)))
    elseif distribution == :bimodal
        return (J=1.0 .+ 2 .* rand(rng, Bool, L),
            h=Float64(h0) .* (1 .+ 2 .* rand(rng, Bool, L)))
    end
    throw(ArgumentError("distribution must be :box or :bimodal"))
end

# K=A+B, with eigenvalues of the 2L BdG matrix equal to ±svdvals(K).
# Add bonds rather than assign: L=2 has two physical bonds between the sites.
function fermion_matrix(J, h, boundary::Int)
    L = length(J)
    K = Matrix(Diagonal(Float64.(h)))
    for i in 1:L-1
        K[i+1, i] -= J[i]
    end
    K[1, L] -= boundary * J[L]
    return K
end

function gap_result(J, h, epsilon_ap)
    epsilon_p = svdvals!(fermion_matrix(J, h, 1))

    # For even L, det(G_p) has sign prod(h)-prod(J). Logs avoid overflow.
    # At equality a zero mode makes either occupancy have the same energy.
    vacuum_even = sum(log, h) >= sum(log, J)
    correction = vacuum_even ? 2minimum(epsilon_p) : 0.0
    E0 = -sum(epsilon_ap)
    E1 = -sum(epsilon_p) + correction
    gap = sum(epsilon_ap .- epsilon_p) + correction

    # A conservative numerical resolution indicator, not a rigorous error bound.
    resolution = 64eps(Float64) * max(abs(E0), abs(E1), 1.0)
    resolved = gap > resolution
    return (; gap, resolved)
end

"""Lowest spin gap E1-E0 for a periodic, even, ferromagnetic Pauli chain.

Uses both fermion boundary sectors (Eqs. 43-45). `gap` retains the raw
Float64 result; `resolved` indicates whether it exceeds the numerical threshold.
Returns `(gap, resolved)` as a named tuple.
Inputs are not mutated. Positive fields and nonnegative bonds are required.
"""
function energy_gap(J::AbstractVector{<:Real}, h::AbstractVector{<:Real})
    check_chain(J, h)
    return gap_result(J, h, svdvals!(fermion_matrix(J, h, -1)))
end

"""Return the spin gap and G[i,j]=⟨(c†ᵢ-cᵢ)(c†ⱼ+cⱼ)⟩ in the AP vacuum.

For K=U*S*V', phi=U' and psi=V', hence G=-V*U' (Eq. 56).
SVD avoids squaring K and the resulting loss of small singular values.
Returns `(gap, resolved, G)` as a named tuple.
"""
function ground_state(J::AbstractVector{<:Real}, h::AbstractVector{<:Real})
    check_chain(J, h)
    F = svd!(fermion_matrix(J, h, -1))
    G = -(F.V * F.U')
    return (; gap_result(J, h, F.S)..., G)
end

"""All-origin C(i,i+r) and log C for distances 0:rmax, with rmax <= L/2.

Rows label origins, columns label distance r+1. AP fermions acquire a minus
sign on crossing the seam, while the spin correlations remain periodic.
Use logabsdet to avoid determinant underflow; a nonpositive determinant is
flagged by NaN in logC (zero gives -Inf). Signed C is retained for diagnosis.
"""
function correlations(G::AbstractMatrix{Float64}; rmax::Int=size(G, 1) ÷ 2)
    L = size(G, 1)
    size(G, 2) == L || throw(DimensionMismatch("G must be square"))
    0 <= rmax <= L ÷ 2 || throw(ArgumentError("require 0 <= rmax <= L/2"))
    C, logC = ones(L, rmax+1), zeros(L, rmax+1)

    for r in 1:rmax
        minor = Matrix{Float64}(undef, r, r)
        for i in 1:L
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

Returns `(gaps, resolved, sample_C, sample_logC, pair_logC)`.
Use rmax=0 for gap-only runs. `pair_logC` is empty unless keep_pairs=true.
Compute means and SEM across sample columns; sites within a sample are correlated.
Average pair logarithms before exponentiating to get the typical correlation.
"""
function disorder_ensemble(L::Int, h0::Real; nsamples::Int=100, seed::Int=1996,
        distribution::Symbol=:box, rmax::Int=L ÷ 2, keep_pairs::Bool=false)
    nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    0 <= rmax <= L ÷ 2 || throw(ArgumentError("invalid rmax"))
    rng = Xoshiro(seed)
    gaps = zeros(nsamples)
    resolved = fill(false, nsamples)
    sample_C, sample_logC = zeros(rmax+1, nsamples), zeros(rmax+1, nsamples)
    pair_logC = Array{Float64}(undef, keep_pairs ? (L, rmax+1, nsamples) : (0, 0, 0))

    for n in 1:nsamples
        J, h = sample_disorder(rng, L, h0; distribution)
        if rmax == 0
            result = energy_gap(J, h)
            sample_C[1, n] = 1.0
            keep_pairs && (pair_logC[:, :, n] .= 0.0)
        else
            result = ground_state(J, h)
            pair = correlations(result.G; rmax)
            sample_C[:, n] = vec(mean(pair.C; dims=1))
            sample_logC[:, n] = vec(mean(pair.logC; dims=1))
            keep_pairs && (pair_logC[:, :, n] = pair.logC)
        end
        gaps[n], resolved[n] = result.gap, result.resolved
    end

    return (; gaps, resolved, sample_C, sample_logC, pair_logC)
end

end
