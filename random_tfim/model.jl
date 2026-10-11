function check_chain(J, h)
    L = length(h)

    L >= 2 || throw(ArgumentError("require L >= 2"))
    length(J) == L-1 ||
        throw(DimensionMismatch("J must have L-1 bonds"))
    all(x -> isfinite(x) && x >= 0, J) || throw(ArgumentError("J must be finite and nonnegative"))
    all(x -> isfinite(x) && x > 0, h) || throw(ArgumentError("h must be finite and positive"))
    return nothing
end

function check_field_distribution(field_distribution::Symbol)
    field_distribution in (:uniform, :fixed) ||
        throw(ArgumentError("field_distribution must be :uniform or :fixed"))
    return field_distribution
end

"""Draw box disorder for an open chain with L-1 bonds.

Select field_distribution=:uniform for h ~ U(0,h0), or :fixed for h=h0/e.
J ~ U(0,1) in both modes; both consume the same RNG draws.
"""
function sample_disorder(rng::AbstractRNG, L::Int, h0::Real;
        field_distribution::Symbol=:uniform)
    L >= 2 || throw(ArgumentError("require L >= 2"))
    isfinite(h0) && h0 > 0 || throw(ArgumentError("h0 must be finite and positive"))
    check_field_distribution(field_distribution)
    # 1-rand excludes zero, which would give an exactly degenerate chain.
    J, h = 1 .- rand(rng, L-1), Float64(h0) .* (1 .- rand(rng, L))
    field_distribution == :fixed && fill!(h, Float64(h0) / exp(1))
    return (; J, h)
end

# K=A+B; the open-chain matrix is lower bidiagonal.
fermion_matrix(J, h) = Bidiagonal(Float64.(h), -Float64.(J), :L)

function gap_result(gap, energy_scale)
    # A conservative numerical resolution indicator, not a rigorous error bound.
    resolved = gap > 64eps(Float64) * max(energy_scale, 1.0)
    return (; gap, resolved)
end

gap_result(epsilon::AbstractVector) = gap_result(2minimum(epsilon), sum(epsilon))

"""Lowest open-chain spin gap E1-E0 = 2minimum(epsilon).

J has L-1 bonds and h has L fields. `gap` retains the raw
Float64 result; `resolved` indicates whether it exceeds the numerical threshold.
Returns `(gap, resolved)` as a named tuple.
Inputs are not mutated. Positive fields and nonnegative bonds are required.
Uses full SVDs, as do ground_state and disorder_ensemble, to keep gap values
and resolution flags consistent near the numerical threshold.
"""
function energy_gap(J::AbstractVector{<:Real}, h::AbstractVector{<:Real})
    check_chain(J, h)
    return gap_result(svd!(fermion_matrix(J, h)).S)
end

"""Return the spin gap and G[i,j]=⟨(c†ᵢ-cᵢ)(c†ⱼ+cⱼ)⟩.

Use the open-chain vacuum with L-1 bonds and L fields.
For K=U*S*V', phi=U' and psi=V', hence G=-V*U' (Eq. 56).
SVD avoids squaring K and the resulting loss of small singular values.
Returns `(gap, resolved, G)` as a named tuple.
"""
function ground_state(J::AbstractVector{<:Real}, h::AbstractVector{<:Real})
    check_chain(J, h)
    F = svd!(fermion_matrix(J, h))
    G = -(F.V * F.U')
    result = gap_result(F.S)
    return (; result..., G)
end
