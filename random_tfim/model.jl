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
    # Match the ensemble's full SVD, including rounding near the resolution threshold.
    epsilon_p = svd!(fermion_matrix(J, h, 1)).S
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
Uses full SVDs, as do ground_state and disorder_ensemble, to keep gap values
and resolution flags consistent near the numerical threshold.
"""
function energy_gap(J::AbstractVector{<:Real}, h::AbstractVector{<:Real}; boundary::Symbol=:periodic)
    check_chain(J, h, boundary)
    bc = Val(boundary)
    return gap_result(J, h, svd!(fermion_matrix(J, h, bc)).S, bc)
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
