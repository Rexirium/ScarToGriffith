"""Ground-state von Neumann entropy across the OBC cut `1:cut | cut+1:L`.

    entanglement_entropy(J, h; cut=length(h) ÷ 2)
    entanglement_entropy(G; cut=size(G, 1) ÷ 2)

Returns a Float64 in nats (natural logarithms). Divide by log(2) for bits.
The matrix interface reuses `ground_state(J, h).G`; it assumes the same pure
Gaussian ground state, not an arbitrary correlation matrix. Inputs are unchanged.
Require 0 <= cut <= L; the empty and whole-chain partitions have zero entropy.
The coupling interface requires L >= 2, finite J >= 0 and finite h > 0.

For Majoranas ordered as all odd indices followed by all even indices,
Gamma[a,b] = (im/2)<[gamma_a,gamma_b]> = [0 -G'; G 0].
Restricting to A=1:cut gives eigenvalues +/-nu of im*Gamma_A, where
nu = svdvals(G[A,A]). Each pair contributes one binary entropy with
p=(1-nu)/2. This evaluates the covariance spectrum without forming a
2cut-by-2cut matrix or double-counting Majorana pairs. For cut > L/2,
use B=cut+1:L instead, since the pure state has S(A)=S(B).

Uses the finite-chain parity ground state selected by the existing SVD,
including when its gap is numerically unresolved; not a symmetry-broken state.
"""
function entanglement_entropy(G::AbstractMatrix{Float64}; cut::Int=size(G, 1) ÷ 2)
    L = size(G, 1)
    size(G, 2) == L || throw(DimensionMismatch("G must be square"))
    0 <= cut <= L || throw(ArgumentError("require 0 <= cut <= L"))
    all(isfinite, G) || throw(ArgumentError("G must be finite"))
    return entanglement_entropy_unchecked(G, cut)
end

# Caller guarantees square, finite G and 0 <= cut <= L; spectrum checks remain here.
function entanglement_entropy_unchecked(G::AbstractMatrix{Float64}, cut::Int)
    L = size(G, 1)
    (cut == 0 || cut == L) && return 0.0

    inds = cut <= L ÷ 2 ? (1:cut) : (cut+1:L)
    # Indexing creates a copy, so the in-place SVD leaves G unchanged.
    nu = svdvals!(G[inds, inds])
    # Only roundoff excursions above one may be clipped to the physical interval.
    maximum(nu) <= 1 + 64eps(Float64)*L ||
        throw(ArgumentError("restricted covariance singular values must not exceed one"))
    entropy = 0.0
    for value in nu
        p = (1 - clamp(value, 0.0, 1.0)) / 2
        iszero(p) && continue
        entropy -= p*log(p) + (1-p)*log1p(-p)
    end
    return entropy
end

function entanglement_entropy(J::AbstractVector{<:Real}, h::AbstractVector{<:Real};
        cut::Int=length(h) ÷ 2)
    0 <= cut <= length(h) || throw(ArgumentError("require 0 <= cut <= L"))
    return entanglement_entropy(ground_state(J, h).G; cut)
end
