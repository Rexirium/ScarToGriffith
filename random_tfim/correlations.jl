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
