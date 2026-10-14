# Gaussian state relative to the selected lowest odd-parity state.
function offset_pairing_setup(F, j)
    L = length(F.S)
    order = sortperm(F.S)
    depth = min(j, L+1-j)
    u = j <= cld(L, 2) ? F.U[1:depth, order] : F.V[L-depth+1:L, order]
    v = j <= cld(L, 2) ? F.V[1:depth-1, order] : F.U[L-depth+2:L, order]
    Gu, Gv = u'*u, v'*v
    P, Q = Gu+Gv-I, Gu-Gv
    P[:, 1], Q[:, 1] = copy(Q[:, 1]), copy(P[:, 1])
    factors = lu!(P; check=false)
    issuccess(factors) || return nothing
    Z = -(factors\Q)
    Z = (Z-Z')/2
    all(isfinite, Z) || return nothing
    beta = 2 .* F.S[order]
    beta[1] = -beta[1]
    return (; Z, beta)
end

# Same DGESDD('N') backend/ABI as LinearAlgebra.svdvals!, reusing all work arrays.
function pairing_svdvals!(A::Matrix{Float64}, ws; query=false)
    n = size(A, 1)
    size(A, 2) == n == length(ws.singular) || throw(DimensionMismatch("SVD workspace size"))
    lwork = query ? BlasInt(-1) : BlasInt(length(ws.work))
    ccall((@blasfunc(dgesdd_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
         Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
         Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Clong),
        'N', n, n, A, max(1, n), ws.singular, ws.dummy, 1, ws.dummy, 1,
        ws.work, lwork, ws.iwork, ws.info, 1)
    LAPACK.chklapackerror(ws.info[])
    return ws.singular
end

function pairing_svd_workspace(A)
    n = size(A, 1)
    ws = (; singular=zeros(n), dummy=zeros(1), work=zeros(1),
        iwork=zeros(BlasInt, 8n), info=Ref{BlasInt}(0))
    pairing_svdvals!(A, ws; query=true)
    resize!(ws.work, round(BlasInt, nextfloat(ws.work[1])))
    return ws
end

function pairing_offset_autocorrelation(pairing, times, weight, gap)
    (; Z, beta) = pairing
    W = zeros(size(Z))
    ws = pairing_svd_workspace(W)
    decays = Vector{Float64}(undef, length(beta))
    C = Vector{Float64}(undef, length(times))
    logweight = log(weight)
    for (n, tau) in enumerate(times)
        isfinite(maximum(beta)*tau) || throw(ArgumentError("time-frequency product overflows Float64"))
        for k in 2:length(beta)
            decays[k] = exp(-beta[k]*tau/2)
        end
        a = 0.0
        for b in axes(Z, 2), k in 1:b-1
            # The swapped lowest mode has negative beta: never form its growing exponential.
            x = k == 1 ? Z[k, b]*exp(-(beta[1]+beta[b])*tau/2) :
                (Z[k, b]*decays[k])*decays[b]
            W[k, b], W[b, k] = x, -x
            a += abs2(x)
        end
        for k in axes(Z, 1)
            W[k, k] = 0.0
        end
        if a <= 1e-8
            # a <= expm1(g) <= expm1(a); relative truncation error <= (expm1(a)-a)/a.
            logvalue = log(a)
        else
            singular = pairing_svdvals!(W, ws)
            g = sum(s -> s <= 1 ? log1p(s*s) : 2log(hypot(1, s)), singular)/2
            logvalue = g + log(-expm1(-g))
        end
        # Combine in log space to avoid an overflowing expm1 times an underflowing prefactor.
        C[n] = exp(logweight-gap*tau+logvalue)
    end
    return C
end
