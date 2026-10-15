# Gaussian state relative to the selected lowest odd-parity state.
function offset_pairing_setup(F, j)
    order = sortperm(F.S)
    u_modes, v_modes = short_string_modes(F, j)
    u, v = u_modes[:, order], v_modes[:, order]
    Gu, Gv = u'*u, v'*v
    P, Q = Gu+Gv-I, Gu-Gv
    P[:, 1], Q[:, 1] = copy(Q[:, 1]), copy(P[:, 1])
    matrix_norm = opnorm(P, 1)
    factors = lu!(P; check=false)
    issuccess(factors) || return nothing
    # Tiny reference overlaps can produce finite but unusable pairing coefficients.
    reciprocal_condition = LAPACK.gecon!('1', factors.factors, matrix_norm)
    isfinite(reciprocal_condition) && reciprocal_condition > sqrt(eps(Float64)) || return nothing
    Z = -(factors\Q)
    Z = (Z-Z')/2
    all(isfinite, Z) || return nothing
    beta = 2 .* F.S[order]
    beta[1] = -beta[1]
    return (; Z, beta)
end

function pairing_offset_autocorrelation(pairing, times, weight, gap)
    (; Z, beta) = pairing
    W = zeros(size(Z))
    decays = Vector{Float64}(undef, length(beta))
    C = Vector{Float64}(undef, length(times))
    logweight = log(weight)
    max_frequency = maximum(beta)
    for (n, tau) in enumerate(times)
        isfinite(max_frequency*tau) || throw(ArgumentError("time-frequency product overflows Float64"))
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
            singular = svdvals!(W)
            g = sum(s -> s <= 1 ? log1p(s*s) : 2log(hypot(1, s)), singular)/2
            logvalue = g + log(-expm1(-g))
        end
        # Combine in log space to avoid an overflowing expm1 times an underflowing prefactor.
        C[n] = exp(logweight-gap*tau+logvalue)
    end
    return C
end
