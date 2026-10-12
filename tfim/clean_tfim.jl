include(joinpath(@__DIR__, "..", "random_tfim", "RandomTFIM.jl"))
using .RandomTFIM

"""Zero-temperature midpoint autocorrelation with the lowest odd-state term removed.

Clean open Pauli chain, J=1, uniform physical field h (no h0/e rescaling).
C_offset(tau) = C(tau) - |<GS_-|sigma_z(j)|GS_+>|^2 exp(-Delta*tau).
Uses j=cld(L,2); this local correlator is not a site average.
"""
function clean_autocorrelation(h, times; L=128)
    L >= 2 || throw(ArgumentError("require L >= 2"))
    autocorrelation(ones(L-1), fill(Float64(h), L), times;
        time_domain=:imaginary, offset=true, rcond_tol=1.0)
end
