using HDF5
include(joinpath(@__DIR__, "../../random_tfim/fit_autocorrelation.jl"))
input = joinpath(@__DIR__, "../../random_tfim/results/full_fixed_20261003_230245.h5")
rows = ["L,h0,raw_SEM_short_weight,log_SEM_short_weight,n,tau_max"]
h5open(input, "r") do f
    for L in read(f["parameters/sizes"]), h0 in sort(read(f["parameters/fields"]))
        g = f["L$(L)/h$(h0)"]
        t, y, s = read(g["imaginary_time"]), read(g["autocorrelation_mean"]), read(g["autocorrelation_sem"])
        keep = autocorrelation_fit_mask(t, y) .& isfinite.(s) .& (s .> 0)
        t, y, s = t[keep], y[keep], s[keep]
        raw, logw = (minimum(s) ./ s).^2, (minimum(s ./ y) .* y ./ s).^2
        @assert all(isfinite, raw) && all(isfinite, logw)
        push!(rows, join((L, h0, sum(raw[t .< 1])/sum(raw), sum(logw[t .< 1])/sum(logw), length(t), maximum(t)), ','))
    end
end
write(joinpath(@__DIR__, "sem_weight_audit.csv"), join(rows, '\n') * "\n")
