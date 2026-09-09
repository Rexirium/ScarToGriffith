#!/usr/bin/env julia

using Printf
using SpinMonteCarlo
using Statistics

function parse_arg(::Type{T}, index::Int, default::T, label::String) where {T}
    length(ARGS) < index && return default
    try
        return parse(T, ARGS[index])
    catch error
        throw(ArgumentError("invalid $label: $(ARGS[index]) ($(sprint(showerror, error)))"))
    end
end

L = parse_arg(Int, 1, 8, "L")
T = parse_arg(Float64, 2, 2.269185314213022, "T")
mcs = parse_arg(Int, 3, 8192, "MCS")
thermalization = parse_arg(Int, 4, mcs >> 2, "Thermalization")
seed = parse_arg(Int, 5, 20260908, "Seed")

L > 1 || throw(ArgumentError("L must be greater than 1"))
T > 0 || throw(ArgumentError("T must be positive"))
mcs > 0 || throw(ArgumentError("MCS must be positive"))
thermalization >= 0 || throw(ArgumentError("Thermalization must be nonnegative"))

param = Parameter(
    "Model" => Ising,
    "Lattice" => "square lattice",
    "L" => L,
    "T" => T,
    "J" => 1.0,
    "Update Method" => SW_update!,
    "MCS" => mcs,
    "Thermalization" => thermalization,
    "Seed" => seed,
)

result = runMC(param)

println("# SpinMonteCarlo version: ", pkgversion(SpinMonteCarlo))
println("# L,T,MCS,Thermalization,Seed,observable,mean,standard_error")
for observable in ("Energy", "|Magnetization|", "Binder Ratio", "Specific Heat")
    jk = result[observable]
    @printf("%d,%.15g,%d,%d,%d,%s,%.15g,%.15g\n",
            L, T, mcs, thermalization, seed, observable, mean(jk), stderror(jk))
end
