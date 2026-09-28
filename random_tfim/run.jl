using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "local"))

using LinearAlgebra, Statistics, HDF5
if !isdefined(Main, :RandomTFIM)
    include(joinpath(@__DIR__, "RandomTFIM.jl"))
    using .RandomTFIM
end

include("results_io.jl")

"""Run a small demonstration or a paper-sized ensemble; save portable HDF5 data.

Both modes compute gaps, spatial correlations and imaginary- and real-time autocorrelations with SEM.
Usage: julia random_tfim/run.jl [demo|full] [samples] [output.h5] [open|periodic]
"""
function main(args)
    mode = isempty(args) ? "demo" : args[1]
    mode in ("demo", "full") ||
        throw(ArgumentError("unknown mode: $mode"))
    length(args) <= 4 || throw(ArgumentError("expected at most four arguments"))
    boundary = length(args) >= 4 ? Symbol(args[4]) : :periodic
    RandomTFIM.check_boundary(boundary)
    default_samples = mode == "demo" ? 20 : 10_000
    nsamples = length(args) >= 2 ? parse(Int, args[2]) : default_samples
    output = length(args) >= 3 ? abspath(args[3]) : joinpath(@__DIR__, "results", "$mode.h5")
    ispath(output) && throw(ArgumentError("output already exists: $output"))
    nsamples > 0 || throw(ArgumentError("samples must be positive"))

    sizes = mode == "demo" ? (16, 32) : (16, 32, 64, 128)
    fields = 10 .^ range(-1, 1, 101)
    sample_fields = select_sample_fields(fields)
    times = collect(0.0:0.2:20.0)
    BLAS.set_num_threads(1)
    mkpath(dirname(output))

    h5open(output, "w") do file
        attributes(file)["complete"] = false
        attributes(file)["nsamples"] = nsamples
        attributes(file)["distribution"] = "box"
        attributes(file)["boundary"] = string(boundary)

        parameters = create_group(file, "parameters")
        parameters["sizes"] = collect(sizes)
        parameters["fields"] = collect(fields)
        parameters["sample_fields"] = sample_fields

        for (k, h0) in enumerate(fields), L in sizes
            rmax = L ÷ 2
            seed = 1996 + 1000k + L
            seconds = @elapsed result = disorder_ensemble(L, h0, times; nsamples, seed,
                rmax, boundary, keep_samples=h0 in sample_fields)
            group = create_group(file, "L$(L)/h$(h0)")
            write_observables(group, result, times)
            attributes(group)["j"] = L ÷ 2
            attributes(group)["seed"] = seed
            println("L=$L h0=$h0 samples=$nsamples time=$(round(seconds; digits=3))s")
            flush(file)
        end
        complete = attributes(file)["complete"]
        try
            write(complete, true)
        finally
            close(complete)
        end
    end
    println("Saved: $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
