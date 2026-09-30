using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "local"))

using LinearAlgebra, Statistics, HDF5, Dates
if !isdefined(Main, :RandomTFIM)
    include(joinpath(@__DIR__, "RandomTFIM.jl"))
    using .RandomTFIM
end

include("results_io.jl")

"""Run a small demonstration or a paper-sized ensemble; save portable HDF5 data.

Both modes compute gaps, spatial correlations and imaginary- and real-time autocorrelations with SEM.
Usage: julia random_tfim/run.jl [demo|full] [samples] [output.h5] [open|periodic] [uniform|fixed]
Output filenames receive a _yyyymmdd_HHMMSS timestamp before the extension.
"""
function main(args)
    cfg = parse_config(args)
    (; nsamples, output, boundary, field_distribution, sizes, fields, sample_fields, times) = cfg
    BLAS.set_num_threads(1)
    mkpath(dirname(output))

    h5open(output, "w") do file
        write_metadata(file, cfg)

        for (k, h0) in enumerate(fields), L in sizes
            rmax = L ÷ 2
            seed = 1996 + 1000k + L
            seconds = @elapsed result = disorder_ensemble(L, h0, times; nsamples, seed,
                rmax, boundary, field_distribution, keep_samples=h0 in sample_fields)
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
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
