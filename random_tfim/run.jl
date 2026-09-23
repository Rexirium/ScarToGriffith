using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "local"))

using LinearAlgebra, Statistics, HDF5
if !isdefined(Main, :RandomTFIM)
    include(joinpath(@__DIR__, "RandomTFIM.jl"))
    using .RandomTFIM
end

"""Run a small demonstration or a paper-sized ensemble; save portable HDF5 data.

Usage: julia random_tfim/run.jl [demo|gap|correlation|bimodal] [samples] [output.h5]
"""
function main(args)
    mode = isempty(args) ? "demo" : args[1]
    mode in ("demo", "gap", "correlation", "bimodal") ||
        throw(ArgumentError("unknown mode: $mode"))
    length(args) <= 3 || throw(ArgumentError("expected at most three arguments"))
    default_samples = mode == "demo" ? 20 : mode == "correlation" ? 10_000 : 50_000
    nsamples = length(args) >= 2 ? parse(Int, args[2]) : default_samples
    output = length(args) >= 3 ? abspath(args[3]) : joinpath(@__DIR__, "results", "$mode.h5")
    ispath(output) && throw(ArgumentError("output already exists: $output"))
    nsamples > 0 || throw(ArgumentError("samples must be positive"))

    sizes = mode == "demo" ? (16, 32) : (16, 32, 64, 128)
    fields = mode == "correlation" ? (1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 3.0) : (1.0, 3.0)
    distribution = mode == "bimodal" ? :bimodal : :box
    BLAS.set_num_threads(1)
    mkpath(dirname(output))

    h5open(output, "w") do file
        attributes(file)["complete"] = false
        attributes(file)["julia_version"] = string(VERSION)
        attributes(file)["active_project"] = Base.active_project()
        attributes(file)["mode"] = mode
        attributes(file)["distribution"] = string(distribution)
        attributes(file)["boundary"] = "periodic spins; even L; Pauli normalization"
        attributes(file)["blas"] = string(BLAS.get_config())
        attributes(file)["blas_threads"] = BLAS.get_num_threads()
        attributes(file)["precision"] = "Float64; unresolved log gaps are NaN"

        for (k, h0) in enumerate(fields), L in sizes
            rmax = mode in ("gap", "bimodal") ? 0 : L ÷ 2
            seed = 1996 + 1000k + L
            seconds = @elapsed result = disorder_ensemble(L, h0; nsamples, seed,
                distribution, rmax, keep_pairs=(mode == "correlation"))
            group = create_group(file, "L$(L)/h$(h0)")
            for name in (:gaps, :resolved, :sample_C, :sample_logC)
                group[string(name)] = getproperty(result, name)
            end
            # Derived quantities belong to the output/analysis layer.
            group["r"] = collect(0:rmax)
            group["loggaps"] = map((gap, ok) -> ok ? log(gap) : NaN,
                result.gaps, result.resolved)
            group["average"] = vec(mean(result.sample_C; dims=2))
            mean_log = vec(mean(result.sample_logC; dims=2))
            group["mean_log"] = mean_log
            group["typical"] = exp.(mean_log)
            group["sem"] = nsamples > 1 ?
                vec(std(result.sample_C; dims=2)) / sqrt(nsamples) : fill(NaN, rmax+1)
            group["log_sem"] = nsamples > 1 ?
                vec(std(result.sample_logC; dims=2)) / sqrt(nsamples) : fill(NaN, rmax+1)
            isempty(result.pair_logC) || (group["pair_logC"] = result.pair_logC)
            attributes(group)["seed"] = seed
            attributes(group)["L"] = L
            attributes(group)["h0"] = h0
            attributes(group)["nsamples"] = nsamples
            attributes(group)["seconds"] = seconds
            attributes(group)["unresolved_gaps"] = count(!, result.resolved)
            println("L=$L h0=$h0 samples=$nsamples time=$(round(seconds; digits=3))s",
                " unresolved=$(count(!, result.resolved))")
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
