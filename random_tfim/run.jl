using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "local"))

using LinearAlgebra, Statistics, HDF5
if !isdefined(Main, :RandomTFIM)
    include(joinpath(@__DIR__, "RandomTFIM.jl"))
    using .RandomTFIM
end

function mean_sem(samples::AbstractVector)
    n = length(samples)
    return (average=mean(samples), sem=n > 1 ? std(samples) / sqrt(n) : NaN)
end

function mean_sem(samples::AbstractMatrix)
    n = size(samples, 2)
    return (average=vec(mean(samples; dims=2)),
        sem=n > 1 ? vec(std(samples; dims=2)) / sqrt(n) : fill(NaN, size(samples, 1)))
end

function write_autocorrelation(group, sample_Ct, times)
    group["times"] = times
    # Separate real/imaginary datasets are portable across HDF5 readers.
    for (suffix, part) in (("real", real), ("imag", imag))
        samples = part.(sample_Ct)
        stats = mean_sem(samples)
        group["sample_C_$suffix"] = samples
        group["average_$suffix"] = stats.average
        group["sem_$suffix"] = stats.sem
    end
end

"""Run a small demonstration or a paper-sized ensemble; save portable HDF5 data.

Both modes compute gaps, spatial correlations and time autocorrelations with SEM.
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
    fields = (1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 3.0)
    times = collect(0.0:0.2:10.0)
    BLAS.set_num_threads(1)
    mkpath(dirname(output))

    h5open(output, "w") do file
        attributes(file)["complete"] = false
        attributes(file)["julia_version"] = string(VERSION)
        attributes(file)["active_project"] = Base.active_project()
        attributes(file)["mode"] = mode
        attributes(file)["distribution"] = "box"
        attributes(file)["boundary"] = "$boundary spins; even L; Pauli normalization"
        attributes(file)["precision"] = "Float64/ComplexF64; unresolved log gaps are NaN"
        attributes(file)["observable"] = "gap; <sigma_z(i) sigma_z(i+r)>; <sigma_z(j,t) sigma_z(j,0)>; j=L/2; ground state"
        attributes(file)["uncertainty"] = "SEM across independent disorder samples; corrected sample variance; NaN for one sample"
        attributes(file)["blas"] = string(BLAS.get_config())
        attributes(file)["blas_threads"] = BLAS.get_num_threads()

        for (k, h0) in enumerate(fields), L in sizes
            rmax = L ÷ 2
            seed = 1996 + 1000k + L
            seconds = @elapsed result = disorder_ensemble(L, h0; nsamples, seed,
                rmax, boundary, times)
            group = create_group(file, "L$(L)/h$(h0)")
            for name in (:gaps, :resolved, :sample_C, :sample_logC)
                group[string(name)] = getproperty(result, name)
            end
            # Derived quantities belong to the output/analysis layer.
            loggaps = map((gap, ok) -> ok ? log(gap) : NaN,
                result.gaps, result.resolved)
            group["loggaps"] = loggaps
            for (prefix, samples) in (("gap", result.gaps), ("loggap", loggaps))
                stats = mean_sem(samples)
                group["$(prefix)_average"] = stats.average
                group["$(prefix)_sem"] = stats.sem
            end
            spatial, log_spatial = mean_sem(result.sample_C), mean_sem(result.sample_logC)
            group["average"] = spatial.average
            group["sem"] = spatial.sem
            group["mean_log"] = log_spatial.average
            group["log_sem"] = log_spatial.sem
            group["typical"] = exp.(log_spatial.average)
            # Transform the log-mean ± one SEM; these are not confidence bounds.
            group["typical_lower"] = exp.(log_spatial.average .- log_spatial.sem)
            group["typical_upper"] = exp.(log_spatial.average .+ log_spatial.sem)
            write_autocorrelation(group, result.sample_Ct, times)
            attributes(group)["j"] = L ÷ 2
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
