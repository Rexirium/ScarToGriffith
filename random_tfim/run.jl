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

function write_correlations(group, result, times)
    RandomTFIM.time_scale(times, :imaginary)
    group["correlation_mean"] = result.C_mean
    group["correlation_sem"] = result.C_sem
    group["log_correlation_mean"] = result.logC_mean
    group["log_correlation_sem"] = result.logC_sem
    group["imaginary_time"] = times
    group["real_time"] = times
    # Both time domains provide real-valued sample means and SEM upstream.
    group["autocorrelation_mean"] = result.Ct_mean
    group["autocorrelation_sem"] = result.Ct_sem
    group["log_autocorrelation_mean"] = result.logCt_mean
    group["log_autocorrelation_sem"] = result.logCt_sem
    group["real_autocorrelation_mean"] = result.real_Ct_mean
    group["real_autocorrelation_sem"] = result.real_Ct_sem
    group["real_log_autocorrelation_mean"] = result.real_logCt_mean
    group["real_log_autocorrelation_sem"] = result.real_logCt_sem
end

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
    times = collect(0.0:0.2:20.0)
    BLAS.set_num_threads(1)
    mkpath(dirname(output))

    h5open(output, "w") do file
        attributes(file)["complete"] = false
        attributes(file)["julia_version"] = string(VERSION)
        attributes(file)["active_project"] = Base.active_project()
        attributes(file)["mode"] = mode
        attributes(file)["distribution"] = "box"
        attributes(file)["boundary"] = "$boundary spins; even L; Pauli normalization"
        attributes(file)["precision"] = "Float64 outputs; ComplexF64 internal Pfaffian; unresolved log gaps are NaN"
        attributes(file)["observable"] = "gap; <sigma_z(i) sigma_z(i+r)>; <sigma_z(j,tau) sigma_z(j,0)>; real(<sigma_z(j,t) sigma_z(j,0)>); j=L/2; ground state"
        attributes(file)["time_domain"] = "imaginary and real"
        attributes(file)["time_definition"] = "tau >= 0; sigma_z(tau) = exp(tau*H) sigma_z exp(-tau*H); t >= 0; sigma_z(t) = exp(im*t*H) sigma_z exp(-im*t*H); real part only; hbar=1"
        attributes(file)["uncertainty"] = "SEM across independent disorder samples; corrected sample variance; NaN for one sample"
        attributes(file)["blas"] = string(BLAS.get_config())
        attributes(file)["blas_threads"] = BLAS.get_num_threads()

        parameters = create_group(file, "parameters")
        parameters["sizes"] = collect(sizes)
        parameters["fields"] = collect(fields)

        for (k, h0) in enumerate(fields), L in sizes
            rmax = L ÷ 2
            seed = 1996 + 1000k + L
            seconds = @elapsed result = disorder_ensemble(L, h0, times; nsamples, seed,
                rmax, boundary)
            group = create_group(file, "L$(L)/h$(h0)")
            group["gap_samples"] = result.gaps
            group["gap_resolved"] = result.resolved
            # Derived quantities belong to the output/analysis layer.
            loggaps = map((gap, ok) -> ok ? log(gap) : NaN,
                result.gaps, result.resolved)
            group["log_gap_samples"] = loggaps
            for (prefix, samples) in (("gap", result.gaps), ("log_gap", loggaps))
                stats = mean_sem(samples)
                group["$(prefix)_mean"] = stats.average
                group["$(prefix)_sem"] = stats.sem
            end
            write_correlations(group, result, times)
            attributes(group)["j"] = L ÷ 2
            attributes(group)["seed"] = seed
            attributes(group)["seconds"] = seconds
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
