# Direct Slurm runs use the server environment; workers inherit it via exeflags.
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "server"))
end

module RandomTFIMSlurm

using Distributed, HDF5, LinearAlgebra, Statistics

include("RandomTFIM.jl")
using .RandomTFIM

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

function parse_config(args)
    mode = isempty(args) ? "demo" : args[1]
    mode in ("demo", "full") || throw(ArgumentError("unknown mode: $mode"))
    length(args) <= 4 || throw(ArgumentError("expected at most four arguments"))

    boundary = length(args) >= 4 ? Symbol(args[4]) : :periodic
    RandomTFIM.check_boundary(boundary)
    default_samples = mode == "demo" ? 20 : 10_000
    nsamples = length(args) >= 2 ? parse(Int, args[2]) : default_samples
    output = length(args) >= 3 ? abspath(args[3]) : joinpath(@__DIR__, "results", "$mode.h5")
    nsamples > 0 || throw(ArgumentError("samples must be positive"))
    ispath(output) && throw(ArgumentError("output already exists: $output"))

    sizes = mode == "demo" ? (16, 32) : (16, 32, 64, 128)
    fields = 10 .^ range(-1, 1, 101)
    times = collect(0.0:0.2:20.0)
    return (; mode, nsamples, output, boundary, sizes, fields, times)
end

function compute_case(job, cfg)
    BLAS.set_num_threads(1)
    seconds = @elapsed result = disorder_ensemble(job.L, job.h0, cfg.times; nsamples=cfg.nsamples, seed=job.seed, rmax=job.L ÷ 2,
        boundary=cfg.boundary)
    return (; L=job.L, h0=job.h0, seed=job.seed, result,
        worker_id=myid(), seconds)
end

function write_case(file, data, cfg)
    group = create_group(file, "L$(data.L)/h$(data.h0)")
    result = data.result
    group["gap_samples"] = result.gaps
    group["gap_resolved"] = result.resolved

    loggaps = map((gap, ok) -> ok ? log(gap) : NaN, result.gaps, result.resolved)
    group["log_gap_samples"] = loggaps
    for (prefix, samples) in (("gap", result.gaps), ("log_gap", loggaps))
        stats = mean_sem(samples)
        group["$(prefix)_mean"] = stats.average
        group["$(prefix)_sem"] = stats.sem
    end

    write_correlations(group, result, cfg.times)
    attributes(group)["j"] = data.L ÷ 2
    attributes(group)["seed"] = data.seed
    attributes(group)["seconds"] = data.seconds

    println("L=$(data.L) h0=$(data.h0) samples=$(cfg.nsamples) worker=$(data.worker_id)",
        " time=$(round(data.seconds; digits=3))s unresolved=$(count(!, result.resolved))")
    flush(file)
end

function write_metadata(file, cfg)
    attributes(file)["complete"] = false
    attributes(file)["julia_version"] = string(VERSION)
    attributes(file)["active_project"] = Base.active_project()
    attributes(file)["mode"] = cfg.mode
    attributes(file)["distribution"] = "box"
    attributes(file)["boundary"] = "$(cfg.boundary) spins; even L; Pauli normalization"
    attributes(file)["precision"] = "Float64 outputs; ComplexF64 internal Pfaffian; unresolved log gaps are NaN"
    attributes(file)["observable"] = "gap; <sigma_z(i) sigma_z(i+r)>; <sigma_z(j,tau) sigma_z(j,0)>; real(<sigma_z(j,t) sigma_z(j,0)>); j=L/2; ground state"
    attributes(file)["time_domain"] = "imaginary and real"
    attributes(file)["time_definition"] = "tau >= 0; sigma_z(tau) = exp(tau*H) sigma_z exp(-tau*H); t >= 0; sigma_z(t) = exp(im*t*H) sigma_z exp(-im*t*H); real part only; hbar=1"
    attributes(file)["uncertainty"] = "SEM across independent disorder samples; corrected sample variance; NaN for one sample"
    attributes(file)["blas"] = string(BLAS.get_config())
    attributes(file)["blas_threads"] = BLAS.get_num_threads()

    parameters = create_group(file, "parameters")
    parameters["sizes"] = collect(cfg.sizes)
    parameters["fields"] = collect(cfg.fields)
end

function run_scan(cfg, pids)
    myid() == 1 || error("run_scan must run on the manager process")
    !isempty(pids) && all(pid -> pid != 1 && pid in workers(), pids) ||
        throw(ArgumentError("pids must contain active worker processes"))
    ispath(cfg.output) && error("Output already exists: $(cfg.output)")
    mkpath(dirname(cfg.output))

    @sync for pid in pids
        @async remotecall_wait(Base.include, pid, Main, abspath(@__FILE__))
    end
    worker_threads = [remotecall_fetch(Threads.nthreads, pid) for pid in pids]
    all(==(8), worker_threads) || error("Every worker must start with exactly 8 Julia threads; got $worker_threads")
    @info "Worker threads" pids worker_threads

    jobs = [(L=L, h0=Float64(h0), seed=1996 + 1000k + L)
        for (k, h0) in enumerate(cfg.fields) for L in cfg.sizes]
    job_queue = Channel{Any}(length(jobs))
    foreach(job -> put!(job_queue, job), jobs)
    close(job_queue)
    results = Channel{Any}(length(pids))
    BLAS.set_num_threads(1)

    h5open(cfg.output, "w") do file
        write_metadata(file, cfg)
        writer = @async try
            for data in results
                write_case(file, data, cfg)
            end
        finally
            isopen(results) && close(results)
        end

        try
            @sync for pid in pids
                @async for job in job_queue
                    isopen(results) || break
                    data = remotecall_fetch(compute_case, pid, job, cfg)
                    put!(results, data)
                end
            end
        finally
            isopen(results) && close(results)
            wait(writer)
        end

        complete = attributes(file)["complete"]
        try
            write(complete, true)
        finally
            close(complete)
        end
    end
    println("Saved: $(cfg.output)")
    return cfg.output
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using Distributed, SlurmClusterManager

    config = RandomTFIMSlurm.parse_config(ARGS)
    project = dirname(Base.active_project())
    flags = `--project=$project --threads=8 --startup-file=no`
    pids = addprocs(SlurmManager(); exeflags=flags)

    try
        RandomTFIMSlurm.run_scan(config, pids)
    finally
        rmprocs(pids)
    end
end
