# Direct Slurm runs use the server environment; workers inherit it via exeflags.
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "server"))
end

module RandomTFIMSlurm

include("scan_common.jl")

function run_scan(cfg, pids)
    myid() == 1 || error("run_scan must run on the manager process")
    !isempty(pids) && all(pid -> pid != 1 && pid in workers(), pids) ||
        throw(ArgumentError("pids must contain active worker processes"))

    @sync for pid in pids
        @async remotecall_wait(Base.include, pid, Main, abspath(@__FILE__))
    end
    worker_threads = [remotecall_fetch(Threads.nthreads, pid) for pid in pids]
    all(==(cfg.worker_threads), worker_threads) || error("Every worker must start with $(cfg.worker_threads) Julia threads; got $worker_threads")
    @info "Worker threads" pids worker_threads

    jobs = collect(scan_jobs(cfg))
    job_queue = Channel{eltype(jobs)}(length(jobs))
    foreach(job -> put!(job_queue, job), jobs)
    close(job_queue)
    results = Channel{NamedTuple}(length(pids))

    return with_results(cfg) do file
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

    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using Distributed, SlurmClusterManager

    length(ARGS) <= 1 || throw(ArgumentError("Usage: run_slurm.jl [config.toml]"))
    config = RandomTFIMSlurm.read_config(get(ARGS, 1, joinpath(@__DIR__, "scan.toml")))
    project = dirname(Base.active_project())
    flags = `--project=$project --threads=$(config.worker_threads) --startup-file=no`
    pids = addprocs(SlurmManager(); exeflags=flags)

    try
        RandomTFIMSlurm.run_scan(config, pids)
    finally
        rmprocs(pids)
    end
end
