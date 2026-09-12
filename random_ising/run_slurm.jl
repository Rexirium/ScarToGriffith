# 直接运行时选择环境；worker include 时沿用启动参数指定的环境。
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    isempty(ARGS) || ARGS == ["--test"] || error("Usage: run_slurm.jl [--test]")
    Pkg.activate(joinpath(@__DIR__, "..", "julia-env", isempty(ARGS) ? "server" : "local"))
end

module RandomIsingScan

using Distributed
using HDF5
using Random
using TOML

include("observables.jl")

"""读取计算参数；相对 output_dir 按 TOML 文件所在目录解析。"""
function read_config(path=joinpath(@__DIR__, "scan.toml"))
    params = TOML.parsefile(path)
    params["output_dir"] = normpath(joinpath(dirname(abspath(path)), params["output_dir"]))
    return (; (Symbol(key) => value for (key, value) in params)...)
end

# Parameter-based seeds: shared disorder across temperatures, separate MC streams.
function case_seeds(cfg, L, T)
    disorder_seed = rand(MersenneTwister(UInt32[cfg.seed, L, 0]), UInt32)
    bits = reinterpret(UInt64, T)
    mc_seed = rand(MersenneTwister(UInt32[cfg.seed, L, 1,
        bits & 0xffffffff, bits >> 32]), UInt32)
    return disorder_seed, mc_seed
end

"""一个 worker 计算一个 (L,T)，内部由 random_bond_observables 多线程处理无序。"""
function compute_case(job, cfg)
    L, T = job.L, job.T
    disorder_seed, mc_seed = case_seeds(cfg, L, T)
    rng = MersenneTwister(disorder_seed)
    disorder = [ifelse.(rand(rng, 2L^2) .< cfg.p, cfg.Jstrong, cfg.Jweak)
        for _ in 1:cfg.ndisorder]
    started = time_ns()
    out = random_bond_observables(L, T, disorder;
        mcs=cfg.mcs, thermalization=cfg.thermalization, binsize=cfg.binsize,
        max_corr_time=cfg.max_corr_time, seed=mc_seed, details=true)
    return (; L, T, out, disorder=reduce(hcat, disorder), disorder_seed,
        worker_id=myid(), worker_threads=Threads.nthreads(),
        elapsed_seconds=(time_ns() - started) / 1e9)
end

"""仅由主进程调用；同一 L 写入同一文件，每个 T 创建独立 group。"""
function write_case(cfg, data)
    path = joinpath(cfg.output_dir, "L_$(data.L).h5")
    h5open(path, "cw") do file
        if !haskey(file, "disorder")
            attributes(file)["L"] = data.L
            attributes(file)["ndisorder"] = cfg.ndisorder
            attributes(file)["p"] = cfg.p
            attributes(file)["Jstrong"] = cfg.Jstrong
            attributes(file)["Jweak"] = cfg.Jweak
            attributes(file)["disorder_seed"] = data.disorder_seed
            attributes(file)["master_seed"] = cfg.seed
            attributes(file)["format_version"] = 2
            attributes(file)["julia_version"] = string(VERSION)
            attributes(file)["slurm_job_id"] = get(ENV, "SLURM_JOB_ID", "local")
            file["parameters"] = repr(cfg)
            file["temperatures"] = cfg.Ts
            file["disorder"] = data.disorder # Julia 维度：(bond, realization)。
        end
        group_name = "T_$(repr(data.T))"
        haskey(file, group_name) && error("Refusing to overwrite $path/$group_name")
        group = create_group(file, group_name)
        out = data.out
        group["heat_capacity"] = out.heat_capacity
        group["susceptibility"] = out.susceptibility
        group["local_susceptibility"] = stack(out.local_susceptibility)
        group["correlation"] = reduce(hcat, out.correlation)
        group["lags"] = collect(0:cfg.max_corr_time)
        group["realization_seeds"] = out.metadata.seeds
        errors = create_group(group, "errors")
        errors["heat_capacity"] = out.errors.heat_capacity
        errors["susceptibility"] = out.errors.susceptibility
        errors["local_susceptibility"] = stack(out.errors.local_susceptibility)
        for (key, value) in pairs(out.metadata)
            key == :seeds && continue
            attributes(group)[string(key)] =
                value isa Symbol || value isa VersionNumber ? string(value) : value
        end
        attributes(group)["worker_id"] = data.worker_id
        attributes(group)["worker_threads"] = data.worker_threads
        attributes(group)["elapsed_seconds"] = data.elapsed_seconds
        attributes(group)["correlation_definition"] = "sum_i s_i(t)*s_i(0)/L^2; t=0 at end of SW thermalization"
        attributes(group)["complete"] = true # 仅在所有数据写完之后标记。
    end
    return path
end

"""
    run_scan(cfg, pids)

在已有 worker 上动态分配 (L,T) 任务；结果入队后领取下一项，由主进程串行写盘。
输出目录必须不存在，避免覆盖之前的扫描；worker 的生命周期由调用者管理。
"""
function run_scan(cfg, pids)
    Threads.nthreads() >= 2 || error("Start the manager with --threads=2 for concurrent scheduling and writing")
    cfg.result_buffer isa Integer && cfg.result_buffer > 0 || error("result_buffer must be a positive integer")
    !isempty(cfg.Ls) && !isempty(cfg.Ts) && allunique(cfg.Ls) && allunique(cfg.Ts) ||
        error("Ls and Ts must be nonempty and contain no duplicates")
    all(L -> 2 <= L <= typemax(UInt32), cfg.Ls) && all(T -> isfinite(T) && T > 0, cfg.Ts) ||
        error("Invalid sizes or temperatures")
    cfg.ndisorder > 0 && 0 <= cfg.p <= 1 && 0 <= cfg.seed <= typemax(UInt32) ||
        error("Invalid disorder count, probability, or seed")
    all(J -> isfinite(J) && J >= 0, (cfg.Jstrong, cfg.Jweak)) || error("Invalid couplings")
    cfg.binsize > 0 && cfg.mcs >= 2cfg.binsize && cfg.mcs % cfg.binsize == 0 &&
        cfg.thermalization >= 0 && 0 <= cfg.max_corr_time <= cfg.mcs || error("Invalid MC parameters")
    myid() == 1 || error("run_scan must run on the manager process")
    !isempty(pids) && all(pid -> pid != 1 && pid in workers(), pids) ||
        throw(ArgumentError("pids must contain active worker processes"))
    ispath(cfg.output_dir) && error("Output directory already exists: $(cfg.output_dir)")
    mkpath(dirname(cfg.output_dir))
    mkdir(cfg.output_dir)
    @sync for pid in pids
        @async remotecall_wait(Base.include, pid, Main, abspath(@__FILE__))
    end

    jobs = [(L=Int(L), T=Float64(T)) for L in cfg.Ls for T in cfg.Ts]
    queue = Channel{eltype(jobs)}(length(jobs))
    foreach(job -> put!(queue, job), jobs)
    close(queue)
    results = Channel{NamedTuple}(cfg.result_buffer)
    @sync begin
        # 只有这一个任务调用 HDF5；独立线程避免同步写盘阻塞进程调度。
        Threads.@spawn begin
            try
                for data in results
                    path = write_case(cfg, data)
                    @info "Saved MC" worker=data.worker_id L=data.L T=data.T path
                end
            finally
                # 写盘失败也要唤醒因队列满而等待的生产者。
                close(results)
            end
        end
        @async begin
            try
                @sync for pid in pids
                    @async for job in queue
                        isopen(results) || break
                        @info "Starting MC" worker=pid L=job.L T=job.T
                        data = remotecall_fetch(compute_case, pid, job, cfg)
                        put!(results, data) # 满时等待，成功入队后即可调度下一项。
                        data = nothing
                    end
                end
            finally
                # 所有生产者结束后，写入任务排空队列再退出。
                close(results)
            end
        end
    end
    return [joinpath(cfg.output_dir, "L_$L.h5") for L in cfg.Ls]
end

end # module

# --test uses local processes; normal execution uses the Slurm allocation.
if abspath(PROGRAM_FILE) == @__FILE__
    using Distributed, HDF5
    testing = ARGS == ["--test"]
    isempty(ARGS) || testing || error("Usage: run_slurm.jl [--test]")
    config = RandomIsingScan.read_config()
    threads = testing ? 2 : parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
    project = dirname(Base.active_project())
    flags = `--project=$project --threads=$threads --startup-file=no`
    if testing
        pids = addprocs(2; exeflags=flags)
    else
        using SlurmClusterManager
        pids = addprocs(SlurmManager(); exeflags=flags)
    end
    try
        if testing
            mktempdir() do dir
                cfg = merge(config, (
                    Ls=[2, 3], Ts=[1.5, 2.5], ndisorder=3, mcs=32,
                    thermalization=8, binsize=8, max_corr_time=5,
                    output_dir=joinpath(dir, "results")))
                paths = RandomIsingScan.run_scan(cfg, pids)
                @assert length(paths) == 2
                for (L, path) in zip(cfg.Ls, paths)
                    h5open(path, "r") do file
                        for T in cfg.Ts
                            data = RandomIsingScan.compute_case((; L, T), cfg)
                            g = file["T_$(repr(T))"]
                            @assert read(attributes(g)["complete"])
                            @assert read(attributes(g)["worker_threads"]) == 2
                            @assert read(file["disorder"]) == data.disorder
                            @assert read(g["heat_capacity"]) == data.out.heat_capacity
                            @assert read(g["susceptibility"]) == data.out.susceptibility
                            @assert read(g["local_susceptibility"]) == stack(data.out.local_susceptibility)
                            @assert read(g["correlation"]) == reduce(hcat, data.out.correlation)
                        end
                    end
                end
                println("PASS: 2 workers x 2 threads, 4 cases, HDF5 matches serial results")
            end
        else
            RandomIsingScan.run_scan(config, pids)
        end
    finally
        rmprocs(pids)
    end
end
