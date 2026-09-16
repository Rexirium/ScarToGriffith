# 直接运行时选择环境；worker include 时沿用启动参数指定的环境。
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    isempty(ARGS) || error("Usage: run_slurm.jl")
    Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "server"))
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
    if params["Ts"] isa AbstractDict
        t = params["Ts"]
        params["Ts"] = collect(range(t["start"], t["stop"]; length=t["length"]))
    end
    params["output_dir"] = normpath(joinpath(dirname(abspath(path)), params["output_dir"]))
    return (; (Symbol(key) => value for (key, value) in params)...)
end

# 种子由参数决定：同一 L 的不同温度共享无序，但使用独立的 MC 随机数流。
function case_seeds(cfg, L, T)
    disorder_seed = rand(MersenneTwister(UInt32[cfg.seed, L, 0]), UInt32)
    bits = reinterpret(UInt64, T)
    mc_seed = rand(MersenneTwister(UInt32[cfg.seed, L, 1,
        bits & 0xffffffff, bits >> 32]), UInt32)
    return disorder_seed, mc_seed
end

"""按固定顺序生成无序构型；扫描和文件重构共用此实现。"""
function generate_disorder(cfg, L, disorder_seed)
    rng = MersenneTwister(disorder_seed)
    return [ifelse.(rand(rng, 2L^2) .< cfg.p, cfg.Jstrong, cfg.Jweak)
        for _ in 1:cfg.ndisorder]
end

"""读取旧文件的 disorder 或从新文件根属性重构，返回 (bond, realization) 矩阵。
重构需使用写入时的 Julia/Random 版本，以保持随机序列一致。
"""
function read_disorder(file)
    haskey(file, "disorder") && return read(file["disorder"])
    attrs = attributes(file)
    cfg = (; (key => read(attrs[string(key)])
        for key in (:ndisorder, :p, :Jstrong, :Jweak))...)
    return reduce(hcat, generate_disorder(cfg, read(attrs["L"]), read(attrs["disorder_seed"])))
end

"""读取旧温度组的采样种子，或由 seed 和文件根的 ndisorder 重构。
重构需使用写入时的 Julia/Random 版本。
"""
function read_realization_seeds(file, group)
    haskey(group, "realization_seeds") && return read(group["realization_seeds"])
    return rand(MersenneTwister(read(attributes(group)["seed"])), UInt32,
        read(attributes(file)["ndisorder"]))
end

"""一个 worker 计算一个 (L,T)，内部由 random_bond_observables 多线程处理无序。"""
function compute_case(job, cfg)
    L, T = job.L, job.T
    disorder_seed, mc_seed = case_seeds(cfg, L, T)
    disorder = generate_disorder(cfg, L, disorder_seed)

    # 计时从采样开始；各无序构型在当前 worker 内并行计算。
    started = time_ns()
    out = random_bond_observables(L, T, disorder;
        mcs=cfg.mcs, thermalization=cfg.thermalization, binsize=cfg.binsize,
        max_corr_time=cfg.max_corr_time,
        seed=mc_seed, details=true)
    # 第一个无序构型用同一种子重现 SW 静态采样轨迹。
    chi_local, err_local = random_bond_local_susceptibility(L, T, disorder[1];
        mcs=cfg.mcs, thermalization=cfg.thermalization, binsize=cfg.binsize,
        seed=out.metadata.seeds[1])

    # 局域计算后不再需要完整种子数组，避免随结果传回主进程。
    out = merge(out, (; metadata=(; (key => value for (key, value) in
        pairs(out.metadata) if key != :seeds)...)))

    return (; L, T, out, chi_local, err_local, disorder_seed,
        worker_id=myid(),
        elapsed_seconds=(time_ns() - started) / 1e9)
end

"""仅由主进程调用；同一 L 写入同一文件，每个 T 创建独立 group。"""
function write_case(cfg, data)
    path = joinpath(cfg.output_dir, "L_$(data.L).h5")
    h5open(path, "cw") do file
        # 同一 L 的公共参数只写一次；无序由根属性中的种子和参数重构。
        if !haskey(attributes(file), "format_version")
            file_metadata = (
                L=data.L, ndisorder=cfg.ndisorder, p=cfg.p,
                Jstrong=cfg.Jstrong, Jweak=cfg.Jweak,
                disorder_seed=data.disorder_seed, master_seed=cfg.seed,

                mcs=cfg.mcs, thermalization=cfg.thermalization, binsize=cfg.binsize,
                max_corr_time=cfg.max_corr_time,
                boundary=string(data.out.metadata.boundary),

                normalization=string(data.out.metadata.normalization),
                julia_version=string(VERSION),
                version=string(data.out.metadata.version)
            )

            for (key, value) in pairs(file_metadata)
                attributes(file)[string(key)] = value
            end

            file["temperatures"] = cfg.Ts
            attributes(file)["format_version"] = 13 # 公共数据写完后标记初始化完成。
        end

        group_name = "T_$(repr(data.T))"
        haskey(file, group_name) && error("Refusing to overwrite $path/$group_name")
        group = create_group(file, group_name)

        # 静态量逐构型保存；correlation 保存跨构型均值向量。
        out = data.out
        for key in (:heat_capacity, :susceptibility, :U4, :correlation)
            group[string(key)] = getproperty(out, key)
        end
        group["chi_local"] = data.chi_local # L×L，仅对应重构的第一个无序构型。

        # 静态量保存分块误差；自关联保存跨构型样本标准误。
        errors = create_group(group, "errors")
        for (key, value) in pairs(out.errors)
            errors[string(key)] = value
        end
        errors["err_local"] = data.err_local

        # 温度组只保存本次计算的属性；公共参数已存于文件根。
        group_metadata = (T=data.T, seed=out.metadata.seed,
            elapsed_seconds=data.elapsed_seconds)
        for (key, value) in pairs(group_metadata)
            attributes(group)[string(key)] = value
        end

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
    # 启动任务前检查参数和进程，避免运行中途才发现配置错误。
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
        cfg.thermalization >= 0 && cfg.max_corr_time >= 0 || error("Invalid MC parameters")

    myid() == 1 || error("run_scan must run on the manager process")
    !isempty(pids) && all(pid -> pid != 1 && pid in workers(), pids) ||
        throw(ArgumentError("pids must contain active worker processes"))

    # 每次扫描使用新目录；所有 worker 加载同一份计算代码。
    ispath(cfg.output_dir) && error("Output directory already exists: $(cfg.output_dir)")
    mkpath(dirname(cfg.output_dir))
    mkdir(cfg.output_dir)
    @sync for pid in pids
        @async remotecall_wait(Base.include, pid, Main, abspath(@__FILE__))
    end

    # 任务队列预先填满后关闭；结果队列限制等待写盘的数据量。
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

        # 每个 worker 完成并提交结果后，立即领取下一组 (L,T)。
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

# 正式运行使用 Slurm 分配的资源；本地验证见 test_run_slurm.jl。
if abspath(PROGRAM_FILE) == @__FILE__
    using Distributed, SlurmClusterManager

    config = RandomIsingScan.read_config()
    threads = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
    project = dirname(Base.active_project())
    flags = `--project=$project --threads=$threads --startup-file=no`
    pids = addprocs(SlurmManager(); exeflags=flags)

    try
        for pid in pids
            @info "Worker threads" pid threads=remotecall_fetch(Threads.nthreads, pid)
        end
        RandomIsingScan.run_scan(config, pids)
    finally
        # 无论扫描成功还是抛出异常，都释放本次启动的 worker。
        rmprocs(pids)
    end
end
