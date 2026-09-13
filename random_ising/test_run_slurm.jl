# 本地集成测试：两个 worker 各用两个线程，写入临时目录并对照串行结果。
import Pkg
isempty(ARGS) || error("Usage: test_run_slurm.jl")
Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "local"))

using Distributed, HDF5
include("run_slurm.jl")

config = RandomIsingScan.read_config()
project = dirname(Base.active_project())
pids = addprocs(2; exeflags=`--project=$project --threads=2 --startup-file=no`)

try
    mktempdir() do dir
        cfg = merge(config, (
            Ls=[2, 3], Ts=[1.5, 2.5], ndisorder=3, mcs=32,
            thermalization=8, binsize=8, max_corr_time=5, corr_start_time=7,
            output_dir=joinpath(dir, "results")))
        paths = RandomIsingScan.run_scan(cfg, pids)
        @assert length(paths) == 2

        for (L, path) in zip(cfg.Ls, paths)
            h5open(path, "r") do file
                for T in cfg.Ts
                    data = RandomIsingScan.compute_case((; L, T), cfg)
                    g = file["T_$(repr(T))"]
                    @assert read(attributes(file)["version"]) == string(data.out.metadata.version)
                    @assert read(attributes(g)["complete"])
                    @assert read(attributes(g)["worker_threads"]) == 2
                    @assert read(attributes(file)["format_version"]) == 4
                    for key in (:L, :mcs, :thermalization, :binsize, :max_corr_time,
                            :corr_start_time, :boundary, :normalization)
                        value = getproperty(data.out.metadata, key)
                        @assert read(attributes(file)[string(key)]) ==
                            (value isa Symbol ? string(value) : value)
                        @assert !haskey(attributes(g), string(key))
                    end
                    @assert Set(keys(attributes(g))) == Set([
                        "T", "seed", "worker_id", "worker_threads", "elapsed_seconds", "complete"])
                    @assert read(attributes(g)["T"]) == T
                    @assert read(attributes(g)["seed"]) == data.out.metadata.seed
                    @assert read(g["realization_seeds"]) == data.out.metadata.seeds
                    @assert read(file["disorder"]) == data.disorder

                    @assert read(g["heat_capacity"]) == data.out.heat_capacity
                    @assert read(g["susceptibility"]) == data.out.susceptibility
                    @assert read(g["local_susceptibility"]) == data.out.local_susceptibility
                    @assert read(g["correlation"]) == data.out.correlation

                    @assert size(read(g["local_susceptibility"])) == (L, L, cfg.ndisorder)
                    @assert size(read(g["correlation"])) == (cfg.max_corr_time + 1, cfg.ndisorder)
                    @assert read(g["lags"]) == collect(0:cfg.max_corr_time)

                    @assert read(g["errors/heat_capacity"]) == data.out.errors.heat_capacity
                    @assert read(g["errors/susceptibility"]) == data.out.errors.susceptibility
                    @assert read(g["errors/local_susceptibility"]) == data.out.errors.local_susceptibility
                end
            end
        end
        println("PASS: 2 workers x 2 threads, 4 cases, HDF5 matches serial results")
    end
finally
    # 测试失败时也释放本次启动的 worker。
    rmprocs(pids)
end
