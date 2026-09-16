# 本地集成测试：两个 worker 各用两个线程，写入临时目录并对照串行结果。
import Pkg
isempty(ARGS) || error("Usage: test_run_slurm.jl")
Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "local"))

using Distributed, HDF5, Random
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
                @assert !haskey(file, "disorder")
                @assert !haskey(file, "parameters")
                @assert read(file["temperatures"]) == cfg.Ts
                disorder = RandomIsingScan.read_disorder(file)
                disorder_seed, _ = RandomIsingScan.case_seeds(cfg, L, first(cfg.Ts))
                @assert read(attributes(file)["disorder_seed"]) == disorder_seed
                rng = MersenneTwister(disorder_seed)
                expected = reduce(hcat, [ifelse.(rand(rng, 2L^2) .< cfg.p,
                    cfg.Jstrong, cfg.Jweak) for _ in 1:cfg.ndisorder])
                @assert disorder == expected
                @assert size(disorder) == (2L^2, cfg.ndisorder)
                for T in cfg.Ts
                    data = RandomIsingScan.compute_case((; L, T), cfg)
                    @assert !hasproperty(data, :disorder)
                    @assert !hasproperty(data.out.metadata, :seeds)
                    g = file["T_$(repr(T))"]
                    @assert read(attributes(file)["version"]) == string(data.out.metadata.version)
                    @assert read(attributes(g)["complete"])
                    @assert !haskey(attributes(file), "slurm_job_id")
                    @assert read(attributes(file)["format_version"]) == 12
                    for key in (:L, :mcs, :thermalization, :binsize, :max_corr_time,
                            :corr_start_time, :boundary, :normalization)
                        value = getproperty(data.out.metadata, key)
                        @assert read(attributes(file)[string(key)]) ==
                            (value isa Symbol ? string(value) : value)
                        @assert !haskey(attributes(g), string(key))
                    end
                    @assert Set(keys(attributes(g))) == Set([
                        "T", "seed", "elapsed_seconds", "complete"])
                    @assert read(attributes(g)["T"]) == T
                    @assert read(attributes(g)["seed"]) == data.out.metadata.seed
                    @assert !haskey(g, "realization_seeds")
                    seeds = RandomIsingScan.read_realization_seeds(file, g)
                    @assert seeds == rand(MersenneTwister(data.out.metadata.seed), UInt32, cfg.ndisorder)
                    @assert seeds isa Vector{UInt32} && length(seeds) == cfg.ndisorder

                    @assert read(g["heat_capacity"]) == data.out.heat_capacity
                    @assert read(g["susceptibility"]) == data.out.susceptibility
                    @assert read(g["U4"]) == data.out.U4
                    @assert read(g["correlation"]) == data.out.correlation
                    @assert read(g["chi_local"]) == data.chi_local
                    @assert read(g["errors/err_local"]) == data.err_local
                    @assert size(data.chi_local) == size(data.err_local) == (L, L)
                    chi_local, err_local = RandomIsingScan.random_bond_local_susceptibility(
                        L, T, disorder[:, 1];
                        mcs=cfg.mcs, thermalization=cfg.thermalization, binsize=cfg.binsize,
                        seed=seeds[1])
                    @assert data.chi_local == chi_local && data.err_local == err_local
                    # SW local response and heat-bath total response use different
                    # trajectories, so their finite-sample sums need not match.

                    @assert size(read(g["U4"])) == (cfg.ndisorder,)
                    @assert size(read(g["correlation"])) == (cfg.max_corr_time + 1,)
                    @assert read(g["errors/correlation"]) == data.out.errors.correlation
                    @assert size(read(g["errors/correlation"])) == (cfg.max_corr_time + 1,)
                    @assert read(g["correlation"])[1] == 1.0
                    @assert read(g["errors/correlation"])[1] == 0.0
                    @assert !haskey(g, "lags")
                    @assert length(0:read(attributes(file)["max_corr_time"])) == length(read(g["correlation"]))

                    @assert read(g["errors/heat_capacity"]) == data.out.errors.heat_capacity
                    @assert read(g["errors/susceptibility"]) == data.out.errors.susceptibility
                    @assert read(g["errors/U4"]) == data.out.errors.U4
                    @assert size(read(g["errors/U4"])) == (cfg.ndisorder,)
                    @assert !haskey(g, "local_susceptibility")
                    @assert Set(keys(g["errors"])) == Set(["heat_capacity", "susceptibility", "U4", "correlation", "err_local"])
                end
            end
        end
        # 旧文件仍可通过同一读取接口取得原有矩阵。
        h5open(joinpath(dir, "legacy.h5"), "w") do file
            expected = reshape(collect(1.0:24.0), 8, 3)
            file["disorder"] = expected
            @assert RandomIsingScan.read_disorder(file) == expected
            g = create_group(file, "T_1.5")
            g["realization_seeds"] = UInt32[10, 20, 30]
            @assert RandomIsingScan.read_realization_seeds(file, g) == UInt32[10, 20, 30]
        end
        println("PASS: 2 workers x 2 threads, 4 cases, HDF5 matches serial results")
    end
finally
    # 测试失败时也释放本次启动的 worker。
    rmprocs(pids)
end
