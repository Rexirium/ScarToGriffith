include(joinpath(@__DIR__, "..", "run.jl"))
include(joinpath(@__DIR__, "..", "run_slurm.jl"))
using Test

@testset "Scan configuration and HDF5 round trip" begin
    base = read_config()
    defaults = TOML.parsefile(joinpath(@__DIR__, "..", "scan.toml"))
    @test base.sizes == defaults["sizes"]
    @test base.seed == 1996
    @test base.worker_threads == 8
    @test base.field_distribution == Symbol(defaults["field_distribution"])
    @test base.offset == defaults["offset"]
    @test base.fields == 10 .^ range(-1, 1, 101)
    @test base.times == config_grid(defaults["times"])
    @test RandomTFIMSlurm.read_config().fields == base.fields

    mktempdir() do directory
        path = joinpath(directory, "scan.toml")
        defaults = TOML.parsefile(joinpath(@__DIR__, "..", "scan.toml"))
        function save_config(params)
            open(path, "w") do io
                TOML.print(io, params)
            end
        end
        fixed_params = merge(defaults, Dict("field_distribution" => "fixed",
            "output" => "results/test_fixed.h5"))
        save_config(fixed_params)
        fixed = read_config(path)
        @test dirname(fixed.output) == joinpath(directory, "results")
        @test occursin(r"test_fixed_\d{8}_\d{6}\.h5$", fixed.output)
        for (key, value) in (("nsamples", 0), ("boundary", "bad"),
                ("field_distribution", "bad"), ("offset", "true"), ("offset", 1), ("seed", -1), ("worker_threads", 0),
                ("fields", [0.0]), ("times", [-1.0]),
                ("sizes", [1]), ("sizes", [4, 4]), ("fields", [1.0, 1.0]),
                ("fields", Dict("start" => 0, "stop" => 1, "length" => 0)),
                ("times", Dict("start" => 0, "stop" => 1, "length" => 2, "scale" => "bad")))
            save_config(merge(defaults, Dict(key => value)))
            @test_throws ArgumentError read_config(path)
        end
        save_config(merge(defaults, Dict("sizes" => [4, 5], "nsamples" => 65,
            "seed" => 72, "field_distribution" => "fixed", "offset" => false,
            "fields" => [0.5, 1.0],
            "times" => Dict("start" => 0.0, "stop" => 1.0, "length" => 3))))
        cfg = read_config(path)
        @test cfg.times == [0.0, 0.5, 1.0]
        @test cfg.seed == 72
        @test !cfg.offset
        output = redirect_stdout(devnull) do
            main(cfg)
        end
        @test_throws ArgumentError main(cfg)
        failed = merge(cfg, (; output=joinpath(directory, "failed.h5")))
        @test_throws ErrorException with_results(failed) do file
            error("simulated computation failure")
        end
        h5open(failed.output, "r") do file
            @test !read(attributes(file)["complete"])
        end
        h5open(output, "r") do file
            @test read(attributes(file)["complete"])
            @test read(attributes(file)["master_seed"]) == 72
            @test read(attributes(file)["fixed_field_divisor"]) == exp(1)
            @test length(keys(file["L4"])) == 2
            @test read(attributes(file["L4/h0.5"])["seed"]) == 72 + 1000 + 4
            @test read(attributes(file["L5/h0.5"])["j"]) == 3
            @test all(isfinite, read(file["L4/h1.0/autocorrelation_sem"]))
            @test !haskey(file["L4/h0.5"], "gap_samples")
            @test !haskey(file["L4/h1.0"], "gap_samples")
        end
        # Exercise actual worker scheduling and compare every saved value with the local run.
        project = dirname(Base.active_project())
        pids = addprocs(2; exeflags=`--project=$project --threads=2 --startup-file=no`)
        try
            distributed_cfg = merge(cfg, (; worker_threads=2, output=joinpath(directory, "distributed.h5")))
            redirect_stdout(devnull) do
                RandomTFIMSlurm.run_scan(distributed_cfg, pids)
            end
            function compare_results(a, b)
                @test Set(keys(a)) == Set(keys(b))
                @test Set(keys(attributes(a))) == Set(keys(attributes(b)))
                for key in keys(attributes(a))
                    @test isequal(read(attributes(a)[key]), read(attributes(b)[key]))
                end
                for key in keys(a)
                    if a[key] isa HDF5.Group
                        compare_results(a[key], b[key])
                    else
                        @test isequal(read(a[key]), read(b[key]))
                    end
                end
            end
            h5open(output, "r") do local_file
                h5open(distributed_cfg.output, "r") do distributed_file
                    compare_results(local_file, distributed_file)
                end
            end
            for (name, overrides) in (("compute_failure", (; nsamples=0)),
                    ("write_failure", (; fields=[0.5, 0.5])))
                failed_cfg = merge(distributed_cfg, overrides,
                    (; output=joinpath(directory, "$name.h5")))
                @test_throws CompositeException RandomTFIMSlurm.run_scan(failed_cfg, pids)
                h5open(failed_cfg.output, "r") do file
                    @test !read(attributes(file)["complete"])
                end
            end
        finally
            rmprocs(pids)
        end
    end

    for L in (4, 16), field_distribution in (:uniform, :fixed), nsamples in (1, 3, 65), offset in (false, true)
        times = [0.0, 0.3, 1.7]
        cfg = (; base..., field_distribution, nsamples, times, offset)
        h0, seed = 1.8, 72
        rng = Xoshiro(seed)
        samples = [sample_disorder(rng, L, h0; field_distribution) for _ in 1:nsamples]
        gaps = [energy_gap(J, h) for (J, h) in samples]
        observations = (
            gap=[g.gap for g in gaps],
            autocorrelation=hcat([autocorrelation(J, h, times; offset) for (J, h) in samples]...),
            real_autocorrelation=hcat([autocorrelation(J, h, times; offset, time_domain=:real) for (J, h) in samples]...),
            entanglement_entropy=hcat([[entanglement_entropy(J, h; cut) for cut in floor.(Int, (0:8) .* (L / 8))] for (J, h) in samples]...))
        data = compute_case((; L, h0, seed), cfg)
        @test length(data.result) == 16
        for (name, values) in pairs(observations)
            logs = RandomTFIM.correlation_log.(values)
            name == :gap && (logs[.!getproperty.(gaps, :resolved)] .= NaN)
            for (prefix, x) in (("", values), ("log_", logs))
                stats = RandomTFIM.mean_sem(x)
                key = replace(prefix * string(name), "log_real_" => "real_log_")
                @test isapprox(getproperty(data.result, Symbol(key, "_mean")), stats.average; nans=true, atol=1e-13)
                @test isapprox(getproperty(data.result, Symbol(key, "_sem")), stats.sem; nans=true, atol=1e-13)
            end
        end
        mktempdir() do directory
            output = joinpath(directory, "summary.h5")
            with_results(merge(cfg, (; output))) do file
                write_case(file, data, cfg)
            end
            h5open(output, "r") do file
                @test read(attributes(file)["offset"]) == offset
                @test read(attributes(file)["format_version"]) == 2
                @test Set(keys(file["parameters"])) == Set(["sizes", "fields"])
                group = file["L$L/h1.8"]
                @test Set(keys(group)) == union(Set(string.(keys(data.result))),
                    Set(["imaginary_time", "real_time", "subsystem_sizes"]))
                @test read(group["subsystem_sizes"]) == (L == 4 ? [0, 0, 1, 1, 2, 2, 3, 3, 4] : collect(0:2:16))
                for prefix in ("entanglement_entropy", "log_entanglement_entropy"), suffix in ("_mean", "_sem")
                    @test length(read(group[prefix * suffix])) == 9
                end
                @test read(group["entanglement_entropy_mean"])[[1, 9]] == [0.0, 0.0]
                @test read(group["imaginary_time"]) == read(group["real_time"]) == times
                for (key, value) in pairs(data.result)
                    @test isequal(read(group[string(key)]), value)
                end
            end
        end
    end
    empty_time = scan_observables(2, 1.0, Float64[];
        nsamples=1, seed=1, field_distribution=:uniform)
    @test isempty(empty_time.autocorrelation_mean)
    @test isempty(empty_time.real_autocorrelation_mean)
    @test length(empty_time.entanglement_entropy_mean) == 9
end
