include(joinpath(@__DIR__, "..", "run.jl"))
include(joinpath(@__DIR__, "..", "run_slurm.jl"))
using Test

@testset "Scan configuration and HDF5 round trip" begin
    base = read_config()
    @test base.sizes == [16, 32]
    @test base.seed == 1996
    @test base.worker_threads == 8
    @test base.field_distribution == :uniform
    @test base.sample_fields == select_sample_fields(base.fields, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    @test base.fields == 10 .^ range(-1, 1, 101)
    @test base.times == 10 .^ range(-1, 3, 101)
    @test RandomTFIMSlurm.read_config().fields == base.fields

    mktempdir() do directory
        path = joinpath(directory, "scan.toml")
        defaults = TOML.parsefile(joinpath(@__DIR__, "..", "scan.toml"))
        function save_config(params)
            open(path, "w") do io
                TOML.print(io, params)
            end
        end
        fixed_params = merge(defaults, Dict("field_distribution" => "fixed"))
        save_config(fixed_params)
        fixed = read_config(path)
        @test fixed.sample_fields == base.sample_fields
        save_config(merge(fixed_params, Dict("sample_fields" => [0.5, 1.0, 1.5, 2.0, exp(1), 3.0])))
        fixed = read_config(path)
        @test fixed.sample_fields ≈ [10.0^x for x in (-0.3, 0.0, 0.18, 0.3, 0.44, 0.48)]
        @test dirname(fixed.output) == joinpath(directory, "results")
        @test occursin(r"demo_uniform_\d{8}_\d{6}\.h5$", fixed.output)
        for (key, value) in (("nsamples", 0), ("boundary", "bad"),
                ("field_distribution", "bad"), ("seed", -1), ("worker_threads", 0),
                ("sizes", [3]), ("sizes", [4, 4]), ("fields", [1.0, 1.0]),
                ("fields", [0.0]), ("times", [-1.0]), ("sample_fields", [-1.0]),
                ("sample_fields", Dict("uniform" => [1.0])),
                ("fields", Dict("start" => 0, "stop" => 1, "length" => 0)),
                ("times", Dict("start" => 0, "stop" => 1, "length" => 2, "scale" => "bad")))
            save_config(merge(defaults, Dict(key => value)))
            @test_throws ArgumentError read_config(path)
        end
        save_config(merge(defaults, Dict("sizes" => [4, 6], "nsamples" => 1,
            "seed" => 72, "boundary" => "open", "field_distribution" => "fixed",
            "fields" => [0.5, 1.0], "sample_fields" => [0.6],
            "times" => Dict("start" => 0.0, "stop" => 1.0, "length" => 3))))
        cfg = read_config(path)
        @test cfg.times == [0.0, 0.5, 1.0]
        @test cfg.sample_fields == [0.5]
        @test cfg.seed == 72
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
            @test all(isnan, read(file["L4/h1.0/autocorrelation_sem"]))
            @test haskey(file["L4/h0.5"], "gap_samples")
            @test !haskey(file["L4/h1.0"], "gap_samples")
        end
        # Exercise actual worker scheduling and compare every saved value with the local run.
        project = dirname(Base.active_project())
        pids = addprocs(1; exeflags=`--project=$project --threads=2 --startup-file=no`)
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
        finally
            rmprocs(pids)
        end
        save_config(merge(defaults, Dict("sample_fields" => [])))
        @test isempty(read_config(path).sample_fields)
    end

    for boundary in (:open, :periodic), field_distribution in (:uniform, :fixed)
        cfg = (; base..., boundary, field_distribution, nsamples=3, times=[0.0, 0.3, 1.7])
        mktempdir() do directory
            path = joinpath(directory, "ensemble.h5")
            expected = Dict()
            h5open(path, "w") do file
                write_metadata(file, cfg)
                # Selected and summary-only points cover both output layouts.
                for h0 in (base.sample_fields[3], base.fields[2])
                    data = RandomTFIMSlurm.compute_case((; L=6, h0, seed=72), cfg)
                    expected[h0] = disorder_ensemble(6, h0, cfg.times; boundary,
                        field_distribution, nsamples=3, seed=72, keep_samples=h0 in cfg.sample_fields)
                    @test isequal(data.result, expected[h0])
                    redirect_stdout(devnull) do
                        RandomTFIMSlurm.write_case(file, data, cfg)
                    end
                end
            end
            h5open(path, "r") do file
                @test !read(attributes(file)["complete"])
                @test read(attributes(file)["field_distribution"]) == string(field_distribution)
                @test read(attributes(file)["boundary"]) == string(boundary)
                @test read(file["parameters/sample_fields"]) == cfg.sample_fields
                if field_distribution == :fixed
                    @test read(attributes(file)["fixed_field_divisor"]) == exp(1)
                end
                for (h0, result) in expected
                    g = file["L6/h$h0"]
                    @test read(g["imaginary_time"]) == read(g["real_time"]) == cfg.times
                    for (prefix, mean_key, sem_key) in (("correlation", :C_mean, :C_sem),
                            ("log_correlation", :logC_mean, :logC_sem),
                            ("autocorrelation", :Ct_mean, :Ct_sem),
                            ("log_autocorrelation", :logCt_mean, :logCt_sem),
                            ("real_autocorrelation", :real_Ct_mean, :real_Ct_sem),
                            ("real_log_autocorrelation", :real_logCt_mean, :real_logCt_sem))
                        @test isequal(read(g[prefix * "_mean"]), getproperty(result, mean_key))
                        @test isequal(read(g[prefix * "_sem"]), getproperty(result, sem_key))
                    end
                    for key in (:gap_mean, :gap_sem, :log_gap_mean, :log_gap_sem)
                        @test isequal(read(g[string(key)]), getproperty(result, key))
                    end
                    keep = h0 in cfg.sample_fields
                    for key in ("gap_samples", "gap_resolved", "sample_C", "sample_Ct", "sample_real_Ct")
                        @test haskey(g, key) == keep
                    end
                    if keep
                        @test read(g["gap_samples"]) == result.gaps
                        @test read(g["gap_resolved"]) == result.resolved
                        for key in (:sample_C, :sample_Ct, :sample_real_Ct)
                            @test read(g[string(key)]) == getproperty(result, key)
                        end
                    end
                end
            end
        end
    end

end
