include(joinpath(@__DIR__, "..", "run.jl"))
include(joinpath(@__DIR__, "..", "run_slurm.jl"))
using Test

@testset "Scan configuration and HDF5 round trip" begin
    base = parse_config(["demo"])
    @test base.sizes == (16, 32)
    @test parse_config(["full"]).sizes == (16, 32, 64, 128)
    @test base.field_distribution == :uniform
    @test base.sample_fields == select_sample_fields(base.fields)
    for args in (["bad"], ["demo", "0"], ["demo", "1", "unused.h5", "bad"],
                 ["demo", "1", "unused.h5", "open", "bad"])
        @test_throws ArgumentError parse_config(args)
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

    # Exercise the actual CLI once; statistics and both modes are covered above.
    mktempdir() do directory
        output = redirect_stdout(devnull) do
            main(["demo", "1", joinpath(directory, "demo.h5"), "periodic", "fixed"])
        end
        h5open(output, "r") do file
            @test read(attributes(file)["complete"])
            @test read(attributes(file)["fixed_field_divisor"]) == exp(1)
            @test length(keys(file["L16"])) == length(base.fields)
            @test all(isnan, read(file["L16/h1.0/autocorrelation_sem"]))
        end
    end
end
