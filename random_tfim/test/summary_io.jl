include(joinpath(@__DIR__, "..", "run.jl"))
include(joinpath(@__DIR__, "..", "run_slurm.jl"))
using Test

@testset "Selective sample output" begin
    base = RandomTFIMSlurm.parse_config(["demo"])
    selected = base.fields[[1, 36, 51, 66, 86, 101]]
    @test base.sample_fields == select_sample_fields(base.fields) == selected
    @test length(unique(selected)) == 6
    sample_keys = ("gap_samples", "gap_resolved",
        "sample_C", "sample_Ct", "sample_real_Ct")
    for nsamples in (1, 3), boundary in (:open, :periodic)
        cfg = (; base..., nsamples, boundary, times=[0.0, 0.3])
        mktempdir() do directory
            h5open(joinpath(directory, "summary.h5"), "w") do file
                RandomTFIMSlurm.write_metadata(file, cfg)
                @test read(file["parameters/sample_fields"]) == selected
                @test Set(keys(attributes(file))) == Set(["complete", "nsamples", "distribution", "boundary"])
                @test !read(attributes(file)["complete"])
                @test read(attributes(file)["boundary"]) == string(boundary)
                @test read(attributes(file)["distribution"]) == "box"
                @test read(attributes(file)["nsamples"]) == nsamples
                for h0 in [selected; base.fields[2]]
                    job = (; L=6, h0, seed=72)
                    data = RandomTFIMSlurm.compute_case(job, cfg)
                    keep = h0 in selected
                    @test hasproperty(data.result, :gaps) == keep
                    @test hasproperty(data.result, :sample_Ct) == keep
                    @test !hasproperty(data.result, :sample_logCt)
                    local_group = create_group(file, "local/h$h0")
                    write_observables(local_group, data.result, cfg.times)
                    RandomTFIMSlurm.write_case(file, data, cfg)
                    group = file["L6/h$h0"]
                    @test Set(keys(attributes(group))) == Set(["j", "seed"])
                    @test Set(keys(group)) == Set(keys(local_group))
                    for key in keys(group)
                        @test isequal(read(group[key]), read(local_group[key]))
                    end
                    for field in (:gap_mean, :gap_sem, :log_gap_mean, :log_gap_sem)
                        @test isequal(read(group[string(field)]), getproperty(data.result, field))
                    end
                    @test !haskey(group, "log_gap_samples")
                    for key in sample_keys
                        @test haskey(group, key) == keep
                    end
                    if keep
                        @test read(group["gap_samples"]) == data.result.gaps
                        @test read(group["gap_resolved"]) == data.result.resolved
                        for field in (:sample_C, :sample_Ct, :sample_real_Ct)
                            @test read(group[string(field)]) == getproperty(data.result, field)
                            @test size(read(group[string(field)]), 2) == nsamples
                        end
                    end
                    @test read(group["imaginary_time"]) == read(group["real_time"]) == cfg.times
                end
            end
        end
    end
end

@testset "Real-time HDF5 contains only real correlation statistics" begin
    times = [0.0, 0.3, 1.7]
    result = disorder_ensemble(6, 1.8, times; nsamples=3, seed=72)
    mktempdir() do directory
        path = joinpath(directory, "real.h5")
        h5open(path, "w") do file
            for (name, writer) in (("local", write_correlations), ("slurm", RandomTFIMSlurm.write_correlations))
                writer(create_group(file, name), result, times)
            end
        end
        h5open(path, "r") do file
            for name in ("local", "slurm")
                group = file[name]
                @test read(group["real_time"]) == times
                @test read(group["imaginary_time"]) == times
                for (key, field) in (("autocorrelation_mean", :Ct_mean), ("autocorrelation_sem", :Ct_sem),
                        ("log_autocorrelation_mean", :logCt_mean), ("log_autocorrelation_sem", :logCt_sem),
                        ("real_autocorrelation_mean", :real_Ct_mean),
                        ("real_autocorrelation_sem", :real_Ct_sem),
                        ("real_log_autocorrelation_mean", :real_logCt_mean),
                        ("real_log_autocorrelation_sem", :real_logCt_sem))
                    values = read(group[key])
                    @test values isa Vector{Float64}
                    @test isequal(values, getproperty(result, field))
                end
            end
        end
    end
end
