include(joinpath(@__DIR__, "..", "run.jl"))
include(joinpath(@__DIR__, "..", "run_slurm.jl"))
using Test

@testset "Summary output interfaces" begin
    for nsamples in (1, 3), boundary in (:open, :periodic)
        cfg = (; nsamples, boundary, times=[0.0, 0.3])
        job = (; L=6, h0=1.0, seed=72)
        data = RandomTFIMSlurm.compute_case(job, cfg)
        @test !hasproperty(data.result, :sample_Ct)
        mktempdir() do directory
            h5open(joinpath(directory, "summary.h5"), "w") do file
                local_group = create_group(file, "local")
                write_correlations(local_group, data.result, cfg.times)
                RandomTFIMSlurm.write_case(file, data, cfg)
                group = file["L6/h1.0"]
                for (name, field) in (("correlation_mean", :C_mean),
                        ("correlation_sem", :C_sem), ("log_correlation_mean", :logC_mean),
                        ("log_correlation_sem", :logC_sem), ("autocorrelation_mean", :Ct_mean),
                        ("autocorrelation_sem", :Ct_sem))
                    @test isequal(read(group[name]), getproperty(data.result, field))
                    @test isequal(read(local_group[name]), read(group[name]))
                end
                @test read(group["gap_samples"]) == data.result.gaps
                @test read(group["gap_resolved"]) == data.result.resolved
                @test read(local_group["imaginary_time"]) == read(group["imaginary_time"]) == cfg.times
            end
        end
    end
end
