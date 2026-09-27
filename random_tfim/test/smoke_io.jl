using Test, HDF5, LinearAlgebra
include(joinpath(@__DIR__, "..", "run.jl"))

@testset "Demo/full: all observables and disorder SEM" begin
    mktempdir() do directory
        for mode in ("demo", "full"), boundary in (:periodic, :open)
            output = joinpath(directory, "$(mode)_$(boundary).h5")
            main([mode, "2", output, string(boundary)])
            h5open(output, "r") do file
                @test read(attributes(file)["complete"])
                @test read(attributes(file)["mode"]) == mode
                @test read(attributes(file)["time_domain"]) == "imaginary"
                @test occursin("exp(tau*H)", read(attributes(file)["time_definition"]))
                @test occursin("sigma_z(j,tau)", read(attributes(file)["observable"]))
                @test startswith(read(attributes(file)["boundary"]), string(boundary))
                sizes = mode == "demo" ? (16, 32) : (16, 32, 64, 128)
                fields = 10 .^ range(-1, 1, 101)
                @test Set(keys(file)) == Set(["parameters"; ["L$L" for L in sizes]])
                @test Set(keys(file["parameters"])) == Set(["sizes", "fields"])
                @test read(file["parameters/sizes"]) == collect(sizes)
                @test read(file["parameters/fields"]) == fields
                for L in sizes
                    @test Set(keys(file["L$L"])) == Set("h$h" for h in fields)
                    for h in keys(file["L$L"])
                        @test Set(keys(file["L$L/$h"])) == Set([
                            "gap_samples", "gap_resolved", "log_gap_samples",
                            "gap_mean", "gap_sem", "log_gap_mean", "log_gap_sem",
                            "correlation_mean", "correlation_sem",
                            "log_correlation_mean", "log_correlation_sem",
                            "imaginary_time", "autocorrelation_mean", "autocorrelation_sem"])
                    end
                    for h in keys(file["L$L"]), name in ("pair_logC", "r", "pair_counts",
                        "sample_C", "sample_logC", "sample_C_real", "sample_C_imag",
                        "typical", "typical_lower", "typical_upper",
                        "average_real", "average_imag", "sem_real", "sem_imag")
                        @test !haskey(file["L$L/$h"], name)
                    end
                    for h in keys(file["L$L"]), name in ("L", "h0", "nsamples", "unresolved_gaps")
                        @test !haskey(attributes(file["L$L/$h"]), name)
                    end
                end
                group = file["L16/h1.0"]
                @test read(attributes(group)["j"]) == 8
                gaps, resolved = read(group["gap_samples"]), read(group["gap_resolved"])
                logs = read(group["log_gap_samples"])
                @test isequal(logs, map((gap, ok) -> ok ? log(gap) : NaN, gaps, resolved))
                for (prefix, samples) in (("gap", gaps), ("log_gap", logs))
                    @test read(group["$(prefix)_mean"]) ≈ mean(samples)
                    @test read(group["$(prefix)_sem"]) ≈ std(samples) / sqrt(2)
                end
                times = read(group["imaginary_time"])
                @test times == collect(0.0:0.1:20.0)
                expected = disorder_ensemble(16, 1.0; times, nsamples=2,
                    seed=read(attributes(group)["seed"]), boundary)
                spatial, log_spatial = expected.sample_C, expected.sample_logC
                @test read(group["correlation_mean"]) ≈ vec(mean(spatial; dims=2))
                @test read(group["correlation_sem"]) ≈ vec(std(spatial; dims=2)) / sqrt(2)
                @test read(group["log_correlation_mean"]) ≈ vec(mean(log_spatial; dims=2))
                @test read(group["log_correlation_sem"]) ≈ vec(std(log_spatial; dims=2)) / sqrt(2)
                samples = expected.sample_Ct
                @test samples isa Matrix{Float64}
                @test size(samples) == (length(times), 2)
                @test all(isfinite, samples)
                average = read(group["autocorrelation_mean"])
                sem = read(group["autocorrelation_sem"])
                @test average isa Vector{Float64}
                @test sem isa Vector{Float64}
                @test average ≈ vec(mean(samples; dims=2))
                @test sem ≈ vec(std(samples; dims=2)) / sqrt(2)
                @test gaps ≈ expected.gaps
                @test average[1] ≈ 1 atol=1e-12
            end
            @test_throws ArgumentError main([mode, "2", output])
        end
        single = joinpath(directory, "single.h5")
        main(["demo", "1", single])
        h5open(single, "r") do file
            group = file["L16/h1.0"]
            for name in ("gap_sem", "log_gap_sem")
                @test isnan(read(group[name]))
            end
            for name in ("correlation_sem", "log_correlation_sem", "autocorrelation_sem")
                @test all(isnan, read(group[name]))
            end
        end
    end
    @test isnan(mean_sem([1.0, NaN]).average)
    @test isnan(mean_sem([1.0, NaN]).sem)
    for mode in ("gap", "correlation", "autocorrelation", "bad")
        @test_throws ArgumentError main([mode])
    end
    @test_throws ArgumentError main(["demo", "1", "unused.h5", "bad"])
    @test_throws ArgumentError main(["demo", "0"])
end
