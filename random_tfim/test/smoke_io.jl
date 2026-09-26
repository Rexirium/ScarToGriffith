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
                @test startswith(read(attributes(file)["boundary"]), string(boundary))
                sizes = mode == "demo" ? (16, 32) : (16, 32, 64, 128)
                @test Set(keys(file)) == Set("L$L" for L in sizes)
                for L in sizes
                    @test Set(keys(file["L$L"])) == Set("h$h" for h in (1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 3.0))
                    for h in keys(file["L$L"]), name in ("pair_logC", "r", "pair_counts")
                        @test !haskey(file["L$L/$h"], name)
                    end
                end
                group = file["L16/h1.0"]
                @test read(attributes(group)["j"]) == 8
                gaps, resolved = read(group["gaps"]), read(group["resolved"])
                logs = read(group["loggaps"])
                @test isequal(logs, map((gap, ok) -> ok ? log(gap) : NaN, gaps, resolved))
                for (prefix, samples) in (("gap", gaps), ("loggap", logs))
                    @test read(group["$(prefix)_average"]) ≈ mean(samples)
                    @test read(group["$(prefix)_sem"]) ≈ std(samples) / sqrt(2)
                end
                spatial, log_spatial = read(group["sample_C"]), read(group["sample_logC"])
                @test read(group["average"]) ≈ vec(mean(spatial; dims=2))
                @test read(group["sem"]) ≈ vec(std(spatial; dims=2)) / sqrt(2)
                @test read(group["mean_log"]) ≈ vec(mean(log_spatial; dims=2))
                @test read(group["log_sem"]) ≈ vec(std(log_spatial; dims=2)) / sqrt(2)
                @test read(group["typical"]) ≈ exp.(read(group["mean_log"]))
                @test read(group["typical_lower"]) ≈ exp.(read(group["mean_log"]) .- read(group["log_sem"]))
                @test read(group["typical_upper"]) ≈ exp.(read(group["mean_log"]) .+ read(group["log_sem"]))
                times = read(group["times"])
                @test times == collect(0.0:0.2:10.0)
                for suffix in ("real", "imag")
                    samples = read(group["sample_C_$suffix"])
                    @test size(samples) == (length(times), 2)
                    @test all(isfinite, samples)
                    @test read(group["average_$suffix"]) ≈ vec(mean(samples; dims=2))
                    @test read(group["sem_$suffix"]) ≈ vec(std(samples; dims=2)) / sqrt(2)
                end
                samples = read(group["sample_C_real"]) + 1im * read(group["sample_C_imag"])
                expected = disorder_ensemble(16, 1.0; times, nsamples=2,
                    seed=read(attributes(group)["seed"]), boundary)
                @test samples ≈ expected.sample_Ct
                @test gaps ≈ expected.gaps
                @test spatial ≈ expected.sample_C
                @test log_spatial ≈ expected.sample_logC
                @test samples[1, :] ≈ ones(2) atol=1e-12
            end
            @test_throws ArgumentError main([mode, "2", output])
        end
        single = joinpath(directory, "single.h5")
        main(["demo", "1", single])
        h5open(single, "r") do file
            group = file["L16/h1.0"]
            for name in ("gap_sem", "loggap_sem")
                @test isnan(read(group[name]))
            end
            for name in ("sem", "log_sem", "sem_real", "sem_imag", "typical_lower", "typical_upper")
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
