include(joinpath(@__DIR__, "..", "run.jl"))
using Test

@testset "Demo/full: all observables and disorder SEM" begin
    mktempdir() do directory
        for mode in ("demo", "full"), boundary in (:periodic, :open)
            output = joinpath(directory, "$(mode)_$(boundary).h5")
            redirect_stdout(devnull) do
                main([mode, "2", output, string(boundary)])
            end
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
                        group = file["L$L/$h"]
                        @test Set(keys(group)) == Set([
                            "gap_samples", "gap_resolved", "log_gap_samples",
                            "gap_mean", "gap_sem", "log_gap_mean", "log_gap_sem",
                            "correlation_mean", "correlation_sem",
                            "log_correlation_mean", "log_correlation_sem",
                            "imaginary_time", "autocorrelation_mean", "autocorrelation_sem"])
                        @test Set(keys(attributes(group))) == Set(["j", "seed", "seconds"])
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
                @test times == collect(0.0:0.2:20.0)
                expected = disorder_ensemble(16, 1.0; keep_samples=true, times, nsamples=2,
                    seed=read(attributes(group)["seed"]), boundary)
                for (prefix, samples) in (("correlation", expected.sample_C),
                        ("log_correlation", expected.sample_logC), ("autocorrelation", expected.sample_Ct))
                    average, sem = read(group["$(prefix)_mean"]), read(group["$(prefix)_sem"])
                    @test average isa Vector{Float64} && sem isa Vector{Float64}
                    @test average ≈ vec(mean(samples; dims=2)) nans=true
                    @test sem ≈ vec(std(samples; dims=2)) / sqrt(2) nans=true
                end
                @test size(expected.sample_Ct) == (length(times), 2)
                @test all(isfinite, expected.sample_Ct)
                @test gaps ≈ expected.gaps
                @test read(group["autocorrelation_mean"])[1] ≈ 1 atol=1e-12
            end
            @test_throws ArgumentError main([mode, "2", output])
        end
        single = joinpath(directory, "single.h5")
        redirect_stdout(devnull) do
            main(["demo", "1", single])
        end
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
