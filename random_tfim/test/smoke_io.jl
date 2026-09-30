include(joinpath(@__DIR__, "..", "run.jl"))
using Test

@testset "Demo/full: all observables and disorder SEM" begin
    mktempdir() do directory
        for (mode, boundary, field_distribution) in (("demo", :periodic, :uniform),
                ("demo", :open, :fixed), ("full", :periodic, :fixed), ("full", :open, :uniform))
            output = joinpath(directory, "$(mode)_$(boundary).h5")
            output = redirect_stdout(devnull) do
                main([mode, "2", output, string(boundary), string(field_distribution)])
            end
            h5open(output, "r") do file
                @test Set(keys(attributes(file))) == Set(["complete", "nsamples", "distribution", "field_distribution", "boundary", (field_distribution == :fixed ? ["fixed_field_divisor"] : String[])...])
                @test read(attributes(file)["field_distribution"]) == string(field_distribution)
                if field_distribution == :fixed
                    @test read(attributes(file)["fixed_field_divisor"]) == exp(1)
                end
                @test read(attributes(file)["complete"])
                @test read(attributes(file)["boundary"]) == string(boundary)
                sizes = mode == "demo" ? (16, 32) : (16, 32, 64, 128)
                fields = 10 .^ range(-1, 1, 101)
                @test Set(keys(file)) == Set(["parameters"; ["L$L" for L in sizes]])
                @test Set(keys(file["parameters"])) == Set(["sizes", "fields", "sample_fields"])
                @test read(file["parameters/sizes"]) == collect(sizes)
                @test read(file["parameters/fields"]) == fields
                selected = select_sample_fields(fields)
                @test read(file["parameters/sample_fields"]) == selected
                @test read(attributes(file)["nsamples"]) == 2
                for L in sizes
                    @test Set(keys(file["L$L"])) == Set("h$h" for h in fields)
                    for h in keys(file["L$L"])
                        group = file["L$L/$h"]
                        expected_keys = Set([
                            "gap_mean", "gap_sem", "log_gap_mean", "log_gap_sem",
                            "correlation_mean", "correlation_sem",
                            "log_correlation_mean", "log_correlation_sem",
                            "imaginary_time", "autocorrelation_mean", "autocorrelation_sem",
                            "log_autocorrelation_mean", "log_autocorrelation_sem",
                            "real_time", "real_autocorrelation_mean", "real_autocorrelation_sem",
                            "real_log_autocorrelation_mean", "real_log_autocorrelation_sem"])
                        if parse(Float64, h[2:end]) in selected
                            union!(expected_keys, ("gap_samples", "gap_resolved",
                                "sample_C", "sample_Ct", "sample_real_Ct"))
                        end
                        @test Set(keys(group)) == expected_keys
                        @test Set(keys(attributes(group))) == Set(["j", "seed"])
                    end
                end
                group = file["L16/h1.0"]
                @test read(attributes(group)["j"]) == 8
                gaps, resolved = read(group["gap_samples"]), read(group["gap_resolved"])
                @test !haskey(group, "log_gap_samples")
                logs = map((gap, ok) -> ok ? log(gap) : NaN, gaps, resolved)
                for (prefix, samples) in (("gap", gaps), ("log_gap", logs))
                    @test read(group["$(prefix)_mean"]) ≈ mean(samples)
                    @test read(group["$(prefix)_sem"]) ≈ std(samples) / sqrt(2)
                end
                times = read(group["imaginary_time"])
                @test times == 10 .^ range(-1, 3, 101)
                expected = disorder_ensemble(16, 1.0, times; keep_samples=true, nsamples=2,
                    seed=read(attributes(group)["seed"]), boundary, field_distribution)
                for (prefix, avg, err) in (("correlation", :C_mean, :C_sem),
                        ("log_correlation", :logC_mean, :logC_sem),
                        ("autocorrelation", :Ct_mean, :Ct_sem),
                        ("log_autocorrelation", :logCt_mean, :logCt_sem),
                        ("real_autocorrelation", :real_Ct_mean, :real_Ct_sem),
                        ("real_log_autocorrelation", :real_logCt_mean, :real_logCt_sem))
                    average, sem = read(group["$(prefix)_mean"]), read(group["$(prefix)_sem"])
                    @test average isa Vector{Float64} && sem isa Vector{Float64}
                    @test average ≈ getproperty(expected, avg) nans=true
                    @test sem ≈ getproperty(expected, err) nans=true
                end
                @test size(expected.sample_Ct) == (length(times), 2)
                @test all(isfinite, expected.sample_Ct)
                @test gaps ≈ expected.gaps
                @test 0 < read(group["autocorrelation_mean"])[1] <= 1
            end
        end
        single = joinpath(directory, "single.h5")
        single = redirect_stdout(devnull) do
            main(["demo", "1", single])
        end
        h5open(single, "r") do file
            group = file["L16/h1.0"]
            for name in ("gap_sem", "log_gap_sem")
                @test isnan(read(group[name]))
            end
            for name in ("correlation_sem", "log_correlation_sem", "autocorrelation_sem", "log_autocorrelation_sem",
                    "real_autocorrelation_sem", "real_log_autocorrelation_sem")
                @test all(isnan, read(group[name]))
            end
        end
    end
    @test isnan(RandomTFIM.mean_sem([1.0, NaN]).average)
    @test isnan(RandomTFIM.mean_sem([1.0, NaN]).sem)
    for mode in ("gap", "correlation", "autocorrelation", "bad")
        @test_throws ArgumentError main([mode])
    end
    @test_throws ArgumentError main(["demo", "1", "unused.h5", "bad"])
    @test_throws ArgumentError main(["demo", "0"])
end
