using Test
include(joinpath(@__DIR__, "..", "plot_results.jl"))

@testset "Plot selection, missing probability, and uncertainty masks" begin
    fields = [0.1, 0.501, 1.0, 1.995, 5.012, 10.0]
    @test representative_fields(reverse(fields), [0.1, 0.5, 1, 2, 5, 10]) == fields
    @test representative_fields(filter(>=(1), fields), [1, 2, 5, 10]) == fields[3:end]
    @test_throws ErrorException representative_fields([1.0], [1, 2])
    @test gap_density([-2.0, -1.0, 0.0, NaN], [-2.0, -1.0, 0.0]) == [0.25, 0.5]
    values = uncertainty_values([1.0, 0.1, 0.0, NaN], [0.2, 0.2, 0.1, 0.1]; positive=true)
    @test values.center[1:2] == [1.0, 0.1]
    @test all(isnan, values.center[3:4])
    @test values.lower[1] == 0.8
    @test all(isnan, values.lower[2:4]) && all(isnan, values.upper[2:4])
    values = uncertainty_values([-2.0, -3.0], [0.5, NaN])
    @test values.center == [-2.0, -3.0]
    @test values.lower[1] == -2.5 && isnan(values.lower[2])
end

@testset "Plot reader: parameter arrays and legacy files" begin
    mktempdir() do directory
        input = joinpath(directory, "ensemble.h5")
        sizes, fields = [16, 32], [1.0, 1.3]
        h5open(input, "w") do f
            HDF5.attributes(f)["complete"] = true
            HDF5.attributes(f)["boundary"] = "periodic"
            for L in sizes, h0 in fields
                g = create_group(f, "L$L/h$h0")
                average = exp.(-h0 .* (0:L÷2))
                logaverage = log.(average)
                # Two identical samples: known means and exactly zero SEM.
                g["sample_C"], g["sample_logC"] = repeat(average, 1, 2), repeat(logaverage, 1, 2)
                g["average"], g["mean_log"] = average, logaverage
                g["sem"], g["log_sem"] = zeros(length(average)), zeros(length(average))
                g["loggaps"], g["resolved"] = [-1.0, -2.0], [true, true]
            end
        end
        legacy = read_plot_data(input)
        @test legacy.sizes == sizes && legacy.fields == fields
        h5open(input, "r+") do f
            p = create_group(f, "parameters")
            p["sizes"], p["fields"] = reverse(sizes), reverse(fields)
            for L in sizes, h0 in fields, key in ("sample_C", "sample_logC")
                HDF5.delete_object(f["L$L/h$h0"], key)
            end
        end
        current = read_plot_data(input)
        @test current.sizes == reverse(sizes) && current.fields == reverse(fields)
        @test isequal(current.records, reverse(legacy.records))
        @test all(d.n == 2 for d in current.records)
        # New descriptive names must return the same records as both old layouts.
        h5open(input, "r+") do f
            for L in sizes, h0 in fields
                g = f["L$L/h$h0"]
                for (old, new) in (("loggaps", "log_gap_samples"), ("resolved", "gap_resolved"),
                    ("average", "correlation_mean"), ("sem", "correlation_sem"),
                    ("mean_log", "log_correlation_mean"), ("log_sem", "log_correlation_sem"))
                    g[new] = read(g[old])
                    HDF5.delete_object(g, old)
                end
            end
        end
        renamed = read_plot_data(input)
        @test isequal(renamed, current)
        # Reconstruct logs from raw samples, retaining unresolved samples as NaN.
        h5open(input, "r+") do f
            for L in sizes, h0 in fields
                g = f["L$L/h$h0"]
                g["gap_samples"] = [exp(-1.0), -eps(Float64)]
                HDF5.delete_object(g, "log_gap_samples")
                write(g["gap_resolved"], [true, false])
            end
        end
        raw = read_plot_data(input)
        for d in raw.records
            @test d.gaps[1] == -1.0
            @test isnan(d.gaps[2])
            @test d.resolved == [true, false]
        end
        # Most scan points now contain statistics only.
        h5open(input, "r+") do f
            HDF5.attributes(f)["nsamples"] = 2
            for L in sizes, key in ("gap_samples", "gap_resolved")
                HDF5.delete_object(f["L$L/h1.3"], key)
            end
        end
        selective = read_plot_data(input)
        @test length(selective.records) == length(current.records)
        for d in selective.records
            @test d.n == 2
            @test isempty(d.gaps) == isempty(d.resolved) == (d.h0 == 1.3)
        end
    end
end
