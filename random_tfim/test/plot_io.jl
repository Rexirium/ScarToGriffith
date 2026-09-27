using Test
include(joinpath(@__DIR__, "..", "plot_results.jl"))

@testset "Plot reader: parameter arrays and legacy files" begin
    mktempdir() do directory
        input = joinpath(directory, "ensemble.h5")
        sizes, fields = [16, 32], [1.0, 1.3]
        h5open(input, "w") do f
            HDF5.attributes(f)["complete"] = true
            HDF5.attributes(f)["boundary"] = "periodic"
            for L in sizes, h0 in fields
                g = create_group(f, "L$L/h$h0")
                C = repeat(exp.(-h0 .* collect(0:L÷2)), 1, 2)
                g["sample_C"], g["sample_logC"] = C, log.(C)
                g["average"] = vec(mean(C; dims=2))
                g["sem"] = vec(std(C; dims=2)) / sqrt(2)
                g["mean_log"] = vec(mean(log.(C); dims=2))
                g["log_sem"] = vec(std(log.(C); dims=2)) / sqrt(2)
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
    end
end
