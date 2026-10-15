using Test
include(joinpath(@__DIR__, "..", "plot_imaginary_time.jl"))

@testset "Imaginary-time field scan" begin
    mktempdir() do directory
        input = joinpath(directory, "field_scan.h5")
        sizes, fields = [32, 256], [0.5, 1.0, 2.0]
        t = exp.(range(log(0.1), log(10.0); length=31))
        h5open(input, "w") do f
            HDF5.attributes(f)["offset"] = true
            f["parameters/fields"] = fields
            for L in sizes, h0 in fields
                L == 256 && h0 == 0.5 && continue
                g = create_group(f, "L$(L)/h$(h0)")
                predictor = h0 == 1.0 ? log.(t) : t
                y = (h0 == 1.0 ? 1e-10 : 0.5) .* exp.(-(L/64) .* predictor)
                g["imaginary_time"] = t
                g["autocorrelation_mean"] = y
                g["autocorrelation_sem"] = fill(NaN, length(y))
            end
        end
        scan = plot_imaginary_field_scan(input, sizes; tmin=1.0)
        @test scan.fields == fields
        @test scan.comparison_size == 256
        @test length(scan.records) == 5
        for r in scan.records
            @test r.result.best.model == (r.h0 == 1.0 ? :power : :exponential)
            @test r.result.best.rate ≈ r.L/64
            @test r.result.best.reduced_chi_square < 1e-20
            @test r.result.tmin >= 1.0
        end
        @test length(scan.crossings) == 1
        @test 1 < only(scan.crossings).h0 < 2
        for k in 1:4
            ax = content(scan.fig[panel_position(k, 2)...])
            @test ax.xscale[] === log10
            @test ax.limits[][1] == (0.1, 10.0)
            @test ax.yscale[] === (k in (1, 2, 4) ? log10 : identity)
        end
        full_scan = plot_imaginary_field_scan(input, sizes; tmin=0.0, tmax=Inf)
        h5open(input, "r") do f
            for r in full_scan.records
                g = f["L$(r.L)/h$(r.h0)"]
                @test r.result.n == length(t)
                @test r.result.power.rate ≈ autocorrelation_log_slope(t, read(g["autocorrelation_mean"]))
            end
        end
    end
end

@testset "Uncertainty masks" begin
    values = uncertainty_values([1.0, 0.1, 0.0, NaN], [0.2, 0.2, 0.1, 0.1]; positive=true)
    @test values.center[1:2] == [1.0, 0.1]
    @test all(isnan, values.center[3:4])
    @test values.lower[1] == 0.8
    @test all(isnan, values.lower[2:4]) && all(isnan, values.upper[2:4])
    values = uncertainty_values([-2.0, -3.0], [0.5, NaN])
    @test values.center == [-2.0, -3.0]
    @test values.lower[1] == -2.5 && isnan(values.lower[2])
end
