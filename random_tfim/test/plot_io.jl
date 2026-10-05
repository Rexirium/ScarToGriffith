using Test
include(joinpath(@__DIR__, "..", "plot_results.jl"))

@testset "Imaginary-time axis floor" begin
    mktempdir() do directory
        input = joinpath(directory, "axis_floor.h5")
        sizes = [16, 32, 64, 128]
        h5open(input, "w") do f
            for (L, tail) in zip(sizes, [1e-7, 1e-9, 1e-8, 0.0])
                g = create_group(f, "L$L/h1.0")
                # A below-floor point at zero time must not affect the axis.
                g["imaginary_time"] = g["real_time"] = [0.0, 1.0, 10.0]
                g["autocorrelation_mean"] = g["real_autocorrelation_mean"] = [1e-10, 1.0, tail]
                g["autocorrelation_sem"] = g["real_autocorrelation_sem"] = zeros(3)
            end
        end
        data = (; input, field_distribution="uniform")
        for real_time in (false, true)
            fig = plot_autocorrelation(data, [1.0], sizes; real_time)
            for (j, L) in enumerate(sizes)
                row, col = panel_position(j, 2)
                ax = content(fig[row, col])
                expected = !real_time && L == 32 ? (nothing, (1e-8, nothing)) : (nothing, nothing)
                @test ax.limits[] == expected
            end
        end
    end
end

@testset "Missing probability and uncertainty masks" begin
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

@testset "Plot reader: current HDF5 format" begin
    mktempdir() do directory
        input = joinpath(directory, "ensemble.h5")
        sizes, fields = [32, 16], [1.3, 1.0]
        for distribution in ("uniform", "fixed")
            h5open(input, "w") do f
                attrs = HDF5.attributes(f)
                attrs["complete"] = true
                attrs["boundary"] = "periodic"
                attrs["field_distribution"] = distribution
                attrs["nsamples"] = 2
                distribution == "fixed" && (attrs["fixed_field_divisor"] = exp(1))
                p = create_group(f, "parameters")
                p["sizes"], p["fields"], p["sample_fields"] = sizes, fields, [1.0]
                for L in sizes, h0 in fields
                    g = create_group(f, "L$L/h$h0")
                    avg = exp.(-h0 .* (0:div(L, 2)))
                    g["correlation_mean"], g["log_correlation_mean"] = avg, log.(avg)
                    g["correlation_sem"], g["log_correlation_sem"] = zeros(length(avg)), zeros(length(avg))
                    if h0 == 1.0
                        g["gap_samples"] = [exp(-1.0), -eps(Float64)]
                        g["gap_resolved"] = [true, false]
                    end
                end
            end
            data = read_plot_data(input)
            @test data.sizes == sizes && data.fields == fields
            @test data.sample_fields == [1.0]
            @test data.field_distribution == distribution
            @test field_distribution_label(data) == (distribution == "fixed" ? "h = h₀/e" : "h ∼ U(0, h₀)")
            @test data.n == 2
            @test length(data.records) == 2
            for d in data.records
                @test d.h0 == 1.0
                @test d.avg ≈ exp.(-d.h0 .* (0:div(d.L, 2)))
                @test d.gaps[1] == -1.0
                @test isnan(d.gaps[2])
                @test d.unresolved == 1
            end
        end
        h5open(input, "r+") do f
            HDF5.delete_object(f["parameters"], "sample_fields")
        end
        @test_throws KeyError read_plot_data(input)
    end
end

@testset "Plot transformations" begin
    times = [0.1, 1.0, 10.0, 100.0]
    @test autocorrelation_log_slope(times, 3 .* times .^ (-0.7)) ≈ 0.7
    @test autocorrelation_log_slope(times, [3 * times[1]^(-0.7), -100.0, 3 * times[3]^(-0.7), -0.001]) ≈ 0.7
    @test autocorrelation_log_slope([0.0; times; Inf; 5.0],
        [1.0; 3 .* times .^ (-0.7); 1.0; 0.0]) ≈ 0.7
    @test_throws ErrorException autocorrelation_log_slope([1.0, 1.0], [1.0, 2.0])
    @test_throws ErrorException autocorrelation_log_slope([0.0, 1.0], [1.0, 1.0])

    gaps, edges = [-2.8, -1.5, -0.2, NaN], [-3.0, -2.0, -1.0, 0.0]
    scale = sqrt(32)
    density = gap_density(gaps ./ scale, edges ./ scale)
    @test density ≈ scale .* gap_density(gaps, edges)
    @test sum(density .* diff(edges ./ scale)) ≈ 3 / 4
    # Re-bin transformed samples on shared edges, independently of the old bins.
    common_edges = [-1.0, -0.5, 0.0]
    @test gap_density(gaps ./ sqrt(16), common_edges) ≈ [0.5, 1.0]
end

@testset "Time-distribution collapse" begin
    times = [1.0, 3.01995, 10.0, 30.1995, 100.0, 301.995]
    base = collect(range(0.01, 2.0; length=1000))
    samples = [base .* t^0.63 for t in times]
    fit = fit_time_collapse(times, samples)
    @test fit.mu ≈ 0.63 atol=1e-12
    @test fit.rms_after < 1e-12
    @test fit.rms_before > fit.rms_after
    @test fit_time_collapse(times, [copy(base) for t in times]).mu ≈ 0 atol=1e-12
    @test fit_time_collapse(times, [vcat(y, NaN) for y in samples]).mu ≈ 0.63
    @test fit_time_collapse(times, [samples[1:4]; [fill(NaN, 1000), fill(NaN, 1000)]]).mu ≈ 0.63
    @test isnan(fit_time_collapse(times, [fill(NaN, 10) for t in times]).mu)
    @test_throws ErrorException fit_time_collapse(ones(6), samples)
    @test_throws ErrorException fit_time_collapse(times, [zeros(10) for t in times])

    mktempdir() do directory
        input = joinpath(directory, "samples.h5")
        h5open(input, "w") do f
            g = create_group(f, "L128/h2.0")
            g["imaginary_time"] = [0.0; times]
            C = vcat(ones(1, length(base)), reduce(hcat, exp.(-y) for y in samples)')
            # 负样本和零值不能取对数，C > 1 对应负的 -ln C。
            C[2, 1:3] = [-0.1, 0.0, 1.1]
            g["sample_Ct"] = C
        end
        panels = read_autocorrelation_distributions((; input), [2.0], 128)
        @test only(panels).times == times
        @test count(!isfinite, only(panels).samples[1]) == 2
        @test isnan(only(panels).samples[1][1])
        @test only(panels).samples[1][3] ≈ -log(1.1)
        @test only(panels).samples[2] ≈ samples[2]
    end
end
