using Test
include(joinpath(@__DIR__, "..", "plot_results.jl"))

@testset "Plot transformations" begin
    times = [0.1, 1.0, 10.0, 100.0]
    @test autocorrelation_log_slope(times, 3 .* times .^ (-0.7)) ≈ 0.7
    @test autocorrelation_log_slope(times, [1, -1, 1, -1] .* 3 .* times .^ (-0.7)) ≈ 0.7
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
    @test_throws ErrorException fit_time_collapse(ones(6), samples)
    @test_throws ErrorException fit_time_collapse(times, [zeros(10) for t in times])

    mktempdir() do directory
        input = joinpath(directory, "samples.h5")
        h5open(input, "w") do f
            g = create_group(f, "L128/h2.0")
            g["imaginary_time"] = [0.0; times]
            C = vcat(ones(1, length(base)), reduce(hcat, exp.(-y) for y in samples)')
            # 负样本取绝对值；零值不能取对数，|C| > 1 对应负的 -ln|C|。
            C[2, 1:3] = [-0.1, 0.0, 1.1]
            g["sample_Ct"] = C
        end
        panels = read_autocorrelation_distributions((; input), [2.0], 128)
        @test only(panels).times == times
        @test count(!isfinite, only(panels).samples[1]) == 1
        @test only(panels).samples[1][1] ≈ -log(0.1)
        @test only(panels).samples[1][3] ≈ -log(1.1)
        @test only(panels).samples[2] ≈ samples[2]
    end
end
