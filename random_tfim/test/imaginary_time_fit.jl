using Test
using LinearAlgebra
include(joinpath(@__DIR__, "..", "fit_autocorrelation.jl"))

@testset "Model-score crossings on logarithmic field axis" begin
    crossings = model_fit_crossings([1., 10., 100.], [0., 2., 0.], ones(3))
    @test [c.h0 for c in crossings] ≈ [sqrt(10), sqrt(1000)]
    @test all(c.reduced_chi_square == 1 for c in crossings)
    @test [(c.left, c.right) for c in crossings] == [(1., 10.), (10., 100.)]
    @test only(model_fit_crossings([1., 2., 3.], [0., 1., 2.], ones(3))).h0 == 2
    @test isempty(model_fit_crossings([1., 2.], [0., 0.], [1., 1.]))
    @test isempty(model_fit_crossings([1., 2., 3.], [0., NaN, 2.], ones(3)))
end

@testset "Shared unweighted log-space fits" begin
    t = exp.(range(log(0.1), log(20.0); length=81))
    for (model, A, rate) in ((:power, 0.3, 0.65), (:exponential, 0.8, 0.4))
        y = A .* exp.(-rate .* (model == :power ? log.(t) : t))
        result = fit_autocorrelation(t, y)
        @test result.best.model == model
        @test result.best.A ≈ A rtol=1e-6
        @test result.best.rate ≈ rate rtol=1e-6
        @test result.best.chi_square < 1e-12
        @test result.best.chi_square ≈ sum(abs2, log.(y) .- log.(decay_prediction(result.best, t))) atol=1e-12
        @test result.power.rate ≈ autocorrelation_log_slope(t, y)
        @test result.n == length(t)
        for fit in (result.power, result.exponential)
            @test fit.reduced_chi_square ≈ fit.chi_square / (result.n - 2)
        end
        @test result.best.reduced_chi_square == min(result.power.reduced_chi_square,
            result.exponential.reduced_chi_square)
    end
    # Nonzero residuals distinguish log-space fitting from the former C-space fit.
    # Check against an independent unweighted design-matrix least-squares solve.
    for model in (:power, :exponential)
        predictor = model == :power ? log.(t) : t
        logC = -0.4 .- 0.7 .* predictor .+ 0.12 .* sin.(eachindex(t))
        y = exp.(logC)
        result = fit_autocorrelation(t, y)
        fit = model == :power ? result.power : result.exponential
        design = hcat(ones(length(t)), -predictor)
        expected = design \ logC
        residual = logC .- design * expected
        @test fit.logA ≈ expected[1] atol=1e-12
        @test fit.rate ≈ expected[2] atol=1e-12
        @test fit.chi_square ≈ sum(abs2, residual) rtol=1e-12
        @test fit.reduced_chi_square ≈ sum(abs2, residual)/(length(t)-2)
        original_chi_square = sum(abs2, y .- decay_prediction(fit, t))
        @test !isapprox(fit.chi_square, original_chi_square; rtol=1e-3)
    end
    result = fit_autocorrelation([0., 1., 2., 3., 4., 5., 6., 7.],
        [1., 0.5, 0.3, 0.2, 1e-10, NaN, 0.1, 0.09])
    @test result.keep == [false, true, true, true, true, false, true, true]
    @test result.n == 6
    # Positive tails below the plot floor must enter both scripts' regressions.
    tail = fit_autocorrelation(t, 1e-10 .* t.^(-0.7))
    @test tail.n == length(t)
    @test tail.power.rate ≈ autocorrelation_log_slope(t, 1e-10 .* t.^(-0.7))
    plateau = fit_autocorrelation(t, fill(0.5, length(t)))
    @test plateau.best.rate ≈ 0 atol=1e-14
    @test plateau.best.A ≈ 0.5
    @test_throws ErrorException fit_autocorrelation([1., 2.], [0.5, 0.4])
end
