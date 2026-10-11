using Test
include(joinpath(@__DIR__, "..", "fit_autocorrelation.jl"))
include(joinpath(@__DIR__, "..", "fit_autocorrelation_offset.jl"))

@testset "Offset decay fits" begin
    t = exp.(range(log(0.1), log(100.0); length=101))
    for model in (:power, :exponential), B in (0.0, 0.03, 0.8), amplitude in (0.7, 1e-20)
        y = amplitude .* ((model == :power ? t.^(-0.65) : exp.(-0.4.*t)) .+ B)
        result = fit_autocorrelation_offset(t, y)
        fit = model == :power ? result.power : result.exponential
        @test result.best.model == model
        @test fit.A ≈ amplitude rtol=1e-5
        @test fit.B/amplitude ≈ B atol=1e-6
        @test fit.rate ≈ (model == :power ? 0.65 : 0.4) rtol=1e-5
        @test fit.chi_square < 1e-12
        @test fit.reduced_chi_square ≈ sum(abs2, log.(y) .- log.(offset_prediction(fit, t)))/(length(t)-3) atol=1e-14
        @test fit.converged
    end
    plateau = fit_autocorrelation_offset(t, fill(0.5, length(t)))
    @test offset_prediction(plateau.best, t) ≈ fill(0.5, length(t))
    @test plateau.best.rate ≈ 0 atol=1e-14
    result = fit_autocorrelation_offset([0., 1., 2., 3., 4., 5., 6., 7.],
        [1., 0.5, 0.3, 0.2, 1e-10, NaN, 0.1, 0.09]; tmax=6)
    @test result.keep == [false, true, true, true, true, false, true, false]
    @test result.n == 5
    @test_throws ErrorException fit_autocorrelation_offset([1., 2., 3.], [0.5, 0.4, 0.3])
    @test_throws ErrorException fit_autocorrelation_offset(ones(4), ones(4))
    @test_throws ErrorException fit_autocorrelation_offset(t, ones(2))
    for model in (:power, :exponential)
        y = (0.7 .* (model == :power ? t.^(-0.65) : exp.(-0.4.*t)) .+ 0.03) .*
            exp.(0.02 .* sin.(eachindex(t)))
        result = fit_autocorrelation_offset(t, y)
        @test result.best.model == model
        for fit in (result.power, result.exponential)
            base = fit_decay(t, y, fit.model)
            rate = fit.model == :power ? min(base.rate, 1.0) : base.rate
            predictor = fit.model == :power ? log.(t) : t
            logA = mean(log.(y) .+ rate .* predictor)
            @test fit.chi_square <= sum(abs2, log.(y) .- (logA .- rate .* predictor)) + 1e-12
            @test fit.B >= 0 && fit.rate >= 0
            @test fit.reduced_chi_square ≈ sum(abs2, log.(y) .- log.(offset_prediction(fit, t)))/(length(t)-3)
        end
    end
end

@testset "Power-law exponent bound" begin
    t = exp.(range(log(0.1), log(100.0); length=101))
    for B in (0.0, 0.03), exponent in (1.0, 1.7)
        y = 0.7 .* t.^(-exponent) .+ B
        fit = fit_autocorrelation_offset(t, y).power
        @test -1 <= fit.alpha <= 0
        @test fit.rate ≈ 1.0 atol=1e-6
        @test fit.converged
        @test fit.chi_square ≈ sum(abs2, log.(y) .- log.(offset_prediction(fit, t))) atol=1e-12
        if B == 0
            @test fit.B == 0
            @test fit.logA ≈ mean(log.(y) .+ log.(t)) atol=1e-10
        end
    end
    fit = fit_autocorrelation_offset(t, 0.7 .* exp.(-1.7 .* t) .+ 0.03).exponential
    @test fit.rate ≈ 1.7 rtol=1e-5
end
