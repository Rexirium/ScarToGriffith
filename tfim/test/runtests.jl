using Test, LinearAlgebra
include(joinpath(@__DIR__, "..", "clean_tfim.jl"))
BLAS.set_num_threads(1)

@testset "Clean open chain with offset" begin
    times = [0.0, 0.1, 1.0, 10.0, 100.0]
    for L in (3, 16), h in (0.2, 0.6, 1.0, 1.4, 5.0)
        C = clean_autocorrelation(h, times; L)
        @test 0 < C[1] < 1
        @test all(-1e-12 .<= C .<= 1+1e-12)
        @test all(diff(C) .<= 1e-12)
        @test C ≈ autocorrelation(ones(L-1), fill(h, L), times;
            j=cld(L,2), time_domain=:imaginary, offset=true) atol=2e-11
        @test C ≈ autocorrelation(ones(L-1), fill(h, L), times;
            j=L+1-cld(L,2), offset=true, rcond_tol=1.0) atol=2e-11
    end
    # Two spins with one OBC bond: subtract the (E-1) transition entirely.
    for h in (0.3, 1.0, 2.0)
        E = sqrt(1 + 4h^2)
        exact = (1-1/E)/2 .* exp.(-(E+1) .* times)
        @test clean_autocorrelation(h, times; L=2) ≈ exact atol=1e-14
    end
    @test abs(only(clean_autocorrelation(0.2, [100.0]; L=128))) < 1e-12
    @test isempty(clean_autocorrelation(1.0, Float64[]))
    @test_throws ArgumentError clean_autocorrelation(1.0, times; L=1)
    @test_throws ArgumentError clean_autocorrelation(-1.0, times)
    @test_throws ArgumentError clean_autocorrelation(1.0, [-1.0])
end
