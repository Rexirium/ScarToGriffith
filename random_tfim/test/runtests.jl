using Test, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
using .RandomTFIM

# Independent oracle: Pauli matrices in the sigma-z product basis.
function spin_oracle(J, h)
    L = length(J)
    H = zeros(2^L, 2^L)
    for s in 0:2^L-1, i in 1:L
        j = mod1(i+1, L)
        zi, zj = 1-2*((s >> (i-1)) & 1), 1-2*((s >> (j-1)) & 1)
        H[s+1, s+1] -= J[i] * zi * zj
        H[xor(s, 1 << (i-1))+1, s+1] -= h[i]
    end
    F = eigen(Symmetric(H))
    psi = F.vectors[:, 1]
    @test norm(H * psi - F.values[1] * psi) < 1e-10
    C = [sum(abs2(psi[s+1]) * (1-2*((s >> (i-1)) & 1)) *
        (1-2*((s >> (j-1)) & 1)) for s in 0:2^L-1) for i in 1:L, j in 1:L]
    return (; E0=F.values[1], E1=F.values[2], C)
end

BLAS.set_num_threads(1)
@testset "Free fermions vs spin ED" begin
    rng = Xoshiro(734)
    for L in (2, 4, 6, 8), h0 in (0.4, 1.0, 3.0), random in (false, true)
        J, h = random ? sample_disorder(rng, L, h0) : (ones(L), fill(h0, L))
        exact = spin_oracle(J, h)
        result = @inferred ground_state(J, h)
        gap = @inferred energy_gap(J, h)
        @test result.gap ≈ exact.E1-exact.E0 atol=1e-11
        @test gap.gap ≈ result.gap atol=1e-11
        @test result.G * result.G' ≈ I atol=1e-12
        pair = @inferred correlations(result.G)
        for r in 0:L÷2, i in 1:L
            @test pair.C[i, r+1] ≈ exact.C[i, mod1(i+r, L)] atol=2e-10
        end
        @test exp.(pair.logC) ≈ pair.C atol=1e-12
    end
end

@testset "Limits and sampling" begin
    J, h = zeros(6), collect(0.5:0.5:3.0)
    state = ground_state(J, h)
    @test state.gap ≈ 2minimum(h)
    @test correlations(state.G).C[:, 2:end] == zeros(6, 3)
    @test !energy_gap(ones(32), fill(0.1, 32)).resolved
    a = @inferred disorder_ensemble(8, 1.0; nsamples=5)
    b = disorder_ensemble(8, 1.0; nsamples=5)
    @test a.gaps == b.gaps
    @test a.sample_C == b.sample_C
    @test a.sample_logC == b.sample_logC
    @test all(exp.(mean(a.sample_logC; dims=2)) .<= mean(a.sample_C; dims=2) .+ 1e-14)
    pairs = disorder_ensemble(4, 1.0; nsamples=3, keep_pairs=true)
    @test size(pairs.pair_logC) == (4, 3, 3)
    @test dropdims(mean(pairs.pair_logC; dims=1); dims=1) ≈ pairs.sample_logC
    @test all(disorder_ensemble(4, 3.0; nsamples=3, rmax=0).resolved)
    @test_throws ArgumentError energy_gap(ones(3), ones(3))
    @test_throws ArgumentError energy_gap(ones(4), zeros(4))
    @test_throws ArgumentError sample_disorder(Xoshiro(1), 4, 1.0; distribution=:bad)
    for h0 in (0.7, 1.0, 3.0)
        J, h = sample_disorder(Xoshiro(9), 6, h0; distribution=:bimodal)
        state = ground_state(J, h)
        oracle = spin_oracle(J, h)
        @test state.gap ≈ oracle.E1-oracle.E0 atol=1e-10
    end
end
