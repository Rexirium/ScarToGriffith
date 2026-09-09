using Test, Random, Statistics, SpinMonteCarlo
include("observables.jl")

# Independent square-torus Hamiltonian, including the L=2 parallel bonds.
function exact_observables(L, T, Js)
    n = L^2
    z = e1 = e2 = m2 = 0.0
    local_chi = zeros(L, L)
    for bits in 0:(2^n - 1)
        s = reshape([iszero(bits & (1 << (i - 1))) ? -1 : 1 for i in 1:n], L, L)
        e = 0.0
        for y in 1:L, x in 1:L
            i = x + L * (y - 1)
            e -= s[x, y] * (Js[2i-1] * s[mod1(x+1,L), y] +
                            Js[2i] * s[x, mod1(y+1,L)])
        end
        w = exp(-e / T)
        m = sum(s)
        z += w
        e1 += w * e
        e2 += w * e^2
        m2 += w * m^2
        local_chi .+= (w * m / T) .* s
    end
    return ((e2/z - (e1/z)^2)/T^2, m2/z/T, local_chi/z)
end

@testset "random-bond responses" begin
    L, T, r = 3, 2.4, 0.3
    disorder = [[i in (1, 2, 5, 9, 10, 14) ? 1.0 : r for i in 1:2L^2], ones(2L^2)]
    original = deepcopy(disorder)
    for update in (:sw, :local)
        out = random_bond_observables(L, T, disorder;
            mcs=65536, thermalization=4096, binsize=256, seed=72, update, details=true)
        @test disorder == original
        @test length(out.heat_capacity) == length(out.susceptibility) == 2
        @test size.(out.local_susceptibility) == [(L,L), (L,L)]
        for a in eachindex(disorder)
            c, chi, loc = exact_observables(L, T, disorder[a])
            @test abs(out.heat_capacity[a] - c) < 6out.errors.heat_capacity[a] + 0.02
            @test abs(out.susceptibility[a] - chi) < 6out.errors.susceptibility[a] + 0.02
            @test all(abs.(out.local_susceptibility[a] - loc) .<
                6 .* out.errors.local_susceptibility[a] .+ 0.02)
            @test sum(out.local_susceptibility[a]) ≈ out.susceptibility[a]
        end
    end
    kwargs = (mcs=8192, thermalization=256, binsize=64, seed=4)
    a = random_bond_observables(3, 2.0, [zeros(18)]; kwargs...)
    @test a == random_bond_observables(3, 2.0, [zeros(18)]; kwargs...)
    @test a[1] == [0.0]
    @test a[2][1] ≈ 9/2 atol=0.2
    @test all(abs.(a[3][1] .- 0.5) .< 0.1)
    free_local = random_bond_observables(3, 2.0, [zeros(18)];
        kwargs..., update=:local)
    @test free_local[2][1] ≈ 9/2 atol=0.5
    @test all(abs.(free_local[3][1] .- 0.5) .< 0.2)
    @test random_bond_observables(3, 2.0, Vector{Float64}[]) ==
        (Float64[], Float64[], Matrix{Float64}[])
    @test_throws ArgumentError random_bond_observables(3, 0.0, disorder)
    @test_throws ArgumentError random_bond_observables(3, 2.0, [ones(17)])
    @test_throws ArgumentError random_bond_observables(3, 2.0, [fill(-0.3,18)])
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; mcs=65)
end
