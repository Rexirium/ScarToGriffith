using Test, Random, Statistics, SpinMonteCarlo
include("observables.jl")

@testset "reused measurements match package estimator" begin
    rng = MersenneTwister(18)
    for L in (2, 3), Js in (zeros(2L^2), rand(rng, 2L^2))
        T = 2.4
        model = Ising(Parameter("Lattice" => "square lattice", "L" => L,
            "Use Indicies as Bond Types" => true, "Seed" => UInt32(18)))
        measurement = Dict{String,Any}()
        local_keys = ["Local Susceptibility $i" for i in 1:L^2]
        for spins in (ones(Int, L^2), -ones(Int, L^2), rand(rng, [-1, 1], L^2))
            model.spins[:] = spins
            expected = simple_estimator(model, T, Js)
            @test rbim_response_estimator!(measurement, model, T, Js; local_keys) === measurement
            @test length(measurement) == length(expected) + L^2
            for (key, value) in expected
                @test measurement[key] ≈ value
            end
            @test [measurement[key] for key in local_keys] ≈ spins .* sum(spins) ./ T
        end
    end
end

@testset "starting-time averaged spin correlation" begin
    spins = Int8[1 1 -1 -1; 1 -1 1 -1]
    @test rbim_time_correlation(spins) ≈ [1.0, -1/3, 0.0, -1.0] atol=1e-12
    @test rbim_time_correlation(ones(Int8, 4, 256)) == ones(101)
    @test rbim_time_correlation(spins; max_corr_time=0) == [1.0]
    @test_throws ArgumentError rbim_time_correlation(spins; max_corr_time=-1)
    @test rbim_time_correlation(reshape(Int8[1, -1], 2, 1)) == [1.0]
    rng = MersenneTwister(81)
    for m in (2, 7, 32, 65, 257)
        spins = rand(rng, Int8[-1, 1], 5, m)
        original = copy(spins)
        expected = [sum(Int(spins[i, k+t]) * Int(spins[i, k])
            for k in 1:m-t, i in 1:5) / (5 * (m-t)) for t in 0:m-1]
        for max_corr_time in (0, 1, 10, 100, m + 5)
            @test rbim_time_correlation(spins; max_corr_time) ≈
                expected[1:min(max_corr_time, m - 1)+1]
        end
        @test spins == original
    end
end

@testset "threaded realizations match serial runMC" begin
    L, T = 3, 2.4
    rng = MersenneTwister(19)
    disorder = [ifelse.(rand(rng, 2L^2) .< 0.5, 1.0, 0.3) for _ in 1:8]
    original = deepcopy(disorder)
    for thermalization in (0, 1, 64)
        out = random_bond_observables(L, T, disorder;
            mcs=256, thermalization, binsize=32, seed=72, details=true,
            max_corr_time=17)
        @test out.metadata.max_corr_time == 17
        @test out.metadata.thermalization_update == :sw
        @test out.metadata.update == :local
        for (a, Js) in enumerate(disorder)
            history = Vector{Vector{Int}}()
            estimator = function (model, temp, bonds, extra)
                push!(history, vec(copy(model.spins)))
                return rbim_response_estimator(model, temp, bonds, extra)
            end
            p = Parameter(
                "Model" => Ising, "Lattice" => "square lattice", "L" => L,
                "Use Indicies as Bond Types" => true, "T" => T, "J" => Js,
                "Update Method" => rbim_heatbath_update!,
                "Estimator" => estimator,
                "MCS" => 256, "Thermalization" => 0, "Binning Size" => 32,
                "Seed" => out.metadata.seeds[a])
            # 独立执行 SW 热化，再让 runMC 仅执行测量，验证切换边界和随机数序列。
            model = Ising(p)
            SpinMonteCarlo.seed!(model, p["Seed"]) # 与 runMC(param) 构造模型后的重新播种一致。
            for _ in 1:thermalization
                SW_update!(model, T, Js)
            end
            result = runMC(model, p)
            @test length(history) == 256
            expected = [sum(history[k+t][i] * history[k][i]
                for k in 1:256-t, i in 1:L^2) / (L^2 * (256-t)) for t in 0:17]
            @test all(isapprox.(out.correlation[a], expected; atol=1e-12, rtol=1e-12))
            c, chi = L^2 * result["Specific Heat"], L^2 * result["Susceptibility"]
            localchi = [result["Local Susceptibility $i"] for i in 1:L^2]
            @test out.heat_capacity[a] == mean(c)
            @test out.susceptibility[a] == mean(chi)
            @test out.errors.heat_capacity[a] == stderror(c)
            @test out.errors.susceptibility[a] == stderror(chi)
            @test out.local_susceptibility[a] == reshape(mean.(localchi), L, L)
            @test out.errors.local_susceptibility[a] == reshape(stderror.(localchi), L, L)
        end
    end
    @test disorder == original
end

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
    out = random_bond_observables(L, T, disorder;
        mcs=65536, thermalization=4096, binsize=256, seed=72, details=true)
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
    kwargs = (mcs=8192, thermalization=256, binsize=64, seed=4)
    a = random_bond_observables(3, 2.0, [zeros(18)]; kwargs...)
    @test a == random_bond_observables(3, 2.0, [zeros(18)]; kwargs...)
    @test a[1] == [0.0]
    @test a[2][1] ≈ 9/2 atol=0.2
    @test all(abs.(a[3][1] .- 0.5) .< 0.1)
    @test length(a) == 4
    @test length(a[4][1]) == 101
    @test a[4][1][1] == 1.0
    # 自由自旋仅在尚未被选中过时保留记忆，每 sweep 随机选点 N 次。
    @test a[4][1][1:4] ≈ [(1 - 1/9)^(9t) for t in 0:3] atol=0.03
    @test random_bond_observables(3, 2.0, Vector{Float64}[]) ==
        (Float64[], Float64[], Matrix{Float64}[], Vector{Float64}[])
    @test_throws ArgumentError random_bond_observables(3, 0.0, disorder)
    @test_throws ArgumentError random_bond_observables(3, 2.0, [ones(17)])
    @test_throws ArgumentError random_bond_observables(3, 2.0, [fill(-0.3,18)])
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; mcs=65)
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; max_corr_time=-1)
end
