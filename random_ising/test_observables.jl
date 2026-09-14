using Test, Random, Statistics, SpinMonteCarlo
include("observables.jl")

@testset "reused measurements match package estimator" begin
    rng = MersenneTwister(18)
    for L in (2, 3), Js in (zeros(2L^2), rand(rng, 2L^2))
        T = 2.4
        model = Ising(Parameter("Lattice" => "square lattice", "L" => L,
            "Use Indicies as Bond Types" => true, "Seed" => UInt32(18)))
        measurement = Dict{String,Any}()
        for spins in (ones(Int, L^2), -ones(Int, L^2), rand(rng, [-1, 1], L^2))
            model.spins[:] = spins
            expected = simple_estimator(model, T, Js)
            @test rbim_response_estimator!(measurement, model, T, Js) === measurement
            @test length(measurement) == length(expected)
            for (key, value) in expected
                @test measurement[key] ≈ value
            end
        end
    end
end

@testset "fixed-reference spin overlap" begin
    reference = Int8[1, 1, -1, -1]
    original = copy(reference)
    @test rbim_time_correlation(reference, reference) == 1.0
    @test rbim_time_correlation(-reference, reference) == -1.0
    @test rbim_time_correlation(Int8[1, -1, 1, -1], reference) == 0.0
    @test rbim_time_correlation(ones(Int8, 1024), ones(Int8, 1024)) == 1.0
    @test reference == original
end

@testset "threaded realizations match serial runMC" begin
    L, T = 3, 2.4
    rng = MersenneTwister(19)
    disorder = [ifelse.(rand(rng, 2L^2) .< 0.5, 1.0, 0.3) for _ in 1:8]
    original = deepcopy(disorder)
    for thermalization in (0, 1, 64), (corr_start_time, max_corr_time) in
            ((0, 0), (0, 17), (0, 256), (1, 17), (23, 233), (256, 0))
        out = random_bond_observables(L, T, disorder;
            mcs=256, thermalization, binsize=32, seed=72, details=true,
            max_corr_time, corr_start_time)
        @test size(out.correlation) == size(out.errors.correlation) == (max_corr_time + 1,)
        expected_samples = Matrix{Float64}(undef, max_corr_time + 1, length(disorder))
        @test out.metadata.max_corr_time == max_corr_time
        @test out.metadata.corr_start_time == corr_start_time
        for (a, Js) in enumerate(disorder)
            history = Vector{Vector{Int}}()
            estimator = function (model, temp, bonds, extra)
                push!(history, vec(copy(model.spins)))
                return simple_estimator(model, temp, bonds, extra)
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
            initial = vec(copy(model.spins)) # 热化结束即时间零点。
            result = runMC(model, p)
            @test length(history) == 256
            # 独立保存完整轨迹，验证在线计算的起点、终点及全部交叠值。
            trajectory = hcat(initial, history...)
            expected = [sum(trajectory[i, corr_start_time+t+1] * trajectory[i, corr_start_time+1]
                for i in 1:L^2) / L^2 for t in 0:max_corr_time]
            expected_samples[:, a] = expected
            c, chi = L^2 * result["Specific Heat"], L^2 * result["Susceptibility"]
            @test out.heat_capacity[a] == mean(c)
            @test out.susceptibility[a] == mean(chi)
            @test out.errors.heat_capacity[a] == stderror(c)
            @test out.errors.susceptibility[a] == stderror(chi)
            @test out.U4[a] == 1 - mean(result["Binder Ratio"]) / 3
            @test out.errors.U4[a] == stderror(result["Binder Ratio"]) / 3
        end
        @test out.correlation ≈ vec(mean(expected_samples; dims=2))
        @test out.errors.correlation ≈ vec(std(expected_samples; dims=2)) ./ sqrt(length(disorder))
        @test out.correlation[1] == 1.0
        @test out.errors.correlation[1] == 0.0
    end
    @test disorder == original
end

@testset "online local response matches stored block jackknife" begin
    rng = MersenneTwister(31)
    for L in (2, 3), thermalization in (0, 16), binsize in (1, 16)
        T, seed, mcs = 2.4, 73, 128
        Js = rand(rng, 2L^2)
        original = copy(Js)
        chi, err = random_bond_local_susceptibility(L, T, Js;
            mcs, thermalization, binsize, seed)
        @test size(chi) == size(err) == (L, L)
        @test Js == original

        # Store a serial runMC trajectory independently, then explicitly form
        # all delete-one-block estimates instead of using online variance.
        history = Vector{Vector{Float64}}()
        estimator = function (model, temp, bonds, extra)
            push!(history, vec(model.spins .* (sum(model.spins) / temp)))
            return simple_estimator(model, temp, bonds, extra)
        end
        param = Parameter("Lattice" => "square lattice", "L" => L,
            "Use Indicies as Bond Types" => true, "Seed" => seed,
            "T" => T, "J" => Js, "Update Method" => rbim_heatbath_update!,
            "Estimator" => estimator, "MCS" => mcs, "Thermalization" => 0,
            "Binning Size" => binsize)
        model = Ising(param)
        SpinMonteCarlo.seed!(model, seed)
        for _ in 1:thermalization
            SW_update!(model, T, Js)
        end
        runMC(model, param)
        nblocks = mcs ÷ binsize
        blocks = dropdims(mean(reshape(hcat(history...), L^2, binsize, nblocks);
            dims=2); dims=2)
        expected = mean(blocks; dims=2)
        leave_one_out = (sum(blocks; dims=2) .- blocks) ./ (nblocks - 1)
        deviations = leave_one_out .- mean(leave_one_out; dims=2)
        expected_error = sqrt.((nblocks - 1) / nblocks .* sum(abs2, deviations; dims=2))
        @test chi ≈ reshape(expected, L, L)
        @test err ≈ reshape(expected_error, L, L)
    end

    kwargs = (mcs=32768, thermalization=256, binsize=128, seed=14)
    chi, err = random_bond_local_susceptibility(3, 2.0, zeros(18); kwargs...)
    @test (chi, err) == random_bond_local_susceptibility(3, 2.0, zeros(18); kwargs...)
    @test all(isfinite, err) && all(err .>= 0)
    @test all(abs.(chi .- 0.5) .< 6 .* err .+ 0.02) # Independent spins: chi_i = 1/T.
    for (L, T, Js) in ((1, 2.0, ones(2)), (3, 0.0, ones(18)),
            (3, Inf, ones(18)), (3, 2.0, ones(17)),
            (3, 2.0, fill(-1.0, 18)), (3, 2.0, fill(NaN, 18)))
        @test_throws ArgumentError random_bond_local_susceptibility(L, T, Js)
    end
    for options in ((mcs=64, binsize=64), (mcs=65,), (binsize=0,), (thermalization=-1,))
        @test_throws ArgumentError random_bond_local_susceptibility(3, 2.0, ones(18); options...)
    end
end

# Independent square-torus Hamiltonian, including the L=2 parallel bonds.
function exact_observables(L, T, Js)
    n = L^2
    z = e1 = e2 = m2 = m4 = 0.0
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
        m4 += w * m^4
    end
    return ((e2/z - (e1/z)^2)/T^2, m2/z/T, 1 - (m4/z)/(3(m2/z)^2))
end

@testset "random-bond responses" begin
    L, T, r = 3, 2.4, 0.3
    disorder = [[i in (1, 2, 5, 9, 10, 14) ? 1.0 : r for i in 1:2L^2], ones(2L^2)]
    original = deepcopy(disorder)
    out = random_bond_observables(L, T, disorder;
        mcs=65536, thermalization=4096, binsize=256, seed=72, details=true)
    @test disorder == original
    @test length(out.heat_capacity) == length(out.susceptibility) == 2
    @test size(out.U4) == (2,)
    @test keys(out.errors) == (:heat_capacity, :susceptibility, :U4, :correlation)
    @test size(out.errors.U4) == (2,)
    @test all(x -> isfinite(x) && x >= 0, out.errors.U4)
    @test !hasproperty(out, :local_susceptibility)
    for a in eachindex(disorder)
        c, chi, u4 = exact_observables(L, T, disorder[a])
        @test abs(out.heat_capacity[a] - c) < 6out.errors.heat_capacity[a] + 0.02
        @test abs(out.susceptibility[a] - chi) < 6out.errors.susceptibility[a] + 0.02
        @test isapprox(out.U4[a], u4; atol=0.02)
    end
    kwargs = (mcs=8192, thermalization=256, binsize=64, seed=4)
    a = random_bond_observables(3, 2.0, [zeros(18)]; kwargs...)
    @test a == random_bond_observables(3, 2.0, [zeros(18)]; kwargs...)
    @test a[1] == [0.0]
    @test a[2][1] ≈ 9/2 atol=0.2
    @test size(a[3]) == (1,)
    @test isapprox(a[3][1], 2/27; atol=0.04)
    @test length(a) == 4
    @test size(a[4]) == (101,)
    @test a[4][1] == 1.0
    @test all(abs.(a[4]) .<= 1)
    single = random_bond_observables(3, 2.0, [zeros(18)]; kwargs..., details=true)
    @test single.correlation == a[4]
    @test size(single.errors.correlation) == (101,)
    @test all(isnan, single.errors.correlation)
    empty = random_bond_observables(3, 2.0, Vector{Float64}[]; details=true)
    @test isempty(empty.heat_capacity) && isempty(empty.susceptibility) && isempty(empty.U4)
    @test size(empty.correlation) == size(empty.errors.correlation) == (101,)
    @test all(isnan, empty.correlation) && all(isnan, empty.errors.correlation)
    @test isequal(random_bond_observables(3, 2.0, Vector{Float64}[]),
        (Float64[], Float64[], Float64[], fill(NaN, 101)))
    @test_throws ArgumentError random_bond_observables(3, 0.0, disorder)
    @test_throws ArgumentError random_bond_observables(3, 2.0, [ones(17)])
    @test_throws ArgumentError random_bond_observables(3, 2.0, [fill(-0.3,18)])
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; mcs=65)
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; max_corr_time=-1)
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; max_corr_time=8193)
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; corr_start_time=-1)
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; corr_start_time=8193, max_corr_time=0)
    @test_throws ArgumentError random_bond_observables(3, 2.0, disorder; corr_start_time=8192, max_corr_time=1)
end
