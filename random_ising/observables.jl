using SpinMonteCarlo
using Random
using Statistics

"""
    random_bond_observables(L, T, disorder; kwargs...)
        -> (heat_capacity, susceptibility, U4, correlation)

Sample each supplied quenched bond realization of the zero-field square Ising
model H = -sum(J_ij*s_i*s_j), s_i = ±1, kB = 1, with periodic boundaries.
`disorder` is a vector of vectors of finite nonnegative ACTUAL couplings, not binary
labels. Each vector has length 2L² in SpinMonteCarlo square-lattice bond order:
site i=x+L*(y-1) contributes J[2i-1] toward +x and J[2i] toward +y, wrapping
periodically. L≥2; L=2 retains the torus's parallel bonds.
Realizations are neither mutated nor regenerated. The supplied couplings fully
specify the Hamiltonian; no parameters of the bond distribution are needed.

For N=L^2 and M=sum(s), the static outputs are:
- heat_capacity[a] = (〈H²〉-〈H〉²)/T², total heat capacity;
- susceptibility[a] = 〈M²〉/T, total uniform-field susceptibility;
- U4[a] = 1 - <M^4>/(3<M^2>^2), the dimensionless Binder cumulant.
U4 has length num_disorder and uses the package's jackknife Binder Ratio,
consistent with critical_temp.jl. Its jackknife error is the Binder Ratio error / 3.
Finite-volume zero-field spin-inversion symmetry gives <M>=0 exactly.
Divide total heat capacity and susceptibility by N to obtain per-site values.

For each realization a, C_a(t) is the fixed-origin spin overlap
sum(s_i(t0+t)*s_i(t0) for i=1:N) / N, with t0 at the end of SW measurement,
for t=0:max_corr_time. `correlation` returns the sample mean across realizations,
as a vector of length max_corr_time+1. With details=true, `errors.correlation`
is the corrected sample standard deviation divided by sqrt(num_disorder).
For nonempty input C(0)=1; its SEM is zero when num_disorder >= 2.
With one sample the SEM is NaN; with no samples both vectors contain NaN.
Time zero is the end of all thermalization + mcs SW updates.
Each realization uses one trajectory, without averaging over starting times
or independent trajectories, and without subtracting spin means.
Time is measured in random-site heat-bath sweeps after SW measurement.
Overlaps are computed online in O(N*max_corr_time) time, retaining only one
reference configuration per active realization. Sample overlaps are retained
in a (max_corr_time+1) × num_disorder workspace until the final reduction.

Keywords: `mcs=8192` measured sweeps, `thermalization=1024` discarded sweeps,
`binsize=64` consecutive sweeps per block, `seed=1234`,
`max_corr_time=100` additional heat-bath sweeps (nonnegative, independent of mcs).
MCS must contain at least two complete blocks for jackknife errors.
Use more blocks when checking convergence. Thermalization and static measurements
both use Swendsen-Wang. After runMC finishes, its final configuration is copied
as the reference, then max_corr_time random-site heat-bath sweeps are performed.
Each heat-bath sweep draws N=L^2 sites with replacement and resamples each selected
spin from its conditional Boltzmann distribution. These sweeps do not contribute
to the static measurements; SW steps are excluded from correlation time.
Realizations run concurrently with `Threads.@threads`; start Julia with
`--threads=auto` to enable multiple threads. Seeds and output order do not
depend on thread scheduling. With one Julia thread, execution is serial.

With `details=true`, return a named tuple containing the four arrays,
`errors` arrays from block jackknife for heat capacity, susceptibility and U4,
the across-sample SEM for correlation, and `metadata` including package version
and one seed per realization. Sampling and block jackknife use `runMC`.
In SpinMonteCarlo v1.2.2, memory scales as O(N+mcs): raw measurements are
retained by the driver before binning.
Static errors do not include disorder averaging. Correlation errors reflect
across-sample variation, including the single-trajectory sampling noise.
Check convergence by increasing
thermalization, MCS and block size, and comparing independent seeds.

Example:
```julia
rng = MersenneTwister(10)
L, T, p, r = 16, 2.0, 0.5, 0.3
Js = [ifelse.(rand(rng, 2L^2) .< p, 1.0, r) for _ in 1:4]
C, chi, U4, correlation = random_bond_observables(L, T, Js)
```
"""
function random_bond_observables(L::Integer, T::Real,
        disorder::AbstractVector{<:AbstractVector};
        mcs::Int=8192, thermalization::Int=1024, max_corr_time::Int=128,
        binsize::Int=64, seed::Integer=1234,
        details::Bool=false)

    # 只检查晶格、温度和统计计算的必要条件。
    L >= 2 || throw(ArgumentError("L must be at least 2"))
    isfinite(T) && T > 0 || throw(ArgumentError("T must be finite and positive"))
    binsize > 0 && mcs >= 2binsize && mcs % binsize == 0 ||
        throw(ArgumentError("mcs must contain at least two complete bins"))
    thermalization >= 0 || throw(ArgumentError("thermalization must be nonnegative"))
    max_corr_time >= 0 || throw(ArgumentError("max_corr_time must be nonnegative"))

    # 包内更新按键编号索引，且此版本的 SW 算法要求非负耦合。
    num_disorder = length(disorder)
    for (a, Js) in enumerate(disorder)
        length(Js) == 2L^2 || throw(ArgumentError("realization $a must have 2L² bonds"))
        all(j -> isfinite(j) && j >= 0, Js) ||
            throw(ArgumentError("realization $a must contain finite nonnegative couplings"))
    end

    # 每个构型分配独立种子，结果顺序与输入顺序一致。
    rng = MersenneTwister(seed)
    seeds = rand(rng, UInt32, num_disorder)

    C, chi, U4 = zeros(num_disorder), zeros(num_disorder), zeros(num_disorder)
    dC, dchi, dU4 = similar(C), similar(chi), similar(U4)
    correlations = Matrix{Float64}(undef, max_corr_time + 1, num_disorder)

    # 每个任务独占模型、随机数流和输出位置，避免并发 push!。
    Threads.@threads for a in 1:num_disorder
        Js = disorder[a]
        measurement = Dict{String,Any}()
        estimator = (model, temp, bonds, extra) ->
            rbim_response_estimator!(measurement, model, temp, bonds, extra)
        # runMC handles SW thermalization, static measurements and jackknife.
        param = Parameter(
            "Model" => Ising, "Lattice" => "square lattice", "L" => Int(L),

            # Indicies 是包 v1.2.2 中实际使用的拼写。
            "Use Indicies as Bond Types" => true,
            "T" => Float64(T), "J" => collect(Float64, Js),

            "Update Method" => SW_update!, "Estimator" => estimator,
            "MCS" => mcs, "Thermalization" => thermalization,
            "Binning Size" => binsize, "Seed" => seeds[a],
        )

        model = Ising(param)
        SpinMonteCarlo.seed!(model, seeds[a]) # Match runMC(param)'s reseeding.
        result = runMC(model, param)

        # Start at the final SW state and continue the same RNG with heat bath.
        reference = copy(vec(model.spins))
        correlation = @view correlations[:, a]
        correlation[1] = 1.0
        for t in 1:max_corr_time
            rbim_heatbath_update!(model, Float64(T), param["J"])
            correlation[t + 1] = rbim_time_correlation(vec(model.spins), reference)
        end

        # 内置热容和磁化率按格点归一化，乘以 L² 得到整个系统的量。
        cj = L^2 * result["Specific Heat"]
        chij = L^2 * result["Susceptibility"]
        binder = result["Binder Ratio"]

        C[a], chi[a], U4[a] = mean(cj), mean(chij), 1 - mean(binder) / 3
        dC[a], dchi[a], dU4[a] = stderror(cj), stderror(chij), stderror(binder) / 3
    end

    correlation_mean = num_disorder > 0 ? vec(mean(correlations; dims=2)) :
        fill(NaN, max_corr_time + 1)
    correlation_sem = num_disorder > 1 ?
        vec(std(correlations; dims=2, corrected=true)) ./ sqrt(num_disorder) :
        fill(NaN, max_corr_time + 1)

    details || return (C, chi, U4, correlation_mean)

    return (heat_capacity=C, susceptibility=chi, U4=U4,
        correlation=correlation_mean,
        errors=(heat_capacity=dC, susceptibility=dchi, U4=dU4, correlation=correlation_sem),
        metadata=(L=L, T=T, seed=seed, seeds=seeds, mcs=mcs,
            thermalization=thermalization, binsize=binsize,
            max_corr_time=max_corr_time,
            boundary=:periodic, normalization=:extensive,
            version=pkgversion(SpinMonteCarlo)))
end

"""
    random_bond_local_susceptibility(L, T, Js; kwargs...) -> (chi_local, err_local)

Sample one supplied bond realization with the same Hamiltonian, bond order and
periodic boundaries as `random_bond_observables`. `Js` contains 2L^2 finite
nonnegative actual couplings and is not mutated.

Return two L-by-L Float64 matrices: the zero-field uniform-field response
chi_local[i] = <s_i*M>/T, M=sum(s), and its block standard error. These are
site responses, without division by L^2 or subtraction of sampled spin means.
No other observables or trajectories are recorded.

Keywords: `mcs=8192`, `thermalization=1024`, `binsize=64`, `seed=1234`.
Thermalization and measurements both use SW updates; measurements follow each
SW update. The seed controls this single MC trajectory directly.
With the same per-realization seed and MC parameters, this reproduces the SW
trajectory of `random_bond_observables`; sum(chi_local) matches its susceptibility
up to floating-point rounding.
Require at least two equal complete blocks. Block means are accumulated online
with Welford's algorithm, using O(L^2) storage and O(L^2*mcs) measurement work.
For this linear mean, the block standard error equals delete-one-block jackknife.
Increase binsize to check error convergence; correlated blocks underestimate
uncertainty. Errors describe thermal sampling, not disorder averaging.

For seeded disorder, generate `Js` with a separate `MersenneTwister(disorder_seed)`
and an explicitly chosen bond distribution, then pass it to this function.
"""
function random_bond_local_susceptibility(L::Integer, T::Real,
        Js::AbstractVector; mcs::Int=8192, thermalization::Int=1024,
        binsize::Int=64, seed::Integer=1234)
    L >= 2 || throw(ArgumentError("L must be at least 2"))
    isfinite(T) && T > 0 || throw(ArgumentError("T must be finite and positive"))
    binsize > 0 && mcs >= 2binsize && mcs % binsize == 0 ||
        throw(ArgumentError("mcs must contain at least two complete bins"))
    thermalization >= 0 || throw(ArgumentError("thermalization must be nonnegative"))
    length(Js) == 2L^2 || throw(ArgumentError("Js must have 2L^2 bonds"))
    all(j -> isfinite(j) && j >= 0, Js) ||
        throw(ArgumentError("Js must contain finite nonnegative couplings"))

    param = Parameter("Lattice" => "square lattice", "L" => Int(L),
        "Use Indicies as Bond Types" => true, "Seed" => seed)
    model = Ising(param)
    SpinMonteCarlo.seed!(model, seed) # Match runMC's reseeding after construction.
    temp, couplings = Float64(T), collect(Float64, Js)
    for _ in 1:thermalization
        SW_update!(model, temp, couplings)
    end

    block_sum = zeros(L, L)
    chi_local, err_local = zeros(L, L), zeros(L, L)
    nblocks = mcs ÷ binsize
    for b in 1:nblocks
        fill!(block_sum, 0.0)
        for _ in 1:binsize
            SW_update!(model, temp, couplings)
            magnetization_over_T = sum(model.spins) / temp
            for i in eachindex(block_sum, model.spins)
                block_sum[i] += model.spins[i] * magnetization_over_T
            end
        end
        for i in eachindex(chi_local)
            block_mean = block_sum[i] / binsize
            delta = block_mean - chi_local[i]
            chi_local[i] += delta / b
            err_local[i] += delta * (block_mean - chi_local[i])
        end
    end
    # err_local holds the sum of squared deviations until this final conversion.
    err_local .= sqrt.(err_local ./ (nblocks * (nblocks - 1)))
    return chi_local, err_local
end

# 当前构型与固定参考构型的格点平均交叠，用 Int 累加避免 Int8 溢出。
function rbim_time_correlation(spins::AbstractVector, reference::AbstractVector)
    total = 0
    for i in eachindex(spins, reference)
        total += Int(spins[i]) * Int(reference[i])
    end
    return total / length(spins)
end

# 每个 sweep 有放回地随机选点 N 次，零局域场时以等概率重采样 ±1。
function rbim_heatbath_update!(model, T, Js)
    ns = numsites(model)
    for _ in 1:ns
        i = rand(model.rng, 1:ns)
        field = 0.0
        for (j, b) in neighbors(model, i)
            field += Js[bondtype(model, b)] * model.spins[j]
        end

        probability_up = (1 + tanh(field / T)) / 2
        model.spins[i] = rand(model.rng) < probability_up ? 1 : -1
    end

    return nothing
end

function rbim_response_estimator!(measurement::Dict{String,Any}, model::Ising,
        T::Real, Js::AbstractArray, extra=nothing)
    ns = numsites(model)
    magnetization = sum(model.spins)
    m = magnetization / ns
    energy = 0.0
    for b in bonds(model)
        s1, s2 = source(b), target(b)
        energy += ifelse(model.spins[s1] == model.spins[s2], -1.0, 1.0) * Js[bondtype(b)]
    end
    energy /= ns
    measurement["Magnetization"] = m
    measurement["|Magnetization|"] = abs(m)
    measurement["Magnetization^2"] = m^2
    measurement["Magnetization^4"] = m^4
    measurement["Energy"] = energy
    measurement["Energy^2"] = energy^2

    return measurement
end

#=
using CairoMakie
let
    L, T = 10, 1.0
    Ns, Nb = L * L, 2 * L * L
    Ndis = 1000

    disorder = [ifelse.(rand(Nb) .< 0.5, 1.0, 0.0) for _ in 1:Ndis]

    out = random_bond_observables(L, T, disorder; details=true)
    chi, U4 = out.susceptibility, out.U4

    fig = Figure(size=(650, 800), figure_padding=12)
    rowgap!(fig.layout, 10)
    ax = Axis(fig[1, 1]; xlabel="χ", ylabel="Probability density",
        title="Susceptibility distribution (L=$L, T=$T)")
    hist!(ax, chi; bins=30, normalization=:pdf)

    ax_u4 = Axis(fig[2, 1]; xlabel="U4", ylabel="Probability density",
        title="Binder cumulant distribution (L=$L, T=$T)")
    hist!(ax_u4, U4; bins=30, normalization=:pdf)

    # 对每个时间间隔，计算所有无序构型的均值及其标准误。
    correlation_mean, correlation_sem = out.correlation, out.errors.correlation
    lags = 0:length(correlation_mean)-1
    ax_correlation = Axis(fig[3, 1]; xlabel="Lag (MC steps)", ylabel="Autocorrelation",
        title="Mean autocorrelation (L=$L, T=$T)")
    band!(ax_correlation, lags, correlation_mean .- correlation_sem,
        correlation_mean .+ correlation_sem; color=(:dodgerblue, 0.25), label="±1 SEM")
    lines!(ax_correlation, lags, correlation_mean; color=:dodgerblue, label="Mean")
    axislegend(ax_correlation)
    display(fig)
end
=#
