using SpinMonteCarlo
using Random
using Statistics

"""
    random_bond_observables(L, T, disorder; kwargs...)
        -> (heat_capacity, susceptibility, local_susceptibility, correlation)

Sample each supplied quenched bond realization of the zero-field square Ising
model H = -sum(J_ij*s_i*s_j), s_i = ±1, kB = 1, with periodic boundaries.
`disorder` is a vector of vectors of finite nonnegative ACTUAL couplings, not binary
labels. Each vector has length 2L² in SpinMonteCarlo square-lattice bond order:
site i=x+L*(y-1) contributes J[2i-1] toward +x and J[2i] toward +y, wrapping
periodically. L≥2; L=2 retains the torus's parallel bonds.
Realizations are neither mutated nor regenerated. The supplied couplings fully
specify the Hamiltonian; no parameters of the bond distribution are needed.

For N=L² and M=sum(s), outputs are extensive quantities:
- heat_capacity[a] = (〈H²〉-〈H〉²)/T², total heat capacity;
- susceptibility[a] = 〈M²〉/T, total uniform-field susceptibility;
- local_susceptibility[a][x,y] = 〈s[x,y]*M〉/T, an L×L response map.
Finite-volume zero-field spin-inversion symmetry gives 〈s_i〉=〈M〉=0 exactly.
This is NOT the |M|-subtracted susceptibility or the N×N pair-response matrix.
The local map sums to the total susceptibility. Divide total heat capacity and
total susceptibility by N to obtain per-site values.

`correlation[a][t+1]` is the fixed-origin spin overlap
sum(s_i(t)*s_i(0) for i=1:N) / N, with the reference taken at the end of thermalization,
for t=0:max_corr_time. Each realization returns a Vector{Float64} of length
max_corr_time+1, with C(0)=1. Time zero is the end of thermalization.
Each realization uses one trajectory, without averaging over starting times
or independent trajectories, and without subtracting spin means.
Time is measured in random-site heat-bath sweeps after thermalization.
Overlaps are computed online in O(N*max_corr_time) time, retaining only one
reference configuration and the output: O(N+max_corr_time) correlation storage.

Keywords: `mcs=8192` measured sweeps, `thermalization=1024` discarded sweeps,
`binsize=64` consecutive sweeps per block, `seed=1234`,
`max_corr_time=100` maximum correlation lag (inclusive), with 0 <= max_corr_time <= mcs.
MCS must contain at least two complete blocks for jackknife errors.
Use more blocks when checking convergence. Thermalization always uses Swendsen-Wang.
After thermalization, each measurement sweep performs
N=L² random-site heat-bath updates per sweep, sampling sites with replacement.
Each selected spin is drawn from its conditional Boltzmann distribution.
One correlation time unit is one measurement sweep; thermalization steps are excluded.
Realizations run concurrently with `Threads.@threads`; start Julia with
`--threads=auto` to enable multiple threads. Seeds and output order do not
depend on thread scheduling. With one Julia thread, execution is serial.

With `details=true`, return a named tuple containing the four arrays,
`errors` arrays from block jackknife for the three static responses (no correlation
error estimate), and `metadata` including package version
and one seed per realization. Sampling and block jackknife use `runMC`.
In SpinMonteCarlo v1.2.2, memory scales as O(N*mcs): raw measurements are
retained by the driver before binning.
Errors do not include disorder averaging. Check convergence by increasing
thermalization, MCS and block size, and comparing independent seeds.

Example:
```julia
rng = MersenneTwister(10)
L, T, p, r = 16, 2.0, 0.5, 0.3
Js = [ifelse.(rand(rng, 2L^2) .< p, 1.0, r) for _ in 1:4]
C, chi, chi_local, correlation = random_bond_observables(L, T, Js)
```
"""
function random_bond_observables(L::Integer, T::Real,
        disorder::AbstractVector{<:AbstractVector};
        mcs::Int=8192, thermalization::Int=1024, max_corr_time::Int=100,
        binsize::Int=64, seed::Integer=1234,
        details::Bool=false)

    # 只检查晶格、温度和统计计算的必要条件。
    L >= 2 || throw(ArgumentError("L must be at least 2"))
    isfinite(T) && T > 0 || throw(ArgumentError("T must be finite and positive"))
    binsize > 0 && mcs >= 2binsize && mcs % binsize == 0 ||
        throw(ArgumentError("mcs must contain at least two complete bins"))
    thermalization >= 0 || throw(ArgumentError("thermalization must be nonnegative"))
    0 <= max_corr_time <= mcs ||
        throw(ArgumentError("max_corr_time must satisfy 0 <= max_corr_time <= mcs"))

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

    C, chi = zeros(num_disorder), zeros(num_disorder)
    maps = Vector{Matrix{Float64}}(undef, num_disorder)
    correlations = Vector{Vector{Float64}}(undef, num_disorder)
    dC, dchi = similar(C), similar(chi)
    dmaps = Vector{Matrix{Float64}}(undef, num_disorder)

    # 固定名称只生成一次，所有任务只读共享，避免每个测量步重复分配字符串。
    local_keys = ["Local Susceptibility $i" for i in 1:L^2]

    # 每个任务独占模型、随机数流和输出位置，避免并发 push!。
    Threads.@threads for a in 1:num_disorder
        Js = disorder[a]
        reference = Vector{Int8}(undef, L^2)
        correlation = Vector{Float64}(undef, max_corr_time + 1)
        correlation[1] = 1.0
        step = Ref(0)
        updates = Ref(0)
        staged_update! = function (model, temp, bonds)
            # 第一次热浴更新前保存参考态，即 SW 热化刚结束的时间零点。
            if updates[] == thermalization
                copyto!(reference, vec(model.spins))
            end
            # 计数器由当前构型独占；前 thermalization 步仅用于 SW 热化。
            updates[] += 1
            if updates[] <= thermalization
                return SW_update!(model, temp, bonds)
            end
            return rbim_heatbath_update!(model, temp, bonds)
        end
        # runMC 立即累积标量值；字典由每个无序构型独占，可在下一步覆盖。
        measurement = Dict{String,Any}()
        estimator = function (model, temp, bonds, extra)
            # 首次测量对应热化后的时间 1，只计算前 max_corr_time 步的交叠。
            step[] += 1
            if step[] <= max_corr_time
                correlation[step[] + 1] =
                    rbim_time_correlation(vec(model.spins), reference)
            end
            return rbim_response_estimator!(measurement, model, temp, bonds, extra; local_keys)
        end
        # 热化、测量、分块与 jackknife 都交给 runMC。
        param = Parameter(
            "Model" => Ising, "Lattice" => "square lattice", "L" => Int(L),

            # Indicies 是包 v1.2.2 中实际使用的拼写。
            "Use Indicies as Bond Types" => true,
            "T" => Float64(T), "J" => collect(Float64, Js),

            "Update Method" => staged_update!, "Estimator" => estimator,
            "MCS" => mcs, "Thermalization" => thermalization,
            "Binning Size" => binsize, "Seed" => seeds[a],
        )

        result = runMC(param)
        correlations[a] = correlation

        # 内置热容和磁化率按格点归一化，乘以 L² 得到整个系统的量。
        cj = L^2 * result["Specific Heat"]
        chij = L^2 * result["Susceptibility"]
        localj = [result[key] for key in local_keys]

        C[a], chi[a] = mean(cj), mean(chij)
        dC[a], dchi[a] = stderror(cj), stderror(chij)

        # 晶格编号的 x 坐标变化最快，与 Julia 的矩阵存储顺序一致。
        maps[a] = reshape(mean.(localj), L, L)
        dmaps[a] = reshape(stderror.(localj), L, L)
    end

    details || return (C, chi, maps, correlations)

    return (heat_capacity=C, susceptibility=chi, local_susceptibility=maps,
        correlation=correlations,
        errors=(heat_capacity=dC, susceptibility=dchi, local_susceptibility=dmaps),
        metadata=(L=L, T=T, seed=seed, seeds=seeds, mcs=mcs,
            thermalization=thermalization, thermalization_update=:sw,
            binsize=binsize, update=:local,
            max_corr_time=max_corr_time, corr_t0=0,
            boundary=:periodic, normalization=:extensive,
            version=pkgversion(SpinMonteCarlo)))
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

function rbim_response_estimator(model::Ising, T::Real, Js::AbstractArray, extra=nothing;
        local_keys=("Local Susceptibility $i" for i in 1:numsites(model)))
    return rbim_response_estimator!(Dict{String,Any}(), model, T, Js, extra; local_keys)
end

function rbim_response_estimator!(measurement::Dict{String,Any}, model::Ising,
        T::Real, Js::AbstractArray, extra=nothing;
        local_keys=("Local Susceptibility $i" for i in 1:numsites(model)))
    # 沿用 SpinMonteCarlo simple_estimator 的归一化和逐键数值，直接覆盖已有字典。
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

    # 零场对称系综下 χᵢ = 〈sᵢM〉/T，每步只计算一次公共因子。
    magnetization_over_T = magnetization / T
    for (i, key) in enumerate(local_keys)
        measurement[key] = model.spins[i] * magnetization_over_T
    end

    return measurement
end

#=
using CairoMakie
let
    L, T = 10, 1.0
    Ns, Nb = L * L, 2 * L * L
    Ndis = 1000

    disorder = [ifelse.(rand(Nb) .< 0.5, 1.0, 0.0) for _ in 1:Ndis]

    C, chi, localchi, correlation = random_bond_observables(L, T, disorder)

    fig = Figure(size=(650, 800), figure_padding=12)
    rowgap!(fig.layout, 10)
    ax = Axis(fig[1, 1]; xlabel="χ", ylabel="Probability density",
        title="Susceptibility distribution (L=$L, T=$T)")
    hist!(ax, chi; bins=30, normalization=:pdf)

    # 展示一个无序构型在采样结束后得到的时间平均局域磁化率。
    realization = 1
    heatmap_layout = GridLayout(fig[2, 1]; tellwidth=false)
    colgap!(heatmap_layout, 8)
    ax_local = Axis(heatmap_layout[1, 1]; xlabel="x", ylabel="y",
        width=230, height=230, aspect=DataAspect(),
        title="Time-averaged local susceptibility (realization $realization)")
    hm = heatmap!(ax_local, 1:L, 1:L, localchi[realization]; colormap=:viridis)
    Colorbar(heatmap_layout[1, 2], hm; label="Local susceptibility", width=12)

    # 对每个时间间隔，计算所有无序构型的均值及其标准误。
    correlations = reduce(hcat, correlation)
    correlation_mean = vec(mean(correlations; dims=2))
    correlation_sem = vec(std(correlations; dims=2)) ./ sqrt(Ndis)
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