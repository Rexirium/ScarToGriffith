using SpinMonteCarlo
using Random
using Statistics

"""
    random_bond_observables(L, T, disorder; kwargs...)
        -> (heat_capacity, susceptibility, local_susceptibility)

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

Keywords: `mcs=8192` measured sweeps, `thermalization=2048` discarded sweeps,
`binsize=64` consecutive sweeps per block, `seed=1234`, `update=:sw` or `:local`.
MCS must contain at least two complete blocks for jackknife errors.
Use more blocks when checking convergence. The local option runs a package
Metropolis sweep with probability 1/2, then a random-site heat-bath update on
every step. This avoids deterministic cycles for decoupled spins.
The default is Swendsen-Wang.

With `details=true`, return a named tuple containing the three arrays, matching
`errors` arrays from block jackknife, and `metadata` including package version
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
C, chi, chi_local = random_bond_observables(L, T, Js)
```
"""
function random_bond_observables(L::Integer, T::Real,
        disorder::AbstractVector{<:AbstractVector};
        mcs::Int=8192, thermalization::Int=2048, binsize::Int=64,
        seed::Integer=1234, update::Symbol=:sw, details::Bool=false)

    # 只检查晶格、温度和统计计算的必要条件。
    L >= 2 || throw(ArgumentError("L must be at least 2"))
    isfinite(T) && T > 0 || throw(ArgumentError("T must be finite and positive"))
    binsize > 0 && mcs >= 2binsize && mcs % binsize == 0 ||
        throw(ArgumentError("mcs must contain at least two complete bins"))
    thermalization >= 0 || throw(ArgumentError("thermalization must be nonnegative"))
    update in (:sw, :local) || throw(ArgumentError("update must be :sw or :local"))

    # 包内更新按键编号索引，且此版本的 SW 算法要求非负耦合。
    for (a, Js) in enumerate(disorder)
        length(Js) == 2L^2 || throw(ArgumentError("realization $a must have 2L² bonds"))
        all(j -> isfinite(j) && j >= 0, Js) ||
            throw(ArgumentError("realization $a must contain finite nonnegative couplings"))
    end

    # 每个构型分配独立种子，结果顺序与输入顺序一致。
    rng = MersenneTwister(seed)
    seeds = rand(rng, UInt32, length(disorder))

    C, chi = zeros(length(disorder)), zeros(length(disorder))
    maps = Matrix{Float64}[]
    dC, dchi = similar(C), similar(chi)
    dmaps = Matrix{Float64}[]

    update! = update == :sw ? SW_update! : rbim_lazy_local_update!

    for (a, Js) in enumerate(disorder)
        # 热化、测量、分块与 jackknife 都交给 runMC。
        param = Parameter(
            "Model" => Ising, "Lattice" => "square lattice", "L" => Int(L),

            # Indicies 是包 v1.2.2 中实际使用的拼写。
            "Use Indicies as Bond Types" => true,
            "T" => Float64(T), "J" => Float64.(collect(Js)),

            "Update Method" => update!, "Estimator" => rbim_response_estimator,
            "MCS" => mcs, "Thermalization" => thermalization,
            "Binning Size" => binsize, "Seed" => seeds[a],
        )

        result = runMC(param)

        # 内置热容和磁化率按格点归一化，乘以 L² 得到整个系统的量。
        cj = L^2 * result["Specific Heat"]
        chij = L^2 * result["Susceptibility"]
        localj = [result["Local Susceptibility $i"] for i in 1:L^2]

        C[a], chi[a] = mean(cj), mean(chij)
        dC[a], dchi[a] = stderror(cj), stderror(chij)

        # 晶格编号的 x 坐标变化最快，与 Julia 的矩阵存储顺序一致。
        push!(maps, reshape(mean.(localj), L, L))
        push!(dmaps, reshape(stderror.(localj), L, L))
    end

    details || return (C, chi, maps)

    return (heat_capacity=C, susceptibility=chi, local_susceptibility=maps,
        errors=(heat_capacity=dC, susceptibility=dchi, local_susceptibility=dmaps),
        metadata=(L=L, T=T, seed=seed, seeds=seeds, mcs=mcs,
            thermalization=thermalization, binsize=binsize, update=update,
            boundary=:periodic, normalization=:extensive,
            version=pkgversion(SpinMonteCarlo)))
end

# 随机单点热浴避免零耦合时固定顺序翻转陷入少数构型的循环。
function rbim_lazy_local_update!(model, T, Js)
    rand(model.rng, Bool) && local_update!(model, T, Js)

    i = rand(model.rng, 1:numsites(model))
    field = 0.0
    for (j, b) in neighbors(model, i)
        field += Js[bondtype(model, b)] * model.spins[j]
    end

    probability_up = (1 + tanh(field / T)) / 2
    model.spins[i] = rand(model.rng) < probability_up ? 1 : -1

    return nothing
end

function rbim_response_estimator(model::Ising, T::Real, Js::AbstractArray, extra=nothing)
    # 保留包内能量和磁化强度的估计量；simple_estimator 可处理零耦合。
    measurement = simple_estimator(model, T, Js, extra)

    # 零场对称系综下 χᵢ = 〈sᵢM〉/T，每个格点作为一个标量观测量。
    magnetization = sum(model.spins)
    for i in 1:numsites(model)
        measurement["Local Susceptibility $i"] = model.spins[i] * magnetization / T
    end

    return measurement
end
