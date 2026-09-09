using SpinMonteCarlo
using Random
using Statistics
using LinearAlgebra

# 找到唯一的下降交点，再用相邻温度点线性插值。
function rbim_crossing(Ts, U, target)
    all(isfinite, U) || throw(ArgumentError("nonfinite Binder data; increase MCS"))

    crossings = findall(i -> U[i] >= target && U[i + 1] < target, 1:(length(Ts) - 1))
    length(crossings) == 1 || throw(ArgumentError(
        "expected one Binder crossing, found $(length(crossings)); widen temperatures or increase sampling"))

    i = only(crossings)
    fraction = (U[i] - target) / (U[i] - U[i + 1])
    return Ts[i] + fraction * (Ts[i + 1] - Ts[i])
end

function rbim_curve(L, Ts, disorder, rng, mcs, thermalization, binsize)
    U = Matrix{Float64}(undef, length(disorder), length(Ts))

    # 同一无序样本在所有温度下使用相同的键耦合。
    for (sample, Js) in enumerate(disorder), (j, T) in enumerate(Ts)
        param = Parameter(
            "Model" => Ising, "Lattice" => "square lattice", "L" => L,
            # The misspelling is the package's actual parameter name (v1.2.2).
            "Use Indicies as Bond Types" => true,
            "T" => T, "J" => Js, "Update Method" => SW_update!,
            # Simple estimator handles J=0 and avoids the old improved energy
            # estimator's 0/0. The package still jackknifes the moment ratio.
            "Estimator" => simple_estimator,
            "MCS" => mcs, "Thermalization" => thermalization,
            "Binning Size" => binsize, "Seed" => rand(rng, UInt32),
        )

        result = runMC(param)
        U[sample, j] = 1 - mean(result["Binder Ratio"]) / 3
    end

    return U
end

"""
    critical_temp(p, r; kwargs...) -> Float64

Estimate the thermodynamic ferromagnetic Tc of the square-lattice random-bond
Ising model, with independent quenched bonds J=1 (probability p) or J=r.
Uses periodic boundaries, spins ±1, kB=1, and SpinMonteCarlo Swendsen–Wang MC.
Requires 0≤p≤1 and r≥0. Negative r requires a different, frustration-aware
algorithm and order parameter. For r=0 and p≤1/2, returns the known Tc=0.

For each L, measure U(T,L) = [1 - <m⁴>/(3<m²>²)]disorder and interpolate
T_L at U=binder_target. Extrapolate T_L = Tc + a*x(L), where
x(L)=L^(-1/nu)*log(L)^log_power. Default nu=1, log_power=0 is leading-order
Ising scaling. This is a finite-size estimate: random bonds have logarithmic
corrections, so vary Lmin, binder_target and log_power (e.g. 0.5) to assess
systematics, particularly near dilution/percolation or strong disorder.
Reference: https://arxiv.org/abs/0804.2788 .

Keywords:
- `sizes=[8,12,16,24,32]`, at least four distinct sizes ≥4.
- `samples=32`: independent disorder realizations AND MC streams per size.
  Each realization is held fixed across temperature; even pure models use
  independent chains to estimate thermal sampling error.
- `mcs=4096`, `thermalization=1024`, `binsize=64`: measurement, discarded
  updates, and jackknife block size. Increase them to check equilibration
  and autocorrelation. The routine does not certify their convergence.
- `temperatures=nothing`: coarse grid, default 21 points from 0.02Jmax to
  3Jmax. The observed bracket is refined with `refine_points=11` temperatures.
- `binder_target=0.5`, `nu=1.0`, `log_power=0.0`: fixed phenomenological
  coupling and chosen scaling form. Exponents are supplied, not inferred.
- `bootstrap=300`, `seed=1234`, `verbose=true`.
- `details=false`: when true, return Tc, statistical CI, fit residuals,
  dropped-smallest-size estimate, T_L, and raw per-sample Binder curves.
  Bootstrap resamples whole disorder curves; failed crossings are counted.
  CI excludes temperature interpolation and scaling-form systematics.

Example: `Tc = critical_temp(0.5, 0.5)`.
Run `rbim_selfcheck()` for a small reproducible MC and scaling check.
"""
function critical_temp(p::Real, r::Real;
        sizes=[8, 12, 16, 24, 32], samples::Int=32,
        mcs::Int=4096, thermalization::Int=1024, binsize::Int=64,
        temperatures=nothing, refine_points::Int=11, binder_target::Real=0.5,
        nu::Real=1.0, log_power::Real=0.0, bootstrap::Int=300,
        seed::Integer=1234, verbose::Bool=true, details::Bool=false)

    isfinite(p) && 0 <= p <= 1 || throw(ArgumentError("p must lie in [0,1]"))
    isfinite(r) && r >= 0 || throw(ArgumentError("r must be finite and nonnegative"))

    Ls = sort(unique(collect(sizes)))
    length(Ls) >= 4 && all(L -> L isa Integer && L >= 4, Ls) ||
        throw(ArgumentError("sizes must contain at least four distinct integers ≥4"))

    samples >= 4 || throw(ArgumentError("samples must be ≥4"))
    binsize > 0 && mcs >= 8binsize && mcs % binsize == 0 ||
        throw(ArgumentError("MCS must contain at least eight complete bins"))
    thermalization >= 0 || throw(ArgumentError("thermalization must be nonnegative"))
    refine_points >= 5 && bootstrap >= 20 ||
        throw(ArgumentError("refine_points must be ≥5 and bootstrap ≥20"))

    isfinite(binder_target) && 0 < binder_target < 2/3 ||
        throw(ArgumentError("binder_target must lie in (0,2/3)"))
    isfinite(nu) && nu > 0 && isfinite(log_power) ||
        throw(ArgumentError("nu must be positive and log_power finite"))

    # 稀释方格模型在键渗流阈值及以下没有有限温度铁磁相变。
    if r == 0 && p <= 0.5
        return details ? (Tc=0.0, ci=(0.0, 0.0), method=:percolation_limit) : 0.0
    end

    Jmax = p == 1 ? 1.0 : p == 0 ? Float64(r) : max(1.0, Float64(r))
    Ts = temperatures === nothing ? collect(range(0.02Jmax, 3Jmax; length=21)) :
        Float64.(collect(temperatures))
    length(Ts) >= 5 && all(t -> isfinite(t) && t > 0, Ts) && all(diff(Ts) .> 0) ||
        throw(ArgumentError("temperatures must contain ≥5 strictly increasing positive finite values"))

    rng = MersenneTwister(seed)
    grids = Vector{Float64}[]
    curves = Matrix{Float64}[]
    TL = Float64[]

    for L in Ls
        # Exactly 2L² undirected nearest-neighbor bonds on a square torus.
        disorder = [ifelse.(rand(rng, 2L^2) .< p, 1.0, Float64(r)) for _ in 1:samples]

        # 先粗扫定位交点，再在附近加密温度网格。
        coarse = rbim_curve(L, Ts, disorder, rng, mcs, thermalization, binsize)
        center = rbim_crossing(Ts, vec(mean(coarse; dims=1)), binder_target)
        j = searchsortedlast(Ts, center)
        fine = collect(range(
            Ts[max(1, j - 1)], Ts[min(length(Ts), j + 2)]; length=refine_points))

        U = rbim_curve(L, fine, disorder, rng, mcs, thermalization, binsize)
        push!(grids, fine)
        push!(curves, U)
        push!(TL, rbim_crossing(fine, vec(mean(U; dims=1)), binder_target))

        verbose && @info "Random-bond Ising" L T_L=last(TL)
    end

    # ponytail: leading two-parameter FSS; increase sizes and compare correction
    # choices before using this estimate for strong-disorder precision work.
    x = Float64.(Ls) .^ (-1 / nu) .* log.(Ls) .^ log_power
    X = hcat(ones(length(Ls)), x)
    rank(X) == 2 || throw(ArgumentError("degenerate size scaling coordinates"))

    Tc, slope = X \ TL
    isfinite(Tc) && 0 < Tc < last(Ts) || error(
        "Tc extrapolation lies outside the sampled range; increase sizes/sampling and check scaling")

    # 整条曲线重采样，保留同一无序样本在不同温度间的关联。
    draws = Float64[]
    failed = 0

    for _ in 1:bootstrap
        try
            Tb = map(eachindex(Ls)) do i
                indices = rand(rng, 1:samples, samples)
                U = vec(mean(curves[i][indices, :]; dims=1))
                rbim_crossing(grids[i], U, binder_target)
            end
            push!(draws, (X \ Tb)[1])
        catch err
            err isa ArgumentError || rethrow()
            failed += 1
        end
    end

    failed <= bootstrap ÷ 5 || error(
        "$failed/$bootstrap bootstrap fits failed; increase samples/MCS or widen the temperature grid")

    ci = Tuple(quantile(draws, [0.025, 0.975]))
    dropped = (X[2:end, :] \ TL[2:end])[1]

    failed > 0 && @warn "Bootstrap crossing failures" failed bootstrap
    abs(dropped - Tc) > (ci[2] - ci[1]) / 2 &&
        @warn "Size-cut drift exceeds the statistical CI half-width; increase sizes" Tc dropped
    verbose && @info "Tc estimate (statistical CI; check FSS systematics separately)" Tc ci dropped

    details || return Tc

    return (Tc=Tc, ci=ci, stderr=std(draws), sizes=Ls, pseudocritical=TL,
        slope=slope, residuals=TL - X * [Tc, slope], dropped_smallest=dropped,
        failed_resamples=failed, bootstrap=bootstrap, temperatures=grids, binder=curves,
        p=p, r=r, seed=seed, samples=samples, mcs=mcs, thermalization=thermalization,
        binsize=binsize, binder_target=binder_target, nu=nu, log_power=log_power,
        version=pkgversion(SpinMonteCarlo))
end

"""Small deterministic geometry, coupling, MC and scaling checks (no production scan)."""
function rbim_selfcheck()
    L = 4
    param = Parameter("Lattice" => "square lattice", "L" => L,
        "Use Indicies as Bond Types" => true)
    lat = generatelattice(param)
    @assert numsites(lat) == L^2 && numbonds(lat) == 2L^2
    @assert sort(bondtype.(bonds(lat))) == collect(1:2L^2)
    @assert all(s -> length(collect(neighbors(s))) == 4, sites(lat))

    model = Ising(lat, MersenneTwister(7))
    model.spins .= 1
    Js = repeat([1.0, 0.25], L^2)
    @assert simple_estimator(model, 2.0, Js)["Energy"] == -sum(Js)/L^2

    U = rbim_curve(L, [0.3, 8.0], [ones(2L^2)], MersenneTwister(1), 1024, 256, 32)
    @assert U[1, 1] > 0.64 && U[1, 2] < 0.3

    # Synthetic fixed-Binder temperatures with an exactly known intercept.
    ls = [8, 12, 16, 24]
    X = hcat(ones(4), 1 ./ ls)
    @assert isapprox((X \ (2.3 .+ 0.4 ./ ls))[1], 2.3; atol=1e-12)
    @assert rbim_crossing([1., 2., 3.], [0.65, 0.5, 0.1], 0.5) == 2.0
    @assert critical_temp(0.4, 0.0) == 0.0

    return true
end
