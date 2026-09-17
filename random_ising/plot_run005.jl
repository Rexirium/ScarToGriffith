# Run: julia --project=@v1.13 --startup-file=no random_ising/plot_run005.jl
# Uses the existing global environment; does not activate/install project packages.
using CairoMakie, HDF5, Statistics, Printf, LaTeXStrings
using HDF5: attributes

const RESULT_DIR = joinpath(@__DIR__, "results", "run_005")
const OUT_DIR = joinpath(RESULT_DIR, "figures")
const SIZES = [10, 16, 24, 32, 40, 50]
# T=2.80 includes the high-temperature paramagnetic regime.
const MAP_T = [0.30, 0.70, 1.02, 2.80]
const DIST_T = [0.30, 0.50, 0.70, 1.02, 1.50, 2.00, 2.80]
const DIST_L = [10, 24, 40, 50]
const FIT_T = [1.2, 1.3, 1.4, 1.5, 1.6]
const COLORS = Makie.wong_colors()

mkpath(OUT_DIR)
set_theme!(Theme(fontsize=18, linewidth=2.2,
    fonts=(regular="Times New Roman", bold="Times New Roman Bold",
           italic="Times New Roman Italic", bold_italic="Times New Roman Bold Italic")))

data = Dict(L => h5open(joinpath(RESULT_DIR, "L_$L.h5"), "r") do f
    @assert read(attributes(f)["format_version"]) == 13
    @assert read(attributes(f)["ndisorder"]) == 1000
    names = filter(k -> startswith(k, "T_"), keys(f))
    sort!(names; by=k -> read(attributes(f[k])["T"]))
    @assert length(names) == 150
    rows = map(names) do name
        g = f[name]
        @assert read(attributes(g)["complete"])
        C, chi = read(g["heat_capacity"]), read(g["susceptibility"])
        local_chi = read(g["chi_local"])
        @assert length(C) == length(chi) == 1000
        @assert all(isfinite, C) && all(isfinite, chi) && all(>(0), chi)
        @assert size(local_chi) == (L, L) && all(isfinite, local_chi)
        # Different disorder samples are independent; their scatter already
        # includes MC noise. Adding the stored MC errors again double counts it.
        (; name, T=read(attributes(g)["T"]), C=mean(C), Csem=std(C)/sqrt(length(C)),
           chi=mean(chi), chisem=std(chi)/sqrt(length(chi)), samples=chi, local_chi)
    end
    @assert [r.T for r in rows] ≈ read(f["temperatures"])
    rows
end for L in SIZES)

function at_temperature(L, T)
    rows = data[L]
    i = argmin(abs.([r.T for r in rows] .- T))
    @assert isapprox(rows[i].T, T; atol=1e-10)
    rows[i]
end

function save_figure(name, fig)
    save(joinpath(OUT_DIR, name * ".png"), fig; px_per_unit=2)
    println("Saved ", name)
end

fig = Figure(size=(1260, 560))
axc = Axis(fig[1, 1], title="(a) Total heat capacity", xlabel=L"T",
    ylabel=L"\langle C\rangle_{\mathrm{dis}}")
axχ = Axis(fig[1, 2], title="(b) Total susceptibility", xlabel=L"T",
    ylabel=L"\langle \chi\rangle_{\mathrm{dis}}", yscale=log10)
for (i, L) in enumerate(SIZES)
    rows = data[L]
    ts = [r.T for r in rows]
    for (ax, key, err) in [(axc, :C, :Csem), (axχ, :chi, :chisem)]
        ys, es = getproperty.(rows, key), getproperty.(rows, err)
        @assert key != :chi || all(ys .> es)
        band!(ax, ts, ys .- es, ys .+ es; color=(COLORS[i], 0.24))
        lines!(ax, ts, ys; color=COLORS[i], label=L"L = %$L")
    end
end
xlims!(axc, 0, 3); xlims!(axχ, 0, 3)
axislegend(axc; position=:rt, nbanks=2, labelsize=15)
Label(fig[2, 1:2], "1,000 disorder realizations · shading: ±1 SEM · extensive observables", fontsize=16)
save_figure("01_mean_observables", fig)

fig = Figure(size=(1100, 950))
for (i, T) in enumerate(MAP_T)
    row, col = fldmod(i-1, 2) .+ 1
    panel = fig[row, col] = GridLayout()
    z = at_temperature(50, T).local_chi
    # Independent symmetric color limits preserve both signs and spatial detail.
    lim = maximum(abs, z)
    temperature_label = @sprintf("%.2f", T)
    ax = Axis(panel[1, 1], title=L"\mathrm{(%$(Char('a'+i-1)))}\quad L=50,\quad T=%$temperature_label",
        xlabel=L"x", ylabel=L"y", aspect=DataAspect(),
        xticks=[1, 10, 20, 30, 40, 50], yticks=[1, 10, 20, 30, 40, 50])
    hm = heatmap!(ax, 1:50, 1:50, z; colormap=:balance, colorrange=(-lim, lim))
    Colorbar(panel[1, 2], hm; label=L"\chi_i", width=18, ticklabelsize=13)
end
Label(fig[0, 1:2], "Local susceptibility · disorder realization #1", fontsize=22)
Label(fig[3, 1:2], "Same disorder at all temperatures · independent color scales · signed estimates retained", fontsize=16)
save_figure("02_local_susceptibility", fig)

fig = Figure(size=(1220, 960))
axes = Axis[]
# Estimate density of the logarithm explicitly: density integrates to one in d(log₁₀ χ).
for (i, L) in enumerate(DIST_L)
    row, col = fldmod(i-1, 2) .+ 1
    ax = Axis(fig[row, col], title=L"\mathrm{(%$(Char('a'+i-1)))}\quad L=%$L",
        xlabel=L"\log_{10}\chi", ylabel=L"p(\log_{10}\chi)")
    push!(axes, ax)
    samples = [log10.(at_temperature(L, T).samples) for T in DIST_T]
    for (j, T) in enumerate(DIST_T)
        temperature_label = @sprintf("%.2f", T)
        density!(ax, samples[j]; color=(COLORS[j], 0.15),
            strokecolor=COLORS[j], strokewidth=2, label=L"T = %$temperature_label")
    end
end
Legend(fig[0, 1:2], axes[1]; orientation=:horizontal, nbanks=1, labelsize=16)
Label(fig[3, 1:2], "1,000 disorder realizations per curve · kernel density estimates · unit-area densities", fontsize=16)
save_figure("03_susceptibility_distributions", fig)

fig = Figure(size=(1220, 960))
correlation_axes = Axis[]
for (i, L) in enumerate(DIST_L)
    row, col = fldmod(i-1, 2) .+ 1
    ax = Axis(fig[row, col], title=L"\mathrm{(%$(Char('a'+i-1)))}\quad L=%$L",
        xlabel=L"t\;[\mathrm{sweeps}]", ylabel=L"|\langle C(t)\rangle_{\mathrm{dis}}|",
        xscale=log10, yscale=log10)
    push!(correlation_axes, ax)
    h5open(joinpath(RESULT_DIR, "L_$L.h5"), "r") do f
        maxlag = read(attributes(f)["max_corr_time"])
        @assert maxlag == 4096
        for (j, T) in enumerate(DIST_T)
            g = f[at_temperature(L, T).name]
            correlation = read(g["correlation"])
            sem = read(g["errors/correlation"])
            @assert length(correlation) == length(sem) == maxlag + 1
            @assert all(isfinite, correlation) && all(isfinite, sem) && all(>=(0), sem)
            @assert isapprox(correlation[1], 1) && iszero(sem[1])
            # Take the absolute value after disorder averaging; zeros are gaps on log axes.
            lag = 1:maxlag
            ys = abs.(correlation[2:end])
            lower, upper = ys .- sem[2:end], ys .+ sem[2:end]
            # The absolute-value image of a zero-crossing SEM interval reaches zero.
            # Omit those bands rather than impose an artificial positive floor.
            invalid_band = lower .<= 0
            lower[invalid_band] .= NaN
            upper[invalid_band] .= NaN
            ys[ys .== 0] .= NaN
            temperature_label = @sprintf("%.2f", T)
            band!(ax, lag, lower, upper; color=(COLORS[j], 0.20))
            lines!(ax, lag, ys; color=COLORS[j], label=L"T=%$temperature_label")
        end
        xlims!(ax, 1, maxlag)
    end
end
linkyaxes!(correlation_axes...)
ylims!(correlation_axes[1], 10.0^(-2.5), nothing)
Legend(fig[0, 1:2], correlation_axes[1]; orientation=:horizontal, labelsize=16)
Label(fig[3, 1:2], L"C(t)=L^{-2}\sum_i s_i(t_0+t)s_i(t_0)", fontsize=18)
Label(fig[4, 1:2], "Reference state: final SW configuration", fontsize=16)
Label(fig[5, 1:2], "1,000 disorder realizations · shading: transformed ±1 SEM (zero-crossing bands omitted)", fontsize=16)
save_figure("04_autocorrelation", fig)

fig = Figure(size=(1150, 780))
axfit = Axis(fig[1, 1], title=L"L=50",
    xlabel=L"(\ln t)^2", ylabel=L"\ln\langle C(t)\rangle_{\mathrm{dis}}")
fit_rows = NamedTuple[]
h5open(joinpath(RESULT_DIR, "L_50.h5"), "r") do f
    for (j, T) in enumerate(FIT_T)
        g = f[at_temperature(50, T).name]
        correlation, sem = read(g["correlation"]), read(g["errors/correlation"])
        @assert length(correlation) == length(sem) == 4097
        @assert all(isfinite, correlation) && all(isfinite, sem) && all(>=(0), sem)
        c, e = correlation[3:end], sem[3:end]
        # Stop at the first unresolved point; do not reintroduce late noise excursions.
        valid = (0 .< c .< 1) .& (c .> 3 .* e)
        first_invalid = findfirst(.!valid)
        n = isnothing(first_invalid) ? length(c) : first_invalid - 1
        @assert n >= 3 "Too few consecutive resolved points at T=$T"
        t = 2:(n+1)
        x, y = log.(t).^2, log.(c[1:n])
        # Ordinary least squares in the transformed coordinates, with an intercept.
        dx, dy = x .- mean(x), y .- mean(y)
        slope = sum(dx .* dy) / sum(abs2, dx)
        intercept = mean(y) - slope * mean(x)
        fitted = intercept .+ slope .* x
        r2 = 1 - sum(abs2, y .- fitted) / sum(abs2, dy)
        @assert all(isfinite, (slope, intercept, r2))
        push!(fit_rows, (; L=50, T, n, tmin=first(t), tmax=last(t), slope, intercept, r2))
        label = @sprintf("T=%.1f, b=%.3f, R²=%.3f", T, slope, r2)
        lines!(axfit, x, y; color=COLORS[j], label)
        lines!(axfit, x, fitted; color=COLORS[j], linestyle=:dash, linewidth=2.5)
    end
end
Legend(fig[1, 2], axfit; labelsize=15)
Label(fig[2, 1:2], L"\ln C=a+b(\ln t)^2\quad\Longleftrightarrow\quad C=e^a\exp[b(\ln t)^2]", fontsize=18)
Label(fig[3, 1:2], "Solid: disorder mean transformed without absolute value · dashed: ordinary least-squares fit", fontsize=16)
Label(fig[4, 1:2], "From t=2 to first failure of 0<C<1 and C>3 SEM · natural logarithms", fontsize=16)
save_figure("05_autocorrelation_scaling_fit", fig)
open(joinpath(OUT_DIR, "05_autocorrelation_fits.csv"), "w") do io
    println(io, "L,T,n,tmin,tmax,slope,intercept,r2")
    for row in fit_rows
        println(io, join(values(row), ','))
    end
end

open(joinpath(OUT_DIR, "plot_notes.txt"), "w") do io
    println(io, "Source: random_ising/results/run_005; format v13; 6 sizes × 150 temperatures × 1000 disorder samples.")
    println(io, "C and chi are total (extensive) observables, not divided by L^2. chi = <M^2>/T.")
    println(io, "Figure 1: disorder mean ± sample std/sqrt(1000); chi axis logarithmic.")
    println(io, "MC noise is already present in across-realization scatter; stored MC errors are not added again.")
    println(io, "Figure 2: L=50, first disorder realization, T=$MAP_T; x is array dimension 1.")
    println(io, "Same disorder across T. Each panel has its own symmetric linear color scale.")
    println(io, "Negative finite-sampling local estimates are retained, not clipped or replaced.")
    println(io, "Figure 3: L=$DIST_L, T=$DIST_T; density of log10(chi), not density per unit chi.")
    println(io, "Figure 4: L=$DIST_L, T=$DIST_T; absolute value of the stored disorder-averaged fixed-origin spin autocorrelation on log-log axes.")
    println(io, "Absolute value is taken after disorder averaging. t=0 is omitted; exactly zero means are plotted as gaps. Shading maps the original mean ± stored SEM interval through absolute value; intervals touching or crossing zero are omitted, with no artificial positive floor.")
    println(io, "C(t)=sum_i s_i(t0+t)*s_i(t0)/L^2; reference state is the final SW configuration, followed by 4096 heat-bath sweeps. No connected subtraction or time-origin averaging.")
    println(io, "T=2.80 is included to cover the high-temperature paramagnetic regime; the remaining selected temperatures retain run_004 comparison points. T=1.02 is not assumed to be the run_005 heat-capacity maximum.")
    println(io, "A heat-capacity maximum is not asserted to be a critical temperature.")
    println(io, "Figure 5: L=50, T=$FIT_T; y=ln(C) versus x=(ln(t))^2, using the signed disorder mean without absolute value. Both transformed axes are linear. Fit: ln(C)=intercept+slope*(ln(t))^2.")
    println(io, "Plot and unweighted OLS fit use consecutive times from t=2, stopping before the first failure of 0<C<1 and C>3 SEM. Dashed lines: y=intercept+slope*x. Parameters, R², point counts and time windows: 05_autocorrelation_fits.csv.")
    println(io, "Fits are descriptive and window-dependent. Adjacent times are correlated; R² does not establish an asymptotic decay law, and independent-point parameter errors are not reported.")
end
