# julia --project=@v1.13 tfim/plot_imaginary_time.jl [output_dir] [L] [tau_min] [tau_max]
include(joinpath(@__DIR__, "clean_tfim.jl"))
include(joinpath(@__DIR__, "..", "random_tfim", "fit_autocorrelation.jl"))
using CairoMakie, LinearAlgebra, Printf

const FIELDS = [0.2, 0.6, 0.9, 1.0, 1.1, 1.4, 1.7, 2.5, 5.0, 10.0]
const COLORS = ["#0072B2", "#56B4E9", "#009E73", "#333333", "#B3A000",
    "#E69F00", "#D55E00", "#CC79A7", "#8E44AD", "#66A61E"]

function main(args)
    length(args) <= 4 || error("Usage: plot_imaginary_time.jl [out] [L] [tau_min] [tau_max]")
    out = isempty(args) ? joinpath(@__DIR__, "results") : abspath(args[1])
    L = length(args) < 2 ? 128 : parse(Int, args[2])
    tmin = length(args) < 3 ? 1e-1 : parse(Float64, args[3])
    tmax = length(args) < 4 ? 1e3 : parse(Float64, args[4])
    L >= 2 || error("L must be >= 2")
    isfinite(tmin) && isfinite(tmax) && 0 < tmin < tmax || error("Require 0 < tau_min < tau_max < Inf")
    BLAS.set_num_threads(1)
    times = vcat(0.0, exp.(range(log(tmin), log(tmax); length=241)))
    # Preserve the requested endpoints in the fit mask despite exp/log rounding.
    times[2], times[end] = tmin, tmax
    data = ["L,J,h,tau,C,boundary,offset,j"]
    fits = ["L,h,phase,model,A,decay_parameter,log_sse,log_sse_per_dof,n,tau_min,tau_max,selected"]
    set_theme!(Theme(fontsize=21, Axis=(xgridvisible=false, ygridvisible=false)))
    fig = Figure(size=(1450, 900))
    Label(fig[0, 1:2], "Figure 1: Clean TFIM imaginary-time autocorrelations"; fontsize=28)
    ax = Axis(fig[1, 1]; title="L = $L, OBC, j = $(cld(L,2)), J = 1, T = 0", xlabel=L"\tau",
        ylabel=L"C_{\mathrm{offset}}(\tau)",
        xscale=log10, yscale=log10, width=630, height=500,
        titlesize=24, xlabelsize=24, ylabelsize=24)
    labels = typeof(L"\alpha")[]
    for (k, h) in enumerate(FIELDS)
        println("Computing L=$L, h=$h ..."); flush(stdout)
        C = clean_autocorrelation(h, times; L)
        @assert all(isfinite, C) && all(-1e-12 .<= C .<= 1+1e-10)
        @assert all(diff(C) .<= 1e-10)
        append!(data, [join((L, 1.0, h, t, c, "open", true, cld(L,2)), ',') for (t, c) in zip(times, C)])
        # Lowest-mode subtraction leaves cancellation noise near zero.
        result = fit_autocorrelation(times, C; tmin, tmax, floor=1e-12)
        phase = h < 1 ? "FM" : h == 1 ? "Critical" : "PM"
        for fit in (result.power, result.exponential)
            push!(fits, join((L, h, phase, fit.model, fit.A, fit.rate, fit.chi_square,
                fit.reduced_chi_square, result.n, result.tmin, result.tmax,
                fit.model == result.best.model), ','))
        end
        lines!(ax, times[2:end], [c > 0 ? c : NaN for c in C[2:end]];
            color=COLORS[k], linewidth=2.5)
        ft = exp.(range(log(result.tmin), log(result.tmax); length=400))
        fy = decay_prediction(result.best, ft)
        lines!(ax, ft, [y > 0 ? y : NaN for y in fy];
            color=COLORS[k], linestyle=:dash, linewidth=2)
        rate = @sprintf("%.4g", result.best.rate)
        label = result.best.model == :power ?
            L"\mathrm{%$(phase)}\quad h=%$(h)\;|\;\alpha=%$(rate)" :
            L"\mathrm{%$(phase)}\quad h=%$(h)\;|\;\lambda=%$(rate)"
        push!(labels, label)
    end
    xlims!(ax, tmin, tmax)
    ylims!(ax, 1e-8, 1.2)
    Legend(fig[1, 2], [LineElement(color=c, linewidth=2.5) for c in COLORS], labels;
        framevisible=false, labelsize=20, tellheight=false)
    footer = fig[2, 1:2] = GridLayout()
    Label(footer[1, 1], "Solid: OBC midpoint, offset=true; dashed: best descriptive fit"; fontsize=18)
    Label(footer[2, 1],
        "Power: A τ^(−α); exponential: A exp(−λτ); equal weights in log C";
        fontsize=18)
    Label(footer[3, 1], "Fit window: τ ∈ [$tmin, $tmax]; C_offset ≥ 10⁻¹²; plot floor: 10⁻⁸"; fontsize=18)
    Label(footer[4, 1], "Lowest odd-state contribution removed; C_offset(0) = 1 − transition weight"; fontsize=18)
    rowgap!(footer, 4)
    mkpath(out)
    write(joinpath(out, "imaginary_time_data.csv"), join(data, '\n') * "\n")
    write(joinpath(out, "imaginary_time_fits.csv"), join(fits, '\n') * "\n")
    save(joinpath(out, "figure1_imaginary_time_autocorrelation.png"), fig; px_per_unit=1.5)
    println("Saved clean-chain data, fits and Figure 1 to $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
