# julia --project=@v1.13 random_tfim/plot_imaginary_time.jl [Lmax256_fixed_YYYYMMDD_HHMMSS.h5] [output_dir] [tau_min] [tau_max]
# Default fit and plot window: 1 <= tau <= 1e3.
include(joinpath(@__DIR__, "plot_results.jl"))
using Printf

const IMAGINARY_COLORS = ["#0072B2", "#56B4E9", "#009E73", "#66A61E", "#B3A000",
    "#E69F00", "#D55E00", "#CC79A7", "#8E44AD", "#333333"]

function select_imaginary_fields(fields)
    selections = []
    for (phase, targets, allowed) in (
        ("FM", [0.2, 0.6], h -> 0 < h < 1),
        ("Griffiths", [1.1, 1.4, 1.7, 2.1, 2.5], h -> 1 < h < exp(1)),
        ("PM", [3.0, 5.0, 10.0], h -> h > exp(1)))
        available = sort(filter(allowed, fields))
        length(available) >= length(targets) || error("Insufficient $phase fields")
        selected = [available[argmin(abs.(available .- target))] for target in targets]
        allunique(selected) || error("Requested representative fields map to duplicate $phase points")
        append!(selections, [(; h0, phase) for h0 in selected])
    end
    selections
end

function plot_imaginary_field_scan(input, sizes; tmin=1.0, tmax=1e3)
    comparison_size = maximum(sizes)
    records = []
    fields, offset = h5open(input, "r") do f
        fields = sort(filter(>=(1.0), read(f["parameters/fields"])))
        isempty(fields) && error("No fields h0 >= 1 for Figure 2")
        for L in sizes, h0 in fields
            g = f["L$(L)/h$(h0)"]
            (; t, y) = read_imaginary_autocorrelation(g)
            result = fit_autocorrelation(t, y; tmin, tmax)
            push!(records, (; L, h0, result))
        end
        fields, read(HDF5.attributes(f)["offset"])
    end
    fig = Figure(size=(1600, 1100))
    Label(fig[0, 1:2], "Figure 2: Imaginary-time fit parameters versus field"; fontsize=28)
    titles = ["Power-law fit", "Exponential fit", "Best-model reduced χ²"]
    ylabels = [L"1/z", L"\lambda=1/\xi_{\tau}", L"\min(\chi^2_{\mathrm{red,power}},\chi^2_{\mathrm{red,exp}})"]
    axes = [Axis(fig[panel_position(k, 2)...]; title=titles[k], xlabel=L"h_0", ylabel=ylabels[k],
        xscale=log10, yscale=(k == 2 ? log10 : identity), width=560, height=360,
        titlesize=23, xlabelsize=23, ylabelsize=23, xticklabelsize=20, yticklabelsize=20) for k in 1:3]
    for (j, L) in enumerate(sizes)
        results = [r.result for r in records if r.L == L]
        values = ([r.power.rate for r in results], [r.exponential.rate for r in results],
            [r.best.reduced_chi_square for r in results])
        for k in 1:3
            plotted = k == 2 ? [v > 0 ? v : NaN for v in values[k]] : values[k]
            lines!(axes[k], fields, plotted; color=PLOT_COLORS[j], linewidth=2, label="L = $L")
        end
    end
    for ax in axes
        xlims!(ax, first(fields), last(fields))
        axislegend(ax; position=:lt, framevisible=false, labelsize=20)
    end
    comparison = Axis(fig[2, 2]; title="Model comparison (L = $comparison_size)", xlabel=L"h_0",
        ylabel=L"\chi^2_{\mathrm{red}}", xscale=log10, yscale=log10,
        width=560, height=360, titlesize=23, xlabelsize=23, ylabelsize=23,
        xticklabelsize=20, yticklabelsize=20)
    comparison_results = [r.result for r in records if r.L == comparison_size]
    power = [r.power.reduced_chi_square for r in comparison_results]
    exponential = [r.exponential.reduced_chi_square for r in comparison_results]
    lines!(comparison, fields, [v > 0 ? v : NaN for v in power]; color=PLOT_COLORS[1], linewidth=2, label="Power law")
    lines!(comparison, fields, [v > 0 ? v : NaN for v in exponential]; color=PLOT_COLORS[2], linewidth=2, label="Exponential")
    crossings = model_fit_crossings(fields, power, exponential)
    for c in crossings
        vlines!(comparison, [c.h0]; color=(:black, 0.5), linestyle=:dash, linewidth=1)
        c.reduced_chi_square > 0 || continue # Zero scores cannot be shown on a log axis.
        scatter!(comparison, [c.h0], [c.reduced_chi_square]; color=:black, marker=:xcross,
            markersize=14, label=@sprintf("Crossing h₀ ≈ %.5g", c.h0))
    end
    isempty(crossings) && text!(comparison, 0.97, 0.95; text="No crossing in scanned range",
        space=:relative, align=(:right, :top), fontsize=16)
    xlims!(comparison, first(fields), last(fields))
    axislegend(comparison; position=:rt, framevisible=false, labelsize=20)
    Label(fig[3, 1:2], "$(basename(input)) | h = h₀/e | offset=$offset | $(length(fields)) fields, h₀ ≥ 1\n" *
        "Unweighted log-space fits; all finite C > 0; τ ∈ [$tmin, $(isinf(tmax) ? "end" : tmax)]\nCrossings: linear interpolation in log(h₀), no extrapolation";
        fontsize=18)
    (; fig, records, fields, crossings, comparison_size)
end

function imaginary_time_main(args)
    length(args) <= 4 || error("Usage: plot_imaginary_time.jl [input] [out] [tau_min] [tau_max]")
    input = if isempty(args)
        paths = filter(p -> occursin(r"^Lmax256_fixed_\d{8}_\d{6}\.h5$", basename(p)),
            readdir(joinpath(@__DIR__, "results"); join=true))
        isempty(paths) && error("No Lmax256_fixed data found")
        paths[argmax(mtime.(paths))]
    else
        abspath(args[1])
    end
    out = length(args) < 2 ? joinpath(dirname(input), "figures_imaginary_time_" * splitext(basename(input))[1]) : abspath(args[2])
    tmin = length(args) < 3 ? 1.0 : parse(Float64, args[3])
    tmax = length(args) < 4 ? 1e3 : parse(Float64, args[4])
    set_theme!(Theme(fontsize=21, Axis=(xgridvisible=false, ygridvisible=false)))
    rows = String["L,h0,phase,model,A,decay_parameter,chi_square,reduced_chi_square,n,tau_min,tau_max,selected"]
    fig = Figure(size=(1400, 900))
    Label(fig[0, 1], "Figure 1: Imaginary-time autocorrelations and best fits"; fontsize=29)
    selections, sizes, n, offset = h5open(input, "r") do f
        attrs = HDF5.attributes(f)
        # Only the plotted groups are required; unrelated scan points may be absent.
        read(attrs["field_distribution"]) == "fixed" || error("Use fixed-field data (h = h0/e)")
        read(attrs["fixed_field_divisor"]) == exp(1) || error("Expected h = h0/e")
        sizes = sort(read(f["parameters/sizes"]))
        isempty(sizes) && error("No system sizes in input")
        offset = read(attrs["offset"])
        selections = select_imaginary_fields(read(f["parameters/fields"]))
        let L = maximum(sizes)
            cell = fig[1, 1] = GridLayout()
            ax = Axis(cell[1, 1]; title="L = $L", xlabel=L"\tau", ylabel=L"C_{\mathrm{av}}(\tau)",
                xscale=log10, yscale=log10, yautolimitmargin=(0, 0),
                width=560, height=460, aspect=AxisAspect(1.2),
                titlesize=24, xlabelsize=24, ylabelsize=24, xticklabelsize=21, yticklabelsize=21)
            labels = []
            plot_end = 0.0
            for (k, selection) in enumerate(selections)
                (; h0, phase) = selection
                g = f["L$(L)/h$(h0)"]
                (; t, y, sem) = read_imaginary_autocorrelation(g)
                result = fit_autocorrelation(t, y; tmin, tmax)
                valid = isfinite.(t) .& (t .> 0) .& (t .>= tmin) .& (t .<= tmax)
                plot_end = max(plot_end, maximum(t[valid]))
                uncertainty_curve!(ax, t[valid], y[valid], sem[valid], IMAGINARY_COLORS[k]; positive=true)
                ft = exp.(range(log(result.tmin), log(result.tmax); length=300))
                fy = decay_prediction(result.best, ft)
                fy[fy .<= 0] .= NaN
                lines!(ax, ft, fy; color=IMAGINARY_COLORS[k], linestyle=:dash, linewidth=2.5)
                best = result.best
                hlabel, rate_label = @sprintf("%.4g", h0), @sprintf("%.3g", best.rate)
                label = best.model == :power ?
                    L"\mathrm{%$(phase)}\ h_0=%$(hlabel)\;|\;1/z=%$(rate_label)" :
                    L"\mathrm{%$(phase)}\ h_0=%$(hlabel)\;|\;1/\xi_{\tau}=%$(rate_label)"
                push!(labels, label)
                for fit in (result.power, result.exponential)
                    push!(rows, join((L, h0, phase, fit.model, fit.A, fit.rate, fit.chi_square,
                        fit.reduced_chi_square, result.n, result.tmin, result.tmax, fit.model == best.model), ','))
                end
            end
            xlims!(ax, tmin > 0 ? tmin : nothing, isfinite(tmax) ? tmax : plot_end)
            ylims!(ax, 1e-12, 1.2)
            Legend(cell[1, 2], [LineElement(color=c, linewidth=2) for c in IMAGINARY_COLORS], labels;
                nbanks=1, labelsize=20, framevisible=false, patchsize=(28, 15),
                tellwidth=true, tellheight=false)
            colgap!(cell, 20)
        end
        selections, sizes, read(attrs["nsamples"]), offset
    end
    offset_note = offset ? "Lowest odd-parity transition subtracted" : "Full autocorrelation"
    Label(fig[2, 1], "Solid: stored mean ±1 SEM; dashed: minimum log-space SSE/(N−2) fit\nPower: A τ^(-1/z); exponential: A exp(−τ/ξτ)\n" *
        "$(basename(input)) | h = h₀/e | offset=$offset\n$offset_note | unweighted fit: all finite C > 0, τ ∈ [$tmin, $(isinf(tmax) ? "end" : tmax)]"; fontsize=18)
    mkpath(out)
    save(joinpath(out, "figure1_imaginary_time_autocorrelation.png"), fig; px_per_unit=1.5)
    write(joinpath(out, "imaginary_time_fits.csv"), join(rows, '\n') * "\n")
    scan = plot_imaginary_field_scan(input, sizes; tmin, tmax)
    save(joinpath(out, "figure2_imaginary_time_fit_parameters.png"), scan.fig; px_per_unit=1.5)
    scan_rows = ["L,h0,power_rate,exponential_rate,power_reduced_chi_square,exponential_reduced_chi_square,min_reduced_chi_square,selected_model,n,tau_min,tau_max"]
    for (; L, h0, result) in scan.records
        push!(scan_rows, join((L, h0, result.power.rate, result.exponential.rate,
            result.power.reduced_chi_square, result.exponential.reduced_chi_square,
            result.best.reduced_chi_square, result.best.model, result.n, result.tmin, result.tmax), ','))
    end
    write(joinpath(out, "imaginary_time_field_scan.csv"), join(scan_rows, '\n') * "\n")
    crossing_rows = ["L,h0,reduced_chi_square,h0_left,h0_right"]
    for c in scan.crossings
        push!(crossing_rows, join((scan.comparison_size, c.h0, c.reduced_chi_square, c.left, c.right), ','))
    end
    write(joinpath(out, "imaginary_time_model_crossings.csv"), join(crossing_rows, '\n') * "\n")
    write(joinpath(out, "imaginary_time_fit_audit.md"), """
    # Imaginary-time fits

    Input: `$input`; h=h0/e; sizes=$sizes; samples=$n; offset=$offset.
    $offset_note; fit the stored arithmetic mean without further subtraction.
    Figure 1 and imaginary_time_fits.csv contain only L=$(maximum(sizes)); Figure 2 scans all $(length(sizes)) sizes.
    Selected fields: $(join(["$(s.phase): $(s.h0)" for s in selections], "; ")).
    FM: h0<1; Griffiths: 1<h0<e; gapped PM: h0>e.
    Requested fit window: [$tmin, $(isinf(tmax) ? "end of stored time grid" : tmax)]. All finite positive times and finite positive C enter both models.
    Plot uses the same time window, with a 1e-12 vertical plot floor that does not truncate the fits. SEM only controls uncertainty shading.
    Actual windows and point counts appear in imaginary_time_fits.csv.
    Models: C=A*tau^(-alpha), C=A*exp(-lambda*tau); A>0, alpha/lambda>=0; no offset.
    Legend notation: 1/z=alpha; 1/xi_tau=lambda; C=A*exp(-tau/xi_tau).
    Fit natural-log coordinates x=ln(tau), Y=ln(C) with equal weights.
    Power: Y=logA-alpha*x; exponential: Y=logA-lambda*exp(x).
    Both models are linear in logA and the decay rate; use unweighted linear regression with nonnegative rate.
    Minimize chi_square=sum((Y-Y_fit)^2) in log space.
    For fixed n, minimizing this also minimizes reduced_chi_square; the horizontal time grid is exact.
    Select the model with smaller reduced_chi_square=chi_square/(n-2).
    Both models use identical points and have two parameters, so this is equivalent to comparing chi_square.
    CSV records both fits and the selected flag.
    Historical CSV names chi_square/reduced_chi_square denote SSE and SSE/(n-2), equivalent to unit log errors.
    These are not measurement-error-normalized chi-squares. No time covariance or resampling is used.
    Model selection is descriptive, not a goodness-of-fit probability or asymptotic exponent claim.
    In particular FM plateaus and crossover regions need not follow either simple decay law.
    Plot uses stored means/SEM directly, with the existing uncertainty masking; zero time omitted.
    Figure 2: 2x2 panels with in-axis legends, log h0 axes; panels 1 and 3 have linear vertical axes, panels 2 and 4 have log vertical axes (nonpositive values omitted); all $(length(scan.fields)) stored fields h0>=1.
    Each curve is one size. Panel 1: power-law rate 1/z; panel 2: exponential rate 1/xi_tau.
    Both rate panels show the corresponding candidate model regardless of which model wins.
    Panel 3: min(power.reduced_chi_square, exponential.reduced_chi_square) at each (L,h0).
    Panel 4: both candidate reduced chi-squares at L=$(scan.comparison_size). Crossings: $(scan.crossings).
    Intersections are interpolated linearly in log(h0) between adjacent fields with opposite signs of the model-score difference.
    Exact equal scores on grid points are retained; no extrapolation. These are numerical model-score crossings, not phase boundaries.
    Crossing coordinates and bracketing fields are saved in imaginary_time_model_crossings.csv.
    Figure 2 uses the same unweighted fits, masks, and requested time window as Figure 1.
    All field-scan rates, both reduced chi-squares, selected models, and actual windows are in imaginary_time_field_scan.csv.
    Julia: $VERSION; CairoMakie: $(pkgversion(CairoMakie)); HDF5: $(pkgversion(HDF5)).
    """)
    println("Saved figure, both-model fit table, and audit to ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    imaginary_time_main(ARGS)
end
