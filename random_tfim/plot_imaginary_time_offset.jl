# julia --project=@v1.13 random_tfim/plot_imaginary_time_offset.jl [full_fixed.h5] [output_dir] [tau_min] [tau_max]
include(joinpath(@__DIR__, "plot_imaginary_time.jl"))
include(joinpath(@__DIR__, "fit_autocorrelation_offset.jl"))
using Printf

function plot_imaginary_offset_field_scan(input, sizes; tmin=1.0, tmax=1e3)
    128 in sizes || error("Figure 2 requires L=128 for the model comparison")
    records = []
    fields = h5open(input, "r") do f
        fields = sort(filter(>=(1.0), read(f["parameters/fields"])))
        isempty(fields) && error("No fields h0 >= 1 for Figure 2")
        for L in sizes, h0 in fields
            g = f["L$(L)/h$(h0)"]
            result = fit_autocorrelation_offset(read(g["imaginary_time"]),
                read(g["autocorrelation_mean"]); tmin, tmax)
            push!(records, (; L, h0, result))
        end
        fields
    end
    fig = Figure(size=(1600, 1100))
    Label(fig[0, 1:2], "Figure 2: Imaginary-time fit parameters versus field"; fontsize=28)
    titles = ["Power-law + B fit", "Exponential + B fit", "Best-model log SSE/(N-3)"]
    ylabels = [L"-\alpha", L"\lambda=1/\xi_{\tau}", L"\min(\mathrm{SSE}_{\log C})/(N-3)"]
    axes = [Axis(fig[panel_position(k, 2)...]; title=titles[k], xlabel=L"h_0", ylabel=ylabels[k],
        xscale=log10, yscale=identity, width=560, height=360,
        titlesize=23, xlabelsize=23, ylabelsize=23, xticklabelsize=20, yticklabelsize=20) for k in 1:3]
    for (j, L) in enumerate(sizes)
        results = [r.result for r in records if r.L == L]
        values = ([r.power.rate for r in results], [r.exponential.rate for r in results],
            [r.best.reduced_chi_square for r in results])
        for k in 1:3
            lines!(axes[k], fields, values[k]; color=PLOT_COLORS[j], linewidth=2, label="L = $L")
        end
    end
    for ax in axes
        xlims!(ax, first(fields), last(fields))
        axislegend(ax; position=:lt, framevisible=false, labelsize=20)
    end
    comparison = Axis(fig[2, 2]; title="Model comparison (L = 128)", xlabel=L"h_0",
        ylabel=L"\mathrm{SSE}_{\log C}/(N-3)", xscale=log10, yscale=log10,
        width=560, height=360, titlesize=23, xlabelsize=23, ylabelsize=23,
        xticklabelsize=20, yticklabelsize=20)
    results128 = [r.result for r in records if r.L == 128]
    power = [r.power.reduced_chi_square for r in results128]
    exponential = [r.exponential.reduced_chi_square for r in results128]
    lines!(comparison, fields, [v > 0 ? v : NaN for v in power]; color=PLOT_COLORS[1], linewidth=2, label="Power law + B")
    lines!(comparison, fields, [v > 0 ? v : NaN for v in exponential]; color=PLOT_COLORS[2], linewidth=2, label="Exponential + B")
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
    Label(fig[3, 1:2], "$(basename(input)) | h = h₀/e | $(length(fields)) fields, h₀ ≥ 1\n" *
        "Unweighted log-space fits; all finite C > 0; τ ∈ [$tmin, $tmax]\nCrossings: linear interpolation in log(h₀), no extrapolation";
        fontsize=18)
    (; fig, records, fields, crossings)
end

function imaginary_time_offset_main(args)
    length(args) <= 4 || error("Usage: plot_imaginary_time_offset.jl [input] [out] [tau_min] [tau_max]")
    input = if isempty(args)
        paths = filter(p -> occursin(r"^full_fixed_\d{8}_\d{6}\.h5$", basename(p)),
            readdir(joinpath(@__DIR__, "results"); join=true))
        isempty(paths) && error("No full_fixed data found")
        paths[argmax(mtime.(paths))]
    else
        abspath(args[1])
    end
    out = length(args) < 2 ? joinpath(dirname(input), "figures_imaginary_time_offset_fixed") : abspath(args[2])
    tmin = length(args) < 3 ? 1.0 : parse(Float64, args[3])
    tmax = length(args) < 4 ? 1e3 : parse(Float64, args[4])
    set_theme!(Theme(fontsize=21, Axis=(xgridvisible=false, ygridvisible=false)))
    rows = String["L,h0,phase,model,A,B,alpha,decay_parameter,log_sse,log_sse_per_dof,converged,iterations,n,tau_min,tau_max,selected"]
    fig = Figure(size=(1400, 900))
    Label(fig[0, 1], "Figure 1: Background-subtracted autocorrelations and best fits"; fontsize=29)
    selections, sizes, n = h5open(input, "r") do f
        attrs = HDF5.attributes(f)
        read(attrs["complete"]) || error("Incomplete ensemble")
        read(attrs["field_distribution"]) == "fixed" || error("Use full_fixed data (h = h0/e)")
        read(attrs["fixed_field_divisor"]) == exp(1) || error("Expected h = h0/e")
        sizes = sort(read(f["parameters/sizes"]))
        sizes == [16, 32, 64, 128] || error("Expected full sizes [16,32,64,128]")
        selections = select_imaginary_fields(read(f["parameters/fields"]))
        let L = 128
            cell = fig[1, 1] = GridLayout()
            ax = Axis(cell[1, 1]; title="L = $L", xlabel=L"\tau", ylabel=L"C_{\mathrm{av}}(\tau)-B_{\mathrm{best}}",
                xscale=log10, yscale=log10, yautolimitmargin=(0, 0),
                width=560, height=460, aspect=AxisAspect(1.2),
                titlesize=24, xlabelsize=24, ylabelsize=24, xticklabelsize=21, yticklabelsize=21)
            labels = []
            for (k, selection) in enumerate(selections)
                (; h0, phase) = selection
                g = f["L$(L)/h$(h0)"]
                t, y, sem = read(g["imaginary_time"]), read(g["autocorrelation_mean"]), read(g["autocorrelation_sem"])
                result = fit_autocorrelation_offset(t, y; tmin, tmax)
                valid = isfinite.(t) .& (t .> 0) .& (t .>= tmin) .& (t .<= tmax)
                uncertainty_curve!(ax, t[valid], y[valid] .- result.best.B, sem[valid], IMAGINARY_COLORS[k]; positive=true)
                ft = exp.(range(log(result.tmin), log(result.tmax); length=300))
                # Evaluate C_fit-B directly to avoid cancellation near the plateau.
                fy = decay_prediction(result.best, ft)
                fy[fy .<= 0] .= NaN
                lines!(ax, ft, fy; color=IMAGINARY_COLORS[k], linestyle=:dash, linewidth=2.5)
                best = result.best
                hlabel, rate_label = @sprintf("%.4g", h0), @sprintf("%.3g", best.rate)
                label = best.model == :power ?
                    L"\mathrm{%$(phase)}\ h_0=%$(hlabel)\;|\;-\alpha=%$(rate_label)" :
                    L"\mathrm{%$(phase)}\ h_0=%$(hlabel)\;|\;1/\xi_{\tau}=%$(rate_label)"
                push!(labels, label)
                for fit in (result.power, result.exponential)
                    push!(rows, join((L, h0, phase, fit.model, fit.A, fit.B, fit.alpha, fit.rate, fit.chi_square,
                        fit.reduced_chi_square, fit.converged, fit.iterations, result.n, result.tmin, result.tmax, fit.model == best.model), ','))
                end
            end
            xlims!(ax, tmin > 0 ? tmin : nothing, isfinite(tmax) ? tmax : nothing)
            ylims!(ax, 1e-12, 1.2)
            Legend(cell[1, 2], [LineElement(color=c, linewidth=2) for c in IMAGINARY_COLORS], labels;
                nbanks=1, labelsize=20, framevisible=false, patchsize=(28, 15),
                tellwidth=true, tellheight=false)
            colgap!(cell, 20)
        end
        selections, sizes, read(attrs["nsamples"])
    end
    Label(fig[2, 1], "Solid: stored mean − B_best ±1 SEM; dashed: best fit − B_best\nPower: A τ^α; exponential: A exp(−λτ); nonpositive values omitted on log axis\n" *
        "$(basename(input)) | h = h₀/e | unweighted fit: all finite C > 0, τ ∈ [$tmin, $tmax]"; fontsize=18)
    mkpath(out)
    save(joinpath(out, "figure1_imaginary_time_autocorrelation.png"), fig; px_per_unit=1.5)
    write(joinpath(out, "imaginary_time_fits.csv"), join(rows, '\n') * "\n")
    scan = plot_imaginary_offset_field_scan(input, sizes; tmin, tmax)
    save(joinpath(out, "figure2_imaginary_time_fit_parameters.png"), scan.fig; px_per_unit=1.5)
    scan_rows = ["L,h0,power_rate,exponential_rate,power_log_sse_per_dof,exponential_log_sse_per_dof,min_log_sse_per_dof,selected_model,n,tau_min,tau_max,power_A,power_B,alpha,exponential_A,exponential_B,power_converged,exponential_converged"]
    for (; L, h0, result) in scan.records
        push!(scan_rows, join((L, h0, result.power.rate, result.exponential.rate,
            result.power.reduced_chi_square, result.exponential.reduced_chi_square,
            result.best.reduced_chi_square, result.best.model, result.n, result.tmin, result.tmax, result.power.A, result.power.B, result.power.alpha, result.exponential.A, result.exponential.B, result.power.converged, result.exponential.converged), ','))
    end
    write(joinpath(out, "imaginary_time_field_scan.csv"), join(scan_rows, '\n') * "\n")
    crossing_rows = ["L,h0,log_sse_per_dof,h0_left,h0_right"]
    for c in scan.crossings
        push!(crossing_rows, join((128, c.h0, c.reduced_chi_square, c.left, c.right), ','))
    end
    write(joinpath(out, "imaginary_time_model_crossings.csv"), join(crossing_rows, '\n') * "\n")
    failed = [(r.L, r.h0, fit.model) for r in scan.records for fit in (r.result.power, r.result.exponential) if !fit.converged]
    write(joinpath(out, "imaginary_time_fit_audit.md"), """
    # Imaginary-time fits with a constant background

    Input: `$input`; h=h0/e; sizes=$sizes; samples=$n.
    Figure 1: L=128, representative FM/Griffiths/PM fields; Figure 2: all sizes, h0>=1.
    Figure 1 subtracts the selected model's B from the stored mean and fitted curve.
    SEM is unchanged and excludes uncertainty in fitted B; nonpositive shifted values are masked on the log axis.
    Background subtraction is for display only; fitting and model selection still use the original C.
    Models: C=A*tau^alpha+B, C=A*exp(-lambda*tau)+B.
    Constraints: A>0, B>=0, alpha<=0, lambda>=0. Power rate=-alpha; exponential rate=lambda.
    Window: [$tmin, $tmax]; identical finite positive times and C for both models.
    Minimize sum((log(C)-log(C_fit))^2), retaining the original script's unweighted log-space objective.
    SEM is used only for plot shading. Tiny positive tails remain in the fit; plot floor is 1e-12.
    Compare log SSE/(N-3), with three nominal parameters for both models; smaller wins (power wins exact ties).
    This is not measurement-error-normalized chi-square, a fit probability, or a phase-boundary estimator.
    A damped Gauss-Newton solver uses analytic derivatives, scaled times and log-sum-exp predictions.
    Three positive-B starts times three rate starts and the exact B=0 boundary are compared.
    Convergence uses projected gradient or jointly stable score and parameters; CSV records the flag.
    Flat curves make A/B and the decay rate unidentifiable; N-3 is only the nominal degrees of freedom.
    No covariance correction, bootstrap uncertainty, or global-optimum guarantee is supplied.
    Figure 2 panels: -alpha, lambda, best log SSE/(N-3), and both scores at L=128.
    Model-score crossings are interpolated linearly in log(h0), without extrapolation: $(scan.crossings).
    Nonconverged scan candidates: $failed.
    CSV files contain both candidate parameters/scores and model selections.
    Julia: $VERSION; CairoMakie: $(pkgversion(CairoMakie)); HDF5: $(pkgversion(HDF5)).
    """)
    isempty(failed) || @warn "Some offset fits did not meet convergence criteria" failed
    println("Saved figure, both-model fit table, and audit to ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    imaginary_time_offset_main(ARGS)
end
