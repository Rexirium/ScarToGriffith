using CairoMakie, HDF5, Statistics
include(joinpath(@__DIR__, "fit_autocorrelation.jl"))

# Run: julia --project=@v1.13 random_tfim/plot_results.jl [input.h5] [output_dir]
# Stored statistics include the spatial average over all starting sites.
function read_plot_data(input)
    h5open(input, "r") do f
        attrs = HDF5.attributes(f)
        @assert read(attrs["complete"]) "Input ensemble is incomplete"
        sizes = read(f["parameters/sizes"])
        fields = read(f["parameters/fields"])
        sample_fields = read(f["parameters/sample_fields"])
        n = read(attrs["nsamples"])

        records = []
        for L in sizes, h0 in sample_fields
            g = f["L$(L)/h$(h0)"]
            resolved = Bool.(read(g["gap_resolved"]))
            gaps = map((gap, ok) -> ok ? log(gap) : NaN, read(g["gap_samples"]), resolved)
            avg = read(g["correlation_mean"])
            sem = read(g["correlation_sem"])
            logavg = read(g["log_correlation_mean"])
            logsem = read(g["log_correlation_sem"])

            @assert length(avg) == length(sem) == length(logavg) == length(logsem) == L ÷ 2 + 1
            @assert length(resolved) == length(gaps) == n
            @assert all(isfinite, avg) && avg[1] == 1 && logavg[1] == 0
            @assert resolved == isfinite.(gaps)
            push!(records, (; L, h0, gaps, unresolved=count(!, resolved), avg, sem, logavg, logsem))
        end
        field_distribution = read(attrs["field_distribution"])
        field_distribution in ("uniform", "fixed") || error("Unknown field distribution: $field_distribution")
        if field_distribution == "fixed"
            @assert read(attrs["fixed_field_divisor"]) == exp(1) "Expected h = h₀/e"
        end
        (; records, sizes, fields, sample_fields, n, field_distribution, boundary=read(attrs["boundary"]))
    end
end

const PLOT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000"]
const DISTRIBUTION_TIMES = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

field_distribution_label(data) = data.field_distribution == "fixed" ?
    "h = h₀/e" : "h ∼ U(0, h₀)"

field_label(h) = "h₀ = $(round(h; sigdigits=5))"
panel_position(k, ncols) = (div(k - 1, ncols) + 1, (k - 1) % ncols + 1)
plot_record(data, L, h0) = only(d for d in data.records if d.L == L && d.h0 == h0)

# Mask invalid points without connecting curves across missing data. On log axes,
# omit intervals crossing zero instead of inventing a positive uncertainty bound.
function uncertainty_values(y, sem; positive=false)
    @assert length(y) == length(sem)
    center = Float64.(y)
    valid = isfinite.(center) .& (.!positive .| (center .> 0))
    center[.!valid] .= NaN

    lower, upper = center .- sem, center .+ sem
    valid_band = valid .& isfinite.(sem) .& (sem .>= 0) .&
        isfinite.(lower) .& isfinite.(upper) .& (.!positive .| (lower .> 0))
    lower[.!valid_band] .= NaN
    upper[.!valid_band] .= NaN
    (; center, lower, upper)
end

function uncertainty_curve!(ax, x, y, sem, color; positive=false)
    values = uncertainty_values(y, sem; positive)
    band!(ax, x, values.lower, values.upper; color=(color, 0.20))
    lines!(ax, x, values.center; color, linewidth=2)
end

function finish_figure!(fig, labels, ncols, note; nrows=2)
    Legend(fig[nrows + 1, 1:ncols],
        [LineElement(color=PLOT_COLORS[k], linewidth=2) for k in eachindex(labels)],
        labels; orientation=:horizontal, framevisible=false)
    Label(fig[nrows + 2, 1:ncols], note; fontsize=13)
end

function gap_density(gaps, edges)
    counts = zeros(Int, length(edges) - 1)
    for x in gaps
        isfinite(x) || continue
        @assert first(edges) <= x <= last(edges)
        counts[clamp(searchsortedlast(edges, x), 1, length(counts))] += 1
    end
    # 未解析的样本仍计入分母，保留缺失的概率质量。
    density = counts ./ (length(gaps) .* diff(edges))
    @assert isapprox(sum(density .* diff(edges)), count(isfinite, gaps) / length(gaps))
    density[counts .== 0] .= NaN
    density
end

function plot_gap_distributions(data, fields, sizes; scaled=false)
    fig = Figure(size=(1150, 900))
    Label(fig[0, 1:2], scaled ? "Scaled log energy-gap distributions" : "Log energy-gap distributions", fontsize=25)
    for (k, h0) in enumerate(fields)
        row, col = panel_position(k, 2)
        ax = Axis(fig[row, col]; title=field_label(h0),
            xlabel=scaled ? L"\ln\Delta E / L^{1/2}" : L"\ln\Delta E",
            ylabel=scaled ? L"P(\ln\Delta E / L^{1/2})" : L"P(\ln\Delta E)", yscale=log10)
        records = [plot_record(data, L, h0) for L in sizes]

        # 先变换每个样本，再为同一子图的所有 L 使用统一分箱。
        samples = [scaled ? d.gaps ./ sqrt(d.L) : d.gaps for d in records]
        finite_gaps = [x for gaps in samples for x in gaps if isfinite(x)]
        isempty(finite_gaps) && error("No resolved gaps at h0=$h0")
        width = scaled ? 0.1 : 1.0
        lo = floor(minimum(finite_gaps) / width) * width
        hi = max(lo + width, ceil(maximum(finite_gaps) / width) * width)
        edges = collect(range(lo, hi; length=round(Int, (hi - lo) / width) + 1))
        centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

        missing = String[]
        for (j, d) in enumerate(records)
            scatterlines!(ax, centers, gap_density(samples[j], edges); color=PLOT_COLORS[j])
            d.unresolved > 0 && push!(missing, "L=$(d.L): $(d.unresolved)/$(data.n)")
        end
        if !isempty(missing)
            text!(ax, 0.03, 0.98; text="Unresolved\n" * join(missing, "\n"),
                space=:relative, align=(:left, :top), fontsize=12)
        end
    end
    finish_figure!(fig, ["L = $L" for L in sizes], 2,
        "$(basename(data.input)) | $(field_distribution_label(data)) | " * (scaled ? "ln-gap/√L samples; common bin width = 0.1" : "ln-gap bin width = 1") *
        "; density normalized by all samples")
    fig
end

function plot_spatial_correlations(data, fields, sizes; logarithmic=false)
    fig = Figure(size=(1500, 900))
    title = logarithmic ? "Mean log spatial correlations" : "Mean spatial correlations"
    Label(fig[0, 1:3], title, fontsize=25)
    for (k, h0) in enumerate(fields)
        row, col = panel_position(k, 3)
        scale = logarithmic ? identity : log10
        ax = Axis(fig[row, col]; title=field_label(h0),
            xlabel=logarithmic ? L"\sqrt{r}" : L"r",
            ylabel=logarithmic ? L"[\ln C(r)]_{\mathrm{av}}" : L"C_{\mathrm{av}}(r)",
            xscale=scale, yscale=scale)

        for (j, L) in enumerate(sizes)
            d = plot_record(data, L, h0)
            r = collect(0:div(L, 2))
            if logarithmic
                uncertainty_curve!(ax, sqrt.(r), d.logavg, d.logsem, PLOT_COLORS[j])
            else
                uncertainty_curve!(ax, r[2:end], d.avg[2:end], d.sem[2:end],
                    PLOT_COLORS[j]; positive=true)
            end
        end
    end
    finish_figure!(fig, ["L = $L" for L in sizes], 3,
        "$(basename(data.input)) | $(field_distribution_label(data)) | Shading: ±1 disorder SEM; invalid points/bounds omitted")
    fig
end

function plot_autocorrelation(data, fields, sizes; real_time=false)
    fig = Figure(size=(1150, 900))
    Label(fig[0, 1:2], real_time ? "Real-time autocorrelations" : "Imaginary-time autocorrelations", fontsize=25)
    time_key = real_time ? "real_time" : "imaginary_time"
    prefix = real_time ? "real_" : ""

    h5open(data.input, "r") do f
        for (j, L) in enumerate(sizes)
            row, col = panel_position(j, 2)
            ax = Axis(fig[row, col]; title="L = $L", xlabel=real_time ? L"t" : L"\tau",
                ylabel=real_time ? L"C_{\mathrm{av}}(t)" : L"C_{\mathrm{av}}(\tau)",
                xscale=log10, yscale=real_time ? identity : log10,
                # Gapped tails reach subnormal values; log padding can underflow to zero.
                yautolimitmargin=real_time ? (0.05, 0.05) : (0.0, 0.0))
            below_floor = false
            for (k, h0) in enumerate(fields)
                g = f["L$(L)/h$(h0)"]
                times = read(g[time_key])
                y = read(g[prefix * "autocorrelation_mean"])
                sem = read(g[prefix * "autocorrelation_sem"])
                @assert length(times) == length(y) == length(sem)
                keep = isfinite.(times) .& (times .> 0)
                @assert any(keep) "No positive times for L=$L, h0=$h0"
                below_floor |= any(v -> isfinite(v) && 0 < v < 1e-8, y[keep])
                uncertainty_curve!(ax, times[keep], y[keep], sem[keep], PLOT_COLORS[k]; positive=!real_time)
            end
            !real_time && below_floor && ylims!(ax, 1e-8, nothing)
        end
    end
    finish_figure!(fig, field_label.(fields), 2,
        "$(basename(data.input)) | $(field_distribution_label(data)) | Shading: ±1 disorder SEM; positive times only" *
        (real_time ? "; linear vertical axis" : "; stored mean; nonpositive/invalid values omitted");
        nrows=cld(length(sizes), 2))
    fig
end

function plot_autocorrelation_slopes(data, sizes)
    fig = Figure(size=(1000, 700))
    Label(fig[0, 1], "Imaginary-time autocorrelation slopes", fontsize=25)
    ax = Axis(fig[1, 1]; xlabel=L"h_0", ylabel="Effective log-log slope",
        xscale=log10, yscale=identity)
    fields = sort(filter(>=(1.0), data.fields))

    h5open(data.input, "r") do f
        for (j, L) in enumerate(sizes)
            slopes = map(fields) do h0
                g = f["L$(L)/h$(h0)"]
                autocorrelation_log_slope(read(g["imaginary_time"]), read(g["autocorrelation_mean"]))
            end
            lines!(ax, fields, slopes; color=PLOT_COLORS[j], label="L = $L")
        end
    end

    xlims!(ax, 1.0, maximum(fields))
    axislegend(ax; position=:lt, framevisible=false)
    Label(fig[2, 1], "$(basename(data.input)) | $(field_distribution_label(data)) | $(length(fields)) fields; log-log fit of stored mean over all valid positive times";
        fontsize=13)
    fig
end

function read_autocorrelation_distributions(data, fields, L)
    h5open(data.input, "r") do f
        map(fields) do h0
            g = f["L$(L)/h$(h0)"]
            grid = read(g["imaginary_time"])
            # 选取指定六个时刻的最近网格点，拟合和图例均使用实际时间。
            indices = [argmin(abs.(grid .- t)) for t in DISTRIBUTION_TIMES]
            times = grid[indices]
            C = read(g["sample_Ct"])
            @assert size(C, 1) == length(grid)
            # 直接对每个正样本取负对数；非正值和非有限值不能取对数。
            samples = [map(c -> isfinite(c) && c > 0 ? -log(c) : NaN, C[i, :]) for i in indices]
            (; L, h0, times, samples, fit=fit_time_collapse(times, samples))
        end
    end
end

function plot_autocorrelation_distributions(data, panels; scaled=false)
    fig = Figure(size=(1200, 1000))
    title = scaled ? "Rescaled imaginary-time distributions" : "Negative-log imaginary-time distributions"
    Label(fig[0, 1:2], "$title (L = $(first(panels).L))"; fontsize=25)
    for (k, d) in enumerate(panels)
        row, col = panel_position(k, 2)
        title = field_label(d.h0)
        scaled && (title *= isfinite(d.fit.mu) ? " | μ = $(round(d.fit.mu; digits=4))" : " | μ unavailable")
        ax = Axis(fig[row, col]; title,
            xlabel=scaled ? L"x = -\ln C(\tau) / \tau^{\mu}" : L"-\ln C(\tau)",
            ylabel=scaled ? L"P(x)" : L"P(-\ln C(\tau))", yscale=log10,
            yautolimitmargin=(0.05, 0.15))

        missing = String[]
        for (j, (t, y)) in enumerate(zip(d.times, d.samples))
            # 直接对变换后的样本重新统计密度；每条曲线用 60 个等宽箱。
            values = scaled ? y ./ t^d.fit.mu : y
            nmiss = count(!isfinite, y)
            nmiss > 0 && push!(missing, "τ=$(round(t; sigdigits=3)): $nmiss/$(length(y))")
            scaled && !isfinite(d.fit.mu) && continue
            any(isfinite, values) || continue
            lo, hi = extrema(filter(isfinite, values))
            hi == lo && (hi = lo + 1.0)
            edges = range(lo, hi; length=61)
            centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
            scatterlines!(ax, centers, gap_density(values, edges); color=PLOT_COLORS[j],
                markersize=5)
        end
        if !isempty(missing)
            text!(ax, 0.98, 0.98; text="Invalid C\n" * join(missing, "\n"),
                space=:relative, align=(:right, :top), fontsize=11)
        end
    end
    # 同一时间网格共用图例；禁止把不同时间误标为同一条曲线。
    @assert all(d.times == first(panels).times for d in panels)
    finish_figure!(fig, ["τ = $(round(t; sigdigits=4))" for t in first(panels).times], 2,
        "$(basename(data.input)) | $(field_distribution_label(data)) | Use C; density / all samples; nonpositive/nonfinite C omitted" *
        (scaled ? "\nμ: least-squares matching of log quantiles (5–95%) of valid samples" :
            "\n60 equal-width bins per curve; nearest grid points to τ = 1, 3, 10, 30, 100, 300"))
    fig
end

function main(args)
    input = if isempty(args)
        candidates = filter(readdir(joinpath(@__DIR__, "results"); join=true)) do path
            occursin(r"^full_(?:uniform|fixed)_\d{8}_\d{6}\.h5$", basename(path))
        end
        isempty(candidates) && error("No full ensemble found; pass an input HDF5 path")
        candidates[argmax(mtime.(candidates))]
    else
        abspath(args[1])
    end
    out = length(args) < 2 ? joinpath(dirname(input), "figures") : abspath(args[2])
    data = (; read_plot_data(input)..., input)

    sizes = sort(data.sizes)
    length(sizes) in (2, 4) || error("Expected demo or full sizes; got $sizes")
    fields = sort(data.sample_fields)
    length(fields) == 6 || error("Six saved sample fields are required; got $fields")
    # Writer's sorted uniform/fixed sample fields put the critical point third.
    gap_fields = fields[3:6]

    mkpath(out)
    set_theme!(Theme(fontsize=17, linewidth=2, Axis=(xgridvisible=false, ygridvisible=false,)))
    figures = [
        ("gap_distribution", plot_gap_distributions(data, gap_fields, sizes)),
        ("scaled_gap_distribution", plot_gap_distributions(data, gap_fields, sizes; scaled=true)),
        ("average_correlation", plot_spatial_correlations(data, fields, sizes)),
        ("log_correlation_sqrt_r", plot_spatial_correlations(data, fields, sizes; logarithmic=true)),
        ("imaginary_time_autocorrelation", plot_autocorrelation(data, fields, sizes)),
        ("imaginary_time_autocorrelation_slopes", plot_autocorrelation_slopes(data, sizes)),
        ("real_time_autocorrelation", plot_autocorrelation(data, fields, sizes; real_time=true))]

    distributions = read_autocorrelation_distributions(data, gap_fields, maximum(sizes))
    push!(figures,
        ("imaginary_time_log_distribution", plot_autocorrelation_distributions(data, distributions)),
        ("imaginary_time_rescaled_distribution", plot_autocorrelation_distributions(data, distributions; scaled=true)))

    for (name, fig) in figures
        path = joinpath(out, "$name.png")
        save(path, fig; px_per_unit=1.5)
        println("Saved ", path)
    end

    report = ["# Plot audit", "", "Input: `$input`", "Boundary: $(data.boundary)",
        "Field distribution: $(data.field_distribution); $(field_distribution_label(data)); critical h0=$(1.0)",
        "Julia: $VERSION; CairoMakie: $(pkgversion(CairoMakie)); HDF5: $(pkgversion(HDF5))",
        "Sizes: $sizes", "Gap fields (2×2): $gap_fields", "Correlation fields (2×3): $fields",
        "Autocorrelation: $(cld(length(sizes), 2))×2 size panels, the same six fields, from the same input file.", "",
        """
        Real-time autocorrelation: included; linear vertical axis preserves negative values.
        Natural logarithms. Shading = ±1 SEM across independent disorder samples, not a confidence interval.
        Gap density = counts / (all samples × bin width); unresolved mass is not renormalized away.
        Scaled gap density: transform each sample to x = ln(ΔE)/√L, then histogram with common bin width 0.1 for every L; normalize by all samples.
        Imaginary-time mean curves use stored autocorrelation_mean directly, retaining the stored SEM. Real-time curves keep their signs.
        Autocorrelation effective log-log slopes: one curve per L, all $(count(>=(1.0), data.fields)) fields with h0 >= $(1.0); absolute unweighted OLS slope of ln(autocorrelation_mean) versus ln(time), with intercept, over all finite positive times and positive finite means.
        Slopes are descriptive, not automatically 1/z.$(data.field_distribution == "fixed" ? " Fixed fields are gapped for h0 > $(exp(1))." : " Uniform fields retain a Griffiths region for all finite h0 > 1.")
        Mean log spatial correlations are mean(log C), not log(mean C). No resampling.
        Time distributions: L=$(maximum(sizes)), fields=$gap_fields, target times=$DISTRIBUTION_TIMES, actual nearest times=$(first(distributions).times); 60 equal-width bins per curve, normalized by all samples.
        Collapse minimizes sum over times and quantiles of [ln Q_p(-ln C) - μ ln τ - a_p]^2, with independent intercept a_p and p=0.05:0.05:0.95; only positive quantiles of valid samples enter the fit.
        Times with no valid samples are excluded from collapse fitting; fewer than two valid times gives unavailable (NaN) μ. Empty densities are omitted, with missing counts retained.
        Time distributions use each sample directly in the negative logarithm. Nonpositive/nonfinite samples are omitted; missing mass remains in the density. μ and log-quantile RMS are descriptive fits, not an asymptotic exponent determination.
        Nonfinite means break curves. Invalid SEM/bounds omit shading; log axes also omit nonpositive values/bounds.
        Zero distance is omitted on log-log spatial axes; zero time is omitted on log-log time axes.

        | h0 | L | samples | unresolved gaps | nonfinite mean-log distances | negative mean distances |
        |---|---|---|---|---|---|"""]
    for d in data.records
        push!(report, "| $(d.h0) | $(d.L) | $(data.n) | $(d.unresolved) | $(count(!isfinite, d.logavg)) | $(count(<(0), d.avg)) |")
    end
    append!(report, ["", "## Time-distribution collapse", "",
        "| h0 | L | μ | log-quantile RMS before | after | invalid samples at each τ |",
        "|---|---|---|---|---|---|"])
    for d in distributions
        invalid = join(["$(round(t; sigdigits=4)): $(count(!isfinite, y))/$(length(y))" for (t, y) in zip(d.times, d.samples)], "; ")
        push!(report, "| $(d.h0) | $(d.L) | $(d.fit.mu) | $(d.fit.rms_before) | $(d.fit.rms_after) | $invalid |")
        println("Collapse h0=$(d.h0): μ=$(d.fit.mu), log-quantile RMS=$(d.fit.rms_after)")
    end
    write(joinpath(out, "plot_audit.md"), join(report, "\n") * "\n")
    println("Verified summary dimensions, r=0 normalization, and histogram probability mass.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
