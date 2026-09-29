using CairoMakie, HDF5, Statistics

# Prefer descriptive dataset names, retaining support for existing ensembles.
read_plot_dataset(group, name, legacy) = read(group[haskey(group, name) ? name : legacy])

# Run: julia --project=@v1.13 random_tfim/plot_results.jl [input.h5] [output_dir]
# Stored statistics include the spatial average over all starting sites.
function read_plot_data(input)
    h5open(input, "r") do f
        @assert read(HDF5.attributes(f)["complete"]) "Input ensemble is incomplete"
        if haskey(f, "parameters")
            sizes = read(f["parameters/sizes"])
            fields = read(f["parameters/fields"])
        else
            # Existing ensembles predate the parameters group.
            sizes = sort([parse(Int, k[2:end]) for k in keys(f) if startswith(k, "L")])
            fields = sort([parse(Float64, k[2:end]) for k in keys(f["L$(first(sizes))"])])
        end

        records = []
        for L in sizes, h0 in fields
            g = f["L$(L)/h$(h0)"]
            has_samples = haskey(g, "gap_samples") || haskey(g, "log_gap_samples") || haskey(g, "loggaps")
            resolved = has_samples ? Bool.(read_plot_dataset(g, "gap_resolved", "resolved")) : Bool[]
            gaps = if haskey(g, "gap_samples")
                map((gap, ok) -> ok ? log(gap) : NaN, read(g["gap_samples"]), resolved)
            else
                has_samples ? read_plot_dataset(g, "log_gap_samples", "loggaps") : Float64[]
            end
            n = has_samples ? length(gaps) : read(HDF5.attributes(f)["nsamples"])
            avg = read_plot_dataset(g, "correlation_mean", "average")
            sem = read_plot_dataset(g, "correlation_sem", "sem")
            logavg = read_plot_dataset(g, "log_correlation_mean", "mean_log")
            logsem = read_plot_dataset(g, "log_correlation_sem", "log_sem")

            @assert length(avg) == length(sem) == length(logavg) == length(logsem) == L ÷ 2 + 1
            @assert length(resolved) == length(gaps)
            @assert all(isfinite, avg) && avg[1] == 1 && logavg[1] == 0
            @assert resolved == isfinite.(gaps)
            push!(records, (; L, h0, n, gaps, resolved, avg, sem, logavg, logsem,
                invalid_log=count(!isfinite, logavg), negative_C=count(<(0), avg)))
        end
        (; records, sizes, fields, boundary=read(HDF5.attributes(f)["boundary"]))
    end
end

const PLOT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000"]
const DISTRIBUTION_TIMES = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

function representative_fields(fields, targets)
    available = sort(unique(fields))
    isempty(available) && error("No fields available for targets $targets")
    selected = [available[argmin(abs.(available .- h))] for h in targets]
    length(unique(selected)) == length(targets) ||
        error("Input needs distinct representative fields for targets $targets; got $selected")
    selected
end

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

function finish_figure!(fig, labels, ncols, note)
    Legend(fig[3, 1:ncols],
        [LineElement(color=PLOT_COLORS[k], linewidth=2) for k in eachindex(labels)],
        labels; orientation=:horizontal, framevisible=false)
    Label(fig[4, 1:ncols], note; fontsize=13)
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
            nmiss = count(!, d.resolved)
            nmiss > 0 && push!(missing, "L=$(d.L): $nmiss/$(d.n)")
        end
        if !isempty(missing)
            text!(ax, 0.03, 0.98; text="Unresolved\n" * join(missing, "\n"),
                space=:relative, align=(:left, :top), fontsize=12)
        end
    end
    finish_figure!(fig, ["L = $L" for L in sizes], 2,
        "$(basename(data.input)) | " * (scaled ? "ln-gap/√L samples; common bin width = 0.1" : "ln-gap bin width = 1") *
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
        "$(basename(data.input)) | Shading: ±1 disorder SEM; invalid points/bounds omitted")
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
                ylabel=real_time ? L"C_{\mathrm{av}}(t)" : L"|C_{\mathrm{av}}(\tau)|",
                xscale=log10, yscale=real_time ? identity : log10)
            for (k, h0) in enumerate(fields)
                g = f["L$(L)/h$(h0)"]
                times = read(g[time_key])
                y = read(g[prefix * "autocorrelation_mean"])
                real_time || (y = abs.(y))
                sem = read(g[prefix * "autocorrelation_sem"])
                @assert length(times) == length(y) == length(sem)
                keep = isfinite.(times) .& (times .> 0)
                @assert any(keep) "No positive times for L=$L, h0=$h0"
                uncertainty_curve!(ax, times[keep], y[keep], sem[keep], PLOT_COLORS[k]; positive=!real_time)
            end
        end
    end
    finish_figure!(fig, field_label.(fields), 2,
        "$(basename(data.input)) | Shading: ±1 disorder SEM; positive times only" *
        (real_time ? "; linear vertical axis" : "; |stored mean|; zero/invalid values omitted"))
    fig
end

function autocorrelation_log_slope(times, y)
    @assert length(times) == length(y)
    y = abs.(y)
    keep = isfinite.(times) .& (times .> 0) .& isfinite.(y) .& (y .> 0)
    count(keep) >= 2 || error("Log-log regression needs at least two valid points")

    # 带截距的无权最小二乘；取斜率绝对值得到 1/z。
    x, z = log.(times[keep]), log.(y[keep])
    dx = x .- mean(x)
    denominator = sum(abs2, dx)
    denominator > 0 || error("Log-log regression needs distinct times")
    abs(sum(dx .* (z .- mean(z))) / denominator)
end

function plot_autocorrelation_slopes(data, sizes)
    fig = Figure(size=(1000, 700))
    Label(fig[0, 1], "Imaginary-time autocorrelation slopes", fontsize=25)
    ax = Axis(fig[1, 1]; xlabel=L"h_0", ylabel=L"1/z",
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
    Label(fig[2, 1], "$(basename(data.input)) | $(length(fields)) fields; log-log fit of |stored mean| over all valid positive times";
        fontsize=13)
    fig
end

# 用分位数匹配整条分布；对数差异避免所有曲线缩向零时产生虚假的重合。
function fit_time_collapse(times, samples)
    @assert length(times) == length(samples) >= 2
    @assert all(t -> isfinite(t) && t > 0, times)
    quantiles = hcat([quantile(filter(isfinite, y), 0.05:0.05:0.95) for y in samples]...)
    keep = vec(all(quantiles .> 0; dims=2))
    any(keep) || error("Time collapse needs positive quantiles")
    q = log.(quantiles[keep, :])
    q .-= mean(q; dims=2)
    dt = log.(times) .- mean(log.(times))
    denominator = size(q, 1) * sum(abs2, dt)
    denominator > 0 || error("Time collapse needs distinct times")
    mu = sum(q .* dt') / denominator
    (; mu, rms_before=sqrt(mean(abs2, q)), rms_after=sqrt(mean(abs2, q .- mu .* dt')))
end

function read_autocorrelation_distributions(data, fields, L)
    h5open(data.input, "r") do f
        map(fields) do h0
            g = f["L$(L)/h$(h0)"]
            haskey(g, "sample_Ct") || error("Missing sample_Ct for L=$L, h0=$h0")
            grid = read(g["imaginary_time"])
            # 选取指定六个时刻的最近网格点，拟合和图例均使用实际时间。
            times = representative_fields(filter(t -> isfinite(t) && t > 0, grid), DISTRIBUTION_TIMES)
            indices = [findfirst(==(t), grid) for t in times]
            C = read(g["sample_Ct"])
            @assert size(C, 1) == length(grid)
            # 先对每个样本取绝对值，再取负对数；零值和非有限值不能取对数。
            samples = [map(c -> isfinite(c) && abs(c) > 0 ? -log(abs(c)) : NaN, C[i, :]) for i in indices]
            all(y -> any(isfinite, y), samples) || error("No valid samples for L=$L, h0=$h0")
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
        scaled && (title *= " | μ = $(round(d.fit.mu; digits=4))")
        ax = Axis(fig[row, col]; title,
            xlabel=scaled ? L"x = -\ln |C(\tau)| / \tau^{\mu}" : L"-\ln |C(\tau)|",
            ylabel=scaled ? L"P(x)" : L"P(-\ln |C(\tau)|)", yscale=log10,
            yautolimitmargin=(0.05, 0.15))

        missing = String[]
        for (j, (t, y)) in enumerate(zip(d.times, d.samples))
            # 直接对变换后的样本重新统计密度；每条曲线用 60 个等宽箱。
            values = scaled ? y ./ t^d.fit.mu : y
            lo, hi = extrema(filter(isfinite, values))
            hi == lo && (hi = lo + 1.0)
            edges = range(lo, hi; length=61)
            centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
            scatterlines!(ax, centers, gap_density(values, edges); color=PLOT_COLORS[j],
                markersize=5, label="τ = $(round(t; sigdigits=4))")
            nmiss = count(!isfinite, y)
            nmiss > 0 && push!(missing, "τ=$(round(t; sigdigits=3)): $nmiss/$(length(y))")
        end
        if !isempty(missing)
            text!(ax, 0.98, 0.98; text="Invalid C\n" * join(missing, "\n"),
                space=:relative, align=(:right, :top), fontsize=11)
        end
    end
    # 同一时间网格共用图例；禁止把不同时间误标为同一条曲线。
    @assert all(d.times == first(panels).times for d in panels)
    finish_figure!(fig, ["τ = $(round(t; sigdigits=4))" for t in first(panels).times], 2,
        "$(basename(data.input)) | Use |C|; density / all samples; zero/nonfinite C omitted" *
        (scaled ? "\nμ: least-squares matching of log quantiles (5–95%) of valid samples" :
            "\n60 equal-width bins per curve; nearest grid points to τ = 1, 3, 10, 30, 100, 300"))
    fig
end

function main(args)
    input = if isempty(args)
        candidates = filter(readdir(joinpath(@__DIR__, "results"); join=true)) do path
            occursin(r"^full(?:_\d{8}_\d{6})?\.h5$", basename(path))
        end
        isempty(candidates) && error("No full ensemble found; pass an input HDF5 path")
        candidates[argmax(mtime.(candidates))]
    else
        abspath(args[1])
    end
    out = length(args) < 2 ? joinpath(dirname(input), "figures") : abspath(args[2])
    data = (; read_plot_data(input)..., input)

    # 分布图选取存有样本的代表场强；斜率图使用全部 h₀ ≥ 1 的点。
    sizes = sort(data.sizes)
    length(sizes) == 4 || error("Four size panels require exactly four sizes; got $sizes")
    fields = representative_fields(data.fields, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    sampled_fields = [h for h in data.fields if all(!isempty(d.gaps) for d in data.records if d.h0 == h)]
    gap_fields = representative_fields(filter(>=(1.0), sampled_fields), [1.0, 2.0, 5.0, 10.0])

    mkpath(out)
    set_theme!(Theme(fontsize=17, linewidth=2, Axis=(xgridvisible=false, ygridvisible=false,)))
    figures = [
        ("gap_distribution", plot_gap_distributions(data, gap_fields, sizes)),
        ("scaled_gap_distribution", plot_gap_distributions(data, gap_fields, sizes; scaled=true)),
        ("average_correlation", plot_spatial_correlations(data, fields, sizes)),
        ("log_correlation_sqrt_r", plot_spatial_correlations(data, fields, sizes; logarithmic=true)),
        ("imaginary_time_autocorrelation", plot_autocorrelation(data, fields, sizes)),
        ("imaginary_time_autocorrelation_slopes", plot_autocorrelation_slopes(data, sizes))]
    has_real_time = h5open(input, "r") do f
        all(haskey(f["L$(L)/h$(h)"], "real_autocorrelation_mean") for L in sizes, h in fields)
    end
    has_real_time && push!(figures,
        ("real_time_autocorrelation", plot_autocorrelation(data, fields, sizes; real_time=true)))

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
        "Julia: $VERSION; CairoMakie: $(pkgversion(CairoMakie)); HDF5: $(pkgversion(HDF5))",
        "Sizes: $sizes", "Gap fields (2×2): $gap_fields", "Correlation fields (2×3): $fields",
        "Autocorrelation: 2×2 size panels, the same six fields, from the same input file.", "",
        "Real-time autocorrelation: $(has_real_time ? "included; linear vertical axis preserves negative values" : "not available").",
        "Natural logarithms. Shading = ±1 SEM across independent disorder samples, not a confidence interval.",
        "Gap density = counts / (all samples × bin width); unresolved mass is not renormalized away.",
        "Scaled gap density: transform each sample to x = ln(ΔE)/√L, then histogram with common bin width 0.1 for every L; normalize by all samples.",
        "Imaginary-time mean curves use |stored autocorrelation_mean|, retaining the stored SEM; this is not mean(|sample_Ct|). Real-time curves keep their signs.",
        "Autocorrelation slopes (1/z): one curve per L, all $(count(>=(1.0), data.fields)) fields with h0 >= 1; absolute unweighted OLS slope of ln(|autocorrelation_mean|) versus ln(time), with intercept, over all finite positive times and nonzero finite means.",
        "Mean log spatial correlations are mean(log C), not log(mean C). No resampling.",
        "Time distributions: L=$(maximum(sizes)), fields=$gap_fields, target times=$DISTRIBUTION_TIMES, actual nearest times=$(first(distributions).times); 60 equal-width bins per curve, normalized by all samples.",
        "Collapse minimizes sum over times and quantiles of [ln Q_p(-ln |C|) - μ ln τ - a_p]^2, with independent intercept a_p and p=0.05:0.05:0.95; only positive quantiles of valid samples enter the fit.",
        "Time distributions use the absolute value of each sample before the negative logarithm. Zero/nonfinite samples are omitted; missing mass remains in the density. μ and log-quantile RMS are descriptive fits, not an asymptotic exponent determination.",
        "Nonfinite means break curves. Invalid SEM/bounds omit shading; log axes also omit nonpositive values/bounds.",
        "Zero distance is omitted on log-log spatial axes; zero time is omitted on log-log time axes.", "",
        "| h0 | L | samples | unresolved gaps | nonfinite mean-log distances | negative mean distances |",
        "|---|---|---|---|---|---|"]
    report_fields = union(fields, gap_fields)
    for d in data.records
        d.h0 in report_fields || continue
        unresolved = isempty(d.gaps) ? "not stored" : string(count(!, d.resolved))
        push!(report, "| $(d.h0) | $(d.L) | $(d.n) | $unresolved | $(d.invalid_log) | $(d.negative_C) |")
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
