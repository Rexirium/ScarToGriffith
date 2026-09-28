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
            gaps = read_plot_dataset(g, "log_gap_samples", "loggaps")
            resolved = Bool.(read_plot_dataset(g, "gap_resolved", "resolved"))
            n = length(gaps)
            avg = read_plot_dataset(g, "correlation_mean", "average")
            sem = read_plot_dataset(g, "correlation_sem", "sem")
            logavg = read_plot_dataset(g, "log_correlation_mean", "mean_log")
            logsem = read_plot_dataset(g, "log_correlation_sem", "log_sem")
            @assert length(avg) == length(sem) == length(logavg) == length(logsem) == L ÷ 2 + 1
            @assert length(resolved) == n
            @assert all(isfinite, avg) && avg[1] == 1 && logavg[1] == 0
            @assert resolved == isfinite.(gaps)
            push!(records, (; L, h0, n, gaps, resolved, avg, sem, logavg, logsem,
                invalid_log=count(!isfinite, logavg), negative_C=count(<(0), avg)))
        end
        (; records, sizes, fields, boundary=read(HDF5.attributes(f)["boundary"]))
    end
end

const PLOT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000"]

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
    density = counts ./ (length(gaps) .* diff(edges))
    @assert isapprox(sum(density .* diff(edges)), count(isfinite, gaps) / length(gaps))
    density[counts .== 0] .= NaN
    density
end

function plot_gap_distributions(data, fields, sizes)
    fig = Figure(size=(1150, 900))
    Label(fig[0, 1:2], "Log energy-gap distributions", fontsize=25)
    for (k, h0) in enumerate(fields)
        row, col = panel_position(k, 2)
        ax = Axis(fig[row, col]; title=field_label(h0), xlabel=L"\ln\Delta E",
            ylabel=L"P(\ln\Delta E)", yscale=log10)
        records = [only(filter(d -> d.L == L && d.h0 == h0, data.records)) for L in sizes]
        finite_gaps = [x for d in records for x in d.gaps if isfinite(x)]
        isempty(finite_gaps) && error("No resolved gaps at h0=$h0")
        lo = floor(minimum(finite_gaps))
        edges = collect(lo:1.0:max(lo + 1, ceil(maximum(finite_gaps))))
        centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
        missing = String[]
        for (j, d) in enumerate(records)
            lines!(ax, centers, gap_density(d.gaps, edges); color=PLOT_COLORS[j])
            nmiss = count(!, d.resolved)
            nmiss > 0 && push!(missing, "L=$(d.L): $nmiss/$(d.n)")
        end
        if !isempty(missing)
            text!(ax, 0.03, 0.98; text="Unresolved\n" * join(missing, "\n"),
                space=:relative, align=(:left, :top), fontsize=12)
        end
    end
    finish_figure!(fig, ["L = $L" for L in sizes], 2,
        "$(basename(data.input)) | ln-gap bin width = 1; density normalized by all samples")
    fig
end

function plot_spatial_correlations(data, fields, sizes; logarithmic=false)
    fig = Figure(size=(1500, 900))
    title = logarithmic ? "Mean log spatial correlations" : "Mean spatial correlations"
    Label(fig[0, 1:3], title, fontsize=25)
    for (k, h0) in enumerate(fields)
        row, col = panel_position(k, 3)
        ax = if logarithmic
            Axis(fig[row, col]; title=field_label(h0), xlabel=L"\sqrt{r}",
                ylabel=L"[\ln C(r)]_{\mathrm{av}}")
        else
            Axis(fig[row, col]; title=field_label(h0), xlabel=L"r",
                ylabel=L"C_{\mathrm{av}}(r)", xscale=log10, yscale=log10)
        end
        for (j, L) in enumerate(sizes)
            d = only(filter(d -> d.L == L && d.h0 == h0, data.records))
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

function plot_autocorrelation(data, fields, sizes)
    fig = Figure(size=(1150, 900))
    Label(fig[0, 1:2], "Imaginary-time autocorrelations", fontsize=25)
    h5open(data.input, "r") do f
        for (j, L) in enumerate(sizes)
            row, col = panel_position(j, 2)
            ax = Axis(fig[row, col]; title="L = $L", xlabel=L"\tau",
                ylabel=L"C_{\mathrm{av}}(\tau)", xscale=log10, yscale=log10)
            for (k, h0) in enumerate(fields)
                g = f["L$(L)/h$(h0)"]
                times = read(g["imaginary_time"])
                y = read(g["autocorrelation_mean"])
                sem = read(g["autocorrelation_sem"])
                @assert length(times) == length(y) == length(sem)
                keep = isfinite.(times) .& (times .> 0)
                @assert any(keep) "No positive imaginary times for L=$L, h0=$h0"
                uncertainty_curve!(ax, times[keep], y[keep], sem[keep], PLOT_COLORS[k]; positive=true)
            end
        end
    end
    finish_figure!(fig, field_label.(fields), 2,
        "$(basename(data.input)) | Shading: ±1 disorder SEM; nonpositive times/values/bounds omitted")
    fig
end

function main(args)
    input = isempty(args) ? joinpath(@__DIR__, "results", "full.h5") : abspath(args[1])
    out = length(args) < 2 ? joinpath(dirname(input), "figures") : abspath(args[2])
    data = (; read_plot_data(input)..., input)
    sizes = sort(data.sizes)
    length(sizes) == 4 || error("Four size panels require exactly four sizes; got $sizes")
    fields = representative_fields(data.fields, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    gap_fields = representative_fields(filter(>=(1.0), data.fields), [1.0, 2.0, 5.0, 10.0])
    mkpath(out)
    set_theme!(Theme(fontsize=17, linewidth=2, Axis=(xgridvisible=false, ygridvisible=false,)))
    for (name, fig) in (
        ("gap_distribution", plot_gap_distributions(data, gap_fields, sizes)),
        ("average_correlation", plot_spatial_correlations(data, fields, sizes)),
        ("log_correlation_sqrt_r", plot_spatial_correlations(data, fields, sizes; logarithmic=true)),
        ("imaginary_time_autocorrelation", plot_autocorrelation(data, fields, sizes)))
        path = joinpath(out, "$name.png")
        save(path, fig; px_per_unit=1.5)
        println("Saved ", path)
    end
    report = ["# Plot audit", "", "Input: `$input`", "Boundary: $(data.boundary)",
        "Julia: $VERSION; CairoMakie: $(pkgversion(CairoMakie)); HDF5: $(pkgversion(HDF5))",
        "Sizes: $sizes", "Gap fields (2×2): $gap_fields", "Correlation fields (2×3): $fields",
        "Autocorrelation: 2×2 size panels, the same six fields, from the same input file.", "",
        "Natural logarithms. Shading = ±1 SEM across independent disorder samples, not a confidence interval.",
        "Gap density = counts / (all samples × bin width); unresolved mass is not renormalized away.",
        "Mean log correlations are mean(log C), not log(mean C). No refitting or resampling.",
        "Nonfinite means break curves. Invalid SEM/bounds omit shading; log axes also omit nonpositive values/bounds.",
        "Zero distance is omitted on log-log spatial axes; zero time is omitted on log-log time axes.", "",
        "| h0 | L | samples | unresolved gaps | nonfinite mean-log distances | negative mean distances |",
        "|---|---|---|---|---|---|"]
    for d in data.records
        d.h0 in union(fields, gap_fields) || continue
        push!(report, "| $(d.h0) | $(d.L) | $(d.n) | $(count(!, d.resolved)) | $(d.invalid_log) | $(d.negative_C) |")
    end
    write(joinpath(out, "plot_audit.md"), join(report, "\n") * "\n")
    println("Verified summary dimensions, r=0 normalization, and histogram probability mass.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
