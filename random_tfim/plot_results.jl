using CairoMakie, HDF5, Statistics

# Run: julia --project=@v1.13 random_tfim/plot_results.jl [input.h5] [output_dir]
# Sample columns already contain the spatial average over all starting sites.
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
            C, logC = read(g["sample_C"]), read(g["sample_logC"])
            n = size(C, 2)
            gaps, resolved = read(g["loggaps"]), Bool.(read(g["resolved"]))
            @assert size(C) == size(logC) == (L ÷ 2 + 1, n)
            @assert length(gaps) == length(resolved) == n
            @assert all(isfinite, C) && all(C[1, :] .== 1) && all(logC[1, :] .== 0)
            @assert resolved == isfinite.(gaps)
            avg, sem = vec(mean(C; dims=2)), vec(std(C; dims=2)) ./ sqrt(n)
            logavg, logsem = vec(mean(logC; dims=2)), vec(std(logC; dims=2)) ./ sqrt(n)
            @assert isapprox(avg, read(g["average"]))
            @assert all(isequal.(logavg, read(g["mean_log"])))
            push!(records, (; L, h0, n, gaps, resolved, avg, sem, logavg, logsem,
                invalid_log=count(!isfinite, logC), negative_C=count(<(0), C)))
        end
        (; records, sizes, fields, boundary=read(HDF5.attributes(f)["boundary"]))
    end
end

function main(args)
    input = isempty(args) ? joinpath(@__DIR__, "results", "full_sample1000.h5") : abspath(args[1])
    out = length(args) < 2 ? joinpath(dirname(input), "figures") : abspath(args[2])
    data = read_plot_data(input)
    records = data.records
    fields, sizes = data.fields, data.sizes
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    markers = [:circle, :rect, :utriangle, :diamond]
    @assert length(sizes) <= length(colors)
    mkpath(out)
    set_theme!(Theme(fontsize=17, linewidth=2, Axis=(xgridvisible=false, ygridvisible=false,)))

    # Common unit-width ln(gap) bins; denominator includes unresolved samples.
    finite_gaps = [x for d in records for x in d.gaps if isfinite(x)]
    edges = collect(floor(minimum(finite_gaps)):1.0:ceil(maximum(finite_gaps)))
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    gapfig = Figure(size=(1560, 870))
    avgfig = Figure(size=(1560, 870))
    logfig = Figure(size=(1560, 870))
    titles = ["Energy-gap distributions", "Average spatial correlations", "Average log correlations"]
    figs = [gapfig, avgfig, logfig]
    for (fig, title) in zip(figs, titles)
        Label(fig[0, 1:4], title * "  |  Young & Rieger (1996)", fontsize=25)
    end
    report = ["# Plot audit", "", "Input: `$input`", "Julia: $(VERSION); project: `$(Base.active_project())`",
        "CairoMakie: $(pkgversion(CairoMakie)); HDF5: $(pkgversion(HDF5))", "Boundary: $(data.boundary)", "",
        "Natural logarithms. Error bands are ±1 SEM across independent disorder samples.",
        "Gap densities use counts / (total samples × bin width), so missing probability is not renormalized away.",
        "Log correlations use mean(sample_logC), not log(mean(sample_C)). Distances with any nonfinite log sample are omitted.",
        "The average-C plot retains the supplied arithmetic averages; tiny negative individual correlations are counted below as numerical artifacts.", "",
        "| h0 | L | samples | unresolved gaps | nonfinite log entries | negative C entries | last complete log r |",
        "|---|---|---|---|---|---|---|"]
    for (k, h0) in enumerate(fields)
        row, col = (k-1) ÷ 4 + 1, (k-1) % 4 + 1
        title = "h₀ = $h0"
        ag = Axis(gapfig[row, col]; title, xlabel=L"\ln\Delta E", ylabel=L"P(\ln\Delta E)", yscale=log10)
        ac = Axis(avgfig[row, col]; title, xlabel=L"r", ylabel=L"C_{\mathrm{av}}(r)", xscale=log10, yscale=log10)
        al = Axis(logfig[row, col]; title, xlabel=L"\sqrt{r}", ylabel=L"[\ln C(r)]_{\mathrm{av}}")
        xlims!(ag, first(edges), last(edges)); ylims!(ag, 5e-4, 1)
        xlims!(ac, 1, maximum(sizes) ÷ 2)
        positive_means = [v for d in records if d.h0 == h0 for v in d.avg if v > 0]
        # ylims!(ac, 10.0^floor(log10(minimum(positive_means)) - 0.3), 1)
        xlims!(al, 0, sqrt(maximum(sizes) ÷ 2))
        missing = String[]
        for (j, L) in enumerate(sizes)
            d = only(filter(d -> d.L == L && d.h0 == h0, records))
            color, marker = colors[j], markers[j]
            counts = zeros(Int, length(centers))
            for x in filter(isfinite, d.gaps)
                counts[clamp(searchsortedlast(edges, x), 1, length(counts))] += 1
            end
            density = counts ./ (d.n .* diff(edges))
            @assert isapprox(sum(density .* diff(edges)), count(d.resolved) / d.n)
            density[counts .== 0] .= NaN # break empty bins on the log axis
            scatterlines!(ag, centers, density; color, marker, markersize=5)
            nmiss = count(!, d.resolved)
            nmiss > 0 && push!(missing, "L=$L: $nmiss/$(d.n)")

            r = collect(0:L÷2)
            y = copy(d.avg); y[(r .== 0) .| (y .<= 0)] .= NaN
            lower = d.avg .- d.sem
            lower[lower .<= 0] .= NaN
            upper = d.avg .+ d.sem
            upper[upper .<= 0] .= NaN
            band!(ac, r[2:end], lower[2:end], upper[2:end]; color=(color, 0.16))
            scatterlines!(ac, r[2:end], y[2:end]; color, marker, markersize=4)
            valid = isfinite.(d.logavg) .& isfinite.(d.logsem)
            ly = copy(d.logavg); ly[.!valid] .= NaN
            band!(al, sqrt.(r), ly .- d.logsem, ly .+ d.logsem; color=(color, 0.16))
            scatterlines!(al, sqrt.(r), ly; color, marker, markersize=4)
            last_r = r[findlast(valid)]
            if last_r < L ÷ 2
                scatter!(al, [sqrt(last_r)], [ly[last_r+1]]; color, marker=:xcross, markersize=14)
            end
            push!(report, "| $h0 | $L | $(d.n) | $nmiss | $(d.invalid_log) | $(d.negative_C) | $last_r |")
        end
        if !isempty(missing)
            text!(ag, 0.03, 0.98; text="Unresolved\n" * join(missing, "\n"), space=:relative,
                align=(:left, :top), fontsize=12)
        end
    end
    notes = ["Natural-log bins: width 1\nDensity normalized by all samples\nUnresolved gaps omitted\nYoung & Rieger: Figs. 1, 3",
        "All starting sites averaged first\nShading: ±1 disorder SEM\n1 ≤ r ≤ L/2; log–log axes\nYoung & Rieger: Fig. 8",
        "Average of ln C, not ln of average C\nShading: ±1 disorder SEM\n×: last complete point before truncation\nYoung & Rieger: Fig. 9"]
    for (fig, note) in zip(figs, notes)
        panel = GridLayout(fig[2, 4])
        Legend(panel[1, 1], [LineElement(color=c, linewidth=3) for c in colors[1:length(sizes)]],
            ["L = $L" for L in sizes], "Chain length"; framevisible=false)
        Label(panel[2, 1], note; fontsize=16, justification=:left)
        Label(fig[3, 1:4], "$(first(records).n) disorder samples per (L, h₀)  •  $(data.boundary)", fontsize=15)
    end
    for (fig, name) in zip(figs, ["gap_distribution", "average_correlation", "log_correlation_sqrt_r"])
        save(joinpath(out, "$name.png"), fig; px_per_unit=1.5)
        println("Saved ", joinpath(out, name))
    end
    write(joinpath(out, "plot_audit.md"), join(report, "\n") * "\n")
    println("Verified dimensions, r=0 normalization, stored averages, and histogram probability mass.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
