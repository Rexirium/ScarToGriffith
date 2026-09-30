function parse_config(args)
    mode = get(args, 1, "demo")
    mode in ("demo", "full") || throw(ArgumentError("unknown mode: $mode"))
    length(args) <= 5 || throw(ArgumentError("expected at most five arguments"))

    boundary = Symbol(get(args, 4, "periodic"))
    RandomTFIM.check_boundary(boundary)
    field_distribution = Symbol(get(args, 5, "uniform"))
    RandomTFIM.check_field_distribution(field_distribution)
    default_samples = mode == "demo" ? 20 : 10_000
    nsamples = parse(Int, get(args, 2, string(default_samples)))
    output = abspath(get(args, 3, joinpath(@__DIR__, "results", "$(mode)_$(field_distribution).h5")))
    stem, ext = splitext(output)
    output = stem * "_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ext
    nsamples > 0 || throw(ArgumentError("samples must be positive"))
    ispath(output) && throw(ArgumentError("output already exists: $output"))

    sizes = mode == "demo" ? (16, 32) : (16, 32, 64, 128)
    fields = 10 .^ range(-1, 1, 101)
    sample_fields = select_sample_fields(fields)
    times = 10 .^ range(-1, 3, 101)
    return (; mode, nsamples, output, boundary, field_distribution, sizes, fields, sample_fields, times)
end

function write_metadata(file, cfg)
    attributes(file)["complete"] = false
    attributes(file)["nsamples"] = cfg.nsamples
    attributes(file)["distribution"] = "box"
    attributes(file)["field_distribution"] = string(cfg.field_distribution)
    cfg.field_distribution == :fixed && (attributes(file)["fixed_field_divisor"] = exp(1))
    attributes(file)["boundary"] = string(cfg.boundary)

    parameters = create_group(file, "parameters")
    parameters["sizes"] = collect(cfg.sizes)
    parameters["fields"] = collect(cfg.fields)
    parameters["sample_fields"] = cfg.sample_fields
end

# Select existing scan points nearest the six representative plotting fields.
select_sample_fields(fields) = [fields[argmin(abs.(fields .- h))] for h in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)]

function write_correlations(group, result, times)
    group["correlation_mean"] = result.C_mean
    group["correlation_sem"] = result.C_sem
    group["log_correlation_mean"] = result.logC_mean
    group["log_correlation_sem"] = result.logC_sem
    group["imaginary_time"] = times
    group["real_time"] = times
    # Both time domains provide real-valued sample means and SEM upstream.
    group["autocorrelation_mean"] = result.Ct_mean
    group["autocorrelation_sem"] = result.Ct_sem
    group["log_autocorrelation_mean"] = result.logCt_mean
    group["log_autocorrelation_sem"] = result.logCt_sem
    group["real_autocorrelation_mean"] = result.real_Ct_mean
    group["real_autocorrelation_sem"] = result.real_Ct_sem
    group["real_log_autocorrelation_mean"] = result.real_logCt_mean
    group["real_log_autocorrelation_sem"] = result.real_logCt_sem
end


function write_observables(group, result, times)
    for field in (:gap_mean, :gap_sem, :log_gap_mean, :log_gap_sem)
        group[string(field)] = getproperty(result, field)
    end
    write_correlations(group, result, times)
    if hasproperty(result, :gaps)
        group["gap_samples"] = result.gaps
        group["gap_resolved"] = result.resolved
        for field in (:sample_C, :sample_Ct, :sample_real_Ct)
            group[string(field)] = getproperty(result, field)
        end
    end
end
