using TOML, Dates

case_seed(cfg, k, L) = cfg.seed + 1000k + L

function config_grid(value)
    value isa AbstractVector && return Float64.(value)
    scale = get(value, "scale", "linear")
    scale in ("linear", "log10") || throw(ArgumentError("grid scale must be linear or log10"))
    value["length"] isa Integer && value["length"] > 0 ||
        throw(ArgumentError("grid length must be a positive integer"))
    grid = range(value["start"], value["stop"]; length=value["length"])
    return scale == "log10" ? 10.0 .^ grid : collect(Float64, grid)
end

"""Read scan parameters; resolve relative output paths against the TOML directory."""
function read_config(path=joinpath(@__DIR__, "scan.toml"))
    params = TOML.parsefile(path)
    boundary = Symbol(params["boundary"])
    RandomTFIM.check_boundary(boundary)
    field_distribution = Symbol(params["field_distribution"])
    RandomTFIM.check_field_distribution(field_distribution)
    nsamples, seed, worker_threads = params["nsamples"], params["seed"], params["worker_threads"]
    nsamples isa Integer && nsamples > 0 || throw(ArgumentError("nsamples must be a positive integer"))
    seed isa Integer && seed >= 0 || throw(ArgumentError("seed must be a nonnegative integer"))
    worker_threads isa Integer && worker_threads > 0 || throw(ArgumentError("worker_threads must be a positive integer"))
    sizes = params["sizes"]
    !isempty(sizes) && allunique(sizes) && all(L -> L isa Integer && L >= 2 && iseven(L), sizes) ||
        throw(ArgumentError("sizes must be distinct even integers >= 2"))
    fields, times = config_grid(params["fields"]), config_grid(params["times"])
    !isempty(fields) && allunique(fields) && all(h -> isfinite(h) && h > 0, fields) ||
        throw(ArgumentError("fields must be distinct finite positive values"))
    all(t -> isfinite(t) && t >= 0, times) || throw(ArgumentError("times must be finite and nonnegative"))
    targets = params["sample_fields"]
    targets isa AbstractVector && all(h -> h isa Real && isfinite(h) && h > 0, targets) ||
        throw(ArgumentError("sample_fields must be an array of finite positive values"))
    sample_fields = select_sample_fields(fields, targets)
    output = normpath(joinpath(dirname(abspath(path)), params["output"]))
    stem, ext = splitext(output)
    output = stem * "_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ext
    ispath(output) && throw(ArgumentError("output already exists: $output"))
    return (; nsamples, seed, worker_threads, output, boundary, field_distribution, sizes, fields, sample_fields, times)
end

# Mark completion only after all computation and writes succeed.
function with_results(f, cfg)
    ispath(cfg.output) && throw(ArgumentError("output already exists: $(cfg.output)"))
    mkpath(dirname(cfg.output))
    h5open(cfg.output, "w") do file
        write_metadata(file, cfg)
        f(file)
        complete = attributes(file)["complete"]
        try
            write(complete, true)
        finally
            close(complete)
        end
    end
    println("Saved: $(cfg.output)")
    return cfg.output
end

function write_case(file, data, cfg)
    group = create_group(file, "L$(data.L)/h$(data.h0)")
    write_observables(group, data.result, cfg.times)
    attributes(group)["j"] = data.L ÷ 2
    attributes(group)["seed"] = data.seed

    println("L=$(data.L) h0=$(data.h0) samples=$(cfg.nsamples) worker=$(data.worker_id)",
        " time=$(round(data.seconds; digits=3))s")
    flush(file)
end

function write_metadata(file, cfg)
    attributes(file)["complete"] = false
    attributes(file)["nsamples"] = cfg.nsamples
    attributes(file)["master_seed"] = cfg.seed
    attributes(file)["distribution"] = "box"
    attributes(file)["field_distribution"] = string(cfg.field_distribution)
    cfg.field_distribution == :fixed && (attributes(file)["fixed_field_divisor"] = exp(1))
    attributes(file)["boundary"] = string(cfg.boundary)

    parameters = create_group(file, "parameters")
    parameters["sizes"] = collect(cfg.sizes)
    parameters["fields"] = collect(cfg.fields)
    parameters["sample_fields"] = cfg.sample_fields
end

# Select existing scan points nearest the configured representative fields.
function select_sample_fields(fields, targets)
    return unique([fields[argmin(abs.(fields .- h))] for h in targets])
end

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
