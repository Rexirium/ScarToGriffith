using TOML, Dates

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
    haskey(params, "boundary") && throw(ArgumentError("remove boundary: scans use open chains only"))
    field_distribution = Symbol(params["field_distribution"])
    RandomTFIM.check_field_distribution(field_distribution)
    offset = get(params, "offset", true)
    offset isa Bool || throw(ArgumentError("offset must be true or false"))
    nsamples, seed, worker_threads = params["nsamples"], params["seed"], params["worker_threads"]
    nsamples isa Integer && nsamples > 0 || throw(ArgumentError("nsamples must be a positive integer"))
    seed isa Integer && seed >= 0 || throw(ArgumentError("seed must be a nonnegative integer"))
    worker_threads isa Integer && worker_threads > 0 || throw(ArgumentError("worker_threads must be a positive integer"))
    sizes = params["sizes"]
    !isempty(sizes) && allunique(sizes) && all(L -> L isa Integer && L >= 2, sizes) ||
        throw(ArgumentError("sizes must be distinct integers >= 2"))
    fields, times = config_grid(params["fields"]), config_grid(params["times"])
    !isempty(fields) && allunique(fields) && all(h -> isfinite(h) && h > 0, fields) ||
        throw(ArgumentError("fields must be distinct finite positive values"))
    all(t -> isfinite(t) && t >= 0, times) || throw(ArgumentError("times must be finite and nonnegative"))
    output = normpath(joinpath(dirname(abspath(path)), params["output"]))
    stem, ext = splitext(output)
    output = stem * "_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ext
    ispath(output) && throw(ArgumentError("output already exists: $output"))
    return (; nsamples, seed, worker_threads, output, field_distribution, offset, sizes, fields, times)
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
    for (key, value) in pairs(data.result)
        group[string(key)] = value
    end
    group["imaginary_time"] = cfg.times
    group["real_time"] = cfg.times
    group["subsystem_sizes"] = entropy_cuts(data.L)
    attributes(group)["j"] = cld(data.L, 2)
    attributes(group)["seed"] = data.seed

    flush(file)
    println("Saved: L=$(data.L) h0=$(data.h0) samples=$(cfg.nsamples) worker=$(data.worker_id)",
        " time=$(round(data.seconds; digits=3))s")
    flush(stdout)
end

function write_metadata(file, cfg)
    attributes(file)["format_version"] = 2
    attributes(file)["offset"] = cfg.offset
    attributes(file)["imaginary_autocorrelation_method"] = cfg.offset ? "pairing_tail" : "full"
    attributes(file)["pairing_tail_threshold"] = 1e-8
    attributes(file)["complete"] = false
    attributes(file)["nsamples"] = cfg.nsamples
    attributes(file)["master_seed"] = cfg.seed
    attributes(file)["distribution"] = "box"
    attributes(file)["field_distribution"] = string(cfg.field_distribution)
    cfg.field_distribution == :fixed && (attributes(file)["fixed_field_divisor"] = exp(1))
    attributes(file)["boundary"] = "open"

    parameters = create_group(file, "parameters")
    parameters["sizes"] = collect(cfg.sizes)
    parameters["fields"] = collect(cfg.fields)
end
