using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia-env", "local"))

include("scan_common.jl")

"""Run an ensemble from read_config(path); save portable HDF5 data.

Compute gaps, spatial correlations and imaginary- and real-time autocorrelations with SEM.
Usage: julia random_tfim/run.jl [config.toml] (default: scan.toml beside this script)
Output filenames receive a _yyyymmdd_HHMMSS timestamp before the extension.
"""
function main(cfg)
    return with_results(cfg) do file
        for job in scan_jobs(cfg)
            write_case(file, compute_case(job, cfg), cfg)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || throw(ArgumentError("Usage: run.jl [config.toml]"))
    main(read_config(get(ARGS, 1, joinpath(@__DIR__, "scan.toml"))))
end
