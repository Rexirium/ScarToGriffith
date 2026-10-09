using Distributed, HDF5, LinearAlgebra

if !isdefined(@__MODULE__, :RandomTFIM)
    include("RandomTFIM.jl")
end
using .RandomTFIM
include("results_io.jl")

scan_jobs(cfg) = ((L=L, h0=h0, seed=case_seed(cfg, k, L))
    for (k, h0) in enumerate(cfg.fields) for L in cfg.sizes)

function compute_case(job, cfg)
    BLAS.set_num_threads(1)
    seconds = @elapsed result = disorder_ensemble(job.L, job.h0, cfg.times; nsamples=cfg.nsamples, seed=job.seed, rmax=job.L ÷ 2,
        boundary=cfg.boundary, field_distribution=cfg.field_distribution, keep_samples=job.h0 in cfg.sample_fields)
    return (; L=job.L, h0=job.h0, seed=job.seed, result,
        worker_id=myid(), seconds)
end

