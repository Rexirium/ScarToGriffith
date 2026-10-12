using Distributed, HDF5, LinearAlgebra, Random

if !isdefined(@__MODULE__, :RandomTFIM)
    include("RandomTFIM.jl")
end
using .RandomTFIM
include("results_io.jl")

entropy_cuts(L) = [k * L ÷ 8 for k in 0:8]

scan_jobs(cfg) = ((L=L, h0=h0, seed=cfg.seed + 1000k + L)
    for (k, h0) in enumerate(cfg.fields) for L in cfg.sizes)

# Fixed chunks keep reductions reproducible across local and worker thread counts.
# Only running moments survive each sample; no sample observables are retained.
function scan_observables(L, h0, times; nsamples, seed, field_distribution, offset::Bool=true)
    nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    RandomTFIM.time_scale(times, :imaginary)
    rng = Xoshiro(seed)
    samples = [sample_disorder(rng, L, h0; field_distribution) for _ in 1:nsamples]
    nt = length(times)
    cuts = entropy_cuts(L)
    width = 1 + 2nt + length(cuts)
    j = cld(L, 2)
    nchunks = min(nsamples, 32)
    chunks = [k:nchunks:nsamples for k in 1:nchunks]
    means = zeros(2width, length(chunks))
    m2 = zeros(size(means))
    Threads.@threads for k in eachindex(chunks)
        local avg, variance, delta
        avg, variance = @view(means[:, k]), @view(m2[:, k])
        values = Vector{Float64}(undef, 2width)
        for (n, index) in enumerate(chunks[k])
            J, h = samples[index]
            F = svd!(RandomTFIM.fermion_matrix(J, h))
            gap = RandomTFIM.gap_result(F.S)
            G = -(F.V * F.U')
            B = !isempty(times) && 1 < j < L ? RandomTFIM.spin_flip_covariance(F, j) : nothing
            values[1] = gap.gap
            values[2:1+nt] = RandomTFIM.autocorrelation(F, times, j, 1.0; B, offset)
            values[2+nt:1+2nt] = RandomTFIM.autocorrelation(F, times, j, 1.0im; B, offset)
            for (i, cut) in enumerate(cuts)
                values[1+2nt+i] = entanglement_entropy(G; cut)
            end
            for i in 1:width
                values[width+i] = RandomTFIM.correlation_log(values[i])
            end
            gap.resolved || (values[width+1] = NaN)
            for i in eachindex(values)
                delta = values[i] - avg[i]
                avg[i] += delta / n
                variance[i] += delta * (values[i] - avg[i])
            end
        end
    end
    avg, variance = copy(means[:, 1]), copy(m2[:, 1])
    count = length(chunks[1])
    for k in 2:length(chunks)
        n = length(chunks[k])
        delta = means[:, k] .- avg
        variance .+= m2[:, k] .+ delta.^2 .* (count * n / (count + n))
        avg .+= delta .* (n / (count + n))
        count += n
    end
    sem = nsamples > 1 ? sqrt.(max.(variance, 0) ./ (nsamples * (nsamples - 1))) : fill(NaN, 2width)
    ranges = ((:gap, :log_gap, 1),
        (:autocorrelation, :log_autocorrelation, 2:1+nt),
        (:real_autocorrelation, :real_log_autocorrelation, 2+nt:1+2nt),
        (:entanglement_entropy, :log_entanglement_entropy, 2+2nt:width))
    return (; (Symbol(name, suffix) => stats[indices .+ shift]
        for (raw, logged, indices) in ranges
        for (name, shift) in ((raw, 0), (logged, width))
        for (suffix, stats) in (("_mean", avg), ("_sem", sem)))...)
end

function compute_case(job, cfg)
    BLAS.set_num_threads(1)
    seconds = @elapsed result = scan_observables(job.L, job.h0, cfg.times;
        nsamples=cfg.nsamples, seed=job.seed, field_distribution=cfg.field_distribution, offset=cfg.offset)
    return (; L=job.L, h0=job.h0, seed=job.seed, result,
        worker_id=myid(), seconds)
end
