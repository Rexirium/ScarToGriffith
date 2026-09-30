# Preserve invalid samples in log statistics; never take abs or drop samples.
correlation_log(x::Real) = isfinite(x) && x > 0 ? log(x) : NaN

function mean_sem(samples::AbstractVector)
    n = length(samples)
    return (average=mean(samples), sem=n > 1 ? std(samples) / sqrt(n) : NaN)
end

function mean_sem(samples::AbstractMatrix)
    n = size(samples, 2)
    average = vec(mean(samples; dims=2))
    sem = n > 1 ? vec(std(samples; dims=2, mean=reshape(average, :, 1))) / sqrt(n) :
        fill(NaN, size(samples, 1))
    return (; average, sem)
end

"""Seeded disorder ensemble; columns of per-sample means label realizations.

Returns scalar gap_mean/gap_sem and log_gap_mean/log_gap_sem,
spatial statistics C_mean/C_sem and logC_mean/logC_sem,
imaginary-time statistics Ct_mean/Ct_sem and logCt_mean/logCt_sem, and
real-time statistics real_Ct_mean/real_Ct_sem and real_logCt_mean/real_logCt_sem.
Unresolved gaps give NaN logarithms, which propagate through log-gap statistics.
Correlation means and SEM are vectors over distance or time. All statistics use
independent realizations (corrected sample variance); SEM is NaN for one realization.
All correlation samples and statistics are real; real-time statistics describe
only the real part. Nonpositive real-time values give NaN logarithms.
Set keep_samples=true to also return gaps and resolved vectors, and sample_C,
sample_Ct and sample_real_Ct matrices (columns are samples). Log samples are not returned.
Each realization is drawn once; gaps, spatial pairs and both time domains share
SVDs in the same sample loop. Both time domains use the required positional times:
a finite, nonnegative grid at j (default L/2). Empty times yields empty temporal
statistics; rmax=0 skips nontrivial spatial pairs.
Realizations are drawn serially, then evaluated with Threads.@threads;
the seed and sample order are independent of the number of Julia threads.
Use julia --threads=N and a single BLAS thread for parallel sample evaluation.
Sites within a sample are correlated, so SEM uses sample columns, not sites.
Average pair logarithms before exponentiating to get the typical correlation.
Select boundary=:periodic (default) or :open. Open-chain sample means use
only the L-r valid origins.
Select field_distribution=:uniform (default) or :fixed (h=h0/e); J remains uniform.
The rcond_tol keyword is forwarded to both imaginary- and real-time
autocorrelations; it has the same meaning and default as in autocorrelation.
Logarithms are taken per pair/time before averaging. Nonpositive or nonfinite
values give NaN, which propagates through
sample and ensemble statistics without dropping sites or realizations.
"""
Base.@constprop :aggressive function disorder_ensemble(L::Int, h0::Real,
        times::AbstractVector{<:Real}; nsamples::Int=100, seed::Int=1996,
        rmax::Int=L ÷ 2, keep_samples::Bool=false, boundary::Symbol=:periodic,
        j::Int=L ÷ 2, rcond_tol::Real=sqrt(eps(Float64)),
        field_distribution::Symbol=:uniform)
    check_boundary(boundary)
    nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    1 <= j <= L || throw(ArgumentError("require 1 <= j <= L"))
    time_scale(times, :imaginary)
    check_rcond_tol(rcond_tol)
    0 <= rmax <= (boundary == :open ? L-1 : L ÷ 2) || throw(ArgumentError("invalid rmax"))

    bc = Val(boundary)
    rng = Xoshiro(seed)
    # Preserve the serial RNG stream; worker threads never share a mutable RNG.
    samples = [sample_disorder(rng, L, h0; boundary, field_distribution) for _ in 1:nsamples]

    gaps = zeros(nsamples)
    resolved = fill(false, nsamples)
    sample_C, sample_logC = zeros(rmax+1, nsamples), zeros(rmax+1, nsamples)
    sample_Ct = Matrix{Float64}(undef, length(times), nsamples)
    sample_real_Ct = similar(sample_Ct)

    # Equal-site spatial correlations are known without a factorization.
    sample_C[1, :] .= 1.0

    # Each iteration owns its workspaces and writes only sample n's output.
    Threads.@threads for n in 1:nsamples
        J, h = samples[n]
        # Share the SVDs across the gap, spatial correlations and dynamics.
        initial = svd!(fermion_matrix(J, h, bc))
        if boundary == :periodic
            evolution = svd!(fermion_matrix(J, h, 1))
            result = gap_result(J, h, initial.S, evolution.S)
        else
            evolution = initial
            result = gap_result(J, h, initial.S, bc)
        end
        gaps[n], resolved[n] = result.gap, result.resolved

        if rmax > 0
            G = -(initial.V * initial.U')
            pair = correlations(G; rmax, boundary)
            logs = correlation_log.(pair)
            for r in 1:rmax
                origins = 1:(boundary == :open ? L-r : L)
                sample_C[r+1, n] = mean(@view pair[origins, r+1])
                sample_logC[r+1, n] = mean(@view logs[origins, r+1])
            end
        end

        sample_Ct[:, n] = autocorrelation(initial, evolution, times, j, 1.0; rcond_tol)
        sample_real_Ct[:, n] = autocorrelation(initial, evolution, times, j, 1.0im; rcond_tol)
    end

    sample_logCt = correlation_log.(sample_Ct)
    sample_real_logCt = correlation_log.(sample_real_Ct)
    spatial, log_spatial = mean_sem(sample_C), mean_sem(sample_logC)
    temporal, log_temporal = mean_sem(sample_Ct), mean_sem(sample_logCt)
    real_temporal, real_log_temporal = mean_sem(sample_real_Ct), mean_sem(sample_real_logCt)
    gap_stats = mean_sem(gaps)
    log_gap_stats = mean_sem(map((gap, ok) -> ok ? log(gap) : NaN, gaps, resolved))
    summary = (; gap_mean=gap_stats.average, gap_sem=gap_stats.sem,
        log_gap_mean=log_gap_stats.average, log_gap_sem=log_gap_stats.sem,
        C_mean=spatial.average, C_sem=spatial.sem,
        logC_mean=log_spatial.average, logC_sem=log_spatial.sem,
        Ct_mean=temporal.average, Ct_sem=temporal.sem,
        logCt_mean=log_temporal.average, logCt_sem=log_temporal.sem,
        real_Ct_mean=real_temporal.average, real_Ct_sem=real_temporal.sem,
        real_logCt_mean=real_log_temporal.average, real_logCt_sem=real_log_temporal.sem)
    return keep_samples ? (; summary..., gaps, resolved, sample_C, sample_Ct, sample_real_Ct) : summary
end
