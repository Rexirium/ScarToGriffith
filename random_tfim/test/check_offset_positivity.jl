using LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "scan_common.jl"))
BLAS.set_num_threads(1)

# Check the first N realizations of each production RNG stream, without changing it.
function check_offset_positivity(nsamples=8; method=:rank_one)
    nsamples > 0 || error("nsamples must be positive")
    method in (:rank_one, :subtract) || error("method must be rank_one or subtract")
    cfg = read_config()
    jobs = collect(scan_jobs(cfg))
    summaries = Vector{NamedTuple}(undef, length(jobs))
    println("Julia=$VERSION method=$method samples_per_case=$nsamples distribution=$(cfg.field_distribution)")
    println("sizes=$(cfg.sizes) fields=$(length(cfg.fields)) times=$(length(cfg.times))")
    println("time_range=$(extrema(cfg.times)) master_seed=$(cfg.seed) BLAS=$(BLAS.get_config())")
    for L in cfg.sizes
        indices = findall(job -> job.L == L, jobs)
        Threads.@threads for index in indices
            job = jobs[index]
            rng = Xoshiro(job.seed)
            negative = tail_negative = negative_samples = nonfinite = zeros_count = 0
            rank_negative = fallback_negative = subtraction_negative = svd_retries = 0
            first_negative = worst = nothing
            for sample in 1:nsamples
                J, h = sample_disorder(rng, job.L, job.h0; field_distribution=cfg.field_distribution)
                F = try
                    svd!(RandomTFIM.fermion_matrix(J, h))
                catch err
                    err isa LinearAlgebra.LAPACKException || rethrow()
                    svd_retries += 1
                    svd!(Matrix(RandomTFIM.fermion_matrix(J, h)); alg=QRIteration())
                end
                j = cld(job.L, 2)
                B = RandomTFIM.spin_flip_covariance(F, j)
                cache = method == :rank_one ? RandomTFIM.rank_one_cache(F, j; B) : nothing
                full = method == :subtract ? RandomTFIM.autocorrelation(F, cfg.times, j, 1.0; offset=false, B) : nothing
                removed = method == :subtract ?
                    RandomTFIM.ground_transition_weight(F, j) .* exp.(-2minimum(F.S) .* cfg.times) : nothing
                C = method == :subtract ? full .- removed :
                    RandomTFIM.autocorrelation(F, cfg.times, j, 1.0; offset=true, B, offset_cache=cache)
                bad = findall(<(0), C)
                nonfinite += count(!isfinite, C)
                zeros_count += count(iszero, C)
                isempty(bad) && continue
                negative_samples += 1
                negative += length(bad)
                tail_negative += count(n -> cfg.times[n] >= 100, bad)
                fallback = method == :subtract ? Int[] :
                    isnothing(cache.X) ? eachindex(C) : last(RandomTFIM.rank_one_autocorrelation(cache, cfg.times, 1.0))
                for n in bad
                    from_fallback = n in fallback
                    fallback_negative += from_fallback
                    rank_negative += method == :rank_one && !from_fallback
                    subtraction_negative += method == :subtract
                    point = (; sample, time_index=n, tau=cfg.times[n], value=C[n], from_fallback)
                    method == :subtract && (point = (; point..., full=full[n], removed=removed[n]))
                    isnothing(first_negative) && (first_negative = point)
                    (isnothing(worst) || point.value < worst.value) && (worst = point)
                end
            end
            summaries[index] = (; job..., negative, tail_negative, negative_samples,
                nonfinite, zeros_count, rank_negative, fallback_negative, subtraction_negative, svd_retries,
                first_negative, worst)
        end
        println("finished L=$L")
        flush(stdout)
    end
    println("TOTAL samples=$(length(jobs)*nsamples) values=$(length(jobs)*nsamples*length(cfg.times))")
    for key in (:negative, :tail_negative, :negative_samples, :nonfinite, :zeros_count,
                :rank_negative, :fallback_negative, :subtraction_negative, :svd_retries)
        println("$key=$(sum(s -> getproperty(s, key), summaries))")
    end
    for L in cfg.sizes
        selected = filter(s -> s.L == L, summaries)
        println("L=$L negative_samples=$(sum(s.negative_samples for s in selected))/$(length(selected)*nsamples) negative_values=$(sum(s.negative for s in selected))")
    end
    for s in summaries
        s.negative > 0 || continue
        println(s)
    end
    # Independently regenerate one affected sample and check the public entry point.
    found = findfirst(s -> s.negative > 0, summaries)
    if !isnothing(found)
        s = summaries[found]
        replay_point = s.first_negative
        replay_rng = Xoshiro(s.seed)
        disorder = [sample_disorder(replay_rng, s.L, s.h0; field_distribution=cfg.field_distribution)
            for _ in 1:replay_point.sample][end]
        reproduced = if method == :subtract
            replay_F = svd!(RandomTFIM.fermion_matrix(disorder.J, disorder.h))
            replay_full = only(autocorrelation(disorder.J, disorder.h, [replay_point.tau]; offset=false))
            replay_removed = RandomTFIM.ground_transition_weight(replay_F, cld(s.L, 2)) *
                exp(-2minimum(replay_F.S)*replay_point.tau)
            println("Public API terms: full=$replay_full removed=$replay_removed")
            replay_full - replay_removed
        else
            only(autocorrelation(disorder.J, disorder.h, [replay_point.tau]; offset=true))
        end
        @assert reproduced < 0 && isapprox(reproduced, replay_point.value; rtol=1e-10, atol=0)
        println("Public API reproduction: L=$(s.L) h0=$(s.h0) seed=$(s.seed) sample=$(replay_point.sample) tau=$(replay_point.tau) C=$reproduced")
    end
    return summaries
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 2 || error("Usage: check_offset_positivity.jl [samples_per_case] [rank_one|subtract]")
    check_offset_positivity(parse(Int, get(ARGS, 1, "8")); method=Symbol(get(ARGS, 2, "rank_one")))
end
