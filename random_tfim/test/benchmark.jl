# Optional argument: path to a saved, pre-optimization RandomTFIM.jl.
# No new dependencies; run from any directory after the correctness tests.
using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
BLAS.set_num_threads(1)

module Reference
    if !isempty(ARGS)
        path = abspath(only(ARGS))
        # The optional reference path is supplied at runtime, not statically.
        if isfile(path)
            include(path)
        else
            throw(ArgumentError("reference source file not found: $path"))
        end
    end
end

function measure(f)
    f() # compile and warm up
    runs = [@timed f() for _ in 1:3]
    return (ms=round(1000median(x.time for x in runs); digits=3),
        MB=round(median(x.bytes for x in runs)/1e6; digits=3))
end

function benchmark(reference=nothing)
    rng = Xoshiro(941)
    times = collect(0.0:0.25:10.0)
    println((julia=VERSION, julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(), repeats=3))
    for L in (64, 128), boundary in (:open, :periodic)
        J, h = RandomTFIM.sample_disorder(rng, L, 1.0; boundary)
        # Use the same G in both spatial kernels to isolate their cost/error.
        provider = isnothing(reference) ? RandomTFIM : reference
        G = provider.ground_state(J, h; boundary).G
        if !isnothing(reference)
            old = reference.autocorrelation(J, h, times; boundary).C
            new = RandomTFIM.autocorrelation(J, h, times; boundary).C
            @assert isapprox(old, new; atol=1e-10)
            oldspace = reference.correlations(G; boundary)
            newspace = RandomTFIM.correlations(G; boundary)
            for field in (:C, :logC)
                a, b = getproperty(oldspace, field), getproperty(newspace, field)
                @assert all(isequal(x, y) || isapprox(x, y; atol=1e-10) for (x, y) in zip(a, b))
            end
            println((L=L, boundary, max_dynamic_error=maximum(abs.(old-new))))
        end
        versions = isnothing(reference) ? ((:optimized, RandomTFIM),) :
            ((:baseline, reference), (:optimized, RandomTFIM))
        for (version, model) in versions
            println((L=L, boundary, version,
                time=measure(() -> model.autocorrelation(J, h, times; boundary)),
                space=measure(() -> model.correlations(G; boundary)),
                ensemble_two_samples=measure(() -> model.disorder_ensemble(L, 1.0;
                    nsamples=2, boundary, times))))
        end
    end
end

benchmark(isempty(ARGS) ? nothing : Reference.RandomTFIM)
