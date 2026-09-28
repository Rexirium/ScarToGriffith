# Optional argument: path to a saved, pre-optimization RandomTFIM.jl.
# The reference must support both time_domain=:imaginary and :real.
# No new dependencies; run from any directory after the correctness tests.
using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
BLAS.set_num_threads(1)

module Reference
    if !isempty(ARGS)
        path = abspath(only(ARGS))
        isfile(path) || throw(ArgumentError("reference source file not found: $path"))
        # The optional baseline is loaded at runtime into this module.
        Base.include(@__MODULE__, path)
    end
end

function measure(f)
    f() # compile and warm up
    runs = [@timed f() for _ in 1:7]
    return (ms=round(1000median(x.time for x in runs); digits=3),
        MB=round(median(x.bytes for x in runs)/1e6; digits=3))
end

function check_spatial(old, new)
    old = old isa NamedTuple ? old.C : old # Accept saved sources with the old API.
    @assert size(old) == size(new)
    @assert all(isapprox.(old, new; atol=1e-10, nans=true))
end

function benchmark(reference=nothing)
    rng = Xoshiro(941)
    times = collect(0.0:0.2:20.0)
    versions = isnothing(reference) ? ((:optimized, RandomTFIM),) :
        ((:baseline, reference), (:optimized, RandomTFIM))
    provider = isnothing(reference) ? RandomTFIM : reference
    println((julia=VERSION, julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(), repeats=7))
    for L in (64, 128), boundary in (:open, :periodic)
        J, h = RandomTFIM.sample_disorder(rng, L, 1.0; boundary)
        # Use the same G in both spatial kernels to isolate their cost/error.
        G = provider.ground_state(J, h; boundary).G
        if !isnothing(reference)
            check_spatial(reference.correlations(G; boundary), RandomTFIM.correlations(G; boundary))
        end
        for time_domain in (:imaginary, :real), j in (boundary == :open ? (1, L÷2, L) : (L÷2,))
            if !isnothing(reference)
                old = reference.autocorrelation(J, h, times; boundary, j, time_domain)
                old = real.(old isa NamedTuple ? old.C : old)
                new = RandomTFIM.autocorrelation(J, h, times; boundary, j, time_domain)
                @assert isapprox(old, new; atol=1e-10)
                println((L, boundary, time_domain, j, max_dynamic_error=maximum(abs.(old-new))))
            end
            for (version, model) in versions
                println((L, boundary, time_domain, j, version,
                    time=measure(() -> model.autocorrelation(J, h, times; boundary, j, time_domain))))
            end
        end
        for (version, model) in versions
            println((L=L, boundary, version,
                space=measure(() -> model.correlations(G; boundary)),
                ensemble_two_samples=measure(() -> model.disorder_ensemble(L, 1.0, times; nsamples=2, boundary))))
        end
    end
    # Strong fields used to trigger an O(r^3) LU refactorization per prefix.
    for h0 in (0.1, 1.0, 2.0, 5.0, 10.0)
        J, h = RandomTFIM.sample_disorder(Xoshiro(941), 128, h0)
        G = RandomTFIM.ground_state(J, h).G
        pair = RandomTFIM.correlations(G)
        if !isnothing(reference)
            check_spatial(reference.correlations(G), pair)
        end
        println((L=128, h0, nonpositive=count(<=(0), pair)))
        for (version, model) in versions
            println((L=128, h0, version,
                space=measure(() -> model.correlations(G))))
        end
    end
end

benchmark(isempty(ARGS) ? nothing : getfield(Reference, :RandomTFIM))
