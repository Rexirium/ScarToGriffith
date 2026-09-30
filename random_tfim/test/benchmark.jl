using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
BLAS.set_num_threads(1)

function measure(f)
    f() # compile and warm up
    runs = [@timed f() for _ in 1:7]
    return (ms=round(1000median(x.time for x in runs); digits=3),
        MB=round(median(x.bytes for x in runs)/1e6; digits=3))
end

function benchmark()
    rng = Xoshiro(941)
    times = collect(0.0:0.2:20.0)
    println((julia=VERSION, julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(), repeats=7))
    for L in (64, 128), boundary in (:open, :periodic)
        J, h = RandomTFIM.sample_disorder(rng, L, 1.0; boundary)
        G = RandomTFIM.ground_state(J, h; boundary).G
        for time_domain in (:imaginary, :real), j in (boundary == :open ? (1, L÷2, L) : (L÷2,))
            println((L, boundary, time_domain, j,
                time=measure(() -> RandomTFIM.autocorrelation(J, h, times; boundary, j, time_domain))))
        end
        println((L=L, boundary,
            space=measure(() -> RandomTFIM.correlations(G; boundary)),
            ensemble_two_samples=measure(() -> RandomTFIM.disorder_ensemble(L, 1.0, times; nsamples=2, boundary))))
    end
    # Strong fields used to trigger an O(r^3) LU refactorization per prefix.
    for h0 in (0.1, 1.0, 2.0, 5.0, 10.0)
        J, h = RandomTFIM.sample_disorder(Xoshiro(941), 128, h0)
        G = RandomTFIM.ground_state(J, h).G
        println((L=128, h0, space=measure(() -> RandomTFIM.correlations(G))))
    end
end

benchmark()
