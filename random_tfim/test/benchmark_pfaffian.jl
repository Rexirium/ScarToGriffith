# Standalone experiment; the production Pfaffian implementation is unchanged.
# julia --startup-file=no --threads=1 random_tfim/test/benchmark_pfaffian.jl [measure|tune|verify]
# Blocked rank-2 updates: https://arxiv.org/abs/1102.3440 (Appendix B).
using LinearAlgebra, Random, Statistics, Test, Printf
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
BLAS.set_num_threads(1)

function blocked_logabspfaffian!(A::Matrix{T}; blocksize=16) where {T<:Union{Float64,ComplexF64}}
    n = size(A, 1)
    X, Y = zeros(T, n, blocksize), zeros(T, n, blocksize)
    row = Vector{T}(undef, n)
    logabs, phase = 0.0, one(T)
    @inbounds for start in 1:2blocksize:n-1
        pairs = min(blocksize, (n-start+1) ÷ 2)
        for r in 1:pairs
            k = start + 2(r-1)
            # Materialize only the pivot row; other trailing updates stay deferred.
            materialize_row!(A, X, Y, row, k, r-1)
            _, offset = findmax(abs, @view A[k, k+1:n])
            p = k + offset
            iszero(A[k, p]) && return (-Inf, zero(T))
            if p != k+1
                for q in 1:n
                    A[k+1, q], A[p, q] = A[p, q], A[k+1, q]
                end
                for q in 1:n
                    A[q, k+1], A[q, p] = A[q, p], A[q, k+1]
                end
                # Pivot permutations also act on the deferred updates.
                for q in 1:r-1
                    X[k+1, q], X[p, q] = X[p, q], X[k+1, q]
                    Y[k+1, q], Y[p, q] = Y[p, q], Y[k+1, q]
                end
                phase = -phase
            end
            materialize_row!(A, X, Y, row, k+1, r-1)
            pivot = A[k, k+1]
            magnitude = abs(pivot)
            logabs += log(magnitude)
            phase *= pivot / magnitude
            for a in k+2:n
                X[a, r] = A[k, a] / pivot
                Y[a, r] = A[k+1, a]
            end
        end
        tail = start+2pairs:n
        if !isempty(tail)
            trailing = @view A[tail, tail]
            x, y = @view(X[tail, 1:pairs]), @view(Y[tail, 1:pairs])
            # Transpose, not adjoint, including for complex time evolution.
            mul!(trailing, x, transpose(y), -one(T), one(T))
            mul!(trailing, y, transpose(x), one(T), one(T))
            for b in tail
                A[b, b] = zero(T)
                for a in first(tail):b-1
                    A[b, a] = -A[a, b]
                end
            end
        end
    end
    return logabs, phase
end

function materialize_row!(A, X, Y, row, k, r)
    ids = k+1:size(A, 1)
    isempty(ids) && return
    @inbounds for b in ids
        row[b] = A[k, b]
    end
    if r > 0
        target = @view row[ids]
        mul!(target, @view(X[ids, 1:r]), @view(Y[k, 1:r]), 1, 1)
        mul!(target, @view(Y[ids, 1:r]), @view(X[k, 1:r]), -1, 1)
    end
    @inbounds for b in ids
        A[k, b] = row[b]
        A[b, k] = -row[b]
    end
end

pfvalue(result) = result[2] * exp(result[1])

function recursive_pfaffian(A)
    n = size(A, 1)
    n == 0 && return one(eltype(A))
    return sum((-1)^j * A[1, j] * recursive_pfaffian(A[[2:j-1; j+1:n], [2:j-1; j+1:n]]) for j in 2:n)
end

function validate_kernel()
    rng = Xoshiro(913)
    @testset "Blocked Pfaffian prototype" begin
        for T in (Float64, ComplexF64), n in (2, 4, 6, 8, 16, 62, 126), bs in (1, 4, 8, 16, 32)
            M = randn(rng, T, n, n)
            A = M - transpose(M)
            reference = RandomTFIM.logabspfaffian!(copy(A))
            actual = blocked_logabspfaffian!(copy(A); blocksize=bs)
            @test actual[1] ≈ reference[1] atol=1e-10
            @test actual[2] ≈ reference[2] atol=1e-10
            @test pfvalue(actual)^2 ≈ det(A) rtol=1e-9
            n <= 8 && @test pfvalue(actual) ≈ recursive_pfaffian(A) rtol=1e-10
        end
        for T in (Float64, ComplexF64), bs in (1, 8, 16, 32)
            @test blocked_logabspfaffian!(zeros(T, 16, 16); blocksize=bs) == (-Inf, zero(T))
            A = zeros(T, 20, 20)
            for k in 1:2:19
                A[k, k+1], A[k+1, k] = T(1e-100), T(-1e-100)
            end
            logabs, phase = blocked_logabspfaffian!(A; blocksize=bs)
            @test logabs ≈ 10log(1e-100)
            @test phase == one(T)
        end
    end
end

# Capture the exact matrices constructed by the current production string routine.
module CaptureRun
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
end
const CAPTURED = Ref{Any}(nothing)
@eval CaptureRun.RandomTFIM function logabspfaffian!(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    push!($(CAPTURED)[], copy(A))
    return $(RandomTFIM.logabspfaffian!)(A)
end

function case_matrices(L, scale)
    matrices = Matrix{typeof(scale)}[]
    CAPTURED[] = matrices
    for h0 in (0.1, 1.0, 10.0), seed in (1996, 941)
        J, h = RandomTFIM.sample_disorder(Xoshiro(seed), L, h0; field_distribution=:fixed)
        F = svd!(RandomTFIM.fermion_matrix(J, h))
        CaptureRun.RandomTFIM.string_autocorrelation(F, [0.1, 10.0, 1e3, 1e4], L÷2, false, scale)
    end
    CAPTURED[] = nothing
    return matrices
end

function batch_time(f, matrices; repeats=5)
    work = [similar(A) for A in matrices]
    function run()
        foreach(pair -> copyto!(pair[1], pair[2]), zip(work, matrices))
        # Copying the destructive input is excluded equally for both algorithms.
        return @elapsed foreach(f, work)
    end
    run()
    return median([run() for _ in 1:repeats])
end

function benchmark_kernels()
    println("KERNEL,L,domain,block_pairs,matrices,current_ms,blocked_ms,speedup,max_abs_error,max_relative_error,max_log_error")
    for L in (32, 64, 128, 256), scale in (1.0, 1.0im)
        matrices = case_matrices(L, scale)
        refs = [RandomTFIM.logabspfaffian!(copy(A)) for A in matrices]
        current = batch_time(RandomTFIM.logabspfaffian!, matrices)
        for bs in (8, 16, 32)
            kernel(A) = blocked_logabspfaffian!(A; blocksize=bs)
            results = [kernel(copy(A)) for A in matrices]
            abs_err = maximum(abs(pfvalue(a)-pfvalue(b)) for (a,b) in zip(results, refs))
            relative = [abs(pfvalue(a)-pfvalue(b))/abs(pfvalue(b)) for (a,b) in zip(results, refs) if abs(pfvalue(b)) >= 1e-280]
            logs = [abs(a[1]-b[1]) for (a,b) in zip(results, refs) if isfinite(a[1]) && isfinite(b[1])]
            elapsed = batch_time(kernel, matrices)
            @printf("KERNEL,%d,%s,%d,%d,%.6f,%.6f,%.3f,%.3e,%.3e,%.3e\n", L,
                scale isa Real ? "imaginary" : "real", bs, length(matrices),
                1000current/length(matrices), 1000elapsed/length(matrices), current/elapsed,
                abs_err, maximum(relative; init=0.0), maximum(logs; init=0.0))
            flush(stdout)
        end
    end
end

# Load a separate model namespace so timing never modifies the baseline kernel.
module BlockedRun
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
end
const USE_BLOCKED = Ref(true)
@eval BlockedRun.RandomTFIM function logabspfaffian!(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    return $(USE_BLOCKED)[] ? $(blocked_logabspfaffian!)(A; blocksize=16) :
        $(RandomTFIM.logabspfaffian!)(A)
end

function paired_times(original, candidate; prepare=()->nothing, repeats=7)
    prepare(); original()
    prepare(); candidate()
    old, new = Float64[], Float64[]
    for i in 1:repeats
        for (f, times) in (isodd(i) ? ((original, old), (candidate, new)) : ((candidate, new), (original, old)))
            prepare()
            push!(times, @elapsed f())
        end
    end
    return median(old), median(new)
end

function measure_fixed_block()
    println("PAIRED_KERNEL,L,domain,matrix_n,matrices,current_ms,blocked_ms,speedup,max_abs_error,max_relative_error,max_log_error")
    for L in (32, 64, 128, 256), scale in (1.0, 1.0im)
        matrices = case_matrices(L, scale)
        refs = [RandomTFIM.logabspfaffian!(copy(A)) for A in matrices]
        vals = [blocked_logabspfaffian!(copy(A)) for A in matrices]
        abs_err = maximum(abs(pfvalue(a)-pfvalue(b)) for (a,b) in zip(vals, refs))
        rel_err = maximum((abs(pfvalue(a)-pfvalue(b))/abs(pfvalue(b)) for (a,b) in zip(vals, refs) if abs(pfvalue(b)) >= 1e-280); init=0.0)
        log_err = maximum((abs(a[1]-b[1]) for (a,b) in zip(vals, refs) if isfinite(a[1]) && isfinite(b[1])); init=0.0)
        @test abs_err < 1e-10
        @test rel_err < 1e-8
        # Also compare logarithms and phases when the scalar Pfaffian underflows.
        for (a, b) in zip(vals, refs)
            @test isfinite(a[1]) == isfinite(b[1])
            if isfinite(b[1])
                @test a[1] ≈ b[1] rtol=1e-10 atol=1e-10
                @test a[2] ≈ b[2] atol=1e-10
            end
        end
        work = [similar(A) for A in matrices]
        prepare() = foreach(pair -> copyto!(pair[1], pair[2]), zip(work, matrices))
        old, new = paired_times(() -> foreach(RandomTFIM.logabspfaffian!, work),
            () -> foreach(blocked_logabspfaffian!, work); prepare)
        @printf("PAIRED_KERNEL,%d,%s,%d,%d,%.6f,%.6f,%.3f,%.3e,%.3e,%.3e\n",
            L, scale isa Real ? "imaginary" : "real", 2L-2, length(matrices),
            1000old/length(matrices), 1000new/length(matrices), old/new, abs_err, rel_err, log_err)
        flush(stdout)
    end
end

function benchmark_autocorrelation()
    ts = 10.0 .^ range(-1, 4; length=101)
    println("AUTOCORRELATION,L,h0,domain,fallback_points,current_ms,blocked_ms,speedup,max_abs_error")
    for L in (128, 256), h0 in (0.1, 1.0, 10.0)
        J, h = RandomTFIM.sample_disorder(Xoshiro(1996), L, h0; field_distribution=:fixed)
        F = svd!(RandomTFIM.fermion_matrix(J, h))
        j = L÷2
        B = RandomTFIM.spin_flip_covariance(F, j)
        transition = (; weight=RandomTFIM.ground_transition_weight(F, j), gap=2minimum(F.S))
        for scale in (1.0, 1.0im)
            # Use identical compiled LU/setup code for both timings. Only the
            # Pfaffian dispatch changes, with the same tiny branch cost in each.
            function run(blocked)
                USE_BLOCKED[] = blocked
                return BlockedRun.RandomTFIM.autocorrelation(F, ts, j, scale; B, offset=true, transition)
            end
            original() = run(false)
            candidate() = run(true)
            a, b = original(), candidate()
            @test a == RandomTFIM.autocorrelation(F, ts, j, scale; B, offset=true, transition)
            @test isapprox(a, b; rtol=1e-9, atol=1e-12)
            CAPTURED[] = Matrix{typeof(scale)}[]
            CaptureRun.RandomTFIM.autocorrelation(F, ts, j, scale; B, offset=true, transition)
            fallback = length(CAPTURED[])
            CAPTURED[] = nothing
            old, new = paired_times(original, candidate; repeats=5)
            @printf("AUTOCORRELATION,%d,%.1f,%s,%d,%.3f,%.3f,%.3f,%.3e\n", L, h0,
                scale isa Real ? "imaginary" : "real", fallback, 1000old, 1000new, old/new, maximum(abs.(a-b)))
            flush(stdout)
        end
    end
end

function verify_physics()
    # Test-process-only replacement; load the existing physics checks without
    # re-including (and thus resetting) the model namespace.
    @eval RandomTFIM function logabspfaffian!(A::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
        return $(blocked_logabspfaffian!)(A; blocksize=16)
    end
    path = joinpath(@__DIR__, "physics.jl")
    source = replace(read(path, String), "include(joinpath(@__DIR__, \"..\", \"RandomTFIM.jl\"))" => "")
    Base.include_string(Main, source, path)
end

function main()
    println((julia=VERSION, cpu=Sys.CPU_NAME, blas=BLAS.get_config(), blas_threads=BLAS.get_num_threads(), julia_threads=Threads.nthreads()))
    validate_kernel()
    mode = isempty(ARGS) ? "measure" : only(ARGS)
    if mode == "verify"
        verify_physics()
    elseif mode == "tune"
        benchmark_kernels()
    elseif mode == "measure"
        measure_fixed_block()
        benchmark_autocorrelation()
    else
        error("expected measure, tune or verify")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
