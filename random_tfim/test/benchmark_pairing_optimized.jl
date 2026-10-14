# Reuse the previous experiment and leave production dispatch unchanged.
# julia --project=julia-env/local --startup-file=no --threads=1 random_tfim/test/benchmark_pairing_optimized.jl
include("benchmark_pairing_offset.jl")
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra: BlasInt, libblastrampoline

# Same DGESDD('N') backend as Julia's svdvals!, with caller-owned buffers.
# ABI follows the installed LinearAlgebra LAPACK wrapper, including character length.
function cached_svdvals!(A::Matrix{Float64}, ws; query=false)
    n = size(A, 1)
    size(A, 2) == n == length(ws.singular) || throw(DimensionMismatch("SVD workspace size"))
    lwork = query ? BlasInt(-1) : BlasInt(length(ws.work))
    ccall((@blasfunc(dgesdd_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
         Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
         Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Clong),
        'N', n, n, A, max(1, n), ws.singular, ws.dummy, 1, ws.dummy, 1,
        ws.work, lwork, ws.iwork, ws.info, 1)
    LAPACK.chklapackerror(ws.info[])
    return ws.singular
end

function svd_workspace(A)
    n = size(A, 1)
    ws = (; singular=zeros(n), dummy=zeros(1), work=zeros(1),
        iwork=zeros(BlasInt, 8n), info=Ref{BlasInt}(0))
    cached_svdvals!(A, ws; query=true)
    resize!(ws.work, round(BlasInt, nextfloat(ws.work[1])))
    return ws
end

function optimized_curve(p, times; factor_exp=true)
    (; Z, beta, gap, weight) = p
    W = zeros(size(Z))
    ws = svd_workspace(W)
    decays = Vector{Float64}(undef, length(beta))
    values = Vector{Float64}(undef, length(times))
    fast = 0
    for (n, tau) in enumerate(times)
        a = 0.0
        if factor_exp
            for k in 2:length(beta)
                decays[k] = exp(-beta[k]*tau/2)
            end
            for b in axes(Z, 2), k in 1:b-1
                # Never form the growing exp(-beta[1]*tau/2).
                x = k == 1 ? Z[k, b]*exp(-(beta[1]+beta[b])*tau/2) :
                    (Z[k, b]*decays[k])*decays[b]
                W[k, b], W[b, k] = x, -x
                a += abs2(x)
            end
        else
            for b in axes(Z, 2), k in 1:b-1
                x = Z[k, b]*exp(-(beta[k]+beta[b])*tau/2)
                W[k, b], W[b, k] = x, -x
                a += abs2(x)
            end
        end
        for k in axes(Z, 1)
            W[k, k] = 0.0
        end
        if a <= 1e-8
            fast += 1
            value = a
        else
            singular = cached_svdvals!(W, ws)
            value = expm1(sum(s -> log1p(s*s), singular)/2)
        end
        values[n] = weight*exp(-gap*tau)*value
    end
    return (; values, fast)
end

function compare_curves(old, new)
    @test all(isfinite, new.values)
    @test all(>=(0), new.values)
    @test old.fast == new.fast
    normal = findall(>=(floatmin(Float64)), old.values)
    err = isempty(normal) ? 0.0 : maximum(abs.(new.values[normal]./old.values[normal] .- 1))
    @test err <= 1e-10
    # Compare the underflow region in absolute units, not unreliable relative units.
    tiny = findall(<(floatmin(Float64)), old.values)
    @test all(abs.(new.values[tiny]-old.values[tiny]) .<= 1e-10*floatmin(Float64))
    return err
end

# Rotate order each repetition to reduce systematic warm-cache/clock bias.
function measure_methods(functions; repeats=7)
    names = sort(collect(keys(functions)))
    foreach(name -> functions[name](), names)
    runs = Dict(name=>Dict{String,Any}[] for name in names)
    for repeat in 1:repeats, k in eachindex(names)
        name = names[mod1(k+repeat-1, length(names))]
        GC.gc()
        r = @timed functions[name]()
        push!(runs[name], Dict("seconds"=>r.time, "bytes"=>r.bytes,
            "allocations"=>Base.gc_alloc_count(r.gcstats), "gc_seconds"=>r.gctime))
    end
    return Dict(name=>Dict("runs"=>runs[name],
        "median_seconds"=>median(r["seconds"] for r in runs[name]),
        "min_seconds"=>minimum(r["seconds"] for r in runs[name]),
        "max_seconds"=>maximum(r["seconds"] for r in runs[name]),
        "median_bytes"=>median(r["bytes"] for r in runs[name]),
        "median_allocations"=>median(r["allocations"] for r in runs[name])) for name in names)
end

function run_optimized_experiment()
    for n in (2, 6, 32), magnitude in (0.0, 1e-150, 1.0, 1e100)
        A = magnitude .* randn(Xoshiro(94), n, n)
        ws = svd_workspace(A)
        expected = svdvals(A)
        @test cached_svdvals!(copy(A), ws) ≈ expected rtol=1e-14
        @test cached_svdvals!(copy(A), ws) ≈ expected rtol=1e-14
    end
    report = Dict{String,Any}("complete"=>false, "julia"=>string(VERSION),
        "cpu"=>Sys.CPU_NAME, "blas"=>string(BLAS.get_config()),
        "blas_threads"=>BLAS.get_num_threads(), "julia_threads"=>Threads.nthreads(),
        "performance"=>Dict{String,Any}[], "accuracy"=>Dict{String,Any}[],
        "source_sha256"=>Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__, file))))
            for file in ("benchmark_pairing_offset.jl", "benchmark_pairing_optimized.jl", "../autocorrelation.jl")))
    reference_path = joinpath(@__DIR__, "..", "results", "tail_reference.out")
    report["reference_sha256"] = bytes2hex(sha256(read(reference_path)))
    reference = TOML.parsefile(reference_path)
    @assert reference["complete"]
    for row in reference["cases"]
        F = svd!(RandomTFIM.fermion_matrix(row["J"], row["h"]))
        p = pairing_setup(F, row["j"])
        times = reference["times"]
        old = pairing_curve(p, times; hybrid=true)
        workspace_only = optimized_curve(p, times; factor_exp=false)
        new = optimized_curve(p, times)
        @test workspace_only.values == old.values
        difference = compare_curves(old, new)
        push!(report["accuracy"], Dict("L"=>row["L"], "h0"=>row["h0"], "seed"=>row["seed"],
            "relative_change"=>difference, "quality"=>quality(new.values, parse.(BigFloat, row["reference"]))))
    end
    println("SVD kernel and all 4824 reference points checked"); flush(stdout)
    # Include zero, unsorted/repeated times, reflection, and both sides of the hybrid threshold.
    for L in (6, 7), j in 2:L-1
        J, h = sample_disorder(Xoshiro(812), L, 1.0; field_distribution=:fixed)
        p = pairing_setup(svd!(RandomTFIM.fermion_matrix(J, h)), j)
        times = [0.0, 100.0, 0.1, 1e4, 0.1]
        compare_curves(pairing_curve(p, times; hybrid=true), optimized_curve(p, times))
    end
    for a in (0.999e-8, 1.001e-8)
        Z = zeros(4, 4)
        Z[1, 2] = Z[3, 4] = sqrt(a/2)
        Z[2, 1] = Z[4, 3] = -sqrt(a/2)
        p = (; Z, beta=zeros(4), gap=0.0, weight=1.0)
        compare_curves(pairing_curve(p, [0.0]; hybrid=true), optimized_curve(p, [0.0]))
    end
    times = collect(10.0 .^ range(-1, 4; length=201))
    output = joinpath(@__DIR__, "..", "results", "pairing_optimized_benchmark_20261014.out")
    report["times"] = times
    for L in (32, 64, 128), h0 in (0.1, 1.0, 10.0)
        J, h = sample_disorder(Xoshiro(812), L, h0; field_distribution=:fixed)
        j = cld(L, 2)
        p = pairing_setup(svd!(RandomTFIM.fermion_matrix(J, h)), j)
        old = pairing_curve(p, times; hybrid=true)
        new = optimized_curve(p, times)
        difference = compare_curves(old, new)
        functions = Dict(
            "production"=>(() -> autocorrelation(J, h, times; j, offset=true)),
            "hybrid_before"=>(() -> pairing_curve(pairing_setup(svd!(RandomTFIM.fermion_matrix(J, h)), j), times; hybrid=true)),
            "workspace_only"=>(() -> optimized_curve(pairing_setup(svd!(RandomTFIM.fermion_matrix(J, h)), j), times; factor_exp=false)),
            "hybrid_optimized"=>(() -> optimized_curve(pairing_setup(svd!(RandomTFIM.fermion_matrix(J, h)), j), times)))
        measurements = measure_methods(functions)
        push!(report["performance"], Dict("L"=>L, "h0"=>h0, "seed"=>812,
            "fast_points"=>new.fast, "relative_change"=>difference, "methods"=>measurements))
        open(io -> TOML.print(io, report), output, "w")
        println("benchmark L=$L h0=$h0 done"); flush(stdout)
    end
    report["complete"] = true
    open(io -> TOML.print(io, report), output, "w")
    println("Saved: $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_optimized_experiment()
end
