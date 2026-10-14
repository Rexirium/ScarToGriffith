# Standalone experiment; production dispatch is unchanged.
# julia --project=julia-env/local --startup-file=no --threads=1 random_tfim/test/benchmark_pairing_offset.jl
using LinearAlgebra, Random, Statistics, TOML, SHA, Test
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
using .RandomTFIM
BLAS.set_num_threads(1)

# Float64 adaptation of pairing_reference in tail_reference.jl.
function pairing_setup(F, j)
    L = length(F.S)
    order = sortperm(F.S)
    depth = min(j, L+1-j)
    u = j <= cld(L, 2) ? F.U[1:depth, order] : F.V[L-depth+1:L, order]
    v = j <= cld(L, 2) ? F.V[1:depth-1, order] : F.U[L-depth+2:L, order]
    Gu, Gv = u'*u, v'*v
    P, Q = Gu+Gv-I, Gu-Gv
    P[:, 1], Q[:, 1] = copy(Q[:, 1]), copy(P[:, 1])
    Z = -(P\Q)
    Z = (Z-Z')/2
    beta = 2 .* F.S[order]
    gap = beta[1]
    beta[1] = -gap
    return (; Z, beta, gap, weight=RandomTFIM.ground_transition_weight(F, j))
end

function pairing_curve(p, times; hybrid=false)
    (; Z, beta, gap, weight) = p
    W = similar(Z)
    values = Vector{Float64}(undef, length(times))
    fast = 0
    for (n, tau) in enumerate(times)
        a = 0.0
        for b in axes(Z, 2), k in 1:b-1
            x = Z[k, b]*exp(-(beta[k]+beta[b])*tau/2)
            W[k, b], W[b, k] = x, -x
            a += abs2(x)
        end
        # Avoid 0*exp(+gap*tau) on the swapped mode's zero diagonal.
        for k in axes(Z, 1)
            W[k, k] = 0.0
        end
        if hybrid && a <= 1e-8
            fast += 1
            value = a
        else
            singular = svdvals!(W)
            value = expm1(sum(s -> log1p(s*s), singular)/2)
        end
        values[n] = weight*exp(-gap*tau)*value
    end
    return (; values, fast)
end

function quality(values, reference)
    normal = reference .>= BigFloat(floatmin(Float64))
    representable = reference .>= BigFloat(nextfloat(0.0))
    errors = abs.(BigFloat.(values)./reference .- 1)
    good_errors = errors[normal]
    return Dict("negative"=>count(<(0), values), "zero"=>count(iszero, values),
        "nonfinite"=>count(!isfinite, values), "normal_points"=>count(normal),
        "subnormal_reference_points"=>count(representable .& .!normal),
        "below_float64_points"=>count(.!representable),
        "zero_representable"=>count(iszero.(values) .& representable),
        "zero_normal"=>count(iszero.(values) .& normal),
        "normal_over_1e8"=>count(>(big"1e-8"), good_errors),
        "max_normal_relative_error"=>isempty(good_errors) ? NaN : Float64(maximum(good_errors)),
        "median_normal_relative_error"=>isempty(good_errors) ? NaN : Float64(median(good_errors)))
end

function measure(f; repeats=5)
    f() # Exclude compilation; GC before each independently measured run.
    runs = map(1:repeats) do _
        GC.gc()
        r = @timed f()
        (; seconds=r.time, bytes=r.bytes, allocations=Base.gc_alloc_count(r.gcstats), gc_seconds=r.gctime)
    end
    return Dict("median_seconds"=>median(r.seconds for r in runs),
        "min_seconds"=>minimum(r.seconds for r in runs),
        "median_bytes"=>median(r.bytes for r in runs),
        "median_allocations"=>median(r.allocations for r in runs),
        "median_gc_seconds"=>median(r.gc_seconds for r in runs))
end

function small_checks()
    # Independent spin spectral sum: remove only the first excited eigenstate.
    largest_error = 0.0
    checks = 0
    for L in 3:7, h0 in (0.5, 1.0, 3.0)
        J, h = sample_disorder(Xoshiro(812), L, h0; field_distribution=:fixed)
        H = zeros(2^L, 2^L)
        for s in 0:2^L-1, i in 1:L
            if i < L
                H[s+1, s+1] -= J[i]*(1-2*((s >> (i-1)) & 1))*(1-2*((s >> i) & 1))
            end
            H[xor(s, 1 << (i-1))+1, s+1] -= h[i]
        end
        E = eigen(Symmetric(H))
        F = svd!(RandomTFIM.fermion_matrix(J, h))
        times = [0.0, 0.1, 1.0, 10.0]
        for j in 2:L-1
            z = [1-2*((s >> (j-1)) & 1) for s in 0:2^L-1]
            weights = abs2.(E.vectors'*(z .* E.vectors[:, 1]))
            weights[1:2] .= 0
            expected = [sum(weights .* exp.(-(E.values .- E.values[1])*t)) for t in times]
            p = pairing_setup(F, j)
            for hybrid in (false, true)
                actual = pairing_curve(p, times; hybrid).values
                err = maximum(abs.(actual-expected))
                largest_error = max(largest_error, err)
                @test actual ≈ expected atol=2e-9 rtol=2e-9
                checks += length(times)
            end
        end
    end
    # A zero-overlap reference is outside the Thouless chart; record this limitation.
    failures = Dict{String,Any}[]
    F = svd!(RandomTFIM.fermion_matrix(zeros(3), [1.3, 0.4, 1.7, 0.9]))
    for j in (2, 3)
        try
            values = pairing_curve(pairing_setup(F, j), [0.0, 1.0]).values
            push!(failures, Dict("j"=>j, "values"=>values, "error"=>""))
        catch err
            push!(failures, Dict("j"=>j, "error"=>sprint(showerror, err)))
        end
    end
    return Dict("points"=>checks, "max_absolute_error"=>largest_error, "decoupled"=>failures)
end

function run_experiment()
    output = joinpath(@__DIR__, "..", "results", "pairing_offset_benchmark_20261014.out")
    reference_path = joinpath(@__DIR__, "..", "results", "tail_reference.out")
    reference = TOML.parsefile(reference_path)
    @assert reference["complete"] && length(reference["cases"]) == 24
    report = Dict{String,Any}("complete"=>false, "julia"=>string(VERSION),
        "cpu"=>Sys.CPU_NAME, "blas"=>string(BLAS.get_config()), "blas_threads"=>BLAS.get_num_threads(),
        "julia_threads"=>Threads.nthreads(), "hybrid_threshold"=>1e-8,
        "reference_sha256"=>bytes2hex(sha256(read(reference_path))),
        "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "production_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__, "..", "autocorrelation.jl")))),
        "accuracy"=>Dict{String,Any}[], "positivity"=>Dict{String,Any}[], "performance"=>Dict{String,Any}[])
    save() = open(io -> TOML.print(io, report), output, "w")
    report["small_checks"] = small_checks()
    println("small spin checks passed"); flush(stdout)
    for row in reference["cases"]
        F = svd!(RandomTFIM.fermion_matrix(row["J"], row["h"]))
        j = row["j"]
        p = pairing_setup(F, j)
        refs = parse.(BigFloat, row["reference"])
        times = reference["times"]
        exact = pairing_curve(p, times)
        hybrid = pairing_curve(p, times; hybrid=true)
        current = RandomTFIM.autocorrelation(F, times, j, 1.0; offset=true)
        valid = findall(>(floatmin(Float64)), exact.values)
        difference = isempty(valid) ? 0.0 : maximum(abs.(hybrid.values[valid]./exact.values[valid] .- 1))
        @test difference <= 5.1e-9
        push!(report["accuracy"], Dict("L"=>row["L"], "h0"=>row["h0"], "seed"=>row["seed"],
            "current"=>quality(current, refs), "pairing"=>quality(exact.values, refs),
            "hybrid"=>quality(hybrid.values, refs), "fast_points"=>hybrid.fast,
            "hybrid_vs_pairing_max_relative_error"=>difference,
            "times"=>times, "current_values"=>current, "pairing_values"=>exact.values,
            "hybrid_values"=>hybrid.values))
    end
    save(); println("accuracy: 24 x 201 points completed"); flush(stdout)
    times = collect(10.0 .^ range(-1, 4; length=201))
    report["scan_times"] = times
    for L in (16, 32, 64, 128), h0 in (0.1, 0.3, 1.0, 3.0, 10.0)
        summary = Dict{String,Any}("L"=>L, "h0"=>h0, "samples"=>8, "points"=>1608,
            "setup_failures"=>String[], "hybrid_fast_points"=>0)
        for method in ("current", "pairing", "hybrid")
            summary[method] = Dict("negative"=>0, "zero"=>0, "nonfinite"=>0, "negative_samples"=>0)
        end
        for seed in 1996:2003
            J, h = sample_disorder(Xoshiro(seed), L, h0; field_distribution=:fixed)
            F = svd!(RandomTFIM.fermion_matrix(J, h))
            j = cld(L, 2)
            values = Dict("current"=>RandomTFIM.autocorrelation(F, times, j, 1.0; offset=true))
            try
                p = pairing_setup(F, j)
                values["pairing"] = pairing_curve(p, times).values
                hybrid = pairing_curve(p, times; hybrid=true)
                values["hybrid"] = hybrid.values
                summary["hybrid_fast_points"] += hybrid.fast
            catch err
                push!(summary["setup_failures"], "seed=$seed: $(sprint(showerror, err))")
            end
            for (method, v) in values
                summary[method]["negative"] += count(<(0), v)
                summary[method]["zero"] += count(iszero, v)
                summary[method]["nonfinite"] += count(!isfinite, v)
                summary[method]["negative_samples"] += any(<(0), v)
            end
        end
        push!(report["positivity"], summary)
        println("positivity L=$L h0=$h0 done"); flush(stdout)
    end
    save()
    for L in (32, 64, 128), h0 in (0.1, 1.0, 10.0)
        J, h = sample_disorder(Xoshiro(812), L, h0; field_distribution=:fixed)
        F = svd!(RandomTFIM.fermion_matrix(J, h))
        j = cld(L, 2)
        p = pairing_setup(F, j)
        total_pair(hybrid) = pairing_curve(pairing_setup(svd!(RandomTFIM.fermion_matrix(J, h)), j), times; hybrid)
        r = Dict{String,Any}("L"=>L, "h0"=>h0, "seed"=>812, "nt"=>length(times),
            "fast_points"=>pairing_curve(p, times; hybrid=true).fast,
            "shared_svd"=>measure(() -> svd!(RandomTFIM.fermion_matrix(J, h))),
            "pairing_setup"=>measure(() -> pairing_setup(F, j)),
            "current_given_F"=>measure(() -> RandomTFIM.autocorrelation(F, times, j, 1.0; offset=true)),
            "pairing_given_setup"=>measure(() -> pairing_curve(p, times)),
            "hybrid_given_setup"=>measure(() -> pairing_curve(p, times; hybrid=true)),
            "current_total"=>measure(() -> autocorrelation(J, h, times; j, offset=true)),
            "pairing_total"=>measure(() -> total_pair(false)),
            "hybrid_total"=>measure(() -> total_pair(true)))
        push!(report["performance"], r)
        save(); println("benchmark L=$L h0=$h0 done"); flush(stdout)
    end
    report["complete"] = true
    save()
    println("Saved: $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment()
end
