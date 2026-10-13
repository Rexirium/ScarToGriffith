# Standalone accuracy reference; does not change production dispatch or precision.
# julia --project=julia-env/precision-check --startup-file=no --threads=1 random_tfim/test/tail_reference.jl [--smoke] [--resume]
using LinearAlgebra, Random, GenericLinearAlgebra, Test, TOML, SHA
include(joinpath(@__DIR__, "..", "RandomTFIM.jl"))
using .RandomTFIM
BLAS.set_num_threads(1)

# Reuse the production scalar Pfaffian kernel in an isolated BigFloat-only module.
module ReferencePfaffian
end
let source = read(joinpath(@__DIR__, "..", "autocorrelation.jl"), String)
    kernel = first(split(source, "function check_rcond_tol"))
    @assert occursin("Union{Float64,ComplexF64}", kernel)
    Base.include_string(ReferencePfaffian,
        replace(kernel, "Union{Float64,ComplexF64}" => "BigFloat"), "reference_pfaffian.jl")
end

function reference_curve(J, h, times)
    K = Bidiagonal(BigFloat.(h), -BigFloat.(J), :L)
    f = svd!(copy(K); tol=100eps(BigFloat))
    residual = norm(Matrix(K)-f.U*Diagonal(f.S)*f.V')/norm(K)
    orthogonality = max(norm(f.U'*f.U-I), norm(f.V'*f.V-I))
    @test max(residual, orthogonality) < sqrt(eps(BigFloat))
    L, j = length(h), cld(length(h), 2)
    N = 2j-1
    M = Matrix{BigFloat}(undef, N, L)
    M[1:2:N, :] = f.U[1:j, :]
    M[2:2:N, :] = f.V[1:j-1, :]
    A = M*M'
    for b in 1:N, a in 1:N
        A[a, b] = iseven(a-b) ? zero(BigFloat) : (isodd(b) ? 1 : -1)*A[a, b]
    end
    weight = RandomTFIM.ground_transition_weight(f, j)
    gap = 2minimum(f.S)
    phased, Q, W = similar(M), similar(A), Matrix{BigFloat}(undef, 2N, 2N)
    values = Vector{BigFloat}(undef, length(times))
    for (n, tau) in enumerate(times)
        for mu in 1:L
            decay = exp(-2f.S[mu]*tau)
            for a in 1:N
                phased[a, mu] = M[a, mu]*decay
            end
        end
        mul!(Q, phased, transpose(M))
        for b in 1:N, a in 1:N
            W[a, b] = W[N+a, N+b] = A[a, b]
            W[a, N+b] = (isodd(b) ? 1 : -1)*Q[a, b]
            W[N+b, a] = -W[a, N+b]
        end
        logabs, phase = ReferencePfaffian.logabspfaffian!(W)
        values[n] = phase*exp(logabs)-weight*exp(-gap*tau)
    end
    return (; values, residual, orthogonality, weight, gap, F=f)
end

function converged_reference(J, h, times)
    previous = nothing
    history = Dict{String,Any}[]
    for bits in (256, 512, 1024, 2048, 4096, 8192, 16384)
        started = time()
        result, change = setprecision(bits) do
            r = reference_curve(J, h, times)
            delta = isnothing(previous) || !all(>(0), r.values) ? BigFloat(Inf) :
                maximum(abs.(previous./r.values .- 1))
            r, delta
        end
        push!(history, Dict("bits"=>bits, "seconds"=>time()-started,
            "relative_change"=>string(change), "nonpositive"=>count(<=(0), result.values),
            "svd_residual"=>string(result.residual), "orthogonality"=>string(result.orthogonality)))
        println((stage="precision", bits, change=Float64(change), seconds=round(time()-started; digits=2)))
        flush(stdout)
        if change < big"1e-25"
            @test all(>(0), result.values)
            @test all(diff(result.values) .<= 0)
            return (; result..., bits, history)
        end
        previous = result.values
        GC.gc()
    end
    error("Reference did not converge through 16384 bits; no unvalidated values are accepted")
end

# Independent spin-Hamiltonian oracle in even/odd global-spin-flip sectors.
# Basis representatives have their highest bit zero; the partner is the bit complement.
function spin_reference(J, h, times)
    L = length(h)
    n, mask = 1 << (L-1), (1 << L)-1
    sectors = map((1, -1)) do parity
        H = zeros(BigFloat, n, n)
        for s in 0:n-1
            for i in 1:L-1
                H[s+1, s+1] -= BigFloat(J[i])*(1-2*((s >> (i-1)) & 1))*(1-2*((s >> i) & 1))
            end
            for i in 1:L
                target = xor(s, 1 << (i-1))
                H[(target < n ? target : xor(target, mask))+1, s+1] -=
                    BigFloat(h[i])*(target < n ? 1 : parity)
            end
        end
        f = eigen(Symmetric(H))
        @test norm(H*f.vectors-f.vectors*Diagonal(f.values))/norm(H) < big"1e-120"
        f
    end
    even, odd = sectors
    j = cld(L, 2)
    z = [1-2*((s >> (j-1)) & 1) for s in 0:n-1]
    weights = abs2.(odd.vectors'*(z.*even.vectors[:, 1]))
    gaps = odd.values .- even.values[1]
    @test abs(sum(weights)-1) < big"1e-120"
    @test gaps[1] > 0
    return [sum(weights[2:end].*exp.(-gaps[2:end].*tau)) for tau in times]
end

# A second fermionic route: particle-hole swap of the lowest mode, then the
# Gaussian pair-state norm. log1p/expm1 avoids subtracting the reference state.
function pairing_reference(F, j, times)
    order = sortperm(F.S)
    u, v = F.U[1:j, order], F.V[1:j-1, order]
    Gu, Gv = u'*u, v'*v
    P, Q = Gu+Gv-I, Gu-Gv
    P[:, 1], Q[:, 1] = copy(Q[:, 1]), copy(P[:, 1])
    Z = -(P\Q)
    Z = (Z-Z')/2
    @test norm(P*Z+Q)/norm(Q) < big"1e-25"
    beta = 2 .* F.S[order]
    gap = beta[1]
    beta[1] = -beta[1]
    weight = RandomTFIM.ground_transition_weight(F, j)
    values = map(times) do tau
        scaled = [Z[a, b]*exp(-(beta[a]+beta[b])*tau/2) for a in eachindex(beta), b in eachindex(beta)]
        singular = svd!(scaled; tol=100eps(BigFloat)).S
        g = sum(s -> log1p(s*s), singular)/2
        weight*exp(-gap*tau)*expm1(g)
    end
    return values
end

function accuracy_row(values, reference)
    errors = abs.(BigFloat.(values)./reference .- 1)
    # Separate unavoidable output underflow from cancellation/algorithm errors.
    representable = reference .>= BigFloat(nextfloat(0.0))
    resolved = findall(representable)
    return Dict("values"=>string.(values), "relative_errors"=>string.(errors),
        "max_relative_error"=>string(maximum(errors)),
        "max_representable_relative_error"=>isempty(resolved) ? "NaN" : string(maximum(errors[resolved])),
        "representable_points"=>length(resolved), "negative_points"=>findall(<(0), values),
        "zero_points"=>findall(iszero, values), "nonfinite_points"=>findall(!isfinite, values),
        "zero_representable_points"=>findall(iszero.(values) .& representable),
        "points_over_1e_8"=>findall((errors .> big"1e-8") .& representable))
end

function run_tail_reference()
    all(arg -> arg in ("--smoke", "--resume"), ARGS) || error("Supported flags: --smoke --resume")
    smoke = "--smoke" in ARGS
    seeds, fields = [1996, 941, 2026, 812], [0.1, 0.3, 1.0, 3.0, 10.0]
    cases = [(L=6, h0=1.0, seed=seed) for seed in seeds]
    append!(cases, [(L=32, h0=h0, seed=seed) for h0 in fields for seed in seeds])
    smoke && (cases = cases[1:1])
    times = collect(10.0 .^ range(2, 4; length=201))
    output = joinpath(@__DIR__, "..", "results", smoke ? "tail_reference_smoke.out" : "tail_reference.out")
    sources = ["model.jl", "autocorrelation.jl"]
    hashes = Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__, "..", file)))) for file in sources)
    hashes["tail_reference.jl"] = bytes2hex(sha256(read(@__FILE__)))
    report = Dict{String,Any}("format_version"=>1, "complete"=>false,
        "julia"=>string(VERSION), "generic_linear_algebra"=>string(pkgversion(GenericLinearAlgebra)),
        "cpu"=>Sys.CPU_NAME, "blas_threads"=>BLAS.get_num_threads(),
        "times"=>times, "source_sha256"=>hashes, "field_distribution"=>"fixed", "field_divisor"=>exp(1),
        "relative_convergence_tolerance"=>"1e-25", "expected_cases"=>length(cases), "cases"=>Dict{String,Any}[])
    if "--resume" in ARGS && isfile(output)
        report = TOML.parsefile(output)
        @assert report["source_sha256"] == hashes && report["times"] == times
        @assert report["expected_cases"] == length(cases)
    end
    for job in cases
        any(c -> (c["L"], c["h0"], c["seed"]) == (job.L, job.h0, job.seed), report["cases"]) && continue
        println((stage="case", job..., completed=length(report["cases"]), total=length(cases))); flush(stdout)
        J, h = sample_disorder(Xoshiro(job.seed), job.L, job.h0; field_distribution=:fixed)
        started = time()
        ref = converged_reference(J, h, times)
        row = setprecision(ref.bits) do
            current = autocorrelation(J, h, times; offset=true)
            f = svd!(RandomTFIM.fermion_matrix(J, h))
            j = cld(job.L, 2)
            old = RandomTFIM.autocorrelation(f, times, j, 1.0)
            RandomTFIM.subtract_ground_transition!(old, f, times, j, 1.0, nothing)
            cache = RandomTFIM.rank_one_cache(f, j)
            fallback = isnothing(cache.X) ? collect(eachindex(times)) :
                last(RandomTFIM.rank_one_autocorrelation(cache, times, 1.0))
            Dict{String,Any}("L"=>job.L, "h0"=>job.h0, "seed"=>job.seed, "j"=>j, "J"=>J, "h"=>h,
                "bits"=>ref.bits, "convergence"=>ref.history,
                "reference"=>string.(ref.values), "log_reference"=>string.(log.(ref.values)),
                "lowest_weight"=>string(ref.weight), "gap"=>string(ref.gap),
                "current_float64"=>accuracy_row(current, ref.values),
                "previous_subtraction_float64"=>accuracy_row(old, ref.values), "fallback_points"=>fallback)
        end
        setprecision(ref.bits) do
            indices = [1, cld(length(times), 2), length(times)]
            pair = pairing_reference(ref.F, cld(job.L, 2), [0.0; times[indices]])
            @test abs(pair[1]+ref.weight-1) < big"1e-25"
            error = maximum(abs.(pair[2:end]./ref.values[indices] .- 1))
            @test error < big"1e-25"
            row["pairing_crosscheck_indices"] = indices
            row["pairing_relative_error"] = string(error)
        end
        if job.L <= 6
            ed = setprecision(512) do
                spin_reference(J, h, times)
            end
            error = setprecision(ref.bits) do
                maximum(abs.(ed./ref.values .- 1))
            end
            @test error < big"1e-25"
            row["spin_ED_relative_error"] = string(error)
        end
        row["seconds"] = time()-started
        push!(report["cases"], row)
        report["complete"] = length(report["cases"]) == length(cases)
        open(output, "w") do io
            TOML.print(io, report)
        end
        println((stage="saved", job..., bits=ref.bits, seconds=round(row["seconds"]; digits=2),
            error=parse(Float64, row["current_float64"]["max_relative_error"]))); flush(stdout)
        GC.gc()
    end
    println("Saved: ", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tail_reference()
end
